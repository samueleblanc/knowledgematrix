#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestTransformerRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int]:
        """
            Returns:
                vocab_size: size of the vocabulary
                d_model: dimension of the model
                num_heads: number of attention heads
        """
        vocab_size = random.randint(30,60)
        d_heads = random.randint(3,15)
        num_heads = random.randint(3,5) * 2  # d_model must be divisible by num_heads and by 2
        d_model = num_heads * d_heads
        return vocab_size, d_model, num_heads

    def create_random_model(self) -> tuple[Transformer, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the Transformer model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        vocab_size, d_model, num_heads = self.generate_random_params()
        x = torch.randint(0, vocab_size, (1,1,random.randint(10,20)))
        d_ff = random.randint(10,20)

        model = Transformer(
            vocab_size = vocab_size,
            d_model = d_model,
            d_ff = d_ff,
            num_heads = num_heads
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, vocab_size

    def test_TransformerRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged.
            Test memory usage and time taken.
        """
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")
            
            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_mem = get_memory_usage()
            start_time = time()
            
            model, x, forward_pass, vocab_size = self.create_random_model()
            batch_size = random.randint(16,32)
            
            # Compute output and knowledge matrix
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass.reshape(1,-1) - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()
            mem_used = end_mem - start_mem

            self.assertAlmostEqual(
                first = diff, 
                second = 0, 
                places = None, 
                msg = f"mat.sum(1) and forward_pass differ by {diff}.", 
                delta = 0.1
            )

            print(f"Results:")
            print(f"Model Architecture:")
            print(f"  - Input Length: {x.shape[2]} - Vocab Size: {vocab_size} - D_model: {model.d_model} - Num Heads: {model.num_heads}")
            print(f"  -  Batch Size: {batch_size}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
