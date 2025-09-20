#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


class CNN1D(NN):

    def __init__(
            self, 
            input_shape: tuple[int],  # (Channels, Length, 1)
            num_classes: int,
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.conv1d(self.input_shape[0], 32, kernel_size=5, stride=1, padding=2)
        self.batchnorm1d(32)
        self.relu()
        self.maxpool1d(2)

        self.conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.batchnorm1d(64)
        self.relu()
        self.maxpool1d(2)

        self.conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm1d(128)
        self.relu()
        self.maxpool1d(2)

        self.adaptiveavgpool1d(3)

        self.flatten()

        self.linear(in_features=128*3, out_features=256)
        self.relu()
        self.dropout(0.5)
        self.linear(in_features=256, out_features=num_classes)


def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestCNN1DRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int]:
        """
            Returns:
                w: width of the input image
                num_classes: number of classes in the output layer
        """
        w = random.randint(10,100)
        num_classes = random.randint(10,20)
        return w, num_classes

    def create_random_model(self) -> tuple[CNN1D, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the AlexNet model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        w, num_classes = self.generate_random_params()
        input_shape = (random.randint(1,3),w,1)
        x = torch.rand(input_shape)

        model = CNN1D(
            input_shape = input_shape,
            num_classes = num_classes
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes

    def test_CNN1DRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged.
            Test memory usage and time taken.
        """
        for test_num in range(10):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/10:")
            
            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_mem = get_memory_usage()
            start_time = time()
            
            model, x, forward_pass, num_classes = self.create_random_model()
            batch_size = random.randint(1,10)
            
            # Compute output and knowledge matrix
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

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
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  -  Batch Size: {batch_size}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
