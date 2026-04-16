#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class RMSNormMLP(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            d_model: int = 256,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=d_model)
        self.rmsnorm(d_model)
        self.relu()
        self.linear(in_features=d_model, out_features=d_model)
        self.rmsnorm(d_model)
        self.relu()
        self.linear(in_features=d_model, out_features=num_classes)


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestRMSNormRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int, int]:
        w = random.randint(8, 16)
        num_classes = random.randint(2, 20)
        d_model = random.choice([64, 128, 256])
        return w, num_classes, d_model

    def create_random_model(self) -> tuple[RMSNormMLP, torch.Tensor, torch.Tensor, int, int]:
        w, num_classes, d_model = self.generate_random_params()
        input_shape = (1, w, w)
        x = torch.rand(input_shape)

        model = RMSNormMLP(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=d_model
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes, d_model

    def test_RMSNormRepBuild(self) -> None:
        """
            Test if building the matrix with RMSNorm keeps the network function unchanged.
        """
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, num_classes, d_model = self.create_random_model()
            batch_size = random.randint(1, 8)

            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()
            mem_used = end_mem - start_mem

            self.assertAlmostEqual(
                first=diff,
                second=0,
                places=None,
                msg=f"rep and forward_pass differ by {diff}.",
                delta=0.1
            )

            print(f"Results:")
            print(f"Model Architecture:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}  - d_model: {d_model}")
            print(f"  - Batch Size: {batch_size}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
