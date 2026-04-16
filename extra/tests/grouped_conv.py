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


class DepthwiseSeparableCNN(NN):
    """CNN using depthwise separable convolutions (groups=in_channels)."""

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
        C = input_shape[0]

        # Depthwise conv: groups=in_channels
        self.conv(in_channels=C, out_channels=C, kernel_size=3, stride=1, padding=1, groups=C, bias=False)
        # Pointwise conv
        self.conv(in_channels=C, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu()
        self.adaptiveavgpool((1,1))
        self.flatten()
        self.linear(in_features=32, out_features=num_classes)


class DilatedCNN(NN):
    """CNN using dilated convolutions."""

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
        C = input_shape[0]

        # Dilated conv with dilation=2
        self.conv(in_channels=C, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu()
        # Dilated conv with dilation=4
        self.conv(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu()
        self.adaptiveavgpool((1,1))
        self.flatten()
        self.linear(in_features=32, out_features=num_classes)


class GroupedCNN(NN):
    """CNN using grouped convolutions (like ResNeXt)."""

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
        C = input_shape[0]

        self.conv(in_channels=C, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu()
        # Grouped conv with groups=4
        self.conv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=4)
        self.relu()
        self.adaptiveavgpool((1,1))
        self.flatten()
        self.linear(in_features=32, out_features=num_classes)


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestDepthwiseSeparableConv(unittest.TestCase):
    def test_depthwise_separable(self) -> None:
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Depthwise Separable Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            w = random.randint(16, 32)
            C = random.choice([1, 3])
            num_classes = random.randint(2, 20)
            input_shape = (C, w, w)
            x = torch.rand(input_shape)

            model = DepthwiseSeparableCNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
            model.eval()
            forward_pass = model(x)

            batch_size = random.randint(1, 8)
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1, msg=f"rep and forward_pass differ by {diff}.")

            print(f"  Input: {input_shape}  Classes: {num_classes}  Batch: {batch_size}")
            print(f"  Diff: {diff:.6e}  Time: {end_time-start_time:.4f}s  Mem: {end_mem-start_mem:.2f}MB")


class TestDilatedConv(unittest.TestCase):
    def test_dilated(self) -> None:
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Dilated Conv Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            w = random.randint(16, 32)
            C = random.choice([1, 3])
            num_classes = random.randint(2, 20)
            input_shape = (C, w, w)
            x = torch.rand(input_shape)

            model = DilatedCNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
            model.eval()
            forward_pass = model(x)

            batch_size = random.randint(1, 8)
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1, msg=f"rep and forward_pass differ by {diff}.")

            print(f"  Input: {input_shape}  Classes: {num_classes}  Batch: {batch_size}")
            print(f"  Diff: {diff:.6e}  Time: {end_time-start_time:.4f}s  Mem: {end_mem-start_mem:.2f}MB")


class TestGroupedConv(unittest.TestCase):
    def test_grouped(self) -> None:
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Grouped Conv Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            w = random.randint(16, 32)
            C = random.choice([1, 3])
            num_classes = random.randint(2, 20)
            input_shape = (C, w, w)
            x = torch.rand(input_shape)

            model = GroupedCNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
            model.eval()
            forward_pass = model(x)

            batch_size = random.randint(1, 8)
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1, msg=f"rep and forward_pass differ by {diff}.")

            print(f"  Input: {input_shape}  Classes: {num_classes}  Batch: {batch_size}")
            print(f"  Diff: {diff:.6e}  Time: {end_time-start_time:.4f}s  Mem: {end_mem-start_mem:.2f}MB")


if __name__ == "__main__":
    unittest.main()
