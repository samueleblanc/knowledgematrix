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


class CELUMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.celu()
        self.linear(256, 128)
        self.celu()
        self.linear(128, num_classes)


class HardsigmoidMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.hardsigmoid()
        self.linear(256, 128)
        self.hardsigmoid()
        self.linear(128, num_classes)


class HardswishMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.hardswish()
        self.linear(256, 128)
        self.hardswish()
        self.linear(128, num_classes)


class PReLUMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.prelu()
        self.linear(256, 128)
        self.prelu()
        self.linear(128, num_classes)


class ReLU6MLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.relu6()
        self.linear(256, 128)
        self.relu6()
        self.linear(128, num_classes)


class SoftplusMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(self.get_input_size(), 256)
        self.softplus()
        self.linear(256, 128)
        self.softplus()
        self.linear(128, num_classes)


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


MODEL_CLASSES = [
    ("CELU", CELUMLP),
    ("Hardsigmoid", HardsigmoidMLP),
    ("Hardswish", HardswishMLP),
    ("PReLU", PReLUMLP),
    ("ReLU6", ReLU6MLP),
    ("Softplus", SoftplusMLP),
]


class TestActivationRepresentation(unittest.TestCase):

    def _run_activation_test(self, name, model_cls, num_tests=5):
        print(f"\n{'='*50}")
        print(f"  {name} Activation")
        print(f"{'='*50}")

        for test_num in range(num_tests):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/{num_tests}:")

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            w = random.randint(8, 16)
            num_classes = random.randint(2, 20)
            input_shape = (1, w, w)
            x = torch.rand(input_shape)

            start_mem = get_memory_usage()
            start_time = time()

            model = model_cls(
                input_shape=input_shape,
                num_classes=num_classes
            ).to(DEVICE)
            model.eval()
            forward_pass = model(x)
            model.save = True

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
                msg=f"{name}: rep and forward_pass differ by {diff}.",
                delta=0.1
            )

            print(f"Results:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  - Batch Size: {batch_size}")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")

    def test_celu(self):
        self._run_activation_test("CELU", CELUMLP)

    def test_hardsigmoid(self):
        self._run_activation_test("Hardsigmoid", HardsigmoidMLP)

    def test_hardswish(self):
        self._run_activation_test("Hardswish", HardswishMLP)

    def test_prelu(self):
        self._run_activation_test("PReLU", PReLUMLP)

    def test_relu6(self):
        self._run_activation_test("ReLU6", ReLU6MLP)

    def test_softplus(self):
        self._run_activation_test("Softplus", SoftplusMLP)


if __name__ == "__main__":
    unittest.main()
