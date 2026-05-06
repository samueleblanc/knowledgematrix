#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.inception import Inception

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestInceptionRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int, float]:
        """
            Returns:
                w: width of the input image (>= 75 to satisfy inception-v3 spatial reductions)
                num_classes: number of classes in the output layer
                scale: channel-width multiplier passed to the custom builder
        """
        w = random.choice([75, 80, 96])
        num_classes = random.randint(5, 15)
        scale = random.choice([0.0625, 0.125, 0.1875])
        return w, num_classes, scale

    def create_random_model(self) -> tuple[Inception, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the Inception model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        w, num_classes, scale = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = Inception(
            input_shape=input_shape,
            num_classes=num_classes,
            scale=scale,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes

    def test_InceptionRepBuild(self) -> None:
        """
            Test that the knowledge matrix sums to the network output for
            randomly configured custom Inception-v3-style models.
        """
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, num_classes = self.create_random_model()
            batch_size = random.choice([16, 32, 64, 128])

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
                msg=f"mat.sum(1) and forward_pass differ by {diff}.",
                delta=0.1,
            )

            print(f"Results:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  -  Batch Size: {batch_size}  -  Total Layers: {model.get_num_layers()}")
            print(f"  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")

    def test_InceptionPretrained(self) -> None:
        """
            Test the KM invariant for the pretrained inception_v3 with the
            final classifier replaced for a small num_classes.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained inception_v3 invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_mem = get_memory_usage()
        start_time = time()

        input_shape = (3, 75, 75)
        num_classes = 10
        x = torch.rand(input_shape)

        model = Inception(
            input_shape=input_shape,
            num_classes=num_classes,
            pretrained=True,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=64)
        mat = matrix_computer.forward(x)
        diff = torch.norm(forward_pass - mat.sum(1)).item()

        end_time = time()
        end_mem = get_memory_usage()
        mem_used = end_mem - start_mem

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )

        print(f"Results:")
        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}")
        print(f"Performance:")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
        print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
        print(f"--------------------------------")

    def test_InceptionPretrainedNonRGB(self) -> None:
        """
            Test the KM invariant for the pretrained inception_v3 with the
            first conv replaced for a non-RGB input.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained inception_v3 (1-channel input) invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time()

        input_shape = (1, 75, 75)
        num_classes = 1000
        x = torch.rand(input_shape)

        model = Inception(
            input_shape=input_shape,
            num_classes=num_classes,
            pretrained=True,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=64)
        mat = matrix_computer.forward(x)
        diff = torch.norm(forward_pass - mat.sum(1)).item()

        end_time = time()

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )

        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Relative Error: {diff / torch.norm(forward_pass).item()}")
        print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
