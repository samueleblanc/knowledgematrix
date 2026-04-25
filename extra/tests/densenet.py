#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.densenet import DenseNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestDenseNetRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int, int, tuple[int, ...], int]:
        """
            Returns:
                w: width of the input image
                num_classes: number of classes in the output layer
                growth_rate
                block_config
                num_init_features
        """
        w = random.randint(32, 45)
        num_classes = random.randint(10, 20)
        growth_rate = random.choice([4, 6, 8])
        block_config = tuple(random.randint(1, 3) for _ in range(random.randint(2, 4)))
        num_init_features = random.choice([8, 12, 16])
        return w, num_classes, growth_rate, block_config, num_init_features

    def create_random_model(self) -> tuple[DenseNet, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the DenseNet model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        w, num_classes, growth_rate, block_config, num_init_features = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = DenseNet(
            input_shape=input_shape,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=4,
            drop_rate=0.0,
            compression=0.5,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes

    def test_DenseNetRepBuild(self) -> None:
        """
            Test if building the matrix keeps the network function unchanged
            for randomly configured custom DenseNets.
        """
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, num_classes = self.create_random_model()
            batch_size = random.randint(1, 16)

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
            print(f"Model Architecture:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  -  Batch Size: {batch_size}  -  Total Layers: {model.get_num_layers()}")
            print(f"  -  Concat Destinations: {len(model.concat_skips)}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
            print(f"--------------------------------")

    def test_DenseNetPretrained(self) -> None:
        """
            Test the KM invariant for the pretrained densenet121 with the final
            classifier replaced for a small num_classes.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained densenet121 invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_mem = get_memory_usage()
        start_time = time()

        input_shape = (3, 32, 32)
        num_classes = 10
        x = torch.rand(input_shape)

        model = DenseNet(
            input_shape=input_shape,
            num_classes=num_classes,
            pretrained=True,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=256)
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
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}")
        print(f"Performance:")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
        print(f"  - Relative Error: {diff / torch.norm(forward_pass).item()}")
        print(f"--------------------------------")

    def test_DenseNetPretrainedNonRGB(self) -> None:
        """
            Test the KM invariant for the pretrained densenet121 with the first
            conv replaced for a non-RGB input.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained densenet121 (1-channel input) invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        input_shape = (1, 32, 32)
        num_classes = 1000
        x = torch.rand(input_shape)

        model = DenseNet(
            input_shape=input_shape,
            num_classes=num_classes,
            pretrained=True,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=256)
        mat = matrix_computer.forward(x)
        diff = torch.norm(forward_pass - mat.sum(1)).item()

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )

        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Difference: {diff}  -  Relative Error: {diff / torch.norm(forward_pass).item()}")
        print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
