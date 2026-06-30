#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.googlenet import GoogLeNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

# Seed for reproducible reported numbers. GoogLeNet's custom builder has no
# channel-width (`scale`) knob -- channels are always full GoogLeNet width --
# so the only lever for keeping the knowledge matrix tractable is the input
# spatial size (the KM has C*H*W columns, each propagated through the full
# network). We therefore keep H, W tiny.
random.seed(0)
torch.manual_seed(0)


def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestGoogLeNetRepresentation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int]:
        """
            Returns:
                w: width/height of the (square) input image. Kept small
                    because the custom GoogLeNet has full channel width and
                    the KM has C*H*W columns. Every choice still survives the
                    5 stride-2 reductions (conv1, maxpool1, maxpool2, maxpool3,
                    maxpool4) thanks to ceil_mode pooling + AdaptiveAvgPool.
                num_classes: number of classes in the output layer
        """
        w = random.choice([16, 24, 32])
        num_classes = random.randint(5, 15)
        return w, num_classes

    def create_random_model(self) -> tuple[GoogLeNet, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the (custom) GoogLeNet model
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        w, num_classes = self.generate_random_params()
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = GoogLeNet(
            input_shape=input_shape,
            num_classes=num_classes,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes

    def test_GoogLeNetRepBuild(self) -> None:
        """
            Test that the knowledge matrix sums to the network output for
            randomly configured custom GoogLeNet (Inception-v1) models.
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

            rel_err = diff / torch.norm(forward_pass).item()

            # Scale-invariant guard: the real regression check. The custom
            # GoogLeNet output norm is small (~1e-1), so an absolute delta of
            # 0.1 would let an all-zeros / badly-broken KM pass. The relative
            # error is ~1e-16 in float64, so 1e-9 has a ~7-order safety margin
            # while still failing immediately on any genuine wiring error.
            self.assertLess(
                rel_err,
                1e-9,
                msg=f"Relative error {rel_err} exceeds 1e-9; mat.sum(1) does "
                    f"not reconstruct forward_pass (diff={diff}).",
            )
            self.assertAlmostEqual(
                first=diff,
                second=0,
                places=None,
                msg=f"mat.sum(1) and forward_pass differ by {diff}.",
                delta=0.1,
            )
            # Shape: (num_classes rows, C*H*W columns + 1 bias column). The +1
            # comes from NN.get_matrix_shape (GoogLeNet has bias/BatchNorm).
            self.assertEqual(
                tuple(mat.shape),
                (num_classes, x.numel() + 1),
                msg=f"Knowledge matrix shape {tuple(mat.shape)} != "
                    f"{(num_classes, x.numel() + 1)} (num_classes, C*H*W + bias).",
            )

            print(f"Results:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  -  Batch Size: {batch_size}  -  Total Layers: {model.get_num_layers()}")
            print(f"  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}  -  Matrix Shape: {tuple(mat.shape)}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {rel_err}")
            print(f"--------------------------------")

    def test_GoogLeNetPretrained(self) -> None:
        """
            Test the KM invariant for the pretrained torchvision googlenet
            (IMAGENET1K_V1) with the final classifier replaced for a small
            num_classes. This exercises the ``_build_from_pretrained`` path
            (BasicConv2d re-use, ceil_mode maxpools, torchvision Inception
            sub-module re-use) on a small spatial input so the KM stays
            tractable. The pretrained weights are not meaningful at 32x32 --
            we only validate the linear-decomposition invariant.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained googlenet invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_mem = get_memory_usage()
        start_time = time()

        input_shape = (3, 32, 32)
        num_classes = 10
        x = torch.rand(input_shape)

        model = GoogLeNet(
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

        rel_err = diff / torch.norm(forward_pass).item()

        # Scale-invariant guard (see test_GoogLeNetRepBuild for rationale).
        self.assertLess(
            rel_err,
            1e-9,
            msg=f"Relative error {rel_err} exceeds 1e-9; mat.sum(1) does "
                f"not reconstruct forward_pass (diff={diff}).",
        )
        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )
        self.assertEqual(
            tuple(mat.shape),
            (num_classes, x.numel() + 1),
            msg=f"Knowledge matrix shape {tuple(mat.shape)} != "
                f"{(num_classes, x.numel() + 1)} (num_classes, C*H*W + bias).",
        )

        print(f"Results:")
        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}  -  Matrix Shape: {tuple(mat.shape)}")
        print(f"Performance:")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
        print(f"  - Relative Error: {rel_err}")
        print(f"--------------------------------")

    def test_GoogLeNetPretrainedNonRGB(self) -> None:
        """
            Test the KM invariant for the pretrained torchvision googlenet
            with a non-RGB (1-channel) input. This exercises GoogLeNet's
            dedicated first-conv replacement path (``self.layers[0]`` swapped
            for a fresh ``nn.Conv2d(in_channels, stem_out, 7x7 s2 p3)`` when
            ``input_shape[0] != 3``), which the RGB-only pretrained test never
            reaches. Mirrors ``inception.py::test_InceptionPretrainedNonRGB``.
            The spatial size is kept tiny so the KM stays tractable; the
            pretrained weights are not meaningful at 32x32 -- only the
            linear-decomposition invariant is validated.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained googlenet (1-channel input) invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time()

        input_shape = (1, 32, 32)
        num_classes = 1000
        x = torch.rand(input_shape)

        model = GoogLeNet(
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
        rel_err = diff / torch.norm(forward_pass).item()

        # Scale-invariant guard (see test_GoogLeNetRepBuild for rationale).
        self.assertLess(
            rel_err,
            1e-9,
            msg=f"Relative error {rel_err} exceeds 1e-9; mat.sum(1) does "
                f"not reconstruct forward_pass (diff={diff}).",
        )
        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )
        self.assertEqual(
            tuple(mat.shape),
            (num_classes, x.numel() + 1),
            msg=f"Knowledge matrix shape {tuple(mat.shape)} != "
                f"{(num_classes, x.numel() + 1)} (num_classes, C*H*W + bias).",
        )

        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}  -  Matrix Shape: {tuple(mat.shape)}")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Relative Error: {rel_err}")
        print(f"--------------------------------")

    def test_GoogLeNetSmallestWidth(self) -> None:
        """
            Deterministically exercise the smallest input width w=16 for the
            custom GoogLeNet. The docstring of ``generate_random_params``
            claims every drawn width "survives the 5 stride-2 reductions",
            but the module-level seeding never actually draws w=16 (it draws
            {24, 24, 24, 24, 32}). This test pins w=16 so that claim is
            genuinely covered: a 16x16 input is reduced 8 -> 4 -> 2 -> 1 -> 1
            through conv1, maxpool1/2/3/4 and collapses cleanly under the
            final AdaptiveAvgPool2d((1, 1)) without producing a degenerate
            (zero-sized) feature map.
        """
        print(f"\n--------------------------------")
        print(f"Custom googlenet smallest-width (w=16) invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time()

        w = 16
        num_classes = 7
        input_shape = (3, w, w)
        x = torch.rand(input_shape)

        model = GoogLeNet(
            input_shape=input_shape,
            num_classes=num_classes,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=64)
        mat = matrix_computer.forward(x)
        diff = torch.norm(forward_pass - mat.sum(1)).item()

        end_time = time()
        rel_err = diff / torch.norm(forward_pass).item()

        # Scale-invariant guard (see test_GoogLeNetRepBuild for rationale).
        self.assertLess(
            rel_err,
            1e-9,
            msg=f"Relative error {rel_err} exceeds 1e-9; mat.sum(1) does "
                f"not reconstruct forward_pass (diff={diff}).",
        )
        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )
        self.assertEqual(
            tuple(mat.shape),
            (num_classes, x.numel() + 1),
            msg=f"Knowledge matrix shape {tuple(mat.shape)} != "
                f"{(num_classes, x.numel() + 1)} (num_classes, C*H*W + bias).",
        )

        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Total Layers: {model.get_num_layers()}  -  Concat Destinations: {len(model.concat_skips)}  -  Branch Inputs: {len(model.branch_inputs)}  -  Matrix Shape: {tuple(mat.shape)}")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Relative Error: {rel_err}")
        print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
