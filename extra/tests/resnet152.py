#!/usr/bin/env python
"""
    Knowledge-matrix invariant test for ResNet-152 (Bottleneck variant).

    Mirrors ``extra/tests/resnet18.py``: the ResNet-152 wrapper in
    ``knowledgematrix/models/resnet152.py`` uses the same additive
    ``NN.residual(start, end)`` machinery as ResNet-18 (plus the explicit
    post-residual ReLU and the eval()-pinned BatchNorms documented in that
    file). The fundamental invariant checked is

        mat.sum(1) == model.forward(x)

    i.e. the knowledge matrix reconstructs the network's own logits to machine
    epsilon (float64).

    Tractability note
    -----------------
    ``ResNet152`` exposes a FIXED architecture: the custom (non-pretrained)
    build hardcodes ``_RESNET152_LAYERS`` = (3, 8, 36, 3) bottlenecks and the
    final ``Linear(2048, num_classes)``. There is NO knob for smaller layer
    counts / channel widths -- the only tractability levers are the input
    spatial size, the channel count, ``num_classes`` and the KM ``batch_size``.
    The knowledge-matrix cost scales with ``input_size = C*H*W`` (the number of
    matrix columns), so the input spatial size is kept as small as the
    stride/pool reductions allow. The stem downsamples by /4 (7x7 stride-2 conv
    then 3x3 stride-2 maxpool) and layers 2/3/4 each add a stride-2 stage, and
    the final ``AdaptiveAvgPool2d((1, 1))`` collapses whatever spatial size
    remains -- so an 8x8 input is already legal (it collapses to 1x1 before the
    pool). Inputs are therefore drawn from an 8..12 spatial range. Random
    params are seeded for reproducible reported numbers.
"""
import unittest
import torch
import random
import psutil
import gc
from time import time

from torchvision.models import resnet152 as tv_resnet152, ResNet152_Weights

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.resnet152 import ResNet152

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestResNet152Representation(unittest.TestCase):
    def generate_random_params(self) -> tuple[int, int, int]:
        """
            Returns:
                c: number of input channels (1 or 3)
                w: width/height of the (square) input image -- kept small so the
                   KM, whose column count is C*H*W, stays tractable for the full
                   152-layer network.
                num_classes: number of classes in the output layer
        """
        c = random.choice([1, 3])
        w = random.randint(8, 12)
        num_classes = random.randint(5, 15)
        return c, w, num_classes

    def create_random_model(self) -> tuple[ResNet152, torch.Tensor, torch.Tensor, int]:
        """
            Returns:
                model: the ResNet-152 model (custom / non-pretrained build)
                x: the input tensor
                forward_pass: the output of the forward pass
                num_classes: number of classes in the output layer
        """
        c, w, num_classes = self.generate_random_params()
        input_shape = (c, w, w)
        x = torch.rand(input_shape)

        model = ResNet152(
            input_shape=input_shape,
            num_classes=num_classes,
        ).to(DEVICE)
        model.eval()
        forward_pass = model(x)
        model.save = True

        return model, x, forward_pass, num_classes

    def test_ResNet152RepBuild(self) -> None:
        """
            Test that building the knowledge matrix keeps the network function
            unchanged for the custom (randomly sized) ResNet-152.
        """
        random.seed(0)
        torch.manual_seed(0)

        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Test {test_num + 1}/5:")

            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_mem = get_memory_usage()
            start_time = time()

            model, x, forward_pass, num_classes = self.create_random_model()
            batch_size = random.randint(16, 96)

            # Compute output and knowledge matrix
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()
            mem_used = end_mem - start_mem

            rel_err = diff / torch.norm(forward_pass).item()

            # Scale-invariant guard: the real regression check. The custom
            # ResNet-152 output norm at these tiny inputs is small (~1e-1), so
            # an absolute delta of 0.1 would let an all-zeros / badly-broken KM
            # pass. The relative error is ~1e-15 in float64, so 1e-9 has a
            # ~6-order safety margin while still failing on any wiring error.
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
            # comes from NN.get_matrix_shape (ResNet has bias/BatchNorm).
            self.assertEqual(
                tuple(mat.shape),
                (num_classes, x.numel() + 1),
                msg=f"Knowledge matrix shape {tuple(mat.shape)} != "
                    f"{(num_classes, x.numel() + 1)} (num_classes, C*H*W + bias).",
            )

            print(f"Results:")
            print(f"Model Architecture:")
            print(f"  - Input Shape: ({x.shape[0]}, {x.shape[1]}, {x.shape[2]})  - Output Classes: {num_classes}")
            print(f"  -  Batch Size: {batch_size}  -  Total Layers: {model.get_num_layers()}")
            print(f"  -  Residual Destinations: {len(model.residuals)}  -  Matrix Shape: {tuple(mat.shape)}")
            print(f"Performance:")
            print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
            print(f"  - Relative Error: {rel_err}")
            print(f"--------------------------------")

    def test_ResNet152Pretrained(self) -> None:
        """
            Test the KM invariant for the pretrained torchvision resnet152 with
            the final classifier replaced for a small num_classes.

            This exercises the pretrained construction path (introspection of
            torchvision's resnet152 and the real ``downsample`` Sequential
            modules wired into the residual projections), which the custom
            build's auto-generated 1x1-conv+BN projection does NOT cover.

            Precision note: this module sets ``torch.set_default_dtype(float64)``
            at import time, and torchvision casts the loaded checkpoint to the
            current default dtype, so the backbone weights, the aliased
            ``downsample`` projection modules AND the fresh classifier are all
            already float64 by the time the wrapper is built. The ``.double()``
            call below is therefore a defensive no-op kept only for clarity; it
            is NOT what converts the downsample projections. If the weights
            cannot be obtained (e.g. no network and no local cache), the test
            is skipped rather than failed.
        """
        print(f"\n--------------------------------")
        print(f"Pretrained resnet152 invariant test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_mem = get_memory_usage()
        start_time = time()

        input_shape = (3, 8, 8)
        num_classes = 10
        x = torch.rand(input_shape)

        try:
            model = ResNet152(
                input_shape=input_shape,
                num_classes=num_classes,
                pretrained=True,
            ).to(DEVICE)
        except Exception as exc:  # pragma: no cover - network/cache dependent
            self.skipTest(f"Could not obtain pretrained resnet152 weights: {exc}")

        # Defensive no-op: under set_default_dtype(float64) torchvision already
        # loaded every weight (including the downsample projections) as float64,
        # so this cast changes nothing. Kept only to make the precision explicit.
        model = model.double()
        model.eval()
        forward_pass = model(x)
        model.save = True

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=64)
        mat = matrix_computer.forward(x)
        diff = torch.norm(forward_pass - mat.sum(1)).item()

        end_time = time()
        end_mem = get_memory_usage()
        mem_used = end_mem - start_mem

        rel_err = diff / torch.norm(forward_pass).item()

        # Scale-invariant guard (see test_ResNet152RepBuild for rationale).
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
        print(f"  - Total Layers: {model.get_num_layers()}  -  Residual Destinations: {len(model.residuals)}")
        print(f"  - Matrix Shape: {tuple(mat.shape)}")
        print(f"Performance:")
        print(f"  - Difference: {diff}  -  Time: {end_time-start_time:.6f}s  -  Memory: {mem_used:.4f}MB")
        print(f"  - Relative Error: {rel_err}")
        print(f"--------------------------------")

    def test_ResNet152PretrainedRoundtrip(self) -> None:
        """
            Roundtrip-fidelity test: the wrapper's own forward must reproduce
            torchvision's ``resnet152`` logits to machine epsilon.

            This is the key piece of coverage the KM invariant alone CANNOT
            provide: ``mat.sum(1) == model.forward(x)`` only proves the
            knowledge matrix is a faithful linear decomposition of whatever the
            wrapper computes -- it says nothing about whether the wrapper was
            wired up to match the real network. A transposed block, a dropped
            ReLU, a corrupted BatchNorm running-stat, or a mis-ordered
            downsample projection would all keep the invariant satisfied while
            silently changing the function. Comparing against the torchvision
            reference catches exactly those wiring errors.

            CRITICAL: ``ResNet152_Weights.DEFAULT`` is ``IMAGENET1K_V2`` and the
            wrapper loads DEFAULT internally, so the reference MUST also be V2.
            Comparing against V1 produces a false ~1.0 relative mismatch even
            though the wrapper is correct.

            Run in eval() at float64 (set_default_dtype handles the dtype, so
            both the wrapper and the torchvision reference load float64 weights).
        """
        print(f"\n--------------------------------")
        print(f"Pretrained resnet152 roundtrip-fidelity test:")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time()

        input_shape = (3, 64, 64)
        num_classes = 1000
        x = torch.rand(input_shape)

        try:
            model = ResNet152(
                input_shape=input_shape,
                num_classes=num_classes,
                pretrained=True,
            ).to(DEVICE)
            # MUST be V2 (== DEFAULT): the wrapper loads DEFAULT, so a V1
            # reference would give a spurious ~1.0 mismatch.
            reference = tv_resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        except Exception as exc:  # pragma: no cover - network/cache dependent
            self.skipTest(f"Could not obtain pretrained resnet152 weights: {exc}")

        model.eval()
        reference.eval()

        wrapped = model.forward(x)                 # shape (1, 1000)
        ref_out = reference(x.unsqueeze(0))        # shape (1, 1000)

        diff = torch.norm(wrapped - ref_out).item()
        rel_err = diff / torch.norm(ref_out).item()

        end_time = time()

        self.assertLess(
            rel_err,
            1e-6,
            msg=f"Wrapper forward diverges from torchvision resnet152 (V2): "
                f"relative error {rel_err} exceeds 1e-6 (diff={diff}). If this "
                f"is ~1.0, check the reference is IMAGENET1K_V2, not V1.",
        )

        print(f"  - Input Shape: {input_shape}  - Output Classes: {num_classes}")
        print(f"  - Wrapper vs torchvision(V2): Difference: {diff}  -  Relative Error: {rel_err}")
        print(f"  - Time: {end_time-start_time:.6f}s")
        print(f"--------------------------------")


if __name__ == "__main__":
    unittest.main()
