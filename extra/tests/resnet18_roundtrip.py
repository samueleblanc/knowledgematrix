#!/usr/bin/env python
"""
    Roundtrip-fidelity test for the ResNet18 KM wrapper.

    The existing invariant test (resnet18.py) only checks the wrapper against
    its OWN forward (mat.sum(1) == model.forward(x)), so it cannot detect an
    architectural mismatch between the wrapper and torchvision's resnet18
    (e.g. missing post-residual-add ReLUs or corrupted BatchNorm running
    stats). This test adds the missing guarantee:

      1. ROUNDTRIP FIDELITY: with pretrained ImageNet weights, the wrapper's
         forward must match torchvision.resnet18's forward to ~machine
         precision (relative error < 1e-6) in float64 on CPU.
      2. KM INVARIANT: mat.sum(1) must still equal model.forward(x) to
         machine epsilon (relative error < 1e-9) on a small input.
"""
import unittest

import torch
import torchvision
from torchvision.models import ResNet18_Weights

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.resnet18 import ResNet18

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"


class TestResNet18Roundtrip(unittest.TestCase):
    def test_roundtrip_fidelity(self) -> None:
        """
            The wrapper (pretrained) must reproduce torchvision.resnet18's
            forward pass to ~machine precision. DEFAULT == IMAGENET1K_V1, so
            both sides carry identical weights; we load V1 explicitly on the
            reference side to be unambiguous.
        """
        torch.manual_seed(0)
        x = torch.rand(3, 64, 64)

        # Reference: a SEPARATE torchvision instance with the same weights.
        # Separate instance => the wrapper's build cannot alias / corrupt it.
        tv = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        tv = tv.to(DEVICE).double().eval()

        # Wrapper with pretrained_model=None loads ResNet18_Weights.DEFAULT
        # internally, which IS IMAGENET1K_V1.
        model = ResNet18(
            input_shape=(3, 64, 64),
            num_classes=1000,
            pretrained=True,
        ).to(DEVICE)
        model.eval()

        with torch.no_grad():
            out_wrapper = model.forward(x)
            out_tv = tv(x.unsqueeze(0))

        diff = torch.norm(out_wrapper - out_tv).item()
        rel = diff / torch.norm(out_tv).item()
        print(f"[roundtrip] ||wrapper - tv|| = {diff:.6e}")
        print(f"[roundtrip] ||tv||          = {torch.norm(out_tv).item():.6e}")
        print(f"[roundtrip] relative error  = {rel:.6e}")
        self.assertLess(
            rel,
            1e-6,
            msg=f"wrapper forward deviates from torchvision.resnet18 by relative {rel}.",
        )

    def test_km_invariant(self) -> None:
        """
            mat.sum(1) must equal model.forward(x) to machine epsilon. Uses a
            small (3,16,16) input to keep the per-position KM sweep fast.
        """
        torch.manual_seed(1)
        input_shape = (3, 16, 16)
        x = torch.rand(input_shape)

        model = ResNet18(
            input_shape=input_shape,
            num_classes=1000,
            pretrained=True,
        ).to(DEVICE)
        model.eval()

        forward_pass = model.forward(x)

        matrix_computer = KnowledgeMatrixComputer(model, batch_size=64)
        mat = matrix_computer.forward(x)

        diff = torch.norm(forward_pass - mat.sum(1)).item()
        rel = diff / torch.norm(forward_pass).item()
        print(f"[invariant] ||forward - mat.sum(1)|| = {diff:.6e}")
        print(f"[invariant] relative error           = {rel:.6e}")
        self.assertLess(
            rel,
            1e-9,
            msg=f"mat.sum(1) and forward differ by relative {rel}.",
        )


if __name__ == "__main__":
    unittest.main()
