#!/usr/bin/env python
"""
    Tests for NN.from_torch — automatic conversion of an in-scope pretrained
    torchvision nn.Module into a KM-computable NN.

    Validation gate (primary): ROUNDTRIP FIDELITY. The converted NN's forward
    must reproduce the source module's forward to ~machine precision (relative
    error < 1e-6) in float64 on CPU. The KM invariant alone is blind to wiring
    (mat.sum(1) == NN.forward holds even against a mis-wired NN), so roundtrip
    is the real gate.

    Validation gate (secondary): KM INVARIANT. mat.sum(1) == NN.forward to
    machine epsilon (relative error < 1e-9) on a small input.

    Negative gates: out-of-MVP architectures (SE-gated MobileNetV3, attention/
    reshape ViT) must raise NotImplementedError rather than silently mis-wire.

    Offline: weights=None (no download). BatchNorm running stats are randomized
    so BN is non-trivial and exact stat transfer is actually exercised.
"""
import unittest

import torch
from torch import nn
import torchvision

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"


def _randomize_bn(module: nn.Module) -> nn.Module:
    """
        Make every BatchNorm non-trivial so the roundtrip actually tests exact
        running-mean / running-var / affine transfer (weights=None leaves BN at
        mean=0, var=1, weight=1, bias=0, i.e. an identity that hides bugs).
    """
    with torch.no_grad():
        for mod in module.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                mod.running_mean.normal_()
                mod.running_var.uniform_(0.5, 1.5)
                if mod.weight is not None:
                    mod.weight.normal_()
                if mod.bias is not None:
                    mod.bias.normal_()
    return module


def _build_source(name: str, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    m = getattr(torchvision.models, name)(weights=None)
    _randomize_bn(m)
    return m.to(DEVICE).double().eval()


class TestFromTorchRoundtrip(unittest.TestCase):
    ROUNDTRIP_MODELS = ("vgg11", "alexnet", "resnet18", "resnet50")

    def _check_roundtrip(self, name: str, shape: tuple) -> float:
        torch.manual_seed(123)
        x = torch.rand(shape, device=DEVICE)
        m = _build_source(name)

        nn_model = NN.from_torch(m, input_shape=shape, device=DEVICE)
        nn_model.eval()

        with torch.no_grad():
            out_nn = nn_model.forward(x)
            out_src = m(x.unsqueeze(0))

        diff = torch.norm(out_nn - out_src).item()
        rel = diff / torch.norm(out_src).item()
        print(f"[roundtrip:{name}] shape={shape} rel_err={rel:.3e} "
              f"||src||={torch.norm(out_src).item():.3e}")
        self.assertLess(
            rel, 1e-6,
            msg=f"from_torch({name}) forward deviates from source by relative {rel}.",
        )
        return rel

    def test_roundtrip_vgg11(self) -> None:
        self._check_roundtrip("vgg11", (3, 64, 64))

    def test_roundtrip_alexnet(self) -> None:
        self._check_roundtrip("alexnet", (3, 64, 64))

    def test_roundtrip_resnet18(self) -> None:
        self._check_roundtrip("resnet18", (3, 64, 64))

    def test_roundtrip_resnet50(self) -> None:
        self._check_roundtrip("resnet50", (3, 64, 64))


class TestFromTorchKMInvariant(unittest.TestCase):
    def _check_invariant(self, name: str, shape: tuple) -> float:
        torch.manual_seed(7)
        x = torch.rand(shape, device=DEVICE)
        m = _build_source(name)

        nn_model = NN.from_torch(m, input_shape=shape, device=DEVICE)
        nn_model.eval()

        forward_pass = nn_model.forward(x)
        km = KnowledgeMatrixComputer(nn_model, batch_size=64)
        mat = km.forward(x)

        diff = torch.norm(forward_pass - mat.sum(1)).item()
        rel = diff / torch.norm(forward_pass).item()
        print(f"[invariant:{name}] shape={shape} rel_err={rel:.3e}")
        self.assertLess(
            rel, 1e-9,
            msg=f"{name}: mat.sum(1) and forward differ by relative {rel}.",
        )
        return rel

    def test_invariant_resnet18(self) -> None:
        self._check_invariant("resnet18", (3, 16, 16))

    def test_invariant_vgg11(self) -> None:
        # vgg11 has 5 stride-2 maxpools, so spatial must survive 2**5: use 32.
        self._check_invariant("vgg11", (3, 32, 32))


class TestFromTorchNegative(unittest.TestCase):
    def test_mobilenet_v3_small_raises(self) -> None:
        m = torchvision.models.mobilenet_v3_small(weights=None).double().eval()
        with self.assertRaises(NotImplementedError):
            NN.from_torch(m, input_shape=(3, 64, 64), device=DEVICE)

    def test_vit_b_16_raises(self) -> None:
        m = torchvision.models.vit_b_16(weights=None).double().eval()
        with self.assertRaises(NotImplementedError):
            NN.from_torch(m, input_shape=(3, 224, 224), device=DEVICE)


if __name__ == "__main__":
    unittest.main()
