#!/usr/bin/env python
import unittest
import torch
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class SmallCNN(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.conv(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=8, out_features=num_classes)


class SmallMLP(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=32)
        self.relu()
        self.linear(in_features=32, out_features=num_classes)


class CNNWithResidual(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.conv(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu()
        start_skip = self.get_num_layers()
        self.conv(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu()
        self.conv(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=16, out_features=num_classes)


def build_cnn(input_shape=(3, 8, 8), num_classes=5) -> tuple[SmallCNN, torch.Tensor]:
    model = SmallCNN(input_shape, num_classes).to(DEVICE)
    model.eval()
    x = torch.rand(input_shape)
    return model, x


def build_mlp(input_shape=(1, 6, 6), num_classes=4) -> tuple[SmallMLP, torch.Tensor]:
    model = SmallMLP(input_shape, num_classes).to(DEVICE)
    model.eval()
    x = torch.rand(input_shape)
    return model, x


def build_cnn_res(input_shape=(3, 10, 10), num_classes=7) -> tuple[CNNWithResidual, torch.Tensor]:
    model = CNNWithResidual(input_shape, num_classes).to(DEVICE)
    model.eval()
    x = torch.rand(input_shape)
    return model, x


class TestExtractWeff(unittest.TestCase):

    def test_shape(self) -> None:
        """W_eff has shape (output_size, input_size), no bias column."""
        model, x = build_cnn()
        model.save = True
        mc = KnowledgeMatrixComputer(model, batch_size=8)
        W_eff = mc.forward(x, extract_weff=True)
        expected = (5, 3 * 8 * 8)
        self.assertEqual(tuple(W_eff.shape), expected,
                         f"W_eff.shape = {tuple(W_eff.shape)}, expected {expected}")
        print(f"[test_shape] W_eff.shape = {tuple(W_eff.shape)}  ✓")

    def test_within_region_linearity(self) -> None:
        """W_eff @ x + b_eff == model(x) at machine epsilon."""
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("CNN+Res", build_cnn_res)]:
            model, x = builder()
            model.save = True
            mc = KnowledgeMatrixComputer(model, batch_size=8)
            A = mc.forward(x)
            W_eff = mc.forward(x, extract_weff=True)
            b_eff = A[:, -1]
            out_pred = W_eff @ x.flatten() + b_eff
            out_true = model.forward(x).flatten()
            diff = torch.norm(out_true - out_pred).item()
            print(f"[test_within_region_linearity] {name}: ‖out_true - (W_eff @ x + b_eff)‖ = {diff:.3e}")
            self.assertTrue(
                torch.allclose(out_true, out_pred, atol=1e-6),
                f"{name}: within-region linearity failed, diff={diff}"
            )

    def test_consistency_with_A(self) -> None:
        """A[:, :-1][:, nonzero] == W_eff[:, nonzero] * x[nonzero]."""
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("CNN+Res", build_cnn_res)]:
            model, x = builder()
            model.save = True
            mc = KnowledgeMatrixComputer(model, batch_size=8)
            A = mc.forward(x)
            W_eff = mc.forward(x, extract_weff=True)
            x_flat = x.flatten()
            nonzero = x_flat.abs() > 1e-8
            diff = torch.norm(
                A[:, :-1][:, nonzero] - W_eff[:, nonzero] * x_flat[nonzero]
            ).item()
            print(f"[test_consistency_with_A] {name}: ‖A[:, :-1] - W_eff * x‖ (nonzero) = {diff:.3e}")
            self.assertTrue(
                torch.allclose(
                    A[:, :-1][:, nonzero],
                    W_eff[:, nonzero] * x_flat[nonzero],
                    atol=1e-6,
                ),
                f"{name}: consistency check failed, diff={diff}"
            )

    def test_zero_pixel_handling(self) -> None:
        """W_eff entries are finite even where x_i = 0."""
        model, x = build_cnn()
        # Force several black pixels (all channels at some spatial locations)
        x[:, 0, 0] = 0.0
        x[:, 3, 3] = 0.0
        x[:, 7, 7] = 0.0
        model.save = True
        mc = KnowledgeMatrixComputer(model, batch_size=8)
        W_eff = mc.forward(x, extract_weff=True)
        self.assertTrue(torch.isfinite(W_eff).all(),
                        "W_eff contains non-finite entries on zero pixels")
        # Black-pixel columns should still equal the within-region slope,
        # so reconstruction must still match the forward pass.
        A = mc.forward(x)
        b_eff = A[:, -1]
        out_pred = W_eff @ x.flatten() + b_eff
        out_true = model.forward(x).flatten()
        diff = torch.norm(out_true - out_pred).item()
        print(f"[test_zero_pixel_handling] finite={True}, reconstruction diff = {diff:.3e}")
        self.assertTrue(torch.allclose(out_true, out_pred, atol=1e-6))

    def test_default_unchanged(self) -> None:
        """Backward compatibility: default forward() is bit-for-bit unchanged."""
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("CNN+Res", build_cnn_res)]:
            model, x = builder()
            model.save = True
            mc = KnowledgeMatrixComputer(model, batch_size=8)
            A_default = mc.forward(x)
            A_explicit = mc.forward(x, extract_weff=False)
            diff = torch.norm(A_default - A_explicit).item()
            print(f"[test_default_unchanged] {name}: ‖A_default - A_explicit‖ = {diff:.3e}")
            self.assertTrue(torch.equal(A_default, A_explicit),
                            f"{name}: default path changed, diff={diff}")

    def test_km_invariant_default(self) -> None:
        """Classic KM invariant: mat.sum(1) == model.forward(x)."""
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("CNN+Res", build_cnn_res)]:
            model, x = builder()
            model.save = True
            mc = KnowledgeMatrixComputer(model, batch_size=8)
            mat = mc.forward(x)
            out = model.forward(x)
            diff = torch.norm(out - mat.sum(1)).item()
            print(f"[test_km_invariant_default] {name}: ‖out - mat.sum(1)‖ = {diff:.3e}")
            self.assertTrue(torch.allclose(out, mat.sum(1), atol=1e-6),
                            f"{name}: KM invariant failed, diff={diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
