#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.gradient_matrix import GradientMatrixComputer
from knowledgematrix.neural_net import NN

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class SmallMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=32)
        self.relu()
        self.linear(in_features=32, out_features=num_classes)


class GeluMLP(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=32)
        self.gelu()
        self.linear(in_features=32, out_features=num_classes)


class SmallCNN(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.conv(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=8, out_features=num_classes)


def build_mlp(input_shape=(1, 6, 6), num_classes=4):
    model = SmallMLP(input_shape, num_classes).to(DEVICE)
    model.eval()
    x = torch.rand(input_shape)
    return model, x


def build_cnn(input_shape=(3, 8, 8), num_classes=5):
    model = SmallCNN(input_shape, num_classes).to(DEVICE)
    model.eval()
    x = torch.rand(input_shape)
    return model, x


class TestGuard(unittest.TestCase):
    def test_accepts_pl(self):
        model, _ = build_mlp()
        gmc = GradientMatrixComputer(model)
        self.assertEqual(gmc.input_size, 1 * 6 * 6)

    def test_raises_on_smooth(self):
        model = GeluMLP((1, 6, 6), 4).to(DEVICE)
        model.eval()
        with self.assertRaises(ValueError):
            GradientMatrixComputer(model)

    def test_rejects_bad_backend(self):
        model, _ = build_mlp()
        with self.assertRaises(ValueError):
            GradientMatrixComputer(model, backend="nope")


class TestJacobian(unittest.TestCase):
    def test_weff_shape(self):
        model, x = build_cnn()
        gmc = GradientMatrixComputer(model)
        W = gmc.forward(x, extract_weff=True)
        self.assertEqual(tuple(W.shape), (5, 3 * 8 * 8))

    def test_weff_matches_reference(self):
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp)]:
            model, x = builder()
            gmc = GradientMatrixComputer(model)
            W_grad = gmc.forward(x, extract_weff=True)
            model.save = True
            ref = KnowledgeMatrixComputer(model, batch_size=8).forward(x, extract_weff=True)
            diff = torch.norm(W_grad - ref).item()
            print(f"[weff_matches_reference] {name}: ||W_grad - W_ref|| = {diff:.3e}")
            self.assertTrue(torch.allclose(W_grad, ref, atol=1e-8),
                            f"{name}: Jacobian mismatch, diff={diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
