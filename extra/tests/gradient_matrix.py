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


class CNNWithBN(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.conv(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm(8)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=8, out_features=num_classes)


def build_cnn_bn(input_shape=(3, 8, 8), num_classes=5):
    model = CNNWithBN(input_shape, num_classes).to(DEVICE)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.randn_like(m.running_mean)
            m.running_var = torch.rand_like(m.running_var) + 0.5
            with torch.no_grad():
                m.weight.copy_(torch.randn_like(m.weight))
                m.bias.copy_(torch.randn_like(m.bias))
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


class TestFullMatrix(unittest.TestCase):
    def test_km_invariant(self):
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("BN", build_cnn_bn)]:
            model, x = builder()
            gmc = GradientMatrixComputer(model)
            mat = gmc.forward(x)
            out = model.forward(x).flatten()
            diff = torch.norm(out - mat.sum(1)).item()
            print(f"[km_invariant] {name}: ||out - mat.sum(1)|| = {diff:.3e}")
            self.assertTrue(torch.allclose(out, mat.sum(1), atol=1e-6),
                            f"{name}: KM invariant failed, diff={diff}")

    def test_matches_reference_full(self):
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("BN", build_cnn_bn)]:
            model, x = builder()
            mat_grad = GradientMatrixComputer(model).forward(x)
            model.save = True
            mat_ref = KnowledgeMatrixComputer(model, batch_size=8).forward(x)
            self.assertEqual(mat_grad.shape, mat_ref.shape, f"{name}: shape mismatch")
            diff = torch.norm(mat_grad - mat_ref).item()
            print(f"[matches_reference_full] {name}: ||A_grad - A_ref|| = {diff:.3e}")
            self.assertTrue(torch.allclose(mat_grad, mat_ref, atol=1e-8),
                            f"{name}: full-matrix mismatch, diff={diff}")


class TestBackends(unittest.TestCase):
    def test_backend_parity(self):
        for name, builder in [("CNN", build_cnn), ("MLP", build_mlp), ("BN", build_cnn_bn)]:
            model, x = builder()
            A_func = GradientMatrixComputer(model, backend="func").forward(x)
            A_auto = GradientMatrixComputer(model, batch_size=3, backend="autograd").forward(x)
            diff = torch.norm(A_func - A_auto).item()
            print(f"[backend_parity] {name}: ||A_func - A_autograd|| = {diff:.3e}")
            self.assertTrue(torch.allclose(A_func, A_auto, atol=1e-8),
                            f"{name}: backend mismatch, diff={diff}")


class TestEdgeCases(unittest.TestCase):
    def test_zero_pixel_finite(self):
        model, x = build_cnn()
        x[:, 0, 0] = 0.0
        x[:, 3, 3] = 0.0
        gmc = GradientMatrixComputer(model)
        mat = gmc.forward(x)
        self.assertTrue(torch.isfinite(mat).all(), "non-finite entries on zero pixels")
        out = model.forward(x).flatten()
        self.assertTrue(torch.allclose(out, mat.sum(1), atol=1e-6))

    def test_output_shapes(self):
        model, x = build_mlp()
        gmc = GradientMatrixComputer(model)
        d = 1 * 6 * 6
        self.assertEqual(tuple(gmc.forward(x).shape), (4, d + 1))
        self.assertEqual(tuple(gmc.forward(x, extract_weff=True).shape), (4, d))

    def test_requires_eval(self):
        model, x = build_mlp()
        model.train()
        gmc = GradientMatrixComputer(model)  # guard is structural, allowed in train
        with self.assertRaises(RuntimeError):
            gmc.forward(x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
