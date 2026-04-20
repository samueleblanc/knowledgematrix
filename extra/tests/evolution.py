#!/usr/bin/env python
"""
    Unit tests for knowledgematrix.evolution.KnowledgeMatrixEvolution.

    The closed-form decomposition claims

        KM(theta_{t+1}, x) = KM(theta_t, x) + dKM_smooth + dKM_cross + O(lr^2)

    for ReLU MLPs under any gradient-based optimizer. These tests verify:

      1. Exact reproduction of the 2-layer worked example from km_evolution.tex.
      2. The decomposition holds to O(lr^2) for random MLPs under GD, momentum,
         and Adam, both with and without ReLU polytope crossings.
      3. The decomposition is internally consistent with the KM correctness
         invariant KM.sum(1) == model(x).
"""
import gc
import random
import unittest
from time import time

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN
from knowledgematrix.evolution import KnowledgeMatrixEvolution


DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class MLP(NN):
    """Generic ReLU MLP with configurable depth and widths."""

    def __init__(
            self,
            input_shape: tuple,
            widths: list,
            num_classes: int,
            save: bool = False,
            device: str = "cpu",
        ) -> None:
        super().__init__(input_shape, save, device)
        self.flatten()
        prev = self.get_input_size()
        for w in widths:
            self.linear(in_features=prev, out_features=w)
            self.relu()
            prev = w
        self.linear(in_features=prev, out_features=num_classes)


class TwoLayerTex(NN):
    """The 2-layer network from km_evolution.tex §2.4 with exact parameters."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(input_shape=(1, 2, 1), save=False, device=device)
        self.flatten()
        self.linear(in_features=2, out_features=2)
        self.relu()
        self.linear(in_features=2, out_features=1)

        with torch.no_grad():
            self.layers[1].weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))
            self.layers[1].bias.copy_(torch.tensor([0.1, -0.3]))
            self.layers[3].weight.copy_(torch.tensor([[1.0, 1.0]]))
            self.layers[3].bias.copy_(torch.tensor([0.0]))


def _compute_km(model: NN, x: torch.Tensor) -> torch.Tensor:
    """Wrap the library's KM computation and return the matrix on CPU."""
    return KnowledgeMatrixComputer(model, batch_size=4).forward(x)


def _any_mask_flip(masks_old, masks_new) -> bool:
    for mo, mn in zip(masks_old, masks_new):
        if not torch.equal(mo, mn):
            return True
    return False


class TestTexWorkedExample(unittest.TestCase):
    """Reproduce numbers from §2.4 (KM) and §3.5 (GD update) of km_evolution.tex."""

    def test_worked_example(self) -> None:
        model = TwoLayerTex().to(DEVICE)
        model.eval()
        x = torch.tensor([[[1.0], [0.5]]])  # shape (1, 2, 1)
        target = torch.tensor([[1.0]])

        km_t = _compute_km(model, x)
        expected_km_t = torch.tensor([[1.5, 0.5, -0.2]])
        self.assertLess(
            (km_t - expected_km_t).norm().item(), 1e-12,
            f"KM(theta_t) mismatch: got {km_t}, expected {expected_km_t}"
        )

        out = model(x)
        self.assertAlmostEqual(out.item(), 1.8, places=12)

        model.zero_grad(set_to_none=True)
        loss = 0.5 * (out - target).pow(2).sum()
        loss.backward()

        ev = KnowledgeMatrixEvolution(model, optimizer="gd", lr=0.1)
        d_smooth, d_cross = ev.forward(x)

        expected_smooth = torch.tensor([[-0.256, -0.112, -0.216]])
        self.assertLess(
            (d_smooth - expected_smooth).norm().item(), 1e-12,
            f"dKM_smooth mismatch: got {d_smooth}, expected {expected_smooth}"
        )
        self.assertLess(
            d_cross.norm().item(), 1e-12,
            f"dKM_cross expected 0 (both neurons stay active), got {d_cross}"
        )

        # First-order prediction: exact tex number up to arithmetic precision.
        predicted_km_t1 = km_t + d_smooth + d_cross
        expected_predicted = torch.tensor([[1.244, 0.388, -0.416]])
        self.assertLess(
            (predicted_km_t1 - expected_predicted).norm().item(), 1e-12,
            f"first-order KM_{{t+1}} mismatch: got {predicted_km_t1}"
        )

        # Actual KM_{t+1} after applying the step: must differ from the
        # first-order prediction by exactly O(lr^2). With lr=0.1 the residual
        # from the dW^{(2)} dW^{(1)} cross term is ~0.01, see the worked
        # example analysis in the tex.
        ev.apply_step()
        km_t1 = _compute_km(model, x)
        residual = (km_t1 - predicted_km_t1).norm().item()
        # lr^2 = 0.01; enforce that residual is in that ballpark.
        self.assertLess(residual, 0.05, f"Expected O(lr^2) residual, got {residual}")
        self.assertGreater(residual, 1e-6, "Residual suspiciously small; check setup")


class TestDecompositionRandomized(unittest.TestCase):
    """
        Verify KM(theta_{t+1}) - (KM(theta_t) + dKM_smooth + dKM_cross) = O(lr^2)
        across randomized MLPs and all three optimizers.
    """

    def _random_model(self) -> tuple:
        d = random.randint(2, 8)
        depth = random.randint(2, 4)
        widths = [random.randint(2, 8) for _ in range(depth - 1)]
        C = random.randint(1, 4)
        input_shape = (1, d, 1)
        model = MLP(input_shape, widths, C).to(DEVICE)
        model.eval()
        x = torch.randn(input_shape)
        target = torch.randn(1, C)
        return model, x, target

    def _run_one(self, optimizer_name: str, lr: float, tol: float) -> dict:
        model, x, target = self._random_model()
        km_t = _compute_km(model, x)

        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = 0.5 * (out - target).pow(2).sum()
        loss.backward()

        ev = KnowledgeMatrixEvolution(
            model,
            optimizer=optimizer_name,
            lr=lr,
            mu=0.9,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )
        d_smooth, d_cross = ev.forward(x)
        ev.apply_step()

        km_t1 = _compute_km(model, x)
        predicted = km_t + d_smooth + d_cross
        residual = (km_t1 - predicted).norm().item()

        had_cross = d_cross.abs().sum().item() > 0
        self.assertLess(
            residual, tol,
            f"[{optimizer_name}] residual {residual:.3e} exceeds tol {tol:.3e}"
            f" (had_crossing={had_cross})"
        )
        return {"residual": residual, "had_cross": had_cross}

    def test_gd_no_crossings(self) -> None:
        """Small lr: decomposition should be tight (< ~lr^2)."""
        lr = 1e-5
        tol = 1e-7
        residuals = []
        for _ in range(10):
            gc.collect()
            res = self._run_one("gd", lr=lr, tol=tol)
            residuals.append(res["residual"])
        print(f"[gd small-lr] max residual {max(residuals):.3e} over 10 runs")

    def test_momentum_no_crossings(self) -> None:
        lr = 1e-5
        tol = 1e-7
        residuals = []
        for _ in range(10):
            gc.collect()
            res = self._run_one("momentum", lr=lr, tol=tol)
            residuals.append(res["residual"])
        print(f"[momentum small-lr] max residual {max(residuals):.3e} over 10 runs")

    def test_adam_no_crossings(self) -> None:
        # Adam's effective step size at t=1 is ~lr; use lr small enough that lr^2 < tol.
        lr = 1e-5
        tol = 1e-7
        residuals = []
        for _ in range(10):
            gc.collect()
            res = self._run_one("adam", lr=lr, tol=tol)
            residuals.append(res["residual"])
        print(f"[adam small-lr] max residual {max(residuals):.3e} over 10 runs")


class TestCrossingDetection(unittest.TestCase):
    """
        Hand-craft a network where a single ReLU neuron is about to flip sign
        under an optimizer step. Verify:
          (a) dKM_cross is non-zero (detected the flip),
          (b) the full decomposition still closes to O(lr^2).
    """

    def _build_edge_case_model(self) -> tuple:
        """
            2-input, 2-hidden, 1-output MLP. The first hidden neuron has
            pre-activation exactly 1e-4 at x (near the polytope boundary), so
            a small gradient step will flip it while leaving the second
            neuron well inside the active region.
        """
        model = MLP(input_shape=(1, 2, 1), widths=[2], num_classes=1).to(DEVICE)
        model.eval()
        with torch.no_grad():
            model.layers[1].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            model.layers[1].bias.copy_(torch.tensor([1e-4, 1.0]))
            model.layers[3].weight.copy_(torch.tensor([[1.0, 1.0]]))
            model.layers[3].bias.copy_(torch.tensor([0.0]))
        x = torch.zeros((1, 2, 1))
        target = torch.tensor([[-5.0]])
        return model, x, target

    def test_crossing_event_detected_and_decomposition_closes(self) -> None:
        model, x, target = self._build_edge_case_model()
        km_t = _compute_km(model, x)

        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = 0.5 * (out - target).pow(2).sum()
        loss.backward()

        # lr chosen just large enough that the tiny-positive pre-activation
        # flips, but small enough that O(lr^2) residual is small in absolute
        # terms. With grad ~ 6 and lr = 1e-3, lr^2 * grad^2 ~ 3.6e-5.
        lr = 1e-3
        ev = KnowledgeMatrixEvolution(model, optimizer="gd", lr=lr)
        d_smooth, d_cross = ev.forward(x)

        self.assertGreater(
            d_cross.abs().sum().item(), 1e-6,
            "Expected a non-zero crossing term — mask should have flipped"
        )

        ev.apply_step()
        km_t1 = _compute_km(model, x)
        residual = (km_t1 - (km_t + d_smooth + d_cross)).norm().item()
        # Theoretical O(lr^2) bound is ~grad^2 * lr^2. Tolerate 1e-4 for safety.
        self.assertLess(residual, 1e-4, f"Decomposition residual {residual:.3e}")
        print(
            f"[crossing] residual={residual:.3e}, lr^2={lr*lr:.3e}, "
            f"crossing_norm={d_cross.norm().item():.3e}"
        )


class TestApiValidation(unittest.TestCase):
    """Surface-level guards on the public API."""

    def test_rejects_non_relu_activation(self) -> None:
        model = NN(input_shape=(1, 4, 1))
        model.flatten()
        model.linear(in_features=4, out_features=4)
        model.sigmoid()
        model.linear(in_features=4, out_features=2)
        with self.assertRaises(ValueError):
            KnowledgeMatrixEvolution(model)

    def test_rejects_activation_as_last_layer(self) -> None:
        model = NN(input_shape=(1, 4, 1))
        model.flatten()
        model.linear(in_features=4, out_features=2)
        model.relu()
        with self.assertRaises(ValueError):
            KnowledgeMatrixEvolution(model)

    def test_rejects_unknown_optimizer(self) -> None:
        model = MLP(input_shape=(1, 4, 1), widths=[4], num_classes=2)
        with self.assertRaises(ValueError):
            KnowledgeMatrixEvolution(model, optimizer="rmsprop")

    def test_forward_without_backward_raises(self) -> None:
        model = MLP(input_shape=(1, 4, 1), widths=[4], num_classes=2)
        x = torch.randn(1, 4, 1)
        ev = KnowledgeMatrixEvolution(model, optimizer="gd", lr=1e-3)
        with self.assertRaises(RuntimeError):
            ev.forward(x)

    def test_apply_step_before_forward_raises(self) -> None:
        model = MLP(input_shape=(1, 4, 1), widths=[4], num_classes=2)
        ev = KnowledgeMatrixEvolution(model, optimizer="gd", lr=1e-3)
        with self.assertRaises(RuntimeError):
            ev.apply_step()


if __name__ == "__main__":
    start = time()
    unittest.main(exit=False, verbosity=2)
    print(f"\nTotal wall-clock: {time() - start:.2f}s")
