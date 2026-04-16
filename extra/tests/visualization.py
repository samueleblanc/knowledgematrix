#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN
from knowledgematrix.visualization import (
    attribution_map,
    top_k_contributors,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class SmallCNN(NN):
    def __init__(self, input_shape, num_classes, save=False, device="cpu"):
        super().__init__(input_shape, save, device)
        self.conv(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=8, out_features=num_classes)


class TestVisualization(unittest.TestCase):

    def _build_model_and_matrix(self, input_shape=(3, 8, 8), num_classes=5):
        model = SmallCNN(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
        model.eval()
        x = torch.rand(input_shape)
        forward_out = model(x).flatten()
        model.save = True
        mc = KnowledgeMatrixComputer(model, batch_size=16)
        A = mc.forward(x)
        return model, x, forward_out, A

    def test_attribution_map_shape(self):
        _, _, _, A = self._build_model_and_matrix(input_shape=(3, 8, 8))
        attr_reduced = attribution_map(A, 0, (3, 8, 8), reduce_channels=True)
        self.assertEqual(attr_reduced.shape, (8, 8))
        attr_full = attribution_map(A, 0, (3, 8, 8), reduce_channels=False)
        self.assertEqual(attr_full.shape, (3, 8, 8))

    def test_attribution_map_single_channel(self):
        _, _, _, A = self._build_model_and_matrix(input_shape=(1, 8, 8))
        attr = attribution_map(A, 0, (1, 8, 8), reduce_channels=True)
        self.assertEqual(attr.shape, (8, 8))

    def test_attribution_sum_matches_output(self):
        """The sum of attributions + bias column should reconstruct the output."""
        for _ in range(3):
            model, x, forward_out, A = self._build_model_and_matrix(input_shape=(3, 8, 8), num_classes=5)
            for j in range(5):
                attr = attribution_map(A, j, (3, 8, 8), reduce_channels=False)
                attr_sum = attr.sum()
                if A.shape[1] > 3 * 8 * 8:
                    attr_sum = attr_sum + A[j, -1]
                diff = abs(forward_out[j].item() - attr_sum.item())
                self.assertAlmostEqual(diff, 0, delta=0.1,
                                       msg=f"Attribution sum differs from output by {diff}")

    def test_top_k_count_and_order(self):
        _, _, _, A = self._build_model_and_matrix()
        k = 10
        values, flat_idx, spatial_idx = top_k_contributors(A, 0, k, (3, 8, 8))
        self.assertEqual(values.shape[0], k)
        self.assertEqual(flat_idx.shape[0], k)
        self.assertEqual(spatial_idx.shape, (k, 3))
        # Verify sorted by descending absolute value
        abs_vals = values.abs()
        for i in range(k - 1):
            self.assertGreaterEqual(abs_vals[i].item(), abs_vals[i + 1].item())

    def test_top_k_spatial_indices(self):
        _, _, _, A = self._build_model_and_matrix(input_shape=(3, 8, 8))
        values, flat_idx, spatial_idx = top_k_contributors(A, 0, 5, (3, 8, 8))
        H, W = 8, 8
        for i in range(5):
            c, h, w = spatial_idx[i]
            expected_flat = c * H * W + h * W + w
            self.assertEqual(flat_idx[i].item(), expected_flat.item())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_overlay_attribution(self):
        from knowledgematrix.visualization import overlay_attribution
        _, x, _, A = self._build_model_and_matrix(input_shape=(3, 8, 8))
        attr = attribution_map(A, 0, (3, 8, 8))
        fig = overlay_attribution(x, attr, show=False)
        import matplotlib.figure
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_per_class_comparison(self):
        from knowledgematrix.visualization import per_class_comparison
        _, _, _, A = self._build_model_and_matrix(input_shape=(3, 8, 8), num_classes=5)
        fig = per_class_comparison(A, [0, 2, 4], (3, 8, 8), show=False)
        self.assertEqual(len(fig.axes), 3)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_layer_contribution(self):
        from knowledgematrix.visualization import layer_contribution
        layers = {"Conv1": torch.randn(10, 10), "FC": torch.randn(5, 10)}
        fig = layer_contribution(layers, show=False)
        import matplotlib.figure
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_layer_contribution_list(self):
        from knowledgematrix.visualization import layer_contribution
        layers = [torch.randn(10, 10), torch.randn(5, 10)]
        fig = layer_contribution(layers, show=False)
        self.assertEqual(len(fig.axes), 1)


if __name__ == "__main__":
    unittest.main()
