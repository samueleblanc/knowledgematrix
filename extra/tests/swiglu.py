import unittest, torch
torch.set_default_dtype(torch.float64)
from knowledgematrix.neural_net import NN, SwiGLU

class TestSwiGLUModule(unittest.TestCase):
    def test_forward_matches_reference(self):
        torch.manual_seed(0)
        blk = SwiGLU(8, 16, activation="silu", bias=True, alpha=0.5); blk.eval()
        x = torch.randn(4, 8)
        ref = torch.nn.functional.silu(blk.gate_proj(x)) * blk.value_proj(x)
        self.assertLess(torch.norm(blk(x) - ref).item(), 1e-12)
        self.assertEqual(blk.alpha, 0.5)

    def test_builder_appends(self):
        m = NN(input_shape=(8,1,1)); m.flatten(); m.swiglu(8, 16, alpha=0.25)
        self.assertIsInstance(m.layers[-1], SwiGLU)
        self.assertEqual(m.layers[-1].alpha, 0.25)

class TestSwiGLUSave(unittest.TestCase):
    def test_save_records_ugv(self):
        torch.manual_seed(0)
        m = NN(input_shape=(8,1,1)); m.flatten(); m.swiglu(8, 16); m.eval()
        x = torch.randn(8, 1, 1)
        m.save = True
        _ = m.forward(x)
        i = len(m.layers) - 1
        u, g, v = m.gated_products[i]
        blk = m.layers[i]
        xf = x.reshape(1, -1)
        self.assertLess(torch.norm(u - blk.gate_proj(xf)).item(), 1e-12)
        self.assertLess(torch.norm(g - blk.act(blk.gate_proj(xf))).item(), 1e-12)
        self.assertLess(torch.norm(v - blk.value_proj(xf)).item(), 1e-12)


from knowledgematrix.matrix_computer import _gated_product, DEFAULT_GATED_PRODUCT_ALPHA

class TestGatedProductHelper(unittest.TestCase):
    def test_rowsum_is_g_times_v_any_alpha(self):
        torch.manual_seed(0)
        cols, hid = 5, 7
        M_g = torch.randn(cols, hid); M_v = torch.randn(cols, hid)
        g = M_g.sum(0); v = M_v.sum(0)          # branch "row-sums" over the column axis (dim 0 here)
        # emulate the KM layout: features on last axis, columns on dim 0
        for alpha in (0.0, 0.5, 1.0):
            M_o = _gated_product(M_g, g, M_v, v, alpha)
            self.assertLess(torch.norm(M_o.sum(0) - g * v).item(), 1e-10)
        self.assertEqual(DEFAULT_GATED_PRODUCT_ALPHA, 0.5)


if __name__ == "__main__":
    unittest.main()
