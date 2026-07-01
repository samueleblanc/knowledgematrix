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


from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
import warnings as _w

class TestSwiGLUInvariant(unittest.TestCase):
    def _invariant(self, m, x):
        m.eval()
        out = m.forward(x)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            mat = KnowledgeMatrixComputer(m).forward(x)
        rel = (torch.norm(out - mat.sum(1)) / torch.norm(out)).item()
        self.assertLess(rel, 1e-9, f"KM invariant broken: rel={rel}")
        self.assertEqual(tuple(mat.shape), (out.reshape(-1).shape[0], x.numel() + 1))

    def test_mlp_with_swiglu(self):
        torch.manual_seed(0)
        m = NN(input_shape=(12,1,1)); m.flatten()
        m.linear(12, 12); m.relu(); m.swiglu(12, 24); m.linear(24, 5)
        self._invariant(m, torch.randn(12, 1, 1))


class TestSwiGLUAlphaEdge(unittest.TestCase):
    def _rel(self, m, x):
        m.eval(); out = m.forward(x)
        import warnings as _w
        with _w.catch_warnings(): _w.simplefilter("ignore")
        mat = KnowledgeMatrixComputer(m).forward(x)
        return (torch.norm(out - mat.sum(1)) / torch.norm(out)).item()

    def test_invariant_holds_across_alpha(self):
        for alpha in (0.0, 0.25, 0.5, 1.0):
            torch.manual_seed(1)
            m = NN(input_shape=(10,1,1)); m.flatten(); m.swiglu(10, 20, alpha=alpha); m.linear(20, 4)
            self.assertLess(self._rel(m, torch.randn(10,1,1)), 1e-9, f"alpha={alpha}")

    def test_zero_input_finite(self):
        torch.manual_seed(2)
        m = NN(input_shape=(10,1,1)); m.flatten(); m.swiglu(10, 20); m.linear(20, 4); m.eval()
        x = torch.zeros(10, 1, 1)
        mat = KnowledgeMatrixComputer(m).forward(x)
        self.assertTrue(torch.isfinite(mat).all())

    def test_warning_emitted(self):
        import warnings as _w
        import knowledgematrix.matrix_computer as mc
        mc._gated_alpha_warned = False
        torch.manual_seed(3)
        m = NN(input_shape=(10,1,1)); m.flatten(); m.swiglu(10, 20); m.linear(20, 4); m.eval()
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            KnowledgeMatrixComputer(m).forward(torch.randn(10,1,1))
        self.assertTrue(any("alpha" in str(x.message) for x in w))


if __name__ == "__main__":
    unittest.main()
