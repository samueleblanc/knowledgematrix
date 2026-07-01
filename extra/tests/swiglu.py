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


if __name__ == "__main__":
    unittest.main()
