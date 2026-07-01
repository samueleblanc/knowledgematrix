import unittest, torch, warnings as _w
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

    def test_out_features_set(self):
        blk = SwiGLU(8, 16)
        self.assertEqual(blk.out_features, 16)   # mirrors nn.Linear

    def test_alpha_out_of_range_warns(self):
        # alpha outside [0,1] is allowed (research) but must WARN (non-convex attribution).
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            SwiGLU(4, 8, alpha=1.5)
        self.assertTrue(any("non-convex" in str(x.message) for x in w))
        # in-range alpha does NOT warn at construction
        with _w.catch_warnings(record=True) as w2:
            _w.simplefilter("always")
            SwiGLU(4, 8, alpha=0.5)
        self.assertFalse(any("non-convex" in str(x.message) for x in w2))

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

class TestSwiGLUInvariant(unittest.TestCase):
    def _invariant(self, m, x, batch_size=1):
        m.eval()
        out = m.forward(x)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            mat = KnowledgeMatrixComputer(m, batch_size=batch_size).forward(x)
        rel = (torch.norm(out - mat.sum(1)) / torch.norm(out)).item()
        self.assertLess(rel, 1e-9, f"KM invariant broken: rel={rel}")
        # matrix shape must match the model's advertised shape (rows = flattened output,
        # cols = inputs + bias/constant column iff the net needs one).
        self.assertEqual(tuple(mat.shape), tuple(m.get_matrix_shape()))
        self.assertEqual(mat.shape[0], out.reshape(-1).shape[0])

    def test_mlp_with_swiglu(self):
        torch.manual_seed(0)
        m = NN(input_shape=(12,1,1)); m.flatten()
        m.linear(12, 12); m.relu(); m.swiglu(12, 24); m.linear(24, 5)
        self._invariant(m, torch.randn(12, 1, 1))

    def test_all_gate_activations_middle(self):
        # invariant for EVERY gate activation, gated block in the middle of the net.
        for idx, act in enumerate(["silu", "gelu", "relu", "sigmoid", "tanh"]):
            torch.manual_seed(300 + idx)
            m = NN(input_shape=(10,1,1)); m.flatten()
            m.linear(10, 14); m.relu()
            m.swiglu(14, 18, activation=act)
            m.linear(18, 12); m.relu()
            m.linear(12, 5)
            self._invariant(m, torch.randn(10, 1, 1))

    def test_bias_free_swiglu_network(self):
        # fully bias=False SwiGLU net (silu: act(0)=0) -> no bias/constant column.
        # Exercises _has_gated_product -> False and the no-bias-column path.
        torch.manual_seed(11)
        m = NN(input_shape=(10,1,1)); m.flatten()
        m.swiglu(10, 16, activation="silu", bias=False)
        self.assertFalse(m._has_gated_product())
        self._invariant(m, torch.randn(10, 1, 1))

    def test_batch_size_gt_one(self):
        torch.manual_seed(13)
        m = NN(input_shape=(12,1,1)); m.flatten()
        m.linear(12, 16); m.relu(); m.swiglu(16, 20); m.linear(20, 6)
        self._invariant(m, torch.randn(12, 1, 1), batch_size=4)

    def test_sigmoid_gate_zero_preactivation_invariant(self):
        # Sigmoid gate with a hidden unit at EXACTLY u==0 (act(0)=0.5). The origin-shift
        # c=act(0) must keep the invariant exact where a plain g/u would drop the row.
        torch.manual_seed(0)
        n, hidden = 4, 6
        m = NN(input_shape=(n,1,1)); m.flatten()
        m.swiglu(n, hidden, activation="sigmoid", bias=False)
        blk = m.layers[-1]
        with torch.no_grad():
            blk.gate_proj.weight.zero_()
            blk.gate_proj.weight[0, 0] = 1.0
            blk.gate_proj.weight[0, 1] = -1.0     # row 0 sums to 0 -> u_0 == 0 on ones input
            for k in range(1, hidden):
                blk.gate_proj.weight[k, 0] = 0.5 * (k + 1)
        x = torch.ones(n, 1, 1)
        # confirm the crafted zero pre-activation is EXACTLY zero
        m.save = True; _ = m.forward(x); m.save = False
        u0 = m.gated_products[len(m.layers) - 1][0].squeeze(0)[0].item()
        self.assertEqual(u0, 0.0)
        # sigmoid has act(0)!=0 -> gated block needs the constant column
        self.assertTrue(m._has_gated_product())
        self._invariant(m, x)


class TestSwiGLUAlphaEdge(unittest.TestCase):
    def _rel(self, m, x):
        m.eval(); out = m.forward(x)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            mat = KnowledgeMatrixComputer(m).forward(x)
        return (torch.norm(out - mat.sum(1)) / torch.norm(out)).item()

    def test_invariant_holds_across_alpha(self):
        for alpha in (0.0, 0.25, 0.5, 1.0):
            torch.manual_seed(1)
            m = NN(input_shape=(10,1,1)); m.flatten(); m.swiglu(10, 20, alpha=alpha); m.linear(20, 4)
            self.assertLess(self._rel(m, torch.randn(10,1,1)), 1e-9, f"alpha={alpha}")

    def test_zero_input_bias_free_invariant(self):
        # bias=False so the gate pre-activation u==0 at the zero input IS hit; assert the
        # INVARIANT (not mere finiteness). Trailing biased linear gives a nonzero output.
        torch.manual_seed(2)
        m = NN(input_shape=(10,1,1)); m.flatten()
        m.swiglu(10, 20, activation="silu", bias=False); m.linear(20, 4)
        self.assertLess(self._rel(m, torch.zeros(10, 1, 1)), 1e-9)

    def test_deep_middle_placement_random_alpha(self):
        # gated block in the MIDDLE of a deep net, several layers AFTER it, RANDOM alpha each trial.
        for trial in range(8):
            torch.manual_seed(100 + trial)
            alpha = float(torch.rand(1).item())
            m = NN(input_shape=(12,1,1)); m.flatten()
            m.linear(12, 16); m.relu()
            m.swiglu(16, 20, alpha=alpha)
            m.linear(20, 16); m.relu()
            m.linear(16, 8); m.gelu()
            m.linear(8, 4)
            self.assertLess(self._rel(m, torch.randn(12,1,1)), 1e-9, f"trial={trial} alpha={alpha}")

    def test_multiple_gated_blocks_random_alpha(self):
        # two SwiGLU blocks, each its OWN random alpha, separated + followed by other layers.
        for trial in range(8):
            torch.manual_seed(200 + trial)
            a1 = float(torch.rand(1).item()); a2 = float(torch.rand(1).item())
            m = NN(input_shape=(10,1,1)); m.flatten()
            m.linear(10, 16); m.relu()
            m.swiglu(16, 24, alpha=a1)
            m.linear(24, 16); m.relu()
            m.swiglu(16, 20, alpha=a2)
            m.linear(20, 6)
            self.assertLess(self._rel(m, torch.randn(10,1,1)), 1e-9, f"trial={trial} a1={a1} a2={a2}")

    def test_alpha_endpoints_select_single_branch(self):
        # alpha=0 -> M_o = diag(g)·M_v (value path only); alpha=1 -> M_o = diag(v)·M_g
        # (gate path only). Tests the attribution feature the row-sum invariant cannot see.
        torch.manual_seed(7)
        n, hidden = 3, 4
        m0 = NN(input_shape=(n,1,1)); m0.flatten()
        m0.swiglu(n, hidden, activation="silu", bias=False, alpha=0.0)   # silu -> c=0, no bias col
        m1 = NN(input_shape=(n,1,1)); m1.flatten()
        m1.swiglu(n, hidden, activation="silu", bias=False, alpha=1.0)
        with torch.no_grad():
            m1.layers[-1].gate_proj.weight.copy_(m0.layers[-1].gate_proj.weight)
            m1.layers[-1].value_proj.weight.copy_(m0.layers[-1].value_proj.weight)
        m0.eval(); m1.eval()
        x = torch.randn(n, 1, 1)
        blk = m0.layers[-1]
        m0.save = True; _ = m0.forward(x); m0.save = False
        u, g, v = (t.squeeze(0) for t in m0.gated_products[len(m0.layers) - 1])
        W1 = blk.gate_proj.weight; W2 = blk.value_proj.weight
        xf = x.reshape(-1)
        M_v_ref = W2 * xf                        # (hidden, n): W2[k,i]*x_i
        ratio = g / u                            # silu: c=0
        M_g_ref = (W1 * xf) * ratio[:, None]     # W1[k,i]*x_i*ratio_k
        expected0 = g[:, None] * M_v_ref         # diag(g)·M_v
        expected1 = v[:, None] * M_g_ref         # diag(v)·M_g
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            A0 = KnowledgeMatrixComputer(m0).forward(x)
            A1 = KnowledgeMatrixComputer(m1).forward(x)
        self.assertLess(torch.norm(A0 - expected0).item(), 1e-9)
        self.assertLess(torch.norm(A1 - expected1).item(), 1e-9)

    def test_warning_emitted(self):
        # asserts the reported alpha VALUE; uses a unique alpha so per-value dedup fires
        # regardless of what ran before -- NO manual reset of internal state.
        torch.manual_seed(3)
        m = NN(input_shape=(10,1,1)); m.flatten(); m.swiglu(10, 20, alpha=0.375); m.linear(20, 4); m.eval()
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            KnowledgeMatrixComputer(m).forward(torch.randn(10,1,1))
        msgs = [str(x.message) for x in w]
        self.assertTrue(any("alpha=0.375" in mm for mm in msgs), msgs)

    def test_two_models_different_alpha_each_warn(self):
        # per-alpha dedup: two models with DIFFERENT alpha each warn, no manual reset.
        torch.manual_seed(5)
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            for a in (0.111, 0.222):
                m = NN(input_shape=(8,1,1)); m.flatten(); m.swiglu(8, 12, alpha=a); m.linear(12, 3); m.eval()
                KnowledgeMatrixComputer(m).forward(torch.randn(8,1,1))
        msgs = [str(x.message) for x in w]
        self.assertTrue(any("alpha=0.111" in mm for mm in msgs), msgs)
        self.assertTrue(any("alpha=0.222" in mm for mm in msgs), msgs)


class TestSwiGLUShape(unittest.TestCase):
    def test_get_matrix_shape_swiglu_terminated(self):
        # SwiGLU as the LAST layer must not raise AttributeError in get_matrix_shape.
        m = NN(input_shape=(8,1,1)); m.flatten(); m.linear(8, 10); m.relu(); m.swiglu(10, 16)
        shape = m.get_matrix_shape()
        self.assertEqual(shape[0], 16)   # terminal SwiGLU out_features
        m.eval(); x = torch.randn(8, 1, 1)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            mat = KnowledgeMatrixComputer(m).forward(x)
        self.assertEqual(tuple(mat.shape), tuple(shape))


class TestLlamaFFN(unittest.TestCase):
    def test_swiglu_ffn_with_rmsnorm(self):
        torch.manual_seed(0)
        dim, hidden = 16, 32
        m = NN(input_shape=(dim,1,1)); m.flatten()
        m.rmsnorm(dim)          # pre-norm
        m.swiglu(dim, hidden)   # SiLU(W1 x) * (W2 x)
        m.linear(hidden, dim)   # down-proj
        m.eval()
        x = torch.randn(dim, 1, 1)
        out = m.forward(x)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            mat = KnowledgeMatrixComputer(m).forward(x)
        rel = (torch.norm(out - mat.sum(1)) / torch.norm(out)).item()
        self.assertLess(rel, 1e-9, f"LLaMA-FFN KM invariant broken: rel={rel}")


if __name__ == "__main__":
    unittest.main()
