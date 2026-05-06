#!/usr/bin/env python
import unittest
import random
import gc

import torch

from knowledgematrix.neural_net import MultiHeadAttention, NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class TestGQAProjectionShapes(unittest.TestCase):
    def test_gqa_shrinks_kv_projections(self):
        d_model, num_heads, num_kv_heads = 64, 8, 4
        d_head = d_model // num_heads

        mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        self.assertEqual(mha.num_heads, num_heads)
        self.assertEqual(mha.num_kv_heads, num_kv_heads)
        self.assertEqual(mha.d_head, d_head)

        self.assertEqual(mha.Q.weight.shape, (d_model, d_model))
        self.assertEqual(mha.K.weight.shape, (num_kv_heads * d_head, d_model))
        self.assertEqual(mha.V.weight.shape, (num_kv_heads * d_head, d_model))
        self.assertEqual(mha.O.weight.shape, (d_model, d_model))

    def test_mqa_single_kv_head(self):
        d_model, num_heads = 64, 8
        d_head = d_model // num_heads

        mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=1,
        )

        self.assertEqual(mha.K.weight.shape, (d_head, d_model))
        self.assertEqual(mha.V.weight.shape, (d_head, d_model))


class TestMHABackwardCompat(unittest.TestCase):
    def test_default_num_kv_heads_matches_num_heads(self):
        d_model, num_heads = 64, 8

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.assertEqual(mha.num_kv_heads, num_heads)
        self.assertEqual(mha.kv_repeat, 1)
        self.assertEqual(mha.K.weight.shape, (d_model, d_model))
        self.assertEqual(mha.V.weight.shape, (d_model, d_model))

    def test_forward_output_matches_explicit_equal_kv_heads(self):
        torch.manual_seed(0)
        d_model, num_heads = 32, 4

        mha_default = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        mha_explicit = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
        )
        mha_explicit.Q.load_state_dict(mha_default.Q.state_dict())
        mha_explicit.K.load_state_dict(mha_default.K.state_dict())
        mha_explicit.V.load_state_dict(mha_default.V.state_dict())
        mha_explicit.O.load_state_dict(mha_default.O.state_dict())

        x = torch.randn(1, 1, 6, d_model)
        out_default = mha_default(x)
        out_explicit = mha_explicit(x)

        self.assertTrue(torch.allclose(out_default, out_explicit, atol=1e-10))


class TestMHAValidation(unittest.TestCase):
    def test_num_heads_not_divisible_by_num_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=64, num_heads=8, num_kv_heads=3)

    def test_non_positive_num_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=64, num_heads=8, num_kv_heads=0)


class TestMultiheadattentionBuilder(unittest.TestCase):
    def test_builder_passes_num_kv_heads(self):
        from knowledgematrix.neural_net import NN

        net = NN(input_shape=(1, 1, 32))
        net.multiheadattention(d_model=32, num_heads=8, num_kv_heads=2)

        layer = net.layers[-1]
        self.assertIsInstance(layer, MultiHeadAttention)
        self.assertEqual(layer.num_heads, 8)
        self.assertEqual(layer.num_kv_heads, 2)

    def test_builder_defaults_num_kv_heads_to_num_heads(self):
        from knowledgematrix.neural_net import NN

        net = NN(input_shape=(1, 1, 32))
        net.multiheadattention(d_model=32, num_heads=4)

        layer = net.layers[-1]
        self.assertEqual(layer.num_kv_heads, 4)


class GQAMiniTransformer(NN):
    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads, seq_len, save=False, device="cpu"):
        super().__init__(input_shape=(1, 1, d_model), save=save, device=device)
        self._seq_len = seq_len
        self.embedding(vocab_size, d_model)
        self.positionalencoding(d_model, max_len=seq_len + 1)

        start = self.get_num_layers()
        self.multiheadattention(d_model, num_heads, num_kv_heads=num_kv_heads)
        end = self.get_num_layers()
        self.residual(start, end)
        self.layernorm(d_model)

        self.linear(in_features=d_model, out_features=vocab_size)
        self.softmax()


class TestGQAKnowledgeMatrixInvariant(unittest.TestCase):
    def test_gqa_kv_invariant(self):
        torch.manual_seed(0)
        random.seed(0)
        vocab_size = 40
        num_heads = 8
        num_kv_heads = 4
        d_head = 4
        d_model = num_heads * d_head
        seq_len = 10

        model = GQAMiniTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
        ).to(DEVICE)
        model.eval()

        x = torch.randint(0, vocab_size, (1, 1, seq_len))
        forward_pass = model(x)
        model.save = True

        computer = KnowledgeMatrixComputer(model, batch_size=8)
        mat = computer.forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )
        gc.collect()

    def test_mqa_kv_invariant(self):
        torch.manual_seed(1)
        random.seed(1)
        vocab_size = 40
        num_heads = 8
        num_kv_heads = 1
        d_head = 4
        d_model = num_heads * d_head
        seq_len = 10

        model = GQAMiniTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
        ).to(DEVICE)
        model.eval()

        x = torch.randint(0, vocab_size, (1, 1, seq_len))
        forward_pass = model(x)
        model.save = True

        computer = KnowledgeMatrixComputer(model, batch_size=8)
        mat = computer.forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"mat.sum(1) and forward_pass differ by {diff}.",
            delta=0.1,
        )
        gc.collect()


if __name__ == "__main__":
    unittest.main()
