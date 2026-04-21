#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.neural_net import MultiHeadAttention

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


if __name__ == "__main__":
    unittest.main()
