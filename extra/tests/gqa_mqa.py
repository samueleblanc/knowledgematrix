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


if __name__ == "__main__":
    unittest.main()
