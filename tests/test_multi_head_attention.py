import torch
import unittest

from llm_from_scratch.multi_head_attention import MultiHeadAttention
from llm_from_scratch.config import PositionalEmbeddingStrategy


class TestMultiHeadAttention(unittest.TestCase):
    def test_known_values(self):
        torch.manual_seed(123)

        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
            ]
        )

        batch = torch.stack((inputs, inputs), dim=0)
        d_in = inputs.shape[1]
        d_out = 2
        context_length = batch.shape[1]
        dropout = 0.0
        num_heads = 2

        mha = MultiHeadAttention(
            d_in,
            d_out,
            context_length,
            dropout,
            num_heads,
            PositionalEmbeddingStrategy.ABSOLUTE,
        )
        context_vec = mha(batch)

        expected = torch.tensor(
            [
                [
                    [0.3190, 0.4858],
                    [0.2943, 0.3897],
                    [0.2856, 0.3593],
                    [0.2693, 0.3873],
                    [0.2639, 0.3928],
                    [0.2575, 0.4028],
                ],
                [
                    [0.3190, 0.4858],
                    [0.2943, 0.3897],
                    [0.2856, 0.3593],
                    [0.2693, 0.3873],
                    [0.2639, 0.3928],
                    [0.2575, 0.4028],
                ],
            ]
        )

        self.assertTrue(torch.allclose(context_vec, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
