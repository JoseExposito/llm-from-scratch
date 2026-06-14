import torch
import unittest

from llm_from_scratch.feed_forward import GELU, FeedForward


class TestGELU(unittest.TestCase):
    def test_well_known_values(self):
        input = torch.tensor(
            [
                [-0.1, -1, -15.9876, -10, -200],
                [0, 0.5, 15.98, 200, 1000],
            ]
        )

        expected = torch.tensor(
            [
                [-0.05, -0.16, -0, -0, -0],
                [0, 0.35, 15.98, 200, 1000],
            ]
        )

        gelu = GELU()
        actual = gelu(input)
        self.assertTrue(torch.equal(torch.round(actual, decimals=2), expected))


class TestFeedForward(unittest.TestCase):
    def test_output_dimension(self):
        input = torch.ones(2, 3, 768)

        ff = FeedForward(embedding_dim=768)
        output = ff(input)

        self.assertEqual(output.shape, torch.Size([2, 3, 768]))
