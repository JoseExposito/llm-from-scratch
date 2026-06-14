import torch
import unittest

from llm_from_scratch.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_mean_and_variance(self):
        input = torch.tensor([0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000])
        layer_norm = LayerNorm(input.shape[0])
        output = layer_norm(input)

        mean = output.mean(dim=-1, keepdim=True)
        self.assertEqual(round(mean.item(), 4), 0)

        # La varianza no es exactamente 1 por el valor de epsilon... Pero evita
        # dividir entre 0
        variance = output.var(dim=-1, unbiased=False, keepdim=True)
        self.assertEqual(round(variance.item(), 4), 0.9995)


if __name__ == "__main__":
    unittest.main()
