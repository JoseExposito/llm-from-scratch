import torch
import unittest

from llm_from_scratch.normalization import LayerNorm, RMSNorm


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


class TestRMSNorm(unittest.TestCase):
    def test_output_shape(self):
        input = torch.randn(2, 3, 8)
        rms_norm = RMSNorm(8)
        output = rms_norm(input)
        self.assertEqual(output.shape, input.shape)

    def test_unit_rms(self):
        input = torch.tensor([0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000])
        rms_norm = RMSNorm(input.shape[0])
        output = rms_norm(input)

        rms = torch.sqrt(output.pow(2).mean(dim=-1, keepdim=True))

        self.assertEqual(round(rms.item(), 4), 0.9999)


if __name__ == "__main__":
    unittest.main()
