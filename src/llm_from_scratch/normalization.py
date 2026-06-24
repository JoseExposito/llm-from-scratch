import torch
import torch.nn as nn

from llm_from_scratch.config import NormalizationStrategy


class LayerNorm(nn.Module):
    """Como en el entrenamiento de cualquier red neuronal, durante el
    entrenamiento del LLM puede aparecer problemas de vanishing o exploding
    gradient.

    Esta clase implementa layer normalization para mitigar los problemas con el
    gradiente y mejorar la convergencia de la red.
    Para ello, transforma la salida de la red para que sus valores tengan
    media 0 y varianza 1 (varianza unitaria).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
    """Implementación de Root Mean Square Layer Normalization (RMSNorm).

    A diferencia de LayerNorm, RMSNorm no centra las activaciones restando la
    media, sino que normaliza utilizando únicamente la raíz cuadrada de la media
    de los cuadrados (RMS). Además, no utiliza un parámetro de sesgo (shift).
    Esto reduce el coste computacional y ha demostrado ser igual de efectivo en
    la práctica.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        norm_x = x * torch.rsqrt(rms)
        norm_x = self.scale * norm_x

        return norm_x


def create_norm(
    normalization_strategy: NormalizationStrategy, embedding_dim: int
) -> nn.Module:
    if normalization_strategy == NormalizationStrategy.LAYER_NORM:
        return LayerNorm(embedding_dim)
    elif normalization_strategy == NormalizationStrategy.RMS_NORM:
        return RMSNorm(embedding_dim)

    raise NotImplementedError(
        f"La estrategia de normalización {normalization_strategy} no está soportada"
    )
