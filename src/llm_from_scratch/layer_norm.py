import torch
import torch.nn as nn


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
