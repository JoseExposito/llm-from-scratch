import torch
import torch.nn as nn


class GELU(nn.Module):
    """Aproximación usada por GPT-2 a la función de activación GELU.
    GELU es similar a ReLU, pero los valores negativos no se transforman siempre
    a 0, si no que están suavizados, evitando la "esquina" que ReLU tiene en el
    0 y mejorando el entrenamiento.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Red neuronal utilizada en los bloques transformer.
    Se compone por dos capas lineales y la función de activación GELU definida
    en este mismo fichero.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)
