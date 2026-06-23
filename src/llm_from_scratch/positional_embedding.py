import torch


def precompute_rope_params(
    head_dim: int, context_length: int, theta_base: float = 10_000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precalcula las matrices de coseno y seno utilizadas por RoPE.

    RoPE (Rotary Position Embedding) codifica la posición de cada token
    aplicando una rotación a los vectores de query y key en el mecanismo de
    atención. Las frecuencias de rotación se calculan a partir del índice de
    cada dimensión del embedding.

    Args:
        head_dim: Número de dimensiones de cada cabeza de atención. Debe ser
            par.
        context_length: Número máximo de tokens en la secuencia.
        theta_base: Base utilizada para calcular las frecuencias inversas.

    Returns:
        Tupla (cos, sin) con forma (context_length, head_dim).
    """
    assert head_dim % 2 == 0, "head_dim debe ser par"

    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    positions = torch.arange(context_length)

    # Shape: (context_length, head_dim // 2)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica la rotación RoPE a un tensor de queries o keys.

    Divide cada vector en dos mitades y aplica una rotación 2D a cada par de
    dimensiones, codificando así la posición del token en la secuencia.

    Args:
        x: Tensor con forma (batch, num_heads, seq_len, head_dim).
        cos: Cosenos precalculados con forma (context_length, head_dim).
        sin: Senos precalculados con forma (context_length, head_dim).

    Returns:
        Tensor rotado con la misma forma que x.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim debe ser par"

    x1 = x[..., : head_dim // 2]  # Primer mitad
    x2 = x[..., head_dim // 2 :]  # Segunda mitad

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)
