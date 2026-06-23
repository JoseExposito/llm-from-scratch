import torch
import torch.nn as nn

from llm_from_scratch.config import PositionalEmbeddingStrategy
from llm_from_scratch.positional_embedding import compute_rope, precompute_rope_params


class MultiHeadAttention(nn.Module):
    """Implementación del mecanismo de atención explicado con mucho más detalle
    en el notebook `src/02_attention_mechanism/02_attention_mechanism.ipynb`.

    En vez de utilizar varios mecanismos de atención causal, invocarlos en un
    bucle y concatenar sus resultados, se agrupan los cálculos para una mayor
    eficiencia.
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        positional_embedding_strategy,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out debe ser divisible entre num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        if positional_embedding_strategy == PositionalEmbeddingStrategy.ROPE:
            cos, sin = precompute_rope_params(self.head_dim, context_length)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if hasattr(self, "cos"):
            keys = compute_rope(keys, self.cos, self.sin)
            queries = compute_rope(queries, self.cos, self.sin)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
