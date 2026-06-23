import torch.nn as nn

from llm_from_scratch.config import Config
from llm_from_scratch.feed_forward import FeedForward
from llm_from_scratch.normalization import create_norm
from llm_from_scratch.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=config.embedding_dim,
            d_out=config.embedding_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            dropout=config.dropout_rate,
            qkv_bias=config.query_key_value_bias,
        )
        self.ff = FeedForward(config.embedding_dim)
        self.norm1 = create_norm(config.normalization_strategy, config.embedding_dim)
        self.norm2 = create_norm(config.normalization_strategy, config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
