import torch
import torch.nn as nn

from llm_from_scratch.config import Config
from llm_from_scratch.normalization import create_norm
from llm_from_scratch.transformer_block import TransformerBlock


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(
            config.vocabulary_size, config.embedding_dim
        )

        self.positional_embeddings = nn.Embedding(
            config.context_length, config.embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_transformer_blocks)]
        )

        self.final_norm = create_norm(
            config.normalization_strategy, config.embedding_dim
        )

        self.out_head = nn.Linear(
            config.embedding_dim,
            config.vocabulary_size,
            bias=False,
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        token_embeddings = self.token_embeddings(in_idx)
        positional_embeddings = self.positional_embeddings(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def print_info(self) -> None:
        print("Model information:")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"- Number of parameters: {total_params:,}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"- Size of the model: {total_size_mb:.2f} MB")
