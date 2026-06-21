from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TinyDataset(Dataset):
    """Copia del Dataset implementado con mayor detalle en el notebook
    "src/01_dataset_preparation/dataset_preparation.ipynb".
    """

    def __init__(self, tokenizer, split, window_size):
        assert split == "train" or split == "validation"
        self.window_size = window_size

        ds = load_dataset("roneneldan/TinyStories", split=split)

        # Pre-tokeniza por lotes. HuggingFace cachea el resultado en disco,
        # por lo que solo se tokeniza en la primera ejecución.
        # Se guarda también n_tokens para evitar tener que leer todas las
        # listas de tokens solo para calcular sus longitudes.
        self.ds = ds.map(
            lambda batch: {
                "tokens": (
                    encoded := tokenizer([t + "<|endoftext|>" for t in batch["text"]])[
                        "input_ids"
                    ]
                ),
                "n_tokens": [len(ids) for ids in encoded],
            },
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
        )

        # Índice acumulativo de longitudes para mapear posición global -> ejemplo.
        # n_tokens es una columna de enteros en Arrow, se lee sin cargar los tokens.
        lengths = np.array(self.ds["n_tokens"])
        self.cumulative_lengths = np.cumsum(lengths)
        self.total_tokens = int(self.cumulative_lengths[-1])

    def __len__(self):
        return self.total_tokens // self.window_size

    def __getitem__(self, index):
        start = index * self.window_size
        end = start + self.window_size + 1  # +1 para el target (y)

        tokens = self._get_token_range(start, end)
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

    def _get_token_range(self, start, end):
        first = int(np.searchsorted(self.cumulative_lengths, start, side="right"))
        last = int(np.searchsorted(self.cumulative_lengths, end - 1, side="right"))

        tokens = []
        for i in range(first, min(last + 1, len(self.ds))):
            example_tokens = self.ds[i]["tokens"]
            offset = int(self.cumulative_lengths[i - 1]) if i > 0 else 0
            lo = max(0, start - offset)
            hi = min(len(example_tokens), end - offset)
            tokens.extend(example_tokens[lo:hi])

        return tokens


def create_dataloader(
    tokenizer,
    split,
    batch_size,
    window_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = TinyDataset(tokenizer, split, window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
