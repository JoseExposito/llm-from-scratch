import argparse
import sys
import torch

from llm_from_scratch.config import ConfigFactory
from llm_from_scratch.model import Model
from llm_from_scratch.train import train_model_simple
from llm_from_scratch.data_loader import create_dataloader


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    torch.manual_seed(123)

    config = ConfigFactory.create_config(args.configuration)

    model = Model(config)
    model.print_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    train_loader = create_dataloader(
        tokenizer=config.tokenizer,
        split="train",
        #batch_size=2,
        batch_size=32,
        window_size=config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        tokenizer=config.tokenizer,
        split="validation",
        #batch_size=2,
        batch_size=32,
        window_size=config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # num_epochs = 10
    num_epochs = 1
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Once upon a time",
        tokenizer=config.tokenizer,
    )

    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_from_scratch",
        description="Un LLM implementado desde cero con distintas configuraciones",
    )

    parser.add_argument(
        "--configuration",
        help="Configuración a utilizar por el modelo",
        default="base-model-10M",
        choices=["base-model-10M"],
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
