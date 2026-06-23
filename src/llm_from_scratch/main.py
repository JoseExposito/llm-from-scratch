import argparse
import sys
import torch

from llm_from_scratch.config import ConfigFactory
from llm_from_scratch.model import Model
from llm_from_scratch.train import train_model
from llm_from_scratch.inference import generate_and_print_sample


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    torch.manual_seed(123)

    config = ConfigFactory.create_config(args.configuration)
    model = Model(config)
    model.print_info()

    if args.mode == "training":
        train_model(model, config)
    else:
        assert args.start_context
        generate_and_print_sample(model, config, args.start_context)

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
        choices=["base-model-10M", "rms-norm-10M"],
    )

    parser.add_argument(
        "--mode",
        help="Modo en el que iniciar el modelo: Entrenamiento o inferencia",
        default="inference",
        choices=["training", "inference"],
    )

    parser.add_argument(
        "--start-context",
        help="Durante la inferencia, el texto con el que empezar la predicción",
        required=False,
        type=str,
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
