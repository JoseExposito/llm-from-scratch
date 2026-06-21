import argparse
import sys

def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

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
