from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulation_statistical.history_conditioned_policy import (  # noqa: E402
    DEFAULT_HISTORY_POLICY_MODEL_PATH,
    train_history_conditioned_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the history-conditioned archetype policy on the learning wave."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=str(DEFAULT_HISTORY_POLICY_MODEL_PATH),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_history_conditioned_policy(output_model_path=Path(args.output_model_path).resolve())


if __name__ == "__main__":
    main()
