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

from simulation_statistical.structured_sequence_policy import (  # noqa: E402
    DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH,
    DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_POLICY_MODEL_PATH,
    train_exact_sequence_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the exact-sequence archetype policy on the learning wave."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no_cluster",
        action="store_true",
        help="Train the structured exact-sequence ablation without cluster information.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_path = (
        DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_POLICY_MODEL_PATH
        if bool(args.no_cluster)
        else DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH
    )
    train_exact_sequence_policy(
        output_model_path=Path(args.output_model_path).resolve() if args.output_model_path else default_path,
        use_cluster=not bool(args.no_cluster),
    )


if __name__ == "__main__":
    main()
