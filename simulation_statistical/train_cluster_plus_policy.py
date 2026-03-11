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

from simulation_statistical.cluster_plus_policy import (  # noqa: E402
    DEFAULT_CLUSTER_PLUS_BEHAVIOR_MODEL_PATH,
    DEFAULT_CLUSTER_PLUS_TRAIN_SUMMARY_PATH,
    build_cluster_plus_behavior_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the archetype_cluster_plus behavior model on the learning wave."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=str(DEFAULT_CLUSTER_PLUS_BEHAVIOR_MODEL_PATH),
    )
    parser.add_argument(
        "--summary_output_path",
        type=str,
        default=str(DEFAULT_CLUSTER_PLUS_TRAIN_SUMMARY_PATH),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_cluster_plus_behavior_model(
        output_path=Path(args.output_model_path).resolve(),
        summary_output_path=Path(args.summary_output_path).resolve(),
    )


if __name__ == "__main__":
    main()
