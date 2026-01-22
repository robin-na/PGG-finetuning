#!/usr/bin/env python3
"""Run validation experiments from a manifest file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Add Simulation to path
sys.path.insert(0, str(Path(__file__).parent / "Simulation"))

from config_loader import load_manifest
from main import run_experiment


DEFAULT_MANIFEST_PATH = Path("data/validation_configs.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation experiments from a JSON manifest."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the validation manifest JSON.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to run per experiment.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose per-round output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_manifest(args.manifest_path)
    verbose = not args.quiet

    for experiment_id, config in configs:
        run_experiment(experiment_id, config, num_games=args.num_games, verbose=verbose)


if __name__ == "__main__":
    main()
