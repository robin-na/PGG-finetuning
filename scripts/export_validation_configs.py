#!/usr/bin/env python3
"""Export validation experiment configs into a single manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_OUTPUT_PATH = Path("data/validation_configs.json")


def load_config(config_path: Path, experiment_id: str) -> Dict[str, Any]:
    data = json.loads(config_path.read_text())
    config = dict(data.get("config", {}))

    if "llm_model" not in config:
        config["llm_model"] = data.get("llm_model")
    if "llm_temperature" not in config:
        config["llm_temperature"] = data.get("llm_temperature")

    return {
        "experiment_id": data.get("experiment_id", experiment_id),
        "config": config,
    }


def export_validation_configs(
    experiments_dir: Path,
    output_path: Path,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for experiment_dir in sorted(experiments_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if not experiment_dir.name.startswith("VALIDATION"):
            continue

        config_path = experiment_dir / "config.json"
        if not config_path.exists():
            continue

        entries.append(load_config(config_path, experiment_dir.name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export VALIDATION experiment configs into a single JSON manifest."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Directory that contains experiment subfolders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_validation_configs(args.experiments_dir, args.output_path)


if __name__ == "__main__":
    main()
