"""Utilities for loading PGG configs from a manifest file."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple

from config import PGGConfig


def _filter_config_fields(config_data: dict) -> dict:
    """Filter manifest config data down to PGGConfig fields."""
    field_names = {field.name for field in fields(PGGConfig)}
    return {key: value for key, value in config_data.items() if key in field_names}


def load_manifest(manifest_path: Path) -> List[Tuple[str, PGGConfig]]:
    """Load a validation manifest and return (experiment_id, config) pairs.

    Args:
        manifest_path: Path to a JSON manifest exported by
            scripts/export_validation_configs.py.

    Returns:
        List of (experiment_id, PGGConfig) tuples.
    """
    entries = json.loads(manifest_path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Manifest must be a list, got {type(entries).__name__}")

    configs: List[Tuple[str, PGGConfig]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be a dict.")

        experiment_id = entry.get("experiment_id")
        if not experiment_id:
            raise ValueError(f"Manifest entry {index} missing experiment_id.")

        config_data = dict(entry.get("config", {}))
        config_data.pop("multiplier", None)
        config = PGGConfig(**_filter_config_fields(config_data))
        configs.append((experiment_id, config))

    return configs
