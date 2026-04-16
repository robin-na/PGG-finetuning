from __future__ import annotations

from pathlib import Path

from .chip_bargain import build_bundle as build_chip_bargain
from .common import DatasetBundle
from .longitudinal_trust_game_ht863 import build_bundle as build_longitudinal_trust_game_ht863
from .minority_game_bret_njzas import build_bundle as build_minority_game_bret_njzas
from .multi_game_llm_fvk2c import build_bundle as build_multi_game_llm_fvk2c
from .two_stage_trust_punishment_y2hgu import (
    build_bundle as build_two_stage_trust_punishment_y2hgu,
)


def build_dataset_bundle(dataset_key: str, repo_root: Path) -> DatasetBundle:
    mapping = {
        "chip_bargain": build_chip_bargain,
        "minority_game_bret_njzas": build_minority_game_bret_njzas,
        "longitudinal_trust_game_ht863": build_longitudinal_trust_game_ht863,
        "two_stage_trust_punishment_y2hgu": build_two_stage_trust_punishment_y2hgu,
        "multi_game_llm_fvk2c": build_multi_game_llm_fvk2c,
    }
    if dataset_key not in mapping:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    return mapping[dataset_key](repo_root)


__all__ = ["DatasetBundle", "build_dataset_bundle"]
