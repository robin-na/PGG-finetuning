from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_OUTPUT_ROOT = Path("/workspace/PGG-finetuning/output")
DEFAULT_ANALYSIS_OUTPUT_ROOT = Path("/workspace/PGG-finetuning/analysis_output_Robin")
DEFAULT_HUMAN_ROUNDS_CSV = Path(
    "/workspace/PGG-finetuning/data/raw_data/validation_wave/player-rounds.csv"
)
DEFAULT_HUMAN_CONFIG_CSV = Path(
    "/workspace/PGG-finetuning/data/processed_data/df_analysis_val.csv"
)


@dataclass(frozen=True)
class ConfigPair:
    sim_config: str
    human_config: str


def base_config_name(config_name: str) -> str:
    if "_" not in config_name:
        return config_name
    return "_".join(config_name.split("_")[:-1])


def paired_config_name(config_name: str) -> Optional[str]:
    if config_name.endswith("_C"):
        return f"{base_config_name(config_name)}_T"
    if config_name.endswith("_T"):
        return f"{base_config_name(config_name)}_C"
    return None


def build_pair(sim_config: str) -> Optional[ConfigPair]:
    human_config = paired_config_name(sim_config)
    if not human_config:
        return None
    return ConfigPair(sim_config=sim_config, human_config=human_config)
