from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "simulation_statistical" / "archetype_distribution_embedding"
ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts"
INTERMEDIATE_ROOT = ARTIFACTS_ROOT / "intermediate"
MODEL_ROOT = ARTIFACTS_ROOT / "models"
OUTPUT_ROOT = ARTIFACTS_ROOT / "outputs"


@dataclass(frozen=True)
class WaveInputPaths:
    archetype_jsonl: Path
    config_csv: Path
    player_rounds_csv: Path


DEFAULT_INPUT_PATHS = {
    "learn": WaveInputPaths(
        archetype_jsonl=REPO_ROOT / "Persona" / "archetype_oracle_gpt51_learn.jsonl",
        config_csv=REPO_ROOT / "benchmark_statistical" / "data" / "processed_data" / "df_analysis_learn.csv",
        player_rounds_csv=REPO_ROOT / "benchmark_statistical" / "data" / "raw_data" / "learning_wave" / "player-rounds.csv",
    ),
    "val": WaveInputPaths(
        archetype_jsonl=REPO_ROOT / "Persona" / "archetype_oracle_gpt51_val.jsonl",
        config_csv=REPO_ROOT / "data" / "processed_data" / "df_analysis_val.csv",
        player_rounds_csv=REPO_ROOT / "data" / "raw_data" / "validation_wave" / "player-rounds.csv",
    ),
}


def intermediate_path(name: str) -> Path:
    return INTERMEDIATE_ROOT / name


def model_path(name: str) -> Path:
    return MODEL_ROOT / name


def output_path(name: str) -> Path:
    return OUTPUT_ROOT / name
