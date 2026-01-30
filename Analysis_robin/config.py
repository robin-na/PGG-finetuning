from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_ANALYSIS_OUTPUT_ROOT = REPO_ROOT / "analysis_output_Robin"
DEFAULT_HUMAN_ROUNDS_CSV = REPO_ROOT / "data/raw_data/validation_wave/player-rounds.csv"
DEFAULT_HUMAN_CONFIG_CSV = REPO_ROOT / "data/processed_data/df_analysis_val.csv"
