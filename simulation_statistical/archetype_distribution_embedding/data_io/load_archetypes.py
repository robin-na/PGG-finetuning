from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.io_utils import read_jsonl


def load_archetype_jsonl(path: str | Path, wave: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archetype JSONL not found: {file_path}")

    rows = read_jsonl(file_path)
    if not rows:
        raise ValueError(f"Archetype JSONL is empty: {file_path}")

    df = pd.DataFrame(rows)
    missing = {"experiment", "participant", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required archetype fields {sorted(missing)} in {file_path}")

    df = df.copy()
    df["wave"] = wave
    df["game_id"] = df["experiment"].astype(str)
    df["player_id"] = df["participant"].astype(str)
    df["archetype_text_raw"] = df["text"].fillna("").astype(str)
    df["source_archetype_path"] = str(file_path)
    return df
