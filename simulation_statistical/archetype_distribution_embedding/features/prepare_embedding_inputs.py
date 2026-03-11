from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.io_utils import write_jsonl


def build_embedding_input_records(df: pd.DataFrame) -> list[dict[str, str]]:
    required = {"row_id", "wave", "game_id", "player_id", "archetype_text_clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for embedding export: {sorted(missing)}")

    records: list[dict[str, str]] = []
    for row in df[["row_id", "wave", "game_id", "player_id", "archetype_text_clean"]].to_dict(orient="records"):
        records.append(
            {
                "row_id": row["row_id"],
                "text": row["archetype_text_clean"],
                "wave": row["wave"],
                "game_id": row["game_id"],
                "player_id": row["player_id"],
            }
        )
    return records


def export_embedding_input_jsonl(df: pd.DataFrame, output_path: str | Path) -> None:
    records = build_embedding_input_records(df)
    write_jsonl(output_path, records, append=False)
