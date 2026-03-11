from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.constants import TAG_PARSE_LOSS_THRESHOLD
from simulation_statistical.archetype_distribution_embedding.utils.io_utils import write_text


def _normalize_payload(text: str) -> str:
    text = re.sub(r"<[A-Z_]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_clean_records(
    player_game_df: pd.DataFrame,
    tag_blocks_df: pd.DataFrame,
    loss_threshold: float = TAG_PARSE_LOSS_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tag_agg = (
        tag_blocks_df.groupby("row_id", as_index=False)
        .agg(
            parsed_text=("tag_text", lambda s: "\n\n".join([v for v in s if isinstance(v, str) and v.strip()])),
            untagged_text=("untagged_text", "first"),
            tag_count=("tag", lambda s: int((s != "__NO_TAGS__").sum())),
        )
    )

    merged = player_game_df.merge(tag_agg, on="row_id", how="left")
    merged["parsed_text"] = merged["parsed_text"].fillna("")
    merged["untagged_text"] = merged["untagged_text"].fillna("")
    merged["tag_count"] = merged["tag_count"].fillna(0).astype(int)
    merged["clean_content"] = merged["archetype_text_clean"].map(_normalize_payload)
    merged["parsed_content"] = (
        merged["parsed_text"].fillna("").astype(str) + "\n" + merged["untagged_text"].fillna("").astype(str)
    ).map(_normalize_payload)
    merged["clean_text_non_empty"] = merged["clean_content"].str.len() > 0
    merged["content_loss_fraction"] = merged.apply(
        lambda row: max(len(row["clean_content"]) - len(row["parsed_content"]), 0) / max(len(row["clean_content"]), 1),
        axis=1,
    )
    merged["high_parse_loss"] = merged["content_loss_fraction"] > loss_threshold

    tag_freq = (
        tag_blocks_df.loc[tag_blocks_df["tag"] != "__NO_TAGS__"]
        .groupby(["wave", "tag"])
        .size()
        .reset_index(name="row_count")
        .sort_values(["wave", "row_count", "tag"], ascending=[True, False, True])
    )
    row_diagnostics = merged[
        [
            "row_id",
            "wave",
            "game_id",
            "player_id",
            "clean_text_non_empty",
            "tag_count",
            "content_loss_fraction",
            "high_parse_loss",
        ]
    ].copy()

    summary = {
        "n_rows": int(len(merged)),
        "n_empty_clean_text": int((~merged["clean_text_non_empty"]).sum()),
        "n_high_parse_loss": int(merged["high_parse_loss"].sum()),
        "loss_threshold": loss_threshold,
        "tag_frequency_rows": int(len(tag_freq)),
    }
    return row_diagnostics, tag_freq, summary


def write_validation_report(
    path: str | Path,
    learn_summary: dict[str, Any],
    val_summary: dict[str, Any],
    tag_frequency_df: pd.DataFrame,
) -> None:
    lines = [
        "# Archetype Tag Validation Report",
        "",
        "## Learn summary",
        *[f"- {key}: {value}" for key, value in sorted(learn_summary.items())],
        "",
        "## Validation summary",
        *[f"- {key}: {value}" for key, value in sorted(val_summary.items())],
        "",
        "## Tag frequencies",
    ]
    for wave, group in tag_frequency_df.groupby("wave", sort=False):
        lines.append("")
        lines.append(f"### {wave}")
        for row in group.to_dict(orient="records"):
            lines.append(f"- {row['tag']}: {row['row_count']}")
    write_text(path, "\n".join(lines) + "\n")
