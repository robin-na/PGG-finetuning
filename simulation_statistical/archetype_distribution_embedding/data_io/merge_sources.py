from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.constants import MERGE_UNMATCHED_THRESHOLD


@dataclass
class MergeSummary:
    wave: str
    archetype_rows: int
    raw_player_game_rows: int
    config_game_rows: int
    matched_player_rows: int
    unmatched_player_rows: int
    matched_config_rows: int
    unmatched_config_rows: int
    duplicate_row_ids: int


def _validate_unique(df: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicate_count = int(df.duplicated(columns).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate rows for keys {columns}")


def build_player_game_table(
    archetype_df: pd.DataFrame,
    player_game_df: pd.DataFrame,
    config_df: pd.DataFrame,
    wave: str,
    unmatched_threshold: float = MERGE_UNMATCHED_THRESHOLD,
) -> tuple[pd.DataFrame, MergeSummary]:
    _validate_unique(archetype_df, ["game_id", "player_id"], f"{wave} archetype table")
    _validate_unique(player_game_df, ["game_id", "player_id"], f"{wave} player-game key table")
    _validate_unique(config_df, ["game_id"], f"{wave} config table")

    merged = archetype_df.merge(
        player_game_df,
        on=["game_id", "player_id"],
        how="left",
        indicator="_player_match",
    )
    unmatched_player_rows = int((merged["_player_match"] != "both").sum())
    if len(merged) and unmatched_player_rows / len(merged) > unmatched_threshold:
        raise ValueError(
            f"{wave} merge failed: {unmatched_player_rows}/{len(merged)} archetype rows had no player-round match"
        )

    merged = merged.merge(
        config_df,
        on="game_id",
        how="left",
        indicator="_config_match",
    )
    unmatched_config_rows = int((merged["_config_match"] != "both").sum())
    if len(merged) and unmatched_config_rows / len(merged) > unmatched_threshold:
        raise ValueError(
            f"{wave} merge failed: {unmatched_config_rows}/{len(merged)} archetype rows had no config match"
        )

    merged["row_id"] = merged.apply(
        lambda row: f"{wave}__{row['game_id']}__{row['player_id']}",
        axis=1,
    )
    duplicate_row_ids = int(merged.duplicated("row_id").sum())
    if duplicate_row_ids:
        raise ValueError(f"{wave} merged table has {duplicate_row_ids} duplicate row_id values")

    merged = merged.drop(columns=["_player_match", "_config_match"])
    summary = MergeSummary(
        wave=wave,
        archetype_rows=len(archetype_df),
        raw_player_game_rows=len(player_game_df),
        config_game_rows=len(config_df),
        matched_player_rows=len(merged) - unmatched_player_rows,
        unmatched_player_rows=unmatched_player_rows,
        matched_config_rows=len(merged) - unmatched_config_rows,
        unmatched_config_rows=unmatched_config_rows,
        duplicate_row_ids=duplicate_row_ids,
    )
    return merged, summary
