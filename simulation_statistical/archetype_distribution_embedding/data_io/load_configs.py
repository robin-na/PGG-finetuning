from __future__ import annotations

from pathlib import Path

import pandas as pd


CONFIG_METADATA_COLUMNS = [
    "gameId",
    "gameLobbyId",
    "treatmentId",
    "playerIds",
    "batchId",
    "name",
    "CONFIG_treatmentName",
]


def _collapse_duplicate_games(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    duplicate_mask = df.duplicated("game_id", keep=False)
    if not duplicate_mask.any():
        return df

    collapsed_rows = []
    for _, group in df.groupby("game_id", sort=False):
        unique_rows = group.drop_duplicates()
        if len(unique_rows) != 1:
            raise ValueError(
                f"Config table {path} has conflicting rows for game_id={group['game_id'].iloc[0]}"
            )
        collapsed_rows.append(unique_rows.iloc[0])
    return pd.DataFrame(collapsed_rows).reset_index(drop=True)


def load_config_table(path: str | Path, required_config_cols: list[str]) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config CSV not found: {file_path}")

    df = pd.read_csv(file_path)
    missing = {"gameId", *required_config_cols} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required config fields {sorted(missing)} in {file_path}")

    keep_columns = []
    for column in CONFIG_METADATA_COLUMNS + required_config_cols:
        if column in df.columns and column not in keep_columns:
            keep_columns.append(column)

    out = df[keep_columns].copy()
    out = out.rename(
        columns={
            "gameId": "game_id",
            "gameLobbyId": "game_lobby_id",
            "treatmentId": "treatment_id",
            "playerIds": "player_ids_raw",
            "batchId": "batch_id",
        }
    )
    out["source_config_path"] = str(file_path)
    out = _collapse_duplicate_games(out, file_path)
    return out.reset_index(drop=True)


def load_player_game_keys(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Player-round CSV not found: {file_path}")

    raw = pd.read_csv(file_path)
    missing = {"gameId", "playerId"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required player-round fields {sorted(missing)} in {file_path}")

    raw = raw.copy()
    if "_id" not in raw.columns:
        raw["_id"] = range(len(raw))
    if "createdAt" not in raw.columns:
        raw["createdAt"] = None
    if "batchId" not in raw.columns:
        raw["batchId"] = None

    grouped = (
        raw.groupby(["gameId", "playerId"], as_index=False)
        .agg(
            raw_round_count=("_id", "size"),
            player_round_batch_id=("batchId", "first"),
            first_round_created_at=("createdAt", "min"),
            last_round_created_at=("createdAt", "max"),
        )
        .rename(columns={"gameId": "game_id", "playerId": "player_id"})
    )
    grouped["source_player_rounds_path"] = str(file_path)
    return grouped
