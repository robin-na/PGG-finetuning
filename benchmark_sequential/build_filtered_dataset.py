from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def parse_id_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed if pd.notna(x) and str(x)]
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [
            part.strip().strip("'").strip('"')
            for part in inner.split(",")
            if part.strip().strip("'").strip('"')
        ]
    if "," in text:
        return [
            part.strip().strip("'").strip('"')
            for part in text.split(",")
            if part.strip().strip("'").strip('"')
        ]
    return [text]


def as_str_set(values: Iterable[Any]) -> set[str]:
    return {str(v) for v in values if pd.notna(v)}


def filter_by_set(df: pd.DataFrame, col: str, allowed: set[str]) -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df[df[col].astype(str).isin(allowed)].copy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def copy_non_csv_files(src_root: Path, dst_root: Path) -> None:
    for path in src_root.rglob("*"):
        if path.is_dir() or path.suffix.lower() == ".csv":
            continue
        out_path = dst_root / path.relative_to(src_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)


def filter_processed_data(
    in_dir: Path,
    out_dir: Path,
    allowed_game_ids: set[str],
) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    for path in sorted(in_dir.glob("*.csv")):
        df = pd.read_csv(path)
        before = len(df)
        if "gameId" in df.columns:
            df = filter_by_set(df, "gameId", allowed_game_ids)
        write_csv(df, out_dir / path.name)
        stats[path.name] = {"before": int(before), "after": int(len(df))}
    return stats


def filter_raw_wave(
    in_dir: Path,
    out_dir: Path,
    allowed_game_ids: set[str],
) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        path.name: pd.read_csv(path) for path in sorted(in_dir.glob("*.csv"))
    }
    filtered: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, int]] = {}

    games = tables["games.csv"]
    games_f = filter_by_set(games, "_id", allowed_game_ids)
    filtered["games.csv"] = games_f

    game_ids = as_str_set(games_f["_id"]) if "_id" in games_f.columns else set()
    treatment_ids = as_str_set(games_f["treatmentId"]) if "treatmentId" in games_f.columns else set()
    batch_ids = as_str_set(games_f["batchId"]) if "batchId" in games_f.columns else set()
    player_ids: set[str] = set()
    round_ids: set[str] = set()

    if "playerIds" in games_f.columns:
        for value in games_f["playerIds"]:
            player_ids.update(parse_id_list(value))
    if "roundIds" in games_f.columns:
        for value in games_f["roundIds"]:
            round_ids.update(parse_id_list(value))

    if "game-lobbies.csv" in tables:
        game_lobbies = filter_by_set(tables["game-lobbies.csv"], "gameId", game_ids)
        filtered["game-lobbies.csv"] = game_lobbies
        if "treatmentId" in game_lobbies.columns:
            treatment_ids |= as_str_set(game_lobbies["treatmentId"])
        if "batchId" in game_lobbies.columns:
            batch_ids |= as_str_set(game_lobbies["batchId"])
        if "playerIds" in game_lobbies.columns:
            for value in game_lobbies["playerIds"]:
                player_ids.update(parse_id_list(value))
        lobby_config_ids = (
            as_str_set(game_lobbies["lobbyConfigId"])
            if "lobbyConfigId" in game_lobbies.columns
            else set()
        )
    else:
        lobby_config_ids = set()

    if "rounds.csv" in tables:
        rounds = filter_by_set(tables["rounds.csv"], "gameId", game_ids)
        filtered["rounds.csv"] = rounds
        if "_id" in rounds.columns:
            round_ids |= as_str_set(rounds["_id"])

    if "stages.csv" in tables:
        stages = filter_by_set(tables["stages.csv"], "gameId", game_ids)
        if "roundId" in stages.columns and round_ids:
            stages = stages[stages["roundId"].astype(str).isin(round_ids)].copy()
        filtered["stages.csv"] = stages
        stage_ids = as_str_set(stages["_id"]) if "_id" in stages.columns else set()
    else:
        stage_ids = set()

    if "player-rounds.csv" in tables:
        player_rounds = filter_by_set(tables["player-rounds.csv"], "gameId", game_ids)
        if "roundId" in player_rounds.columns and round_ids:
            player_rounds = player_rounds[
                player_rounds["roundId"].astype(str).isin(round_ids)
            ].copy()
        if "playerId" in player_rounds.columns and player_ids:
            player_rounds = player_rounds[
                player_rounds["playerId"].astype(str).isin(player_ids)
            ].copy()
        filtered["player-rounds.csv"] = player_rounds
        if "playerId" in player_rounds.columns:
            player_ids |= as_str_set(player_rounds["playerId"])

    if "player-stages.csv" in tables:
        player_stages = filter_by_set(tables["player-stages.csv"], "gameId", game_ids)
        if "roundId" in player_stages.columns and round_ids:
            player_stages = player_stages[
                player_stages["roundId"].astype(str).isin(round_ids)
            ].copy()
        if "stageId" in player_stages.columns and stage_ids:
            player_stages = player_stages[
                player_stages["stageId"].astype(str).isin(stage_ids)
            ].copy()
        if "playerId" in player_stages.columns and player_ids:
            player_stages = player_stages[
                player_stages["playerId"].astype(str).isin(player_ids)
            ].copy()
        filtered["player-stages.csv"] = player_stages
        if "playerId" in player_stages.columns:
            player_ids |= as_str_set(player_stages["playerId"])

    if "player-inputs.csv" in tables:
        player_inputs = filter_by_set(tables["player-inputs.csv"], "gameId", game_ids)
        if "playerId" in player_inputs.columns and player_ids:
            player_inputs = player_inputs[
                player_inputs["playerId"].astype(str).isin(player_ids)
            ].copy()
        filtered["player-inputs.csv"] = player_inputs
        if "playerId" in player_inputs.columns:
            player_ids |= as_str_set(player_inputs["playerId"])

    if "players.csv" in tables:
        filtered["players.csv"] = filter_by_set(tables["players.csv"], "_id", player_ids)

    if "treatments.csv" in tables:
        treatments = filter_by_set(tables["treatments.csv"], "_id", treatment_ids)
        filtered["treatments.csv"] = treatments
        factor_ids: set[str] = set()
        if "factorIds" in treatments.columns:
            for value in treatments["factorIds"]:
                factor_ids.update(parse_id_list(value))
    else:
        factor_ids = set()

    if "factors.csv" in tables:
        factors = filter_by_set(tables["factors.csv"], "_id", factor_ids)
        filtered["factors.csv"] = factors
        factor_type_ids = (
            as_str_set(factors["factorTypeId"]) if "factorTypeId" in factors.columns else set()
        )
    else:
        factor_type_ids = set()

    if "factor-types.csv" in tables:
        filtered["factor-types.csv"] = filter_by_set(
            tables["factor-types.csv"], "_id", factor_type_ids
        )

    if "batches.csv" in tables:
        batches = tables["batches.csv"].copy()
        keep_batch = batches["_id"].astype(str).isin(batch_ids) if "_id" in batches.columns else pd.Series(False, index=batches.index)
        if "gameIds" in batches.columns:
            keep_batch = keep_batch | batches["gameIds"].apply(
                lambda value: any(item in game_ids for item in parse_id_list(value))
            )
        filtered["batches.csv"] = batches[keep_batch].copy()

    if "lobby-configs.csv" in tables:
        if lobby_config_ids:
            filtered["lobby-configs.csv"] = filter_by_set(
                tables["lobby-configs.csv"], "_id", lobby_config_ids
            )
        else:
            filtered["lobby-configs.csv"] = tables["lobby-configs.csv"].iloc[0:0].copy()

    for name, df in tables.items():
        out_df = filtered.get(name, df)
        write_csv(out_df, out_dir / name)
        stats[name] = {"before": int(len(df)), "after": int(len(out_df))}
    return stats


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_root = root / "data"
    out_root = root / "benchmark_sequential"
    dst_root = out_root / "data"

    val_df = pd.read_csv(
        src_root / "processed_data" / "df_analysis_val.csv",
        usecols=["gameId", "valid_number_of_starting_players"],
    )
    allowed_game_ids = as_str_set(
        val_df.loc[normalize_bool(val_df["valid_number_of_starting_players"]), "gameId"]
    )

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copy_non_csv_files(src_root, dst_root)

    processed_stats = filter_processed_data(
        in_dir=src_root / "processed_data",
        out_dir=dst_root / "processed_data",
        allowed_game_ids=allowed_game_ids,
    )
    raw_validation_stats = filter_raw_wave(
        in_dir=src_root / "raw_data" / "validation_wave",
        out_dir=dst_root / "raw_data" / "validation_wave",
        allowed_game_ids=allowed_game_ids,
    )
    raw_learning_stats = filter_raw_wave(
        in_dir=src_root / "raw_data" / "learning_wave",
        out_dir=dst_root / "raw_data" / "learning_wave",
        allowed_game_ids=allowed_game_ids,
    )

    for path in sorted(src_root.glob("*.csv")):
        df = pd.read_csv(path)
        write_csv(df, dst_root / path.name)
    for path in sorted((src_root / "exp_config_files").glob("*.csv")):
        df = pd.read_csv(path)
        write_csv(df, dst_root / "exp_config_files" / path.name)

    summary = {
        "criteria": "gameId from processed_data/df_analysis_val.csv with valid_number_of_starting_players == True",
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "selected_game_ids": int(len(allowed_game_ids)),
        "processed_data": processed_stats,
        "raw_data_validation_wave": raw_validation_stats,
        "raw_data_learning_wave": raw_learning_stats,
    }

    summary_path = dst_root / "filter_summary_gameId_valid_starters.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
