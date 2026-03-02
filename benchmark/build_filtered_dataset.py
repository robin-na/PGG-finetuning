from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class WaveConfig:
    wave_name: str
    tag: str
    player_rounds_csv: Path
    df_analysis_csv: Path
    demographics_csv: Path


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def parse_id_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed if str(x)]
    except (ValueError, SyntaxError):
        pass
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        parts = [p.strip().strip("'").strip('"') for p in body.split(",")]
        return [p for p in parts if p]
    if "," in text:
        parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
        return [p for p in parts if p]
    return [text]


def as_str_set(values: Iterable[object]) -> set[str]:
    return {str(v) for v in values if pd.notna(v)}


def has_any_demographic_info(df: pd.DataFrame) -> pd.Series:
    # Match the condition-2 counting definition previously used:
    # a player has demographic info if at least one of age/gender/education is present.
    age_present = df["age"].notna()
    gender_present = df["gender_code"].notna()
    education_present = df["education_code"].notna()
    return age_present | gender_present | education_present


def compute_condition2_games(cfg: WaveConfig) -> tuple[set[str], set[str]]:
    player_rounds = pd.read_csv(cfg.player_rounds_csv)
    df_analysis = pd.read_csv(cfg.df_analysis_csv)
    demographics = pd.read_csv(cfg.demographics_csv)

    valid_games = as_str_set(
        df_analysis.loc[
            normalize_bool(df_analysis["valid_number_of_starting_players"]),
            "gameId",
        ]
    )

    player_rounds["gameId"] = player_rounds["gameId"].astype(str)
    player_rounds["playerId"] = player_rounds["playerId"].astype(str)
    player_rounds = player_rounds[player_rounds["gameId"].isin(valid_games)].copy()

    game_players = (
        player_rounds[["gameId", "playerId"]]
        .drop_duplicates()
        .copy()
    )

    demographics["gameId"] = demographics["gameId"].astype(str)
    demographics["playerId"] = demographics["playerId"].astype(str)
    demographics["has_any_demo"] = has_any_demographic_info(demographics)
    demographics = demographics[demographics["gameId"].isin(valid_games)].copy()

    merged = game_players.merge(
        demographics[["gameId", "playerId", "has_any_demo"]],
        on=["gameId", "playerId"],
        how="left",
    )

    eligible_games = set(
        merged.groupby("gameId")["has_any_demo"]
        .apply(lambda s: s.notna().all() and s.fillna(False).all())
        .loc[lambda s: s]
        .index.astype(str)
    )

    eligible_players = as_str_set(
        player_rounds.loc[player_rounds["gameId"].isin(eligible_games), "playerId"]
    )
    return eligible_games, eligible_players


def filter_by_set(df: pd.DataFrame, col: str, allowed: set[str]) -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df[df[col].astype(str).isin(allowed)].copy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def filter_raw_wave(
    root: Path,
    out_root: Path,
    wave_name: str,
    game_ids: set[str],
    player_ids: set[str],
) -> dict[str, int]:
    in_dir = root / "data" / "raw_data" / wave_name
    out_dir = out_root / "data" / "raw_data" / wave_name

    games_df = pd.read_csv(in_dir / "games.csv")
    games_df = filter_by_set(games_df, "_id", game_ids)
    selected_treatment_ids = as_str_set(games_df.get("treatmentId", pd.Series([], dtype=object)))
    selected_game_lobby_ids = as_str_set(games_df.get("gameLobbyId", pd.Series([], dtype=object)))
    selected_batch_ids = as_str_set(games_df.get("batchId", pd.Series([], dtype=object)))
    write_csv(games_df, out_dir / "games.csv")

    game_lobbies_df = pd.read_csv(in_dir / "game-lobbies.csv")
    game_lobbies_df = filter_by_set(game_lobbies_df, "gameId", game_ids)
    selected_treatment_ids |= as_str_set(
        game_lobbies_df.get("treatmentId", pd.Series([], dtype=object))
    )
    selected_batch_ids |= as_str_set(
        game_lobbies_df.get("batchId", pd.Series([], dtype=object))
    )
    selected_lobby_config_ids = as_str_set(
        game_lobbies_df.get("lobbyConfigId", pd.Series([], dtype=object))
    )
    write_csv(game_lobbies_df, out_dir / "game-lobbies.csv")

    player_rounds_df = pd.read_csv(in_dir / "player-rounds.csv")
    player_rounds_df = filter_by_set(player_rounds_df, "gameId", game_ids)
    player_ids |= as_str_set(player_rounds_df.get("playerId", pd.Series([], dtype=object)))
    write_csv(player_rounds_df, out_dir / "player-rounds.csv")

    rounds_df = pd.read_csv(in_dir / "rounds.csv")
    rounds_df = filter_by_set(rounds_df, "gameId", game_ids)
    selected_round_ids = as_str_set(rounds_df.get("_id", pd.Series([], dtype=object)))
    write_csv(rounds_df, out_dir / "rounds.csv")

    stages_df = pd.read_csv(in_dir / "stages.csv")
    stages_df = filter_by_set(stages_df, "gameId", game_ids)
    if "roundId" in stages_df.columns:
        stages_df = stages_df[stages_df["roundId"].astype(str).isin(selected_round_ids)].copy()
    selected_stage_ids = as_str_set(stages_df.get("_id", pd.Series([], dtype=object)))
    write_csv(stages_df, out_dir / "stages.csv")

    player_stages_df = pd.read_csv(in_dir / "player-stages.csv")
    player_stages_df = filter_by_set(player_stages_df, "gameId", game_ids)
    if "playerId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["playerId"].astype(str).isin(player_ids)
        ].copy()
    if "roundId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["roundId"].astype(str).isin(selected_round_ids)
        ].copy()
    if "stageId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["stageId"].astype(str).isin(selected_stage_ids)
        ].copy()
    write_csv(player_stages_df, out_dir / "player-stages.csv")

    player_inputs_df = pd.read_csv(in_dir / "player-inputs.csv")
    player_inputs_df = filter_by_set(player_inputs_df, "gameId", game_ids)
    if "playerId" in player_inputs_df.columns:
        player_inputs_df = player_inputs_df[
            player_inputs_df["playerId"].astype(str).isin(player_ids)
        ].copy()
    write_csv(player_inputs_df, out_dir / "player-inputs.csv")

    players_df = pd.read_csv(in_dir / "players.csv")
    players_df = filter_by_set(players_df, "_id", player_ids)
    write_csv(players_df, out_dir / "players.csv")

    batches_df = pd.read_csv(in_dir / "batches.csv")
    if "_id" in batches_df.columns:
        batches_df = batches_df[batches_df["_id"].astype(str).isin(selected_batch_ids)].copy()
    write_csv(batches_df, out_dir / "batches.csv")

    treatments_df = pd.read_csv(in_dir / "treatments.csv")
    treatments_df = filter_by_set(treatments_df, "_id", selected_treatment_ids)
    write_csv(treatments_df, out_dir / "treatments.csv")

    selected_factor_ids: set[str] = set()
    if "factorIds" in treatments_df.columns:
        for value in treatments_df["factorIds"]:
            selected_factor_ids.update(parse_id_list(value))

    factors_df = pd.read_csv(in_dir / "factors.csv")
    factors_df = filter_by_set(factors_df, "_id", selected_factor_ids)
    write_csv(factors_df, out_dir / "factors.csv")

    selected_factor_type_ids = as_str_set(
        factors_df.get("factorTypeId", pd.Series([], dtype=object))
    )
    factor_types_df = pd.read_csv(in_dir / "factor-types.csv")
    factor_types_df = filter_by_set(factor_types_df, "_id", selected_factor_type_ids)
    write_csv(factor_types_df, out_dir / "factor-types.csv")

    lobby_configs_df = pd.read_csv(in_dir / "lobby-configs.csv")
    if selected_lobby_config_ids:
        lobby_configs_df = filter_by_set(lobby_configs_df, "_id", selected_lobby_config_ids)
    write_csv(lobby_configs_df, out_dir / "lobby-configs.csv")

    return {
        "games": int(len(games_df)),
        "actions": int(len(player_rounds_df)),
        "players": int(len(players_df)),
        "rounds": int(len(rounds_df)),
    }


def filter_processed_data(
    root: Path,
    out_root: Path,
    learn_game_ids: set[str],
    val_game_ids: set[str],
) -> None:
    in_dir = root / "data" / "processed_data"
    out_dir = out_root / "data" / "processed_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    learn_config_ids: set[str] = set()
    val_config_ids: set[str] = set()

    learn_df = pd.read_csv(in_dir / "df_analysis_learn.csv")
    learn_filtered = filter_by_set(learn_df, "gameId", learn_game_ids)
    write_csv(learn_filtered, out_dir / "df_analysis_learn.csv")
    if "CONFIG_configId" in learn_filtered.columns:
        learn_config_ids = as_str_set(learn_filtered["CONFIG_configId"])

    val_df = pd.read_csv(in_dir / "df_analysis_val.csv")
    val_filtered = filter_by_set(val_df, "gameId", val_game_ids)
    write_csv(val_filtered, out_dir / "df_analysis_val.csv")
    if "CONFIG_configId" in val_filtered.columns:
        val_config_ids = as_str_set(val_filtered["CONFIG_configId"])

    for path in sorted(in_dir.glob("*.csv")):
        if path.name in {"df_analysis_learn.csv", "df_analysis_val.csv"}:
            continue
        df = pd.read_csv(path)
        lower = path.name.lower()
        if "gameId" in df.columns:
            if "learn" in lower:
                df = filter_by_set(df, "gameId", learn_game_ids)
            elif "val" in lower:
                df = filter_by_set(df, "gameId", val_game_ids)
        elif path.name == "df_paired_learn.csv" and "CONFIG_configId" in df.columns:
            df = filter_by_set(df, "CONFIG_configId", learn_config_ids)
        elif path.name == "df_paired_val.csv" and "CONFIG_configId" in df.columns:
            df = filter_by_set(df, "CONFIG_configId", val_config_ids)
        elif "CONFIG_configId" in df.columns:
            df = filter_by_set(df, "CONFIG_configId", learn_config_ids | val_config_ids)
        write_csv(df, out_dir / path.name)


def filter_demographics(
    root: Path,
    out_root: Path,
    learn_game_ids: set[str],
    val_game_ids: set[str],
    learn_player_ids: set[str],
    val_player_ids: set[str],
) -> None:
    in_dir = root / "demographics"
    out_dir = out_root / "data" / "demographics"
    out_dir.mkdir(parents=True, exist_ok=True)

    learn = pd.read_csv(in_dir / "demographics_numeric_learn.csv")
    learn = filter_by_set(learn, "gameId", learn_game_ids)
    learn = filter_by_set(learn, "playerId", learn_player_ids)
    write_csv(learn, out_dir / "demographics_numeric_learn.csv")

    val = pd.read_csv(in_dir / "demographics_numeric_val.csv")
    val = filter_by_set(val, "gameId", val_game_ids)
    val = filter_by_set(val, "playerId", val_player_ids)
    write_csv(val, out_dir / "demographics_numeric_val.csv")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "benchmark"

    learn_cfg = WaveConfig(
        wave_name="learning_wave",
        tag="learn",
        player_rounds_csv=root / "data" / "raw_data" / "learning_wave" / "player-rounds.csv",
        df_analysis_csv=root / "data" / "processed_data" / "df_analysis_learn.csv",
        demographics_csv=root / "demographics" / "demographics_numeric_learn.csv",
    )
    val_cfg = WaveConfig(
        wave_name="validation_wave",
        tag="val",
        player_rounds_csv=root / "data" / "raw_data" / "validation_wave" / "player-rounds.csv",
        df_analysis_csv=root / "data" / "processed_data" / "df_analysis_val.csv",
        demographics_csv=root / "demographics" / "demographics_numeric_val.csv",
    )

    learn_games, learn_players = compute_condition2_games(learn_cfg)
    val_games, val_players = compute_condition2_games(val_cfg)

    learn_stats = filter_raw_wave(
        root=root,
        out_root=out_root,
        wave_name="learning_wave",
        game_ids=learn_games,
        player_ids=learn_players,
    )
    val_stats = filter_raw_wave(
        root=root,
        out_root=out_root,
        wave_name="validation_wave",
        game_ids=val_games,
        player_ids=val_players,
    )
    filter_processed_data(
        root=root,
        out_root=out_root,
        learn_game_ids=learn_games,
        val_game_ids=val_games,
    )
    filter_demographics(
        root=root,
        out_root=out_root,
        learn_game_ids=learn_games,
        val_game_ids=val_games,
        learn_player_ids=learn_players,
        val_player_ids=val_players,
    )

    summary = {
        "criteria": "condition_2_every_player_has_at_least_one_demographic_info",
        "output_root": str(out_root / "data"),
        "learning_wave": learn_stats,
        "validation_wave": val_stats,
        "combined": {
            "games": learn_stats["games"] + val_stats["games"],
            "actions": learn_stats["actions"] + val_stats["actions"],
            "players": learn_stats["players"] + val_stats["players"],
            "rounds": learn_stats["rounds"] + val_stats["rounds"],
        },
    }

    summary_path = out_root / "data" / "subset_summary_condition2.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
