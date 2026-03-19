from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CONFIG_COLUMNS = [
    "CONFIG_numRounds",
    "CONFIG_endowment",
    "CONFIG_allOrNothing",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_MPCR",
]


@dataclass(frozen=True)
class GameConfig:
    num_rounds: int
    endowment: int
    all_or_nothing: bool
    punishment_exists: bool
    reward_exists: bool
    punishment_cost: float
    punishment_magnitude: float
    reward_cost: float
    reward_magnitude: float
    mpcr: float


@dataclass(frozen=True)
class RoundRecord:
    index: int
    contributions: dict[str, int]
    punished: dict[str, dict[str, int]]
    rewarded: dict[str, dict[str, int]]
    round_payoffs: dict[str, float]


@dataclass(frozen=True)
class GameTrajectory:
    game_id: str
    players: list[str]
    config: GameConfig
    rounds: list[RoundRecord]

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def num_players(self) -> int:
        return len(self.players)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_action_dict(value: Any) -> dict[str, int]:
    if pd.isna(value):
        return {}
    if not isinstance(value, str):
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    parsed: dict[str, int] = {}
    for target, units in raw.items():
        try:
            unit_value = int(round(float(units)))
        except (TypeError, ValueError):
            continue
        if unit_value > 0:
            parsed[str(target)] = unit_value
    return parsed


def load_wave_games(
    repo_root: Path,
    wave_name: str,
    processed_suffix: str,
    min_num_rounds_exclusive: int = 0,
) -> list[GameTrajectory]:
    """Load complete games for a wave where no player exits mid-game."""
    player_rounds_path = repo_root / f"data/raw_data/{wave_name}/player-rounds.csv"
    rounds_path = repo_root / f"data/raw_data/{wave_name}/rounds.csv"
    processed_path = repo_root / f"data/processed_data/df_analysis_{processed_suffix}.csv"

    player_rounds = pd.read_csv(
        player_rounds_path,
        usecols=[
            "playerId",
            "roundId",
            "gameId",
            "data.punished",
            "data.rewarded",
            "data.contribution",
            "data.roundPayoff",
        ],
    ).rename(
        columns={
            "data.punished": "punished_raw",
            "data.rewarded": "rewarded_raw",
            "data.contribution": "contribution",
            "data.roundPayoff": "round_payoff",
        }
    )
    rounds = pd.read_csv(rounds_path, usecols=["_id", "index"])
    configs = pd.read_csv(processed_path, usecols=["gameId"] + CONFIG_COLUMNS).drop_duplicates("gameId")

    player_rounds = player_rounds.merge(
        rounds,
        left_on="roundId",
        right_on="_id",
        how="left",
        validate="many_to_one",
    )

    incomplete_games = set(player_rounds.loc[player_rounds["contribution"].isna(), "gameId"])
    player_rounds = player_rounds[
        (~player_rounds["gameId"].isin(incomplete_games)) & player_rounds["gameId"].isin(set(configs["gameId"]))
    ].copy()

    config_map = configs.set_index("gameId")
    games: list[GameTrajectory] = []

    for game_id, game_df in player_rounds.groupby("gameId", sort=False):
        game_df = game_df.sort_values(["index", "playerId"]).copy()
        num_rounds = int(game_df["index"].nunique())
        if num_rounds <= min_num_rounds_exclusive:
            continue

        players = sorted(game_df["playerId"].astype(str).unique().tolist())
        if len(game_df) != len(players) * num_rounds:
            continue

        config_row = config_map.loc[game_id]
        config = GameConfig(
            num_rounds=int(config_row["CONFIG_numRounds"]),
            endowment=int(round(float(config_row["CONFIG_endowment"]))),
            all_or_nothing=_as_bool(config_row["CONFIG_allOrNothing"]),
            punishment_exists=_as_bool(config_row["CONFIG_punishmentExists"]),
            reward_exists=_as_bool(config_row["CONFIG_rewardExists"]),
            punishment_cost=float(config_row["CONFIG_punishmentCost"]),
            punishment_magnitude=float(config_row["CONFIG_punishmentMagnitude"]),
            reward_cost=float(config_row["CONFIG_rewardCost"]),
            reward_magnitude=float(config_row["CONFIG_rewardMagnitude"]),
            mpcr=float(config_row["CONFIG_MPCR"]),
        )

        rounds_by_index: list[RoundRecord] = []
        for round_index, round_df in game_df.groupby("index", sort=True):
            contributions: dict[str, int] = {}
            punished: dict[str, dict[str, int]] = {}
            rewarded: dict[str, dict[str, int]] = {}
            round_payoffs: dict[str, float] = {}

            for row in round_df.itertuples(index=False):
                player_id = str(row.playerId)
                contributions[player_id] = int(round(float(row.contribution)))
                punished[player_id] = _parse_action_dict(row.punished_raw)
                rewarded[player_id] = _parse_action_dict(row.rewarded_raw)
                round_payoffs[player_id] = float(row.round_payoff)

            rounds_by_index.append(
                RoundRecord(
                    index=int(round_index),
                    contributions=contributions,
                    punished=punished,
                    rewarded=rewarded,
                    round_payoffs=round_payoffs,
                )
            )

        games.append(
            GameTrajectory(
                game_id=str(game_id),
                players=players,
                config=config,
                rounds=rounds_by_index,
            )
        )

    games.sort(key=lambda game: game.game_id)
    return games


def load_learning_wave_games(repo_root: Path, min_num_rounds_exclusive: int = 0) -> list[GameTrajectory]:
    """Load complete learning-wave games that have no player exits mid-game."""
    return load_wave_games(
        repo_root=repo_root,
        wave_name="learning_wave",
        processed_suffix="learn",
        min_num_rounds_exclusive=min_num_rounds_exclusive,
    )
