from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = str(SCRIPT_PATH.parent)
ALGORITHMIC_LATENT_ROOT = SCRIPT_PATH.parents[1]
SIMULATION_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
for path in (SCRIPT_DIR, str(SIMULATION_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulation_statistical.common import (  # noqa: E402
    _build_round_index,
    as_bool,
    make_unique_avatar_map,
    parse_dict,
)
from simulation_statistical.paths import BENCHMARK_DATA_ROOT  # noqa: E402


DEFAULT_STATE_TABLE_ROOT = ALGORITHMIC_LATENT_ROOT / "artifacts" / "state_tables"
RATE_BINS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)


def _analysis_csv_for_wave(wave: str) -> Path:
    if str(wave) == "learning_wave":
        return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "processed_data" / "df_analysis_learn.csv"
    if str(wave) == "validation_wave":
        return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "processed_data" / "df_analysis_val.csv"
    raise ValueError(f"Unsupported wave '{wave}'.")


def _rounds_csv_for_wave(wave: str) -> Path:
    return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "raw_data" / str(wave) / "player-rounds.csv"


def _players_csv_for_wave(wave: str) -> Path:
    return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "raw_data" / str(wave) / "players.csv"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _rate_bin_index(rate: float | None) -> int | None:
    if rate is None or pd.isna(rate):
        return None
    clipped = float(max(0.0, min(1.0, float(rate))))
    return int(np.argmin(np.abs(RATE_BINS - clipped)))


def _normalized_rank(target_value: float, peer_values: Sequence[float]) -> float:
    if not peer_values:
        return float("nan")
    arr = np.asarray(peer_values, dtype=float)
    less = float(np.sum(arr < target_value))
    equal = float(np.sum(arr == target_value))
    return float((less + 0.5 * equal) / max(len(arr), 1))


def _visible_round_phase(env: Mapping[str, Any], round_idx: int) -> str:
    show_n_rounds = as_bool(env.get("CONFIG_showNRounds", False))
    num_rounds = max(int(_to_float(env.get("CONFIG_numRounds", 1), default=1.0)), 1)
    if show_n_rounds:
        progress = float(max(int(round_idx) - 1, 0)) / float(max(num_rounds - 1, 1))
        if progress < (1.0 / 3.0):
            return "early"
        if progress < (2.0 / 3.0):
            return "mid"
        return "late"
    round_idx = int(round_idx)
    if round_idx <= 1:
        return "r1"
    if round_idx <= 3:
        return "r2_3"
    if round_idx <= 6:
        return "r4_6"
    return "r7plus"


def _analysis_lookup(df_analysis: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    analysis = df_analysis.copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
    config_columns = [column for column in analysis.columns if column.startswith("CONFIG_")]
    keep_columns = ["gameId", "name", "batchId", "createdAt", "finishedAt"] + config_columns
    keep_columns = [column for column in keep_columns if column in analysis.columns]
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in analysis[keep_columns].iterrows():
        payload = row.to_dict()
        game_id = str(payload.get("gameId"))
        for key, value in list(payload.items()):
            if key.startswith("CONFIG_") and key in {
                "CONFIG_allOrNothing",
                "CONFIG_chat",
                "CONFIG_punishmentExists",
                "CONFIG_rewardExists",
                "CONFIG_showNRounds",
                "CONFIG_showOtherSummaries",
                "CONFIG_showPunishmentId",
                "CONFIG_showRewardId",
            }:
                payload[key] = as_bool(value)
        out[game_id] = payload
    return out


def _avatar_lookup(df_players: pd.DataFrame) -> Dict[str, str]:
    if df_players.empty or "_id" not in df_players.columns:
        return {}
    players = df_players.copy()
    players["_id"] = players["_id"].astype(str)
    return {
        str(row["_id"]): str(row.get("data.avatar", "") or "").strip().upper()
        for _, row in players.iterrows()
        if str(row.get("_id", "")).strip()
    }


def _row_lookup_by_player(round_df: pd.DataFrame) -> Dict[str, Mapping[str, Any]]:
    return {str(row["playerId"]): row for _, row in round_df.iterrows()}


def _contribution_rate(contribution: float, endowment: float) -> float:
    denom = max(float(endowment), 1.0)
    return float(contribution) / denom


def _build_current_round_state(round_df: pd.DataFrame) -> Dict[str, Any]:
    row_by_player = _row_lookup_by_player(round_df)
    contributions_by_player = {
        str(player_id): int(_to_float(row.get("data.contribution"), default=0.0))
        for player_id, row in row_by_player.items()
    }
    punished_by_player = {
        str(player_id): parse_dict(row.get("data.punished"))
        for player_id, row in row_by_player.items()
    }
    rewarded_by_player = {
        str(player_id): parse_dict(row.get("data.rewarded"))
        for player_id, row in row_by_player.items()
    }
    punished_by_others = {
        str(player_id): parse_dict(row.get("data.punishedBy"))
        for player_id, row in row_by_player.items()
    }
    rewarded_by_others = {
        str(player_id): parse_dict(row.get("data.rewardedBy"))
        for player_id, row in row_by_player.items()
    }
    costs_by_player = {
        str(player_id): _to_float(row.get("data.costs"), default=float("nan"))
        for player_id, row in row_by_player.items()
    }
    penalties_by_player = {
        str(player_id): _to_float(row.get("data.penalties"), default=float("nan"))
        for player_id, row in row_by_player.items()
    }
    rewards_by_player = {
        str(player_id): _to_float(row.get("data.rewards"), default=float("nan"))
        for player_id, row in row_by_player.items()
    }
    payoff_by_player = {
        str(player_id): _to_float(row.get("data.roundPayoff"), default=float("nan"))
        for player_id, row in row_by_player.items()
    }
    return {
        "row_by_player": row_by_player,
        "contributions_by_player": contributions_by_player,
        "punished_by_player": punished_by_player,
        "rewarded_by_player": rewarded_by_player,
        "punished_by_others": punished_by_others,
        "rewarded_by_others": rewarded_by_others,
        "costs_by_player": costs_by_player,
        "penalties_by_player": penalties_by_player,
        "rewards_by_player": rewards_by_player,
        "payoff_by_player": payoff_by_player,
    }


def _new_history_tracker(player_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        str(player_id): {
            "own_rates": [],
            "own_rate_bins": [],
            "peer_mean_rates": [],
            "peer_std_rates": [],
            "peer_zero_counts": [],
            "peer_full_counts": [],
            "punish_received_total": 0,
            "reward_received_total": 0,
            "punish_given_total": 0,
            "reward_given_total": 0,
            "visible_peer_mean_costs": [],
            "visible_peer_mean_penalties": [],
            "visible_peer_mean_rewards": [],
            "visible_peer_mean_payoffs": [],
            "punished_by_visible_counts": {},
            "punished_target_visible_counts": {},
            "rewarded_by_visible_counts": {},
            "rewarded_target_visible_counts": {},
        }
        for player_id in player_ids
    }


def _expected_norm_visible(
    *,
    prev_peer_mean_rate: float | None,
    current_peer_mean_rate: float | None,
    default_contrib_prop: float | None,
) -> float | None:
    if prev_peer_mean_rate is not None and not pd.isna(prev_peer_mean_rate):
        return float(prev_peer_mean_rate)
    if default_contrib_prop is not None and not pd.isna(default_contrib_prop):
        return float(default_contrib_prop)
    if current_peer_mean_rate is not None and not pd.isna(current_peer_mean_rate):
        return float(current_peer_mean_rate)
    return None


def build_state_tables(
    *,
    wave: str,
    rounds_csv: Path,
    analysis_csv: Path,
    players_csv: Path,
    output_root: Path,
    max_games: int | None = None,
) -> Dict[str, Any]:
    df_rounds = pd.read_csv(rounds_csv)
    df_rounds = _build_round_index(df_rounds)
    df_rounds["gameId"] = df_rounds["gameId"].astype(str)
    df_rounds["playerId"] = df_rounds["playerId"].astype(str)

    df_analysis = pd.read_csv(analysis_csv)
    env_by_game = _analysis_lookup(df_analysis)

    df_players = pd.read_csv(players_csv) if players_csv.exists() else pd.DataFrame()
    avatar_seed_lookup = _avatar_lookup(df_players)

    contribution_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    games_processed = 0

    grouped_games = list(df_rounds.groupby("gameId", sort=True))
    if max_games is not None:
        grouped_games = grouped_games[: int(max_games)]

    for game_id, game_df in grouped_games:
        env = env_by_game.get(str(game_id))
        if not env:
            continue
        games_processed += 1
        game_df = game_df.copy().sort_values(["roundIndex", "playerId"]).reset_index(drop=True)
        player_ids = list(dict.fromkeys(game_df["playerId"].astype(str).tolist()))
        avatar_seed_map = {player_id: avatar_seed_lookup.get(player_id, "") for player_id in player_ids}
        avatar_by_player = make_unique_avatar_map(player_ids, avatar_seed_map)

        previous_state: Dict[str, Any] | None = None
        history_tracker = _new_history_tracker(player_ids)
        for round_idx, round_df in game_df.groupby("roundIndex", sort=True):
            round_idx = int(round_idx)
            round_df = round_df.copy().sort_values(["playerId"]).reset_index(drop=True)
            current_state = _build_current_round_state(round_df)
            current_player_ids = list(round_df["playerId"].astype(str))

            phase_visible = _visible_round_phase(env, round_idx)
            show_n_rounds = as_bool(env.get("CONFIG_showNRounds", False))
            rounds_remaining_visible = (
                max(int(_to_float(env.get("CONFIG_numRounds", 1), default=1.0)) - round_idx + 1, 0)
                if show_n_rounds
                else None
            )
            default_contrib_prop = (
                _to_float(env.get("CONFIG_defaultContribProp"), default=float("nan"))
                if "CONFIG_defaultContribProp" in env
                else float("nan")
            )

            for player_id in current_player_ids:
                row = current_state["row_by_player"][player_id]
                endowment = _to_float(env.get("CONFIG_endowment"), default=20.0)
                contribution = int(_to_float(row.get("data.contribution"), default=0.0))
                contribution_rate = _contribution_rate(contribution, endowment)

                peers = [peer_id for peer_id in current_player_ids if peer_id != player_id]
                prev_peer_rates: List[float] = []
                prev_summary_costs: List[float] = []
                prev_summary_penalties: List[float] = []
                prev_summary_rewards: List[float] = []
                prev_summary_payoffs: List[float] = []
                history_state = history_tracker[str(player_id)]

                own_prev_contribution = None
                own_prev_contribution_rate = None
                own_prev_costs = None
                own_prev_penalties = None
                own_prev_rewards = None
                own_prev_payoff = None
                punish_received_prev_units = 0
                reward_received_prev_units = 0

                if previous_state is not None:
                    own_prev_contribution = int(previous_state["contributions_by_player"].get(player_id, 0))
                    own_prev_contribution_rate = _contribution_rate(own_prev_contribution, endowment)
                    own_prev_costs = previous_state["costs_by_player"].get(player_id)
                    own_prev_penalties = previous_state["penalties_by_player"].get(player_id)
                    own_prev_rewards = previous_state["rewards_by_player"].get(player_id)
                    own_prev_payoff = previous_state["payoff_by_player"].get(player_id)
                    punish_received_prev_units = int(
                        sum(int(units) for units in previous_state["punished_by_others"].get(player_id, {}).values())
                    )
                    reward_received_prev_units = int(
                        sum(int(units) for units in previous_state["rewarded_by_others"].get(player_id, {}).values())
                    )
                    for peer_id in peers:
                        if peer_id not in previous_state["contributions_by_player"]:
                            continue
                        prev_peer_rates.append(
                            _contribution_rate(previous_state["contributions_by_player"][peer_id], endowment)
                        )
                        if as_bool(env.get("CONFIG_showOtherSummaries", False)):
                            prev_summary_costs.append(previous_state["costs_by_player"].get(peer_id, float("nan")))
                            prev_summary_penalties.append(
                                previous_state["penalties_by_player"].get(peer_id, float("nan"))
                            )
                            prev_summary_rewards.append(
                                previous_state["rewards_by_player"].get(peer_id, float("nan"))
                            )
                            prev_summary_payoffs.append(
                                previous_state["payoff_by_player"].get(peer_id, float("nan"))
                            )

                prev_peer_mean_rate = _safe_mean(prev_peer_rates)
                prev_peer_std_rate = _safe_std(prev_peer_rates)
                expected_norm_visible = _expected_norm_visible(
                    prev_peer_mean_rate=(
                        None if pd.isna(prev_peer_mean_rate) else float(prev_peer_mean_rate)
                    ),
                    current_peer_mean_rate=None,
                    default_contrib_prop=(
                        None if pd.isna(default_contrib_prop) else float(default_contrib_prop)
                    ),
                )
                own_history_mean_contribution_rate = _safe_mean(history_state["own_rates"])
                own_history_mode_contribution_bin5 = (
                    int(pd.Series(history_state["own_rate_bins"]).mode().iloc[0])
                    if history_state["own_rate_bins"]
                    else None
                )
                peer_history_mean_contribution_rate = _safe_mean(history_state["peer_mean_rates"])
                peer_history_mean_peer_std_rate = _safe_mean(history_state["peer_std_rates"])
                peer_history_mean_zero_count = _safe_mean(history_state["peer_zero_counts"])
                peer_history_mean_full_count = _safe_mean(history_state["peer_full_counts"])
                cumulative_punish_received_units = int(history_state["punish_received_total"])
                cumulative_reward_received_units = int(history_state["reward_received_total"])
                cumulative_punish_given_units = int(history_state["punish_given_total"])
                cumulative_reward_given_units = int(history_state["reward_given_total"])

                contribution_rows.append(
                    {
                        "wave": str(wave),
                        "row_id": f"{wave}|{game_id}|r{round_idx}|{player_id}",
                        "stage": "contribution",
                        "gameId": str(game_id),
                        "gameName": str(env.get("name", game_id)),
                        "CONFIG_treatmentName": env.get("CONFIG_treatmentName"),
                        "playerId": str(player_id),
                        "playerAvatar": avatar_by_player.get(player_id, player_id),
                        "roundIndex": int(round_idx),
                        "history_rounds_observed": int(round_idx - 1),
                        "history_available": int(previous_state is not None),
                        "round_phase_visible": phase_visible,
                        "round_phase_visible_code": phase_visible,
                        "rounds_remaining_visible": rounds_remaining_visible,
                        "n_players_current_round": int(len(current_player_ids)),
                        "n_peers_current_round": int(len(peers)),
                        "expected_norm_visible": expected_norm_visible,
                        "expected_norm_visible_bin5": _rate_bin_index(expected_norm_visible),
                        "own_prev_contribution": own_prev_contribution,
                        "own_prev_contribution_rate": own_prev_contribution_rate,
                        "own_prev_contribution_bin5": _rate_bin_index(own_prev_contribution_rate),
                        "own_history_mean_contribution_rate": own_history_mean_contribution_rate,
                        "own_history_mean_contribution_bin5": _rate_bin_index(own_history_mean_contribution_rate),
                        "own_history_mode_contribution_bin5": own_history_mode_contribution_bin5,
                        "peer_prev_mean_contribution_rate": prev_peer_mean_rate,
                        "peer_prev_mean_contribution_bin5": _rate_bin_index(prev_peer_mean_rate),
                        "peer_prev_std_contribution_rate": prev_peer_std_rate,
                        "peer_history_mean_contribution_rate": peer_history_mean_contribution_rate,
                        "peer_history_mean_contribution_bin5": _rate_bin_index(peer_history_mean_contribution_rate),
                        "peer_history_mean_peer_std_rate": peer_history_mean_peer_std_rate,
                        "peer_history_mean_zero_count": peer_history_mean_zero_count,
                        "peer_history_mean_full_count": peer_history_mean_full_count,
                        "punished_prev_any": int(punish_received_prev_units > 0),
                        "rewarded_prev_any": int(reward_received_prev_units > 0),
                        "punish_received_prev_units": int(punish_received_prev_units),
                        "reward_received_prev_units": int(reward_received_prev_units),
                        "cumulative_punish_received_units": cumulative_punish_received_units,
                        "cumulative_reward_received_units": cumulative_reward_received_units,
                        "cumulative_punish_given_units": cumulative_punish_given_units,
                        "cumulative_reward_given_units": cumulative_reward_given_units,
                        "prev_round_costs": own_prev_costs,
                        "prev_round_penalties": own_prev_penalties,
                        "prev_round_rewards": own_prev_rewards,
                        "prev_round_payoff": own_prev_payoff,
                        "prev_peer_summary_visible": int(as_bool(env.get("CONFIG_showOtherSummaries", False))),
                        "visible_prev_peer_mean_costs": _safe_mean(prev_summary_costs),
                        "visible_prev_peer_mean_penalties": _safe_mean(prev_summary_penalties),
                        "visible_prev_peer_mean_rewards": _safe_mean(prev_summary_rewards),
                        "visible_prev_peer_mean_payoff": _safe_mean(prev_summary_payoffs),
                        "visible_history_peer_mean_costs": _safe_mean(history_state["visible_peer_mean_costs"]),
                        "visible_history_peer_mean_penalties": _safe_mean(history_state["visible_peer_mean_penalties"]),
                        "visible_history_peer_mean_rewards": _safe_mean(history_state["visible_peer_mean_rewards"]),
                        "visible_history_peer_mean_payoff": _safe_mean(history_state["visible_peer_mean_payoffs"]),
                        "actual_contribution": int(contribution),
                        "actual_contribution_rate": contribution_rate,
                        "actual_contribution_bin5": _rate_bin_index(contribution_rate),
                        "actual_contribution_is_zero": int(contribution <= 0),
                        "actual_contribution_is_full": int(contribution >= int(round(endowment))),
                        **{key: value for key, value in env.items() if str(key).startswith("CONFIG_")},
                    }
                )

            punish_enabled = as_bool(env.get("CONFIG_punishmentExists", False))
            reward_enabled = as_bool(env.get("CONFIG_rewardExists", False))
            if punish_enabled or reward_enabled:
                for focal_player_id in current_player_ids:
                    own_current_contribution = int(current_state["contributions_by_player"].get(focal_player_id, 0))
                    own_current_rate = _contribution_rate(own_current_contribution, endowment)
                    peer_ids = [peer_id for peer_id in current_player_ids if peer_id != focal_player_id]
                    history_state = history_tracker[str(focal_player_id)]
                    peer_current_rates = [
                        _contribution_rate(current_state["contributions_by_player"].get(peer_id, 0), endowment)
                        for peer_id in peer_ids
                    ]
                    peer_current_mean_rate = _safe_mean(peer_current_rates)
                    peer_current_std_rate = _safe_std(peer_current_rates)
                    peer_current_min_rate = float(np.min(np.asarray(peer_current_rates, dtype=float))) if peer_current_rates else float("nan")
                    peer_current_max_rate = float(np.max(np.asarray(peer_current_rates, dtype=float))) if peer_current_rates else float("nan")

                    prev_peer_rates_for_focal: List[float] = []
                    punish_received_prev_units = 0
                    reward_received_prev_units = 0
                    if previous_state is not None:
                        punish_received_prev_units = int(
                            sum(
                                int(units)
                                for units in previous_state["punished_by_others"].get(focal_player_id, {}).values()
                            )
                        )
                        reward_received_prev_units = int(
                            sum(
                                int(units)
                                for units in previous_state["rewarded_by_others"].get(focal_player_id, {}).values()
                            )
                        )
                        for peer_id in peer_ids:
                            if peer_id not in previous_state["contributions_by_player"]:
                                continue
                            prev_peer_rates_for_focal.append(
                                _contribution_rate(previous_state["contributions_by_player"][peer_id], endowment)
                            )
                    prev_peer_mean_rate = _safe_mean(prev_peer_rates_for_focal)
                    expected_norm_visible = _expected_norm_visible(
                        prev_peer_mean_rate=(
                            None if pd.isna(prev_peer_mean_rate) else float(prev_peer_mean_rate)
                        ),
                        current_peer_mean_rate=(
                            None if pd.isna(peer_current_mean_rate) else float(peer_current_mean_rate)
                        ),
                        default_contrib_prop=(
                            None if pd.isna(default_contrib_prop) else float(default_contrib_prop)
                        ),
                    )
                    own_history_mean_contribution_rate = _safe_mean(history_state["own_rates"])
                    own_history_mode_contribution_bin5 = (
                        int(pd.Series(history_state["own_rate_bins"]).mode().iloc[0])
                        if history_state["own_rate_bins"]
                        else None
                    )
                    peer_history_mean_contribution_rate = _safe_mean(history_state["peer_mean_rates"])
                    peer_history_mean_peer_std_rate = _safe_mean(history_state["peer_std_rates"])
                    peer_history_mean_zero_count = _safe_mean(history_state["peer_zero_counts"])
                    peer_history_mean_full_count = _safe_mean(history_state["peer_full_counts"])
                    cumulative_punish_received_units = int(history_state["punish_received_total"])
                    cumulative_reward_received_units = int(history_state["reward_received_total"])
                    cumulative_punish_given_units = int(history_state["punish_given_total"])
                    cumulative_reward_given_units = int(history_state["reward_given_total"])
                    if expected_norm_visible is None or pd.isna(expected_norm_visible):
                        n_peers_below_expected_current = None
                        n_peers_above_expected_current = None
                    else:
                        n_peers_below_expected_current = int(
                            sum(float(rate) < float(expected_norm_visible) for rate in peer_current_rates)
                        )
                        n_peers_above_expected_current = int(
                            sum(float(rate) > float(expected_norm_visible) for rate in peer_current_rates)
                        )

                    for target_player_id in peer_ids:
                        target_current_contribution = int(
                            current_state["contributions_by_player"].get(target_player_id, 0)
                        )
                        target_current_rate = _contribution_rate(target_current_contribution, endowment)
                        target_peer_values = [
                            _contribution_rate(
                                current_state["contributions_by_player"].get(peer_id, 0),
                                endowment,
                            )
                            for peer_id in peer_ids
                        ]
                        observed_punish_units = int(
                            current_state["punished_by_player"].get(focal_player_id, {}).get(target_player_id, 0)
                        )
                        observed_reward_units = int(
                            current_state["rewarded_by_player"].get(focal_player_id, {}).get(target_player_id, 0)
                        )
                        if observed_punish_units > 0 and observed_reward_units > 0:
                            action_label = "both"
                        elif observed_punish_units > 0:
                            action_label = "punish"
                        elif observed_reward_units > 0:
                            action_label = "reward"
                        else:
                            action_label = "none"

                        target_punished_focal_prev_visible = 0
                        focal_punished_target_prev_visible = 0
                        target_rewarded_focal_prev_visible = 0
                        focal_rewarded_target_prev_visible = 0
                        target_punished_focal_history_visible = int(
                            history_state["punished_by_visible_counts"].get(str(target_player_id), 0) > 0
                        )
                        focal_punished_target_history_visible = int(
                            history_state["punished_target_visible_counts"].get(str(target_player_id), 0) > 0
                        )
                        target_rewarded_focal_history_visible = int(
                            history_state["rewarded_by_visible_counts"].get(str(target_player_id), 0) > 0
                        )
                        focal_rewarded_target_history_visible = int(
                            history_state["rewarded_target_visible_counts"].get(str(target_player_id), 0) > 0
                        )
                        target_punished_focal_history_visible_count = int(
                            history_state["punished_by_visible_counts"].get(str(target_player_id), 0)
                        )
                        focal_punished_target_history_visible_count = int(
                            history_state["punished_target_visible_counts"].get(str(target_player_id), 0)
                        )
                        target_rewarded_focal_history_visible_count = int(
                            history_state["rewarded_by_visible_counts"].get(str(target_player_id), 0)
                        )
                        focal_rewarded_target_history_visible_count = int(
                            history_state["rewarded_target_visible_counts"].get(str(target_player_id), 0)
                        )
                        if previous_state is not None and as_bool(env.get("CONFIG_showPunishmentId", False)):
                            target_punished_focal_prev_visible = int(
                                previous_state["punished_by_player"].get(target_player_id, {}).get(focal_player_id, 0)
                                > 0
                            )
                            focal_punished_target_prev_visible = int(
                                previous_state["punished_by_player"].get(focal_player_id, {}).get(target_player_id, 0)
                                > 0
                            )
                        if previous_state is not None and as_bool(env.get("CONFIG_showRewardId", False)):
                            target_rewarded_focal_prev_visible = int(
                                previous_state["rewarded_by_player"].get(target_player_id, {}).get(focal_player_id, 0)
                                > 0
                            )
                            focal_rewarded_target_prev_visible = int(
                                previous_state["rewarded_by_player"].get(focal_player_id, {}).get(target_player_id, 0)
                                > 0
                            )

                        action_rows.append(
                            {
                                "wave": str(wave),
                                "row_id": f"{wave}|{game_id}|r{round_idx}|{focal_player_id}|{target_player_id}",
                                "stage": "action",
                                "gameId": str(game_id),
                                "gameName": str(env.get("name", game_id)),
                                "CONFIG_treatmentName": env.get("CONFIG_treatmentName"),
                                "playerId": str(focal_player_id),
                                "playerAvatar": avatar_by_player.get(focal_player_id, focal_player_id),
                                "targetPlayerId": str(target_player_id),
                                "targetPlayerAvatar": avatar_by_player.get(target_player_id, target_player_id),
                                "roundIndex": int(round_idx),
                                "history_rounds_observed": int(round_idx - 1),
                                "history_available": int(previous_state is not None),
                                "round_phase_visible": phase_visible,
                                "round_phase_visible_code": phase_visible,
                                "rounds_remaining_visible": rounds_remaining_visible,
                                "n_players_current_round": int(len(current_player_ids)),
                                "n_peers_current_round": int(len(peer_ids)),
                                "own_current_contribution": int(own_current_contribution),
                                "own_current_contribution_rate": own_current_rate,
                                "own_current_contribution_bin5": _rate_bin_index(own_current_rate),
                                "own_history_mean_contribution_rate": own_history_mean_contribution_rate,
                                "own_history_mean_contribution_bin5": _rate_bin_index(own_history_mean_contribution_rate),
                                "own_history_mode_contribution_bin5": own_history_mode_contribution_bin5,
                                "peer_current_mean_contribution_rate": peer_current_mean_rate,
                                "peer_current_mean_contribution_bin5": _rate_bin_index(peer_current_mean_rate),
                                "peer_current_std_contribution_rate": peer_current_std_rate,
                                "peer_current_min_contribution_rate": peer_current_min_rate,
                                "peer_current_max_contribution_rate": peer_current_max_rate,
                                "n_peers_zero_current": int(sum(float(rate) <= 0.0 for rate in peer_current_rates)),
                                "n_peers_full_current": int(sum(float(rate) >= 1.0 for rate in peer_current_rates)),
                                "n_peers_below_expected_current": n_peers_below_expected_current,
                                "n_peers_above_expected_current": n_peers_above_expected_current,
                                "expected_norm_visible": expected_norm_visible,
                                "expected_norm_visible_bin5": _rate_bin_index(expected_norm_visible),
                                "target_current_contribution": int(target_current_contribution),
                                "target_current_contribution_rate": target_current_rate,
                                "target_current_contribution_bin5": _rate_bin_index(target_current_rate),
                                "target_minus_peer_mean_current": (
                                    None
                                    if pd.isna(peer_current_mean_rate)
                                    else float(target_current_rate - peer_current_mean_rate)
                                ),
                                "target_minus_expected_norm_visible": (
                                    None
                                    if expected_norm_visible is None or pd.isna(expected_norm_visible)
                                    else float(target_current_rate - expected_norm_visible)
                                ),
                                "target_current_rank_among_peers": _normalized_rank(
                                    target_current_rate,
                                    target_peer_values,
                                ),
                                "peer_prev_mean_contribution_rate": prev_peer_mean_rate,
                                "peer_prev_mean_contribution_bin5": _rate_bin_index(prev_peer_mean_rate),
                                "peer_history_mean_contribution_rate": peer_history_mean_contribution_rate,
                                "peer_history_mean_contribution_bin5": _rate_bin_index(peer_history_mean_contribution_rate),
                                "peer_history_mean_peer_std_rate": peer_history_mean_peer_std_rate,
                                "peer_history_mean_zero_count": peer_history_mean_zero_count,
                                "peer_history_mean_full_count": peer_history_mean_full_count,
                                "punished_prev_any": int(punish_received_prev_units > 0),
                                "rewarded_prev_any": int(reward_received_prev_units > 0),
                                "punish_received_prev_units": int(punish_received_prev_units),
                                "reward_received_prev_units": int(reward_received_prev_units),
                                "cumulative_punish_received_units": cumulative_punish_received_units,
                                "cumulative_reward_received_units": cumulative_reward_received_units,
                                "cumulative_punish_given_units": cumulative_punish_given_units,
                                "cumulative_reward_given_units": cumulative_reward_given_units,
                                "punishment_id_visible": int(as_bool(env.get("CONFIG_showPunishmentId", False))),
                                "reward_id_visible": int(as_bool(env.get("CONFIG_showRewardId", False))),
                                "target_punished_focal_prev_visible": int(target_punished_focal_prev_visible),
                                "focal_punished_target_prev_visible": int(focal_punished_target_prev_visible),
                                "target_punished_focal_history_visible": int(target_punished_focal_history_visible),
                                "focal_punished_target_history_visible": int(focal_punished_target_history_visible),
                                "target_punished_focal_history_visible_count": target_punished_focal_history_visible_count,
                                "focal_punished_target_history_visible_count": focal_punished_target_history_visible_count,
                                "target_rewarded_focal_prev_visible": int(target_rewarded_focal_prev_visible),
                                "focal_rewarded_target_prev_visible": int(focal_rewarded_target_prev_visible),
                                "target_rewarded_focal_history_visible": int(target_rewarded_focal_history_visible),
                                "focal_rewarded_target_history_visible": int(focal_rewarded_target_history_visible),
                                "target_rewarded_focal_history_visible_count": target_rewarded_focal_history_visible_count,
                                "focal_rewarded_target_history_visible_count": focal_rewarded_target_history_visible_count,
                                "observed_punish_units": int(observed_punish_units),
                                "observed_reward_units": int(observed_reward_units),
                                "observed_any_punish": int(observed_punish_units > 0),
                                "observed_any_reward": int(observed_reward_units > 0),
                                "observed_any_action": int((observed_punish_units > 0) or (observed_reward_units > 0)),
                                "observed_action_label": action_label,
                                **{key: value for key, value in env.items() if str(key).startswith("CONFIG_")},
                            }
                        )

            show_other_summaries = as_bool(env.get("CONFIG_showOtherSummaries", False))
            show_punishment_id = as_bool(env.get("CONFIG_showPunishmentId", False))
            show_reward_id = as_bool(env.get("CONFIG_showRewardId", False))
            for focal_player_id in current_player_ids:
                focal_history = history_tracker[str(focal_player_id)]
                own_rate_this_round = _contribution_rate(
                    current_state["contributions_by_player"].get(focal_player_id, 0),
                    endowment,
                )
                peer_ids = [peer_id for peer_id in current_player_ids if peer_id != focal_player_id]
                peer_rates_this_round = [
                    _contribution_rate(current_state["contributions_by_player"].get(peer_id, 0), endowment)
                    for peer_id in peer_ids
                ]
                focal_history["own_rates"].append(float(own_rate_this_round))
                focal_history["own_rate_bins"].append(int(_rate_bin_index(own_rate_this_round)))
                if peer_rates_this_round:
                    focal_history["peer_mean_rates"].append(_safe_mean(peer_rates_this_round))
                    focal_history["peer_std_rates"].append(_safe_std(peer_rates_this_round))
                    focal_history["peer_zero_counts"].append(
                        int(sum(float(rate) <= 0.0 for rate in peer_rates_this_round))
                    )
                    focal_history["peer_full_counts"].append(
                        int(sum(float(rate) >= 1.0 for rate in peer_rates_this_round))
                    )
                focal_history["punish_received_total"] += int(
                    sum(int(units) for units in current_state["punished_by_others"].get(focal_player_id, {}).values())
                )
                focal_history["reward_received_total"] += int(
                    sum(int(units) for units in current_state["rewarded_by_others"].get(focal_player_id, {}).values())
                )
                focal_history["punish_given_total"] += int(
                    sum(int(units) for units in current_state["punished_by_player"].get(focal_player_id, {}).values())
                )
                focal_history["reward_given_total"] += int(
                    sum(int(units) for units in current_state["rewarded_by_player"].get(focal_player_id, {}).values())
                )
                if show_other_summaries:
                    peer_costs = [
                        current_state["costs_by_player"].get(peer_id, float("nan"))
                        for peer_id in peer_ids
                    ]
                    peer_penalties = [
                        current_state["penalties_by_player"].get(peer_id, float("nan"))
                        for peer_id in peer_ids
                    ]
                    peer_rewards = [
                        current_state["rewards_by_player"].get(peer_id, float("nan"))
                        for peer_id in peer_ids
                    ]
                    peer_payoffs = [
                        current_state["payoff_by_player"].get(peer_id, float("nan"))
                        for peer_id in peer_ids
                    ]
                    focal_history["visible_peer_mean_costs"].append(_safe_mean(peer_costs))
                    focal_history["visible_peer_mean_penalties"].append(_safe_mean(peer_penalties))
                    focal_history["visible_peer_mean_rewards"].append(_safe_mean(peer_rewards))
                    focal_history["visible_peer_mean_payoffs"].append(_safe_mean(peer_payoffs))
                if show_punishment_id:
                    for target_player_id, units in current_state["punished_by_player"].get(focal_player_id, {}).items():
                        focal_history["punished_target_visible_counts"][str(target_player_id)] = (
                            int(focal_history["punished_target_visible_counts"].get(str(target_player_id), 0))
                            + int(units)
                        )
                    for source_player_id, units in current_state["punished_by_others"].get(focal_player_id, {}).items():
                        focal_history["punished_by_visible_counts"][str(source_player_id)] = (
                            int(focal_history["punished_by_visible_counts"].get(str(source_player_id), 0))
                            + int(units)
                        )
                if show_reward_id:
                    for target_player_id, units in current_state["rewarded_by_player"].get(focal_player_id, {}).items():
                        focal_history["rewarded_target_visible_counts"][str(target_player_id)] = (
                            int(focal_history["rewarded_target_visible_counts"].get(str(target_player_id), 0))
                            + int(units)
                        )
                    for source_player_id, units in current_state["rewarded_by_others"].get(focal_player_id, {}).items():
                        focal_history["rewarded_by_visible_counts"][str(source_player_id)] = (
                            int(focal_history["rewarded_by_visible_counts"].get(str(source_player_id), 0))
                            + int(units)
                        )

            previous_state = current_state

    contribution_df = pd.DataFrame(contribution_rows)
    action_df = pd.DataFrame(action_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    contribution_path = output_root / f"{wave}_contribution_stage.parquet"
    action_path = output_root / f"{wave}_action_stage.parquet"
    summary_path = output_root / f"{wave}_state_table_summary.json"

    contribution_df.to_parquet(contribution_path, index=False)
    action_df.to_parquet(action_path, index=False)

    summary = {
        "wave": str(wave),
        "inputs": {
            "rounds_csv": str(rounds_csv),
            "analysis_csv": str(analysis_csv),
            "players_csv": str(players_csv),
        },
        "outputs": {
            "contribution_stage": str(contribution_path),
            "action_stage": str(action_path),
        },
        "n_games_processed": int(games_processed),
        "n_contribution_rows": int(len(contribution_df)),
        "n_action_rows": int(len(action_df)),
        "contribution_columns": contribution_df.columns.tolist(),
        "action_columns": action_df.columns.tolist(),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build visibility-aware contribution/action state tables for the algorithmic-latent simulator."
    )
    parser.add_argument("--wave", type=str, default="learning_wave")
    parser.add_argument("--rounds_csv", type=str, default=None)
    parser.add_argument("--analysis_csv", type=str, default=None)
    parser.add_argument("--players_csv", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_STATE_TABLE_ROOT))
    parser.add_argument("--max_games", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    wave = str(args.wave).strip()
    summary = build_state_tables(
        wave=wave,
        rounds_csv=Path(args.rounds_csv).resolve() if args.rounds_csv else _rounds_csv_for_wave(wave),
        analysis_csv=Path(args.analysis_csv).resolve() if args.analysis_csv else _analysis_csv_for_wave(wave),
        players_csv=Path(args.players_csv).resolve() if args.players_csv else _players_csv_for_wave(wave),
        output_root=Path(args.output_root).resolve(),
        max_games=args.max_games,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
