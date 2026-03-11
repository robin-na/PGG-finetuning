from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from simulation_statistical.algorithmic_latent.inference.build_state_table import (
    RATE_BINS,
    _expected_norm_visible,
    _rate_bin_index,
    _safe_mean,
    _safe_std,
    _visible_round_phase,
)
from simulation_statistical.algorithmic_latent.inference.fit_env_family_mixture import (
    DEFAULT_METADATA_PATH as DEFAULT_ENV_FAMILY_METADATA_PATH,
    DEFAULT_MODEL_OUTPUT_PATH as DEFAULT_ENV_FAMILY_MODEL_PATH,
)
from simulation_statistical.algorithmic_latent.inference.fit_action_rate_calibration import (
    DEFAULT_OUTPUT_PATH as DEFAULT_ACTION_RATE_CALIBRATION_PATH,
)
from simulation_statistical.algorithmic_latent.inference.fit_family_policies import (
    DEFAULT_MODEL_OUTPUT_PATH as DEFAULT_FAMILY_POLICY_MODEL_PATH,
)
from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import as_bool, parse_dict


ALGORITHMIC_LATENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = ALGORITHMIC_LATENT_ROOT / "artifacts"
EPS = 1e-12


@dataclass(frozen=True)
class AlgorithmicLatentPolicyConfig:
    artifacts_root: str | None = None


@dataclass
class AlgorithmicLatentRoundState:
    round_idx: int
    contributions_by_player: Dict[str, int]
    punish_by_player: Dict[str, Dict[str, int]]
    reward_by_player: Dict[str, Dict[str, int]]
    costs_by_player: Dict[str, float]
    penalties_by_player: Dict[str, float]
    rewards_by_player: Dict[str, float]
    payoff_by_player: Dict[str, float]


@dataclass
class AlgorithmicLatentGameState:
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    family_by_player: Dict[str, str]
    cluster_by_player: Dict[str, int]
    archetype_label_by_player: Dict[str, str]
    history_rounds: List[AlgorithmicLatentRoundState] = field(default_factory=list)


def _row_to_feature_dict(row: Mapping[str, Any], feature_names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for feature in feature_names:
        value = row.get(feature)
        if value is None or pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            out[str(feature)] = int(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)):
            out[str(feature)] = float(value)
        else:
            out[str(feature)] = str(value)
    return out


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    if probs.ndim != 1 or probs.size == 0 or total <= 0.0:
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError("Expected a non-empty 1D probability vector.")
        return np.full_like(probs, fill_value=1.0 / probs.size, dtype=float)
    return probs / total


def _metric_by_player(round_rows: pd.DataFrame, column_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for _, row in round_rows.iterrows():
        player_id = str(row.get("playerId"))
        try:
            out[player_id] = float(row.get(column_name))
        except Exception:
            out[player_id] = 0.0
    return out


def _action_dicts_by_player(round_rows: pd.DataFrame, column_name: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for _, row in round_rows.iterrows():
        out[str(row.get("playerId"))] = parse_dict(row.get(column_name))
    return out


def _contrib_by_player(round_rows: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, row in round_rows.iterrows():
        try:
            out[str(row.get("playerId"))] = int(float(row.get("data.contribution", 0) or 0))
        except Exception:
            out[str(row.get("playerId"))] = 0
    return out


def _round_payoff_by_player(round_rows: pd.DataFrame) -> Dict[str, float]:
    return _metric_by_player(round_rows, "data.roundPayoff")


def _player_inbound_units(
    actions_by_player: Mapping[str, Mapping[str, int]],
    target_player_id: str,
) -> int:
    target = str(target_player_id)
    return int(sum(int(targets.get(target, 0)) for targets in actions_by_player.values()))


def _sum_units(values: Mapping[str, int]) -> int:
    return int(sum(int(value) for value in values.values()))


def _available_action_budget_coins(
    *,
    env: Mapping[str, Any],
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
) -> float:
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    own_contrib = float(contributions_by_player.get(str(focal_player_id), 0.0))
    multiplier = float(env.get("CONFIG_multiplier", 0) or 0)
    n_players = max(len(contributions_by_player), 1)
    share = multiplier * float(sum(int(value) for value in contributions_by_player.values())) / n_players
    return float(max(endowment - own_contrib + share, 0.0))


def _sample_from_label_probs(
    *,
    label_probs: Mapping[str, float],
    allowed_labels: Sequence[str],
    rng: np.random.Generator,
) -> str:
    labels = [str(label) for label in allowed_labels]
    probs = np.asarray([float(label_probs.get(label, 0.0)) for label in labels], dtype=float)
    probs = _normalize_probs(probs)
    return str(rng.choice(np.asarray(labels, dtype=object), p=probs))


def _compute_round_financials(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    punish_cost = float(env.get("CONFIG_punishmentCost", 1) or 1)
    reward_cost = float(env.get("CONFIG_rewardCost", 1) or 1)
    punish_magnitude = float(env.get("CONFIG_punishmentMagnitude", 1) or 1)
    reward_magnitude = float(env.get("CONFIG_rewardMagnitude", 1) or 1)

    costs_by_player: Dict[str, float] = {}
    penalties_by_player: Dict[str, float] = {}
    rewards_by_player: Dict[str, float] = {}
    for player_id in player_ids:
        punish_out_units = float(_sum_units(punish_by_player.get(str(player_id), {})))
        reward_out_units = float(_sum_units(reward_by_player.get(str(player_id), {})))
        punish_in_units = float(_player_inbound_units(punish_by_player, str(player_id)))
        reward_in_units = float(_player_inbound_units(reward_by_player, str(player_id)))
        costs_by_player[str(player_id)] = punish_out_units * punish_cost + reward_out_units * reward_cost
        penalties_by_player[str(player_id)] = punish_in_units * punish_magnitude
        rewards_by_player[str(player_id)] = reward_in_units * reward_magnitude
    return costs_by_player, penalties_by_player, rewards_by_player


def _build_round_state(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    round_idx: int,
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
    costs_by_player: Mapping[str, float] | None = None,
    penalties_by_player: Mapping[str, float] | None = None,
    rewards_by_player: Mapping[str, float] | None = None,
) -> AlgorithmicLatentRoundState:
    if costs_by_player is None or penalties_by_player is None or rewards_by_player is None:
        computed_costs, computed_penalties, computed_rewards = _compute_round_financials(
            env=env,
            player_ids=player_ids,
            punish_by_player=punish_by_player,
            reward_by_player=reward_by_player,
        )
        costs_by_player = computed_costs if costs_by_player is None else costs_by_player
        penalties_by_player = computed_penalties if penalties_by_player is None else penalties_by_player
        rewards_by_player = computed_rewards if rewards_by_player is None else rewards_by_player
    return AlgorithmicLatentRoundState(
        round_idx=int(round_idx),
        contributions_by_player={str(k): int(v) for k, v in contributions_by_player.items()},
        punish_by_player={
            str(player_id): {str(target_id): int(units) for target_id, units in targets.items() if int(units) > 0}
            for player_id, targets in punish_by_player.items()
        },
        reward_by_player={
            str(player_id): {str(target_id): int(units) for target_id, units in targets.items() if int(units) > 0}
            for player_id, targets in reward_by_player.items()
        },
        costs_by_player={str(k): float(v) for k, v in costs_by_player.items()},
        penalties_by_player={str(k): float(v) for k, v in penalties_by_player.items()},
        rewards_by_player={str(k): float(v) for k, v in rewards_by_player.items()},
        payoff_by_player={str(k): float(v) for k, v in payoff_by_player.items()},
    )


def _history_stats(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    history_rounds: Sequence[AlgorithmicLatentRoundState],
) -> Dict[str, Any]:
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    focal = str(focal_player_id)
    peers = [str(peer_id) for peer_id in player_ids if str(peer_id) != focal]
    show_other_summaries = as_bool(env.get("CONFIG_showOtherSummaries", False))
    show_punishment_id = as_bool(env.get("CONFIG_showPunishmentId", False))
    show_reward_id = as_bool(env.get("CONFIG_showRewardId", False))

    own_rates: List[float] = []
    own_rate_bins: List[int] = []
    peer_mean_rates: List[float] = []
    peer_std_rates: List[float] = []
    peer_zero_counts: List[int] = []
    peer_full_counts: List[int] = []
    visible_peer_mean_costs: List[float] = []
    visible_peer_mean_penalties: List[float] = []
    visible_peer_mean_rewards: List[float] = []
    visible_peer_mean_payoffs: List[float] = []
    punish_received_total = 0
    reward_received_total = 0
    punish_given_total = 0
    reward_given_total = 0
    punished_by_visible_counts: Dict[str, int] = {}
    punished_target_visible_counts: Dict[str, int] = {}
    rewarded_by_visible_counts: Dict[str, int] = {}
    rewarded_target_visible_counts: Dict[str, int] = {}

    previous_round = history_rounds[-1] if history_rounds else None
    for round_state in history_rounds:
        own_rate = float(round_state.contributions_by_player.get(focal, 0)) / max(endowment, 1.0)
        own_rates.append(own_rate)
        own_rate_bins.append(int(_rate_bin_index(own_rate)))

        peer_rates = [
            float(round_state.contributions_by_player.get(peer_id, 0)) / max(endowment, 1.0)
            for peer_id in peers
        ]
        if peer_rates:
            peer_mean_rates.append(_safe_mean(peer_rates))
            peer_std_rates.append(_safe_std(peer_rates))
            peer_zero_counts.append(int(sum(float(rate) <= 0.0 for rate in peer_rates)))
            peer_full_counts.append(int(sum(float(rate) >= 1.0 for rate in peer_rates)))

        punish_received_total += _player_inbound_units(round_state.punish_by_player, focal)
        reward_received_total += _player_inbound_units(round_state.reward_by_player, focal)
        punish_given_total += _sum_units(round_state.punish_by_player.get(focal, {}))
        reward_given_total += _sum_units(round_state.reward_by_player.get(focal, {}))

        if show_other_summaries:
            peer_costs = [float(round_state.costs_by_player.get(peer_id, 0.0)) for peer_id in peers]
            peer_penalties = [float(round_state.penalties_by_player.get(peer_id, 0.0)) for peer_id in peers]
            peer_rewards = [float(round_state.rewards_by_player.get(peer_id, 0.0)) for peer_id in peers]
            peer_payoffs = [float(round_state.payoff_by_player.get(peer_id, 0.0)) for peer_id in peers]
            visible_peer_mean_costs.append(_safe_mean(peer_costs))
            visible_peer_mean_penalties.append(_safe_mean(peer_penalties))
            visible_peer_mean_rewards.append(_safe_mean(peer_rewards))
            visible_peer_mean_payoffs.append(_safe_mean(peer_payoffs))

        if show_punishment_id:
            for target_player_id, units in round_state.punish_by_player.get(focal, {}).items():
                punished_target_visible_counts[str(target_player_id)] = (
                    int(punished_target_visible_counts.get(str(target_player_id), 0)) + int(units)
                )
            for source_player_id, targets in round_state.punish_by_player.items():
                units = int(targets.get(focal, 0))
                if units > 0:
                    punished_by_visible_counts[str(source_player_id)] = (
                        int(punished_by_visible_counts.get(str(source_player_id), 0)) + units
                    )

        if show_reward_id:
            for target_player_id, units in round_state.reward_by_player.get(focal, {}).items():
                rewarded_target_visible_counts[str(target_player_id)] = (
                    int(rewarded_target_visible_counts.get(str(target_player_id), 0)) + int(units)
                )
            for source_player_id, targets in round_state.reward_by_player.items():
                units = int(targets.get(focal, 0))
                if units > 0:
                    rewarded_by_visible_counts[str(source_player_id)] = (
                        int(rewarded_by_visible_counts.get(str(source_player_id), 0)) + units
                    )

    own_prev_contribution = None
    own_prev_contribution_rate = None
    own_prev_costs = None
    own_prev_penalties = None
    own_prev_rewards = None
    own_prev_payoff = None
    punish_received_prev_units = 0
    reward_received_prev_units = 0
    prev_peer_rates: List[float] = []
    prev_summary_costs: List[float] = []
    prev_summary_penalties: List[float] = []
    prev_summary_rewards: List[float] = []
    prev_summary_payoffs: List[float] = []
    if previous_round is not None:
        own_prev_contribution = int(previous_round.contributions_by_player.get(focal, 0))
        own_prev_contribution_rate = float(own_prev_contribution) / max(endowment, 1.0)
        own_prev_costs = float(previous_round.costs_by_player.get(focal, 0.0))
        own_prev_penalties = float(previous_round.penalties_by_player.get(focal, 0.0))
        own_prev_rewards = float(previous_round.rewards_by_player.get(focal, 0.0))
        own_prev_payoff = float(previous_round.payoff_by_player.get(focal, 0.0))
        punish_received_prev_units = _player_inbound_units(previous_round.punish_by_player, focal)
        reward_received_prev_units = _player_inbound_units(previous_round.reward_by_player, focal)
        prev_peer_rates = [
            float(previous_round.contributions_by_player.get(peer_id, 0)) / max(endowment, 1.0)
            for peer_id in peers
        ]
        if show_other_summaries:
            prev_summary_costs = [float(previous_round.costs_by_player.get(peer_id, 0.0)) for peer_id in peers]
            prev_summary_penalties = [float(previous_round.penalties_by_player.get(peer_id, 0.0)) for peer_id in peers]
            prev_summary_rewards = [float(previous_round.rewards_by_player.get(peer_id, 0.0)) for peer_id in peers]
            prev_summary_payoffs = [float(previous_round.payoff_by_player.get(peer_id, 0.0)) for peer_id in peers]

    return {
        "own_prev_contribution": own_prev_contribution,
        "own_prev_contribution_rate": own_prev_contribution_rate,
        "own_prev_contribution_bin5": _rate_bin_index(own_prev_contribution_rate),
        "own_prev_costs": own_prev_costs,
        "own_prev_penalties": own_prev_penalties,
        "own_prev_rewards": own_prev_rewards,
        "own_prev_payoff": own_prev_payoff,
        "punish_received_prev_units": int(punish_received_prev_units),
        "reward_received_prev_units": int(reward_received_prev_units),
        "prev_peer_mean_rate": _safe_mean(prev_peer_rates),
        "prev_peer_std_rate": _safe_std(prev_peer_rates),
        "prev_summary_costs": prev_summary_costs,
        "prev_summary_penalties": prev_summary_penalties,
        "prev_summary_rewards": prev_summary_rewards,
        "prev_summary_payoffs": prev_summary_payoffs,
        "own_history_mean_contribution_rate": _safe_mean(own_rates),
        "own_history_mean_contribution_bin5": _rate_bin_index(_safe_mean(own_rates)),
        "own_history_mode_contribution_bin5": (
            int(pd.Series(own_rate_bins).mode().iloc[0]) if own_rate_bins else None
        ),
        "peer_history_mean_contribution_rate": _safe_mean(peer_mean_rates),
        "peer_history_mean_contribution_bin5": _rate_bin_index(_safe_mean(peer_mean_rates)),
        "peer_history_mean_peer_std_rate": _safe_mean(peer_std_rates),
        "peer_history_mean_zero_count": _safe_mean(peer_zero_counts),
        "peer_history_mean_full_count": _safe_mean(peer_full_counts),
        "cumulative_punish_received_units": int(punish_received_total),
        "cumulative_reward_received_units": int(reward_received_total),
        "cumulative_punish_given_units": int(punish_given_total),
        "cumulative_reward_given_units": int(reward_given_total),
        "visible_prev_peer_mean_costs": _safe_mean(prev_summary_costs),
        "visible_prev_peer_mean_penalties": _safe_mean(prev_summary_penalties),
        "visible_prev_peer_mean_rewards": _safe_mean(prev_summary_rewards),
        "visible_prev_peer_mean_payoff": _safe_mean(prev_summary_payoffs),
        "visible_history_peer_mean_costs": _safe_mean(visible_peer_mean_costs),
        "visible_history_peer_mean_penalties": _safe_mean(visible_peer_mean_penalties),
        "visible_history_peer_mean_rewards": _safe_mean(visible_peer_mean_rewards),
        "visible_history_peer_mean_payoff": _safe_mean(visible_peer_mean_payoffs),
        "punished_by_visible_counts": punished_by_visible_counts,
        "punished_target_visible_counts": punished_target_visible_counts,
        "rewarded_by_visible_counts": rewarded_by_visible_counts,
        "rewarded_target_visible_counts": rewarded_target_visible_counts,
    }


def _base_feature_row(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    round_idx: int,
    history_rounds: Sequence[AlgorithmicLatentRoundState],
) -> Dict[str, Any]:
    history = _history_stats(
        env=env,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        history_rounds=history_rounds,
    )
    show_n_rounds = as_bool(env.get("CONFIG_showNRounds", False))
    rounds_remaining_visible = (
        max(int(float(env.get("CONFIG_numRounds", 1) or 1)) - int(round_idx) + 1, 0)
        if show_n_rounds
        else None
    )
    default_contrib_prop = env.get("CONFIG_defaultContribProp", None)
    prev_peer_mean_rate = history["prev_peer_mean_rate"]
    expected_norm_visible = _expected_norm_visible(
        prev_peer_mean_rate=None if pd.isna(prev_peer_mean_rate) else float(prev_peer_mean_rate),
        current_peer_mean_rate=None,
        default_contrib_prop=None if pd.isna(default_contrib_prop) else float(default_contrib_prop),
    )
    row: Dict[str, Any] = {
        "round_phase_visible": _visible_round_phase(env, int(round_idx)),
        "round_phase_visible_code": _visible_round_phase(env, int(round_idx)),
        "roundIndex": int(round_idx),
        "history_rounds_observed": int(max(int(round_idx) - 1, 0)),
        "history_available": int(len(history_rounds) > 0),
        "rounds_remaining_visible": rounds_remaining_visible,
        "n_players_current_round": int(len(player_ids)),
        "n_peers_current_round": int(max(len(player_ids) - 1, 0)),
        "expected_norm_visible": expected_norm_visible,
        "expected_norm_visible_bin5": _rate_bin_index(expected_norm_visible),
        "prev_peer_summary_visible": int(as_bool(env.get("CONFIG_showOtherSummaries", False))),
        "own_prev_contribution": history["own_prev_contribution"],
        "own_prev_contribution_rate": history["own_prev_contribution_rate"],
        "own_prev_contribution_bin5": history["own_prev_contribution_bin5"],
        "peer_prev_mean_contribution_rate": history["prev_peer_mean_rate"],
        "peer_prev_mean_contribution_bin5": _rate_bin_index(history["prev_peer_mean_rate"]),
        "peer_prev_std_contribution_rate": history["prev_peer_std_rate"],
        "punished_prev_any": int(history["punish_received_prev_units"] > 0),
        "rewarded_prev_any": int(history["reward_received_prev_units"] > 0),
        "punish_received_prev_units": int(history["punish_received_prev_units"]),
        "reward_received_prev_units": int(history["reward_received_prev_units"]),
        "prev_round_costs": history["own_prev_costs"],
        "prev_round_penalties": history["own_prev_penalties"],
        "prev_round_rewards": history["own_prev_rewards"],
        "prev_round_payoff": history["own_prev_payoff"],
        **{
            key: value
            for key, value in history.items()
            if not key.endswith("_counts")
            and key not in {
                "own_prev_contribution",
                "own_prev_contribution_rate",
                "own_prev_contribution_bin5",
                "prev_peer_mean_rate",
                "prev_peer_std_rate",
                "own_prev_costs",
                "own_prev_penalties",
                "own_prev_rewards",
                "own_prev_payoff",
                "punish_received_prev_units",
                "reward_received_prev_units",
            }
        },
        **{key: value for key, value in env.items() if str(key).startswith("CONFIG_")},
    }
    return row


def _contribution_feature_row(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    round_idx: int,
    history_rounds: Sequence[AlgorithmicLatentRoundState],
) -> Dict[str, Any]:
    return _base_feature_row(
        env=env,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        round_idx=round_idx,
        history_rounds=history_rounds,
    )


def _action_feature_row(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    target_player_id: str,
    round_idx: int,
    history_rounds: Sequence[AlgorithmicLatentRoundState],
    contributions_by_player: Mapping[str, int],
) -> Dict[str, Any]:
    base_row = _base_feature_row(
        env=env,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        round_idx=round_idx,
        history_rounds=history_rounds,
    )
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    focal = str(focal_player_id)
    target = str(target_player_id)
    peer_ids = [str(player_id) for player_id in player_ids if str(player_id) != focal]
    peer_rates = [
        float(contributions_by_player.get(peer_id, 0)) / max(endowment, 1.0)
        for peer_id in peer_ids
    ]
    peer_mean_rate = _safe_mean(peer_rates)
    expected_norm_visible = _expected_norm_visible(
        prev_peer_mean_rate=(
            None if pd.isna(base_row.get("peer_prev_mean_contribution_rate")) else float(base_row["peer_prev_mean_contribution_rate"])
        ),
        current_peer_mean_rate=None if pd.isna(peer_mean_rate) else float(peer_mean_rate),
        default_contrib_prop=None if pd.isna(env.get("CONFIG_defaultContribProp", np.nan)) else float(env.get("CONFIG_defaultContribProp")),
    )
    target_rate = float(contributions_by_player.get(target, 0)) / max(endowment, 1.0)
    history = _history_stats(
        env=env,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        history_rounds=history_rounds,
    )
    target_punished_focal_history_visible_count = int(history["punished_by_visible_counts"].get(target, 0))
    focal_punished_target_history_visible_count = int(history["punished_target_visible_counts"].get(target, 0))
    target_rewarded_focal_history_visible_count = int(history["rewarded_by_visible_counts"].get(target, 0))
    focal_rewarded_target_history_visible_count = int(history["rewarded_target_visible_counts"].get(target, 0))
    previous_round = history_rounds[-1] if history_rounds else None
    target_punished_focal_prev_visible = 0
    focal_punished_target_prev_visible = 0
    target_rewarded_focal_prev_visible = 0
    focal_rewarded_target_prev_visible = 0
    if previous_round is not None and as_bool(env.get("CONFIG_showPunishmentId", False)):
        target_punished_focal_prev_visible = int(previous_round.punish_by_player.get(target, {}).get(focal, 0) > 0)
        focal_punished_target_prev_visible = int(previous_round.punish_by_player.get(focal, {}).get(target, 0) > 0)
    if previous_round is not None and as_bool(env.get("CONFIG_showRewardId", False)):
        target_rewarded_focal_prev_visible = int(previous_round.reward_by_player.get(target, {}).get(focal, 0) > 0)
        focal_rewarded_target_prev_visible = int(previous_round.reward_by_player.get(focal, {}).get(target, 0) > 0)

    target_peer_values = [float(contributions_by_player.get(peer_id, 0)) / max(endowment, 1.0) for peer_id in peer_ids]
    base_row.update(
        {
            "own_current_contribution": int(contributions_by_player.get(focal, 0)),
            "own_current_contribution_rate": float(contributions_by_player.get(focal, 0)) / max(endowment, 1.0),
            "own_current_contribution_bin5": _rate_bin_index(float(contributions_by_player.get(focal, 0)) / max(endowment, 1.0)),
            "peer_current_mean_contribution_rate": peer_mean_rate,
            "peer_current_mean_contribution_bin5": _rate_bin_index(peer_mean_rate),
            "peer_current_std_contribution_rate": _safe_std(peer_rates),
            "peer_current_min_contribution_rate": float(np.min(np.asarray(peer_rates, dtype=float))) if peer_rates else float("nan"),
            "peer_current_max_contribution_rate": float(np.max(np.asarray(peer_rates, dtype=float))) if peer_rates else float("nan"),
            "n_peers_zero_current": int(sum(float(rate) <= 0.0 for rate in peer_rates)),
            "n_peers_full_current": int(sum(float(rate) >= 1.0 for rate in peer_rates)),
            "n_peers_below_expected_current": (
                None
                if expected_norm_visible is None or pd.isna(expected_norm_visible)
                else int(sum(float(rate) < float(expected_norm_visible) for rate in peer_rates))
            ),
            "n_peers_above_expected_current": (
                None
                if expected_norm_visible is None or pd.isna(expected_norm_visible)
                else int(sum(float(rate) > float(expected_norm_visible) for rate in peer_rates))
            ),
            "expected_norm_visible": expected_norm_visible,
            "expected_norm_visible_bin5": _rate_bin_index(expected_norm_visible),
            "target_current_contribution": int(contributions_by_player.get(target, 0)),
            "target_current_contribution_rate": target_rate,
            "target_current_contribution_bin5": _rate_bin_index(target_rate),
            "target_minus_peer_mean_current": None if pd.isna(peer_mean_rate) else float(target_rate - peer_mean_rate),
            "target_minus_expected_norm_visible": (
                None
                if expected_norm_visible is None or pd.isna(expected_norm_visible)
                else float(target_rate - expected_norm_visible)
            ),
            "target_current_rank_among_peers": (
                0.5 if not target_peer_values else float((np.sum(np.asarray(target_peer_values) < target_rate) + 0.5 * np.sum(np.asarray(target_peer_values) == target_rate)) / max(len(target_peer_values), 1))
            ),
            "punishment_id_visible": int(as_bool(env.get("CONFIG_showPunishmentId", False))),
            "reward_id_visible": int(as_bool(env.get("CONFIG_showRewardId", False))),
            "target_punished_focal_prev_visible": int(target_punished_focal_prev_visible),
            "focal_punished_target_prev_visible": int(focal_punished_target_prev_visible),
            "target_rewarded_focal_prev_visible": int(target_rewarded_focal_prev_visible),
            "focal_rewarded_target_prev_visible": int(focal_rewarded_target_prev_visible),
            "target_punished_focal_history_visible": int(target_punished_focal_history_visible_count > 0),
            "focal_punished_target_history_visible": int(focal_punished_target_history_visible_count > 0),
            "target_rewarded_focal_history_visible": int(target_rewarded_focal_history_visible_count > 0),
            "focal_rewarded_target_history_visible": int(focal_rewarded_target_history_visible_count > 0),
            "target_punished_focal_history_visible_count": target_punished_focal_history_visible_count,
            "focal_punished_target_history_visible_count": focal_punished_target_history_visible_count,
            "target_rewarded_focal_history_visible_count": target_rewarded_focal_history_visible_count,
            "focal_rewarded_target_history_visible_count": focal_rewarded_target_history_visible_count,
        }
    )
    return base_row


class AlgorithmicLatentPolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor,
        family_model: Mapping[str, Any],
        family_names: Sequence[str],
        action_rate_calibration: Mapping[str, Any] | None = None,
    ) -> None:
        self.env_model = env_model
        self.family_model = family_model
        self.family_names = [str(name) for name in family_names]
        self.family_to_index = {name: idx + 1 for idx, name in enumerate(self.family_names)}
        self.family_payloads = family_model["families"]
        self.action_rate_calibration = dict(action_rate_calibration or {})

    @classmethod
    def from_config(cls, config: AlgorithmicLatentPolicyConfig) -> "AlgorithmicLatentPolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        env_model_path = artifacts_root / "models" / DEFAULT_ENV_FAMILY_MODEL_PATH.name
        family_model_path = artifacts_root / "models" / DEFAULT_FAMILY_POLICY_MODEL_PATH.name
        metadata_path = artifacts_root / "outputs" / DEFAULT_ENV_FAMILY_METADATA_PATH.name
        action_rate_calibration_path = artifacts_root / "outputs" / DEFAULT_ACTION_RATE_CALIBRATION_PATH.name
        if not env_model_path.exists():
            raise FileNotFoundError(f"Algorithmic-latent env family model not found at {env_model_path}")
        if not family_model_path.exists():
            raise FileNotFoundError(f"Algorithmic-latent family policy bundle not found at {family_model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Algorithmic-latent env family metadata not found at {metadata_path}")
        env_model = DirichletEnvRegressor.load(env_model_path)
        family_model = joblib.load(family_model_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        family_prob_columns = metadata.get("family_probability_columns", [])
        if not family_prob_columns:
            raise ValueError("No family probability columns found in env family metadata.")
        family_names = [str(column).replace("__posterior_prob", "") for column in family_prob_columns]
        action_rate_calibration = None
        if action_rate_calibration_path.exists():
            with open(action_rate_calibration_path, "r", encoding="utf-8") as handle:
                action_rate_calibration = json.load(handle)
        return cls(
            env_model=env_model,
            family_model=family_model,
            family_names=family_names,
            action_rate_calibration=action_rate_calibration,
        )

    def predict_family_distribution(self, env: Mapping[str, Any]) -> list[float]:
        row = {column: env.get(column) for column in REQUIRED_CONFIG_COLUMNS}
        frame = pd.DataFrame([row])
        predicted = self.env_model.predict(frame)[0]
        predicted = _normalize_probs(predicted)
        return predicted.tolist()

    def _sample_family_for_player(self, distribution: Sequence[float], rng: np.random.Generator) -> str:
        return str(rng.choice(np.asarray(self.family_names, dtype=object), p=np.asarray(distribution, dtype=float)))

    def start_game(
        self,
        *,
        env: Mapping[str, Any],
        player_ids: Sequence[str],
        avatar_by_player: Mapping[str, str],
        rng: np.random.Generator,
    ) -> AlgorithmicLatentGameState:
        distribution = self.predict_family_distribution(env)
        family_by_player = {
            str(player_id): self._sample_family_for_player(distribution, rng)
            for player_id in player_ids
        }
        cluster_by_player = {
            str(player_id): int(self.family_to_index[family_name])
            for player_id, family_name in family_by_player.items()
        }
        return AlgorithmicLatentGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player={str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids},
            player_by_avatar={str(avatar): str(player_id) for player_id, avatar in avatar_by_player.items()},
            family_by_player=family_by_player,
            cluster_by_player=cluster_by_player,
            archetype_label_by_player=dict(family_by_player),
            history_rounds=[],
        )

    def _contribution_label_probs(
        self,
        *,
        family_name: str,
        feature_row: Mapping[str, Any],
    ) -> Dict[int, float]:
        payload = self.family_payloads[str(family_name)]
        record = _row_to_feature_dict(feature_row, payload["contribution_features"])
        probs = payload["contribution_model"].predict_proba([record])[0]
        probs = _normalize_probs(probs)
        classes = [int(value) for value in payload["contribution_model"].named_steps["model"].classes_.tolist()]
        return {int(label): float(prob) for label, prob in zip(classes, probs)}

    def sample_contributions_for_round(
        self,
        *,
        game_state: AlgorithmicLatentGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        all_or_nothing = as_bool(game_state.env.get("CONFIG_allOrNothing", False))
        legal_bins = [0, 4] if all_or_nothing else list(range(len(RATE_BINS)))
        out: Dict[str, int] = {}
        for player_id in game_state.player_ids:
            family_name = str(game_state.family_by_player[str(player_id)])
            feature_row = _contribution_feature_row(
                env=game_state.env,
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                round_idx=int(round_idx),
                history_rounds=game_state.history_rounds,
            )
            label_probs = self._contribution_label_probs(family_name=family_name, feature_row=feature_row)
            sampled_bin = int(
                _sample_from_label_probs(
                    label_probs={str(label): label_probs.get(int(label), 0.0) for label in legal_bins},
                    allowed_labels=[str(label) for label in legal_bins],
                    rng=rng,
                )
            )
            sampled_rate = float(RATE_BINS[int(sampled_bin)])
            if all_or_nothing:
                out[str(player_id)] = int(endowment if sampled_bin >= 4 else 0)
            else:
                out[str(player_id)] = int(max(0, min(endowment, round(sampled_rate * endowment))))
        return out

    def _action_label_probs(
        self,
        *,
        family_name: str,
        feature_row: Mapping[str, Any],
    ) -> Dict[str, float]:
        payload = self.family_payloads[str(family_name)]
        record = _row_to_feature_dict(feature_row, payload["action_features"])
        probs = payload["action_model"].predict_proba([record])[0]
        probs = _normalize_probs(probs)
        classes = [str(value) for value in payload["action_model"].named_steps["model"].classes_.tolist()]
        label_probs = {str(label): float(prob) for label, prob in zip(classes, probs)}
        calibration = self.action_rate_calibration.get("global", {})
        punish_scale = float(calibration.get("punish_scale", 1.0))
        reward_scale = float(calibration.get("reward_scale", 1.0))
        none = float(label_probs.get("none", 0.0))
        punish = float(label_probs.get("punish", 0.0)) * max(punish_scale, 0.0)
        reward = float(label_probs.get("reward", 0.0)) * max(reward_scale, 0.0)
        denom = max(none + punish + reward, EPS)
        return {
            "none": none / denom,
            "punish": punish / denom,
            "reward": reward / denom,
        }

    def sample_actions_for_round(
        self,
        *,
        game_state: AlgorithmicLatentGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        punish_enabled = as_bool(game_state.env.get("CONFIG_punishmentExists", False))
        reward_enabled = as_bool(game_state.env.get("CONFIG_rewardExists", False))
        punish_cost = float(game_state.env.get("CONFIG_punishmentCost", 1) or 1)
        reward_cost = float(game_state.env.get("CONFIG_rewardCost", 1) or 1)
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}

        for player_id in game_state.player_ids:
            family_name = str(game_state.family_by_player[str(player_id)])
            available_budget = _available_action_budget_coins(
                env=game_state.env,
                focal_player_id=player_id,
                contributions_by_player=contributions_by_player,
            )
            positive_actions: List[tuple[str, str, float]] = []
            for target_player_id in game_state.player_ids:
                if str(target_player_id) == str(player_id):
                    continue
                feature_row = _action_feature_row(
                    env=game_state.env,
                    player_ids=game_state.player_ids,
                    focal_player_id=player_id,
                    target_player_id=target_player_id,
                    round_idx=int(round_idx),
                    history_rounds=game_state.history_rounds,
                    contributions_by_player=contributions_by_player,
                )
                label_probs = self._action_label_probs(family_name=family_name, feature_row=feature_row)
                allowed_labels = ["none"]
                if punish_enabled:
                    allowed_labels.append("punish")
                if reward_enabled:
                    allowed_labels.append("reward")
                sampled_label = _sample_from_label_probs(
                    label_probs=label_probs,
                    allowed_labels=allowed_labels,
                    rng=rng,
                )
                if sampled_label in {"punish", "reward"}:
                    positive_actions.append(
                        (
                            sampled_label,
                            str(target_player_id),
                            float(label_probs.get(sampled_label, 0.0) - label_probs.get("none", 0.0)),
                        )
                    )

            positive_actions.sort(key=lambda item: item[2], reverse=True)
            punish_allocations: Dict[str, int] = {}
            reward_allocations: Dict[str, int] = {}
            spend = 0.0
            for label, target_player_id, _ in positive_actions:
                action_cost = punish_cost if label == "punish" else reward_cost
                if spend + action_cost > float(available_budget) + 1e-9:
                    continue
                if label == "punish":
                    punish_allocations[str(target_player_id)] = 1
                else:
                    reward_allocations[str(target_player_id)] = 1
                spend += action_cost
            punish_out[str(player_id)] = punish_allocations
            reward_out[str(player_id)] = reward_allocations

        return {"punish": punish_out, "reward": reward_out}

    def record_round(
        self,
        *,
        game_state: AlgorithmicLatentGameState,
        contributions_by_player: Mapping[str, int],
        punish_by_player: Mapping[str, Mapping[str, int]],
        reward_by_player: Mapping[str, Mapping[str, int]],
        payoff_by_player: Mapping[str, float],
        round_idx: Optional[int] = None,
    ) -> None:
        if round_idx is None:
            round_idx = int(len(game_state.history_rounds) + 1)
        game_state.history_rounds.append(
            _build_round_state(
                env=game_state.env,
                player_ids=game_state.player_ids,
                round_idx=int(round_idx),
                contributions_by_player=contributions_by_player,
                punish_by_player=punish_by_player,
                reward_by_player=reward_by_player,
                payoff_by_player=payoff_by_player,
            )
        )

    def record_actual_round(
        self,
        *,
        game_state: AlgorithmicLatentGameState,
        round_rows: pd.DataFrame,
    ) -> None:
        if round_rows is None or round_rows.empty:
            return
        round_idx = int(pd.to_numeric(round_rows["roundIndex"], errors="coerce").dropna().iloc[0])
        contributions_by_player = _contrib_by_player(round_rows)
        punish_by_player = _action_dicts_by_player(round_rows, "data.punished")
        reward_by_player = _action_dicts_by_player(round_rows, "data.rewarded")
        payoff_by_player = _round_payoff_by_player(round_rows)
        costs_by_player = _metric_by_player(round_rows, "data.costs")
        penalties_by_player = _metric_by_player(round_rows, "data.penalties")
        rewards_by_player = _metric_by_player(round_rows, "data.rewards")
        game_state.history_rounds.append(
            _build_round_state(
                env=game_state.env,
                player_ids=game_state.player_ids,
                round_idx=int(round_idx),
                contributions_by_player=contributions_by_player,
                punish_by_player=punish_by_player,
                reward_by_player=reward_by_player,
                payoff_by_player=payoff_by_player,
                costs_by_player=costs_by_player,
                penalties_by_player=penalties_by_player,
                rewards_by_player=rewards_by_player,
            )
        )
