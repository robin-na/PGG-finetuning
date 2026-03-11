from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import _build_round_index, as_bool
from simulation_statistical.history_conditioned_policy import (
    _action_dicts_by_player,
    _contrib_by_player,
    _prepare_cluster_assignments,
    _round_payoff_by_player,
)
from simulation_statistical.trained_policy import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_ENV_MODEL_PATH,
    DEFAULT_LEARN_ANALYSIS_CSV,
    DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    DEFAULT_LEARN_ROUNDS_CSV,
    _distribute_units,
    _load_oracle_treatment_cluster_distributions,
    _normalize_probability_vector,
    _normalized_rank,
)


DEFAULT_CLUSTER_PLUS_BEHAVIOR_MODEL_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "models" / "cluster_plus_behavior_model.pkl"
)
DEFAULT_CLUSTER_PLUS_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "cluster_plus_behavior_model_summary.csv"
)
HISTORY_RATE_BINS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)


@dataclass(frozen=True)
class ArchetypeClusterPlusPolicyConfig:
    artifacts_root: str | None = None
    rebuild_behavior_model: bool = False
    cluster_source: str = "env_model"


@dataclass
class ClusterPlusRoundState:
    round_idx: int
    contributions_by_player: Dict[str, int]
    punish_by_player: Dict[str, Dict[str, int]]
    reward_by_player: Dict[str, Dict[str, int]]
    payoff_by_player: Dict[str, float]


@dataclass
class ArchetypeClusterPlusGameState:
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    cluster_by_player: Dict[str, int]
    history_rounds: List[ClusterPlusRoundState] = field(default_factory=list)


def _round_phase(round_index: int, num_rounds: int) -> str:
    total = max(int(num_rounds), 1)
    progress = float(max(int(round_index) - 1, 0)) / float(max(total - 1, 1))
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "mid"
    return "late"


def _visible_phase(env: Mapping[str, Any], round_idx: int) -> str:
    if as_bool(env.get("CONFIG_showNRounds", False)):
        num_rounds = int(env.get("CONFIG_numRounds", 1) or 1)
        return _round_phase(int(round_idx), num_rounds)
    round_idx = int(round_idx)
    if round_idx <= 1:
        return "r1"
    if round_idx <= 3:
        return "r2_3"
    if round_idx <= 6:
        return "r4_6"
    return "r7plus"


def _rate_bin_label(rate: float) -> int:
    clipped = float(max(0.0, min(1.0, float(rate))))
    return int(np.argmin(np.abs(HISTORY_RATE_BINS - clipped)))


def _relative_peer_ids(player_ids: Sequence[str], focal_player_id: str) -> List[str]:
    focal = str(focal_player_id)
    return [str(player_id) for player_id in player_ids if str(player_id) != focal]


def _sum_units(values: Mapping[str, int]) -> int:
    return int(sum(int(value) for value in values.values()))


def _player_inbound_units(
    actions_by_player: Mapping[str, Mapping[str, int]],
    target_player_id: str,
) -> int:
    target = str(target_player_id)
    return int(sum(int(targets.get(target, 0)) for targets in actions_by_player.values()))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _build_round_state(
    *,
    round_idx: int,
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
) -> ClusterPlusRoundState:
    return ClusterPlusRoundState(
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
        payoff_by_player={
            str(player_id): float(value) if not pd.isna(value) else 0.0
            for player_id, value in payoff_by_player.items()
        },
    )


def _add_sample(store: Dict[tuple, list], keys: Iterable[tuple], value: Any) -> None:
    for key in keys:
        store.setdefault(key, []).append(value)


def _add_rate_count(
    store: Dict[tuple, List[int]],
    keys: Iterable[tuple],
    positive: bool,
) -> None:
    for key in keys:
        pos, total = store.get(key, [0, 0])
        store[key] = [pos + int(bool(positive)), total + 1]


def _history_signature(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    history_rounds: Sequence[ClusterPlusRoundState],
) -> tuple[int, int, int, int] | None:
    if not history_rounds:
        return None
    focal = str(focal_player_id)
    peers = _relative_peer_ids(player_ids, focal)
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    last_round = history_rounds[-1]
    own_contrib_rate = float(last_round.contributions_by_player.get(focal, 0.0)) / max(endowment, 1.0)
    peer_contrib_rates = [
        float(last_round.contributions_by_player.get(peer_id, 0.0)) / max(endowment, 1.0)
        for peer_id in peers
    ]
    peer_mean_rate = _safe_mean(peer_contrib_rates)
    punish_received = int(_player_inbound_units(last_round.punish_by_player, focal) > 0)
    reward_received = int(_player_inbound_units(last_round.reward_by_player, focal) > 0)
    return (
        _rate_bin_label(own_contrib_rate),
        _rate_bin_label(peer_mean_rate),
        punish_received,
        reward_received,
    )


def _current_action_signature(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
) -> tuple[int, int]:
    focal = str(focal_player_id)
    peers = _relative_peer_ids(player_ids, focal)
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    own_contrib_rate = float(contributions_by_player.get(focal, 0.0)) / max(endowment, 1.0)
    peer_contrib_rates = [
        float(contributions_by_player.get(peer_id, 0.0)) / max(endowment, 1.0)
        for peer_id in peers
    ]
    peer_mean_rate = _safe_mean(peer_contrib_rates)
    return (
        _rate_bin_label(own_contrib_rate),
        _rate_bin_label(peer_mean_rate),
    )


def _contribution_keys_plus(
    cluster_id: int,
    phase: str,
    all_or_nothing: bool,
    history_signature: tuple[int, int, int, int] | None,
) -> list[tuple]:
    aon = int(bool(all_or_nothing))
    keys: list[tuple] = []
    if history_signature is None:
        keys.extend(
            [
                ("cluster_phase_start_aon", cluster_id, phase, aon),
                ("cluster_start_aon", cluster_id, aon),
                ("cluster_phase_start", cluster_id, phase),
                ("cluster_start", cluster_id),
            ]
        )
    else:
        own_bin, peer_bin, punish_received, reward_received = history_signature
        keys.extend(
            [
                (
                    "cluster_phase_hist_aon",
                    cluster_id,
                    phase,
                    aon,
                    own_bin,
                    peer_bin,
                    punish_received,
                    reward_received,
                ),
                ("cluster_phase_hist_coarse_aon", cluster_id, phase, aon, own_bin, peer_bin),
                ("cluster_hist_aon", cluster_id, aon, own_bin, peer_bin, punish_received, reward_received),
                ("cluster_hist_coarse_aon", cluster_id, aon, own_bin, peer_bin),
                ("cluster_phase_hist", cluster_id, phase, own_bin, peer_bin, punish_received, reward_received),
                ("cluster_phase_hist_coarse", cluster_id, phase, own_bin, peer_bin),
                ("cluster_hist", cluster_id, own_bin, peer_bin, punish_received, reward_received),
                ("cluster_hist_coarse", cluster_id, own_bin, peer_bin),
            ]
        )
    keys.extend(
        [
            ("cluster_phase_aon", cluster_id, phase, aon),
            ("cluster_aon", cluster_id, aon),
            ("cluster_phase", cluster_id, phase),
            ("cluster", cluster_id),
            ("global_aon", aon),
            ("global",),
        ]
    )
    return keys


def _action_keys_plus(
    cluster_id: int,
    phase: str,
    history_signature: tuple[int, int, int, int] | None,
    current_signature: tuple[int, int],
) -> list[tuple]:
    current_own_bin, current_peer_bin = current_signature
    keys: list[tuple] = []
    if history_signature is None:
        keys.extend(
            [
                ("cluster_phase_start_current", cluster_id, phase, current_own_bin, current_peer_bin),
                ("cluster_start_current", cluster_id, current_own_bin, current_peer_bin),
            ]
        )
    else:
        prev_own_bin, prev_peer_bin, punish_received, reward_received = history_signature
        keys.extend(
            [
                (
                    "cluster_phase_hist_current",
                    cluster_id,
                    phase,
                    prev_own_bin,
                    prev_peer_bin,
                    punish_received,
                    reward_received,
                    current_own_bin,
                    current_peer_bin,
                ),
                (
                    "cluster_phase_hist_current_coarse",
                    cluster_id,
                    phase,
                    prev_own_bin,
                    prev_peer_bin,
                    current_own_bin,
                    current_peer_bin,
                ),
                (
                    "cluster_hist_current",
                    cluster_id,
                    prev_own_bin,
                    prev_peer_bin,
                    punish_received,
                    reward_received,
                    current_own_bin,
                    current_peer_bin,
                ),
                (
                    "cluster_hist_current_coarse",
                    cluster_id,
                    prev_own_bin,
                    prev_peer_bin,
                    current_own_bin,
                    current_peer_bin,
                ),
            ]
        )
    keys.extend(
        [
            ("cluster_phase_current", cluster_id, phase, current_own_bin, current_peer_bin),
            ("cluster_current", cluster_id, current_own_bin, current_peer_bin),
            ("cluster_phase", cluster_id, phase),
            ("cluster", cluster_id),
            ("global",),
        ]
    )
    return keys


def _choice_from_values(
    rng: np.random.Generator,
    values: Sequence[Any],
    default: Any,
) -> Any:
    if not values:
        return default
    index = int(rng.integers(0, len(values)))
    return values[index]


def _lookup_samples(store: Dict[tuple, list], keys: Sequence[tuple]) -> list:
    for key in keys:
        values = store.get(key)
        if values:
            return values
    return []


def _lookup_rate(store: Dict[tuple, list[int]], keys: Sequence[tuple], default: float) -> float:
    for key in keys:
        counts = store.get(key)
        if counts and counts[1] > 0:
            return float(counts[0]) / float(counts[1])
    return float(default)


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


def _prune_overlapping_actions(
    *,
    punish_allocations: Dict[str, int],
    reward_allocations: Dict[str, int],
    peer_contributions: Mapping[str, int],
) -> None:
    overlap = sorted(set(punish_allocations) & set(reward_allocations))
    if not overlap:
        return
    peer_values = list(float(value) for value in peer_contributions.values())
    peer_mean = _safe_mean(peer_values)
    for target_player_id in overlap:
        target_value = float(peer_contributions.get(target_player_id, 0.0))
        if target_value >= peer_mean:
            punish_allocations.pop(target_player_id, None)
        else:
            reward_allocations.pop(target_player_id, None)


def _prune_actions_to_budget(
    *,
    punish_allocations: Dict[str, int],
    reward_allocations: Dict[str, int],
    punish_cost: float,
    reward_cost: float,
    available_budget: float,
    peer_contributions: Mapping[str, int],
    endowment: float,
) -> None:
    def total_cost() -> float:
        punish_units = float(sum(int(units) for units in punish_allocations.values()))
        reward_units = float(sum(int(units) for units in reward_allocations.values()))
        return punish_units * punish_cost + reward_units * reward_cost

    while total_cost() > float(available_budget) + 1e-9:
        candidates: List[Tuple[str, str, float]] = []
        for target_player_id, units in punish_allocations.items():
            if int(units) <= 0:
                continue
            target_value = float(peer_contributions.get(target_player_id, 0.0)) / max(endowment, 1.0)
            candidates.append(("punish", str(target_player_id), target_value))
        for target_player_id, units in reward_allocations.items():
            if int(units) <= 0:
                continue
            target_value = float(peer_contributions.get(target_player_id, 0.0)) / max(endowment, 1.0)
            candidates.append(("reward", str(target_player_id), 1.0 - target_value))
        if not candidates:
            punish_allocations.clear()
            reward_allocations.clear()
            return
        candidates.sort(key=lambda item: item[2], reverse=True)
        mechanism, target_player_id, _ = candidates[0]
        if mechanism == "punish":
            punish_allocations[target_player_id] = int(punish_allocations.get(target_player_id, 0)) - 1
            if punish_allocations[target_player_id] <= 0:
                punish_allocations.pop(target_player_id, None)
        else:
            reward_allocations[target_player_id] = int(reward_allocations.get(target_player_id, 0)) - 1
            if reward_allocations[target_player_id] <= 0:
                reward_allocations.pop(target_player_id, None)


def _sample_ranked_targets(
    *,
    target_rank: float,
    target_count: int,
    units_total: int,
    peer_contributions: Dict[str, int],
    rng: np.random.Generator,
) -> Dict[str, int]:
    if not peer_contributions:
        return {}
    peers = list(peer_contributions.items())
    order = rng.permutation(len(peers))
    peers = [peers[int(index)] for index in order]
    peer_values = [float(value) for _, value in peers]
    ranked = sorted(
        peers,
        key=lambda item: abs(_normalized_rank(float(item[1]), peer_values) - float(target_rank)),
    )
    chosen = ranked[: max(1, min(int(target_count), len(ranked)))]
    allocations = _distribute_units(units_total=units_total, target_count=len(chosen))
    return {
        str(target_player_id): int(units)
        for (target_player_id, _), units in zip(chosen, allocations)
        if int(units) > 0
    }


def build_cluster_plus_behavior_model(
    *,
    output_path: Path = DEFAULT_CLUSTER_PLUS_BEHAVIOR_MODEL_PATH,
    summary_output_path: Path = DEFAULT_CLUSTER_PLUS_TRAIN_SUMMARY_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
) -> dict[str, Any]:
    cluster_assignments = _prepare_cluster_assignments(learn_cluster_weights_path)
    cluster_assignments["gameId"] = cluster_assignments["gameId"].astype(str)
    cluster_assignments["playerId"] = cluster_assignments["playerId"].astype(str)

    rounds = pd.read_csv(learn_rounds_csv)
    rounds = _build_round_index(rounds)
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)

    analysis = pd.read_csv(learn_analysis_csv).copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    env_lookup = (
        analysis.drop_duplicates(subset=["gameId"], keep="first")
        .set_index("gameId")
        .to_dict(orient="index")
    )

    merged = rounds.merge(
        cluster_assignments[["gameId", "playerId", "hard_cluster_id"]],
        on=["gameId", "playerId"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No learning-wave player-round rows remained after joining cluster assignments.")

    contribution_samples: Dict[tuple, list[float]] = {}
    punish_rate_counts: Dict[tuple, list[int]] = {}
    reward_rate_counts: Dict[tuple, list[int]] = {}
    punish_unit_samples: Dict[tuple, list[int]] = {}
    reward_unit_samples: Dict[tuple, list[int]] = {}
    punish_target_samples: Dict[tuple, list[int]] = {}
    reward_target_samples: Dict[tuple, list[int]] = {}
    punish_orientation_samples: Dict[tuple, list[float]] = {}
    reward_orientation_samples: Dict[tuple, list[float]] = {}

    n_contribution_rows = 0
    n_action_rows = 0

    for game_id, game_df in merged.groupby("gameId", sort=True):
        env = env_lookup.get(str(game_id))
        if not env:
            continue
        env = dict(env)
        player_ids = list(
            dict.fromkeys(
                game_df.sort_values(["roundIndex", "playerId"])["playerId"].astype(str).tolist()
            )
        )
        if not player_ids:
            continue
        cluster_by_player = (
            game_df[["playerId", "hard_cluster_id"]]
            .drop_duplicates(subset=["playerId"], keep="first")
            .set_index("playerId")["hard_cluster_id"]
            .astype(int)
            .to_dict()
        )
        history_rounds: List[ClusterPlusRoundState] = []

        round_values = sorted(
            int(value)
            for value in game_df["roundIndex"].dropna().unique().tolist()
            if int(value) > 0
        )
        for round_idx in round_values:
            round_rows = (
                game_df[game_df["roundIndex"] == int(round_idx)]
                .copy()
                .sort_values(["playerId"])
                .reset_index(drop=True)
            )
            if round_rows.empty:
                continue
            contributions_by_player = _contrib_by_player(round_rows)
            punish_by_player = _action_dicts_by_player(round_rows, "data.punished")
            reward_by_player = _action_dicts_by_player(round_rows, "data.rewarded")
            payoff_by_player = _round_payoff_by_player(round_rows)

            phase = _visible_phase(env, int(round_idx))
            all_or_nothing = as_bool(env.get("CONFIG_allOrNothing", False))
            punish_enabled = as_bool(env.get("CONFIG_punishmentExists", False))
            reward_enabled = as_bool(env.get("CONFIG_rewardExists", False))
            endowment = float(env.get("CONFIG_endowment", 20) or 20)

            for player_id in player_ids:
                cluster_id = int(cluster_by_player.get(str(player_id), 1))
                history_signature = _history_signature(
                    env=env,
                    player_ids=player_ids,
                    focal_player_id=player_id,
                    history_rounds=history_rounds,
                )
                contribution_prop = float(contributions_by_player.get(str(player_id), 0)) / max(endowment, 1.0)
                contribution_keys = _contribution_keys_plus(
                    cluster_id=cluster_id,
                    phase=phase,
                    all_or_nothing=all_or_nothing,
                    history_signature=history_signature,
                )
                _add_sample(contribution_samples, contribution_keys, contribution_prop)
                n_contribution_rows += 1

                if not (punish_enabled or reward_enabled):
                    continue

                current_signature = _current_action_signature(
                    env=env,
                    player_ids=player_ids,
                    focal_player_id=player_id,
                    contributions_by_player=contributions_by_player,
                )
                action_keys = _action_keys_plus(
                    cluster_id=cluster_id,
                    phase=phase,
                    history_signature=history_signature,
                    current_signature=current_signature,
                )
                if punish_enabled:
                    punish_units_total = int(_sum_units(punish_by_player.get(str(player_id), {})))
                    punish_target_count = int(sum(1 for v in punish_by_player.get(str(player_id), {}).values() if int(v) > 0))
                    _add_rate_count(punish_rate_counts, action_keys, punish_units_total > 0)
                    if punish_units_total > 0:
                        _add_sample(punish_unit_samples, action_keys, punish_units_total)
                        _add_sample(punish_target_samples, action_keys, punish_target_count)
                        peer_values = [
                            float(value)
                            for peer_id, value in contributions_by_player.items()
                            if str(peer_id) != str(player_id)
                        ]
                        weighted_ranks: list[float] = []
                        for target_player_id, units in punish_by_player.get(str(player_id), {}).items():
                            if str(target_player_id) == str(player_id):
                                continue
                            if str(target_player_id) not in contributions_by_player or int(units) <= 0:
                                continue
                            rank = _normalized_rank(float(contributions_by_player[str(target_player_id)]), peer_values)
                            weighted_ranks.extend([rank] * int(units))
                        if weighted_ranks:
                            _add_sample(punish_orientation_samples, action_keys, float(np.mean(weighted_ranks)))
                    n_action_rows += 1
                if reward_enabled:
                    reward_units_total = int(_sum_units(reward_by_player.get(str(player_id), {})))
                    reward_target_count = int(sum(1 for v in reward_by_player.get(str(player_id), {}).values() if int(v) > 0))
                    _add_rate_count(reward_rate_counts, action_keys, reward_units_total > 0)
                    if reward_units_total > 0:
                        _add_sample(reward_unit_samples, action_keys, reward_units_total)
                        _add_sample(reward_target_samples, action_keys, reward_target_count)
                        peer_values = [
                            float(value)
                            for peer_id, value in contributions_by_player.items()
                            if str(peer_id) != str(player_id)
                        ]
                        weighted_ranks = []
                        for target_player_id, units in reward_by_player.get(str(player_id), {}).items():
                            if str(target_player_id) == str(player_id):
                                continue
                            if str(target_player_id) not in contributions_by_player or int(units) <= 0:
                                continue
                            rank = _normalized_rank(float(contributions_by_player[str(target_player_id)]), peer_values)
                            weighted_ranks.extend([rank] * int(units))
                        if weighted_ranks:
                            _add_sample(reward_orientation_samples, action_keys, float(np.mean(weighted_ranks)))
                    n_action_rows += 1

            history_rounds.append(
                _build_round_state(
                    round_idx=int(round_idx),
                    contributions_by_player=contributions_by_player,
                    punish_by_player=punish_by_player,
                    reward_by_player=reward_by_player,
                    payoff_by_player=payoff_by_player,
                )
            )

    behavior_model = {
        "version": 1,
        "n_clusters": int(cluster_assignments["hard_cluster_id"].max()),
        "cluster_ids": sorted(int(value) for value in cluster_assignments["hard_cluster_id"].dropna().unique().tolist()),
        "history_bins": HISTORY_RATE_BINS.tolist(),
        "contribution_samples": contribution_samples,
        "punish_rate_counts": punish_rate_counts,
        "reward_rate_counts": reward_rate_counts,
        "punish_unit_samples": punish_unit_samples,
        "reward_unit_samples": reward_unit_samples,
        "punish_target_samples": punish_target_samples,
        "reward_target_samples": reward_target_samples,
        "punish_orientation_samples": punish_orientation_samples,
        "reward_orientation_samples": reward_orientation_samples,
        "learn_cluster_weights_path": str(learn_cluster_weights_path),
        "learn_analysis_csv": str(learn_analysis_csv),
        "learn_rounds_csv": str(learn_rounds_csv),
        "history_signature_definition": {
            "prev_own_contrib_bin": HISTORY_RATE_BINS.tolist(),
            "prev_peer_mean_contrib_bin": HISTORY_RATE_BINS.tolist(),
            "prev_received_punish_any": [0, 1],
            "prev_received_reward_any": [0, 1],
        },
        "current_action_signature_definition": {
            "current_own_contrib_bin": HISTORY_RATE_BINS.tolist(),
            "current_peer_mean_contrib_bin": HISTORY_RATE_BINS.tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f"{output_path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
        joblib.dump(behavior_model, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    summary_df = pd.DataFrame(
        [
            {
                "dataset": "contribution",
                "n_rows": int(n_contribution_rows),
                "n_unique_keys": int(len(contribution_samples)),
                "mean_sample_pool_size": float(
                    np.mean([len(values) for values in contribution_samples.values()])
                )
                if contribution_samples
                else 0.0,
            },
            {
                "dataset": "action",
                "n_rows": int(n_action_rows),
                "n_punish_rate_keys": int(len(punish_rate_counts)),
                "n_reward_rate_keys": int(len(reward_rate_counts)),
                "mean_punish_rate": float(
                    np.mean([counts[0] / counts[1] for counts in punish_rate_counts.values() if counts[1] > 0])
                )
                if punish_rate_counts
                else 0.0,
                "mean_reward_rate": float(
                    np.mean([counts[0] / counts[1] for counts in reward_rate_counts.values() if counts[1] > 0])
                )
                if reward_rate_counts
                else 0.0,
            },
        ]
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    return behavior_model


class ArchetypeClusterPlusPolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor | None,
        behavior_model: dict[str, Any],
        cluster_source: str = "env_model",
        oracle_treatment_distributions: Dict[str, list[float]] | None = None,
        oracle_global_distribution: list[float] | None = None,
    ) -> None:
        self.env_model = env_model
        self.behavior_model = behavior_model
        self.cluster_source = str(cluster_source or "env_model")
        self.oracle_treatment_distributions = oracle_treatment_distributions or {}
        self.oracle_global_distribution = oracle_global_distribution or []
        self.cluster_ids = [int(cluster_id) for cluster_id in behavior_model["cluster_ids"]]
        self.contribution_samples = behavior_model["contribution_samples"]
        self.punish_rate_counts = behavior_model["punish_rate_counts"]
        self.reward_rate_counts = behavior_model["reward_rate_counts"]
        self.punish_unit_samples = behavior_model["punish_unit_samples"]
        self.reward_unit_samples = behavior_model["reward_unit_samples"]
        self.punish_target_samples = behavior_model["punish_target_samples"]
        self.reward_target_samples = behavior_model["reward_target_samples"]
        self.punish_orientation_samples = behavior_model["punish_orientation_samples"]
        self.reward_orientation_samples = behavior_model["reward_orientation_samples"]

    @classmethod
    def from_config(cls, config: ArchetypeClusterPlusPolicyConfig) -> "ArchetypeClusterPlusPolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        cluster_source = str(config.cluster_source or "env_model")
        env_model_path = artifacts_root / DEFAULT_ENV_MODEL_PATH.relative_to(DEFAULT_ARTIFACTS_ROOT)
        behavior_model_path = artifacts_root / "models" / DEFAULT_CLUSTER_PLUS_BEHAVIOR_MODEL_PATH.name
        summary_output_path = artifacts_root / "outputs" / DEFAULT_CLUSTER_PLUS_TRAIN_SUMMARY_PATH.name
        if cluster_source == "env_model" and not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        if config.rebuild_behavior_model or not behavior_model_path.exists():
            build_cluster_plus_behavior_model(
                output_path=behavior_model_path,
                summary_output_path=summary_output_path,
            )
        env_model = DirichletEnvRegressor.load(env_model_path) if cluster_source == "env_model" else None
        behavior_model = joblib.load(behavior_model_path)
        oracle_treatment_distributions: Dict[str, list[float]] | None = None
        oracle_global_distribution: list[float] | None = None
        if cluster_source == "val_treatment_oracle":
            oracle_treatment_distributions, oracle_global_distribution = (
                _load_oracle_treatment_cluster_distributions(artifacts_root=artifacts_root)
            )
        return cls(
            env_model=env_model,
            behavior_model=behavior_model,
            cluster_source=cluster_source,
            oracle_treatment_distributions=oracle_treatment_distributions,
            oracle_global_distribution=oracle_global_distribution,
        )

    def predict_cluster_distribution(self, env: Mapping[str, Any]) -> list[float]:
        if self.cluster_source == "val_treatment_oracle":
            treatment_name = str(env.get("CONFIG_treatmentName", "") or "")
            if treatment_name in self.oracle_treatment_distributions:
                return list(self.oracle_treatment_distributions[treatment_name])
            if self.oracle_global_distribution:
                return list(self.oracle_global_distribution)
            raise RuntimeError("Oracle treatment cluster distribution lookup is empty.")
        if self.env_model is None:
            raise RuntimeError("Cluster distribution requested without an environment model.")
        row = {column: env.get(column) for column in REQUIRED_CONFIG_COLUMNS}
        frame = pd.DataFrame([row])
        predicted = self.env_model.predict(frame)[0]
        predicted = np.clip(np.asarray(predicted, dtype=float), 1e-8, None)
        predicted = predicted / predicted.sum()
        return predicted.tolist()

    def start_game(
        self,
        *,
        env: Mapping[str, Any],
        player_ids: Sequence[str],
        avatar_by_player: Mapping[str, str],
        rng: np.random.Generator,
    ) -> ArchetypeClusterPlusGameState:
        distribution = self.predict_cluster_distribution(env)
        cluster_ids = np.arange(1, len(distribution) + 1, dtype=int)
        sampled = list(rng.choice(cluster_ids, size=len(player_ids), p=distribution))
        cluster_by_player = {
            str(player_id): int(cluster_id)
            for player_id, cluster_id in zip(player_ids, sampled)
        }
        return ArchetypeClusterPlusGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player={str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids},
            player_by_avatar={str(avatar): str(player_id) for player_id, avatar in avatar_by_player.items()},
            cluster_by_player=cluster_by_player,
            history_rounds=[],
        )

    def sample_contributions_for_round(
        self,
        *,
        game_state: ArchetypeClusterPlusGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        out: Dict[str, int] = {}
        phase = _visible_phase(game_state.env, int(round_idx))
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        all_or_nothing = as_bool(game_state.env.get("CONFIG_allOrNothing", False))
        for player_id in game_state.player_ids:
            cluster_id = int(game_state.cluster_by_player[str(player_id)])
            history_signature = _history_signature(
                env=game_state.env,
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                history_rounds=game_state.history_rounds,
            )
            sample_pool = _lookup_samples(
                self.contribution_samples,
                _contribution_keys_plus(
                    cluster_id=cluster_id,
                    phase=phase,
                    all_or_nothing=all_or_nothing,
                    history_signature=history_signature,
                ),
            )
            sampled_prop = float(_choice_from_values(rng, sample_pool, 0.5))
            sampled_prop = max(0.0, min(1.0, sampled_prop))
            if all_or_nothing:
                out[str(player_id)] = int(endowment if sampled_prop >= 0.5 else 0)
            else:
                out[str(player_id)] = int(max(0, min(endowment, round(sampled_prop * endowment))))
        return out

    def _sample_action_dict(
        self,
        *,
        mechanism: str,
        cluster_id: int,
        env: Mapping[str, Any],
        round_idx: int,
        history_signature: tuple[int, int, int, int] | None,
        current_signature: tuple[int, int],
        peer_contributions: Dict[str, int],
        available_budget_coins: float,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        phase = _visible_phase(env, int(round_idx))
        action_keys = _action_keys_plus(
            cluster_id=cluster_id,
            phase=phase,
            history_signature=history_signature,
            current_signature=current_signature,
        )
        if mechanism == "punish":
            rate = _lookup_rate(self.punish_rate_counts, action_keys, default=0.0)
            unit_pool = _lookup_samples(self.punish_unit_samples, action_keys)
            target_pool = _lookup_samples(self.punish_target_samples, action_keys)
            orientation_pool = _lookup_samples(self.punish_orientation_samples, action_keys)
            cost = float(env.get("CONFIG_punishmentCost", 1) or 1)
            default_rank = 0.0
        else:
            rate = _lookup_rate(self.reward_rate_counts, action_keys, default=0.0)
            unit_pool = _lookup_samples(self.reward_unit_samples, action_keys)
            target_pool = _lookup_samples(self.reward_target_samples, action_keys)
            orientation_pool = _lookup_samples(self.reward_orientation_samples, action_keys)
            cost = float(env.get("CONFIG_rewardCost", 1) or 1)
            default_rank = 1.0
        if float(rng.random()) >= max(0.0, min(1.0, rate)):
            return {}
        max_units = max(int(np.floor(float(available_budget_coins) / max(cost, 1.0))), 1)
        units_total = int(max(1, min(max_units, _choice_from_values(rng, unit_pool, 1))))
        target_count = int(max(1, _choice_from_values(rng, target_pool, 1)))
        target_rank = float(_choice_from_values(rng, orientation_pool, default_rank))
        target_rank = max(0.0, min(1.0, target_rank))
        return _sample_ranked_targets(
            target_rank=target_rank,
            target_count=target_count,
            units_total=units_total,
            peer_contributions=peer_contributions,
            rng=rng,
        )

    def sample_actions_for_round(
        self,
        *,
        game_state: ArchetypeClusterPlusGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}
        punish_enabled = as_bool(game_state.env.get("CONFIG_punishmentExists", False))
        reward_enabled = as_bool(game_state.env.get("CONFIG_rewardExists", False))
        endowment = float(game_state.env.get("CONFIG_endowment", 20) or 20)

        for player_id in game_state.player_ids:
            cluster_id = int(game_state.cluster_by_player[str(player_id)])
            history_signature = _history_signature(
                env=game_state.env,
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                history_rounds=game_state.history_rounds,
            )
            current_signature = _current_action_signature(
                env=game_state.env,
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                contributions_by_player=contributions_by_player,
            )
            peer_contributions = {
                str(peer_id): int(contributions_by_player[str(peer_id)])
                for peer_id in game_state.player_ids
                if str(peer_id) != str(player_id)
            }
            available_budget = _available_action_budget_coins(
                env=game_state.env,
                focal_player_id=player_id,
                contributions_by_player=contributions_by_player,
            )
            punish_allocations = (
                self._sample_action_dict(
                    mechanism="punish",
                    cluster_id=cluster_id,
                    env=game_state.env,
                    round_idx=int(round_idx),
                    history_signature=history_signature,
                    current_signature=current_signature,
                    peer_contributions=peer_contributions,
                    available_budget_coins=available_budget,
                    rng=rng,
                )
                if punish_enabled
                else {}
            )
            reward_allocations = (
                self._sample_action_dict(
                    mechanism="reward",
                    cluster_id=cluster_id,
                    env=game_state.env,
                    round_idx=int(round_idx),
                    history_signature=history_signature,
                    current_signature=current_signature,
                    peer_contributions=peer_contributions,
                    available_budget_coins=available_budget,
                    rng=rng,
                )
                if reward_enabled
                else {}
            )
            _prune_overlapping_actions(
                punish_allocations=punish_allocations,
                reward_allocations=reward_allocations,
                peer_contributions=peer_contributions,
            )
            _prune_actions_to_budget(
                punish_allocations=punish_allocations,
                reward_allocations=reward_allocations,
                punish_cost=float(game_state.env.get("CONFIG_punishmentCost", 1) or 1),
                reward_cost=float(game_state.env.get("CONFIG_rewardCost", 1) or 1),
                available_budget=available_budget,
                peer_contributions=peer_contributions,
                endowment=endowment,
            )
            punish_out[str(player_id)] = punish_allocations
            reward_out[str(player_id)] = reward_allocations
        return {"punish": punish_out, "reward": reward_out}

    def record_round(
        self,
        *,
        game_state: ArchetypeClusterPlusGameState,
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
        game_state: ArchetypeClusterPlusGameState,
        round_rows: pd.DataFrame,
    ) -> None:
        if round_rows is None or round_rows.empty:
            return
        round_idx = int(pd.to_numeric(round_rows["roundIndex"], errors="coerce").dropna().iloc[0])
        contributions_by_player = _contrib_by_player(round_rows)
        punish_by_player = _action_dicts_by_player(round_rows, "data.punished")
        reward_by_player = _action_dicts_by_player(round_rows, "data.rewarded")
        payoff_by_player = _round_payoff_by_player(round_rows)
        self.record_round(
            game_state=game_state,
            contributions_by_player=contributions_by_player,
            punish_by_player=punish_by_player,
            reward_by_player=reward_by_player,
            payoff_by_player=payoff_by_player,
            round_idx=round_idx,
        )
