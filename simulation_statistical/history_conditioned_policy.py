from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import _build_round_index, as_bool, parse_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = (
    REPO_ROOT / "simulation_statistical" / "archetype_distribution_embedding" / "artifacts"
)
DEFAULT_ENV_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "dirichlet_env_model.pkl"
DEFAULT_HISTORY_POLICY_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "history_conditioned_policy.pkl"
DEFAULT_HISTORY_CONTRIBUTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "history_contribution_train.parquet"
)
DEFAULT_HISTORY_ACTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "history_action_train.parquet"
)
DEFAULT_HISTORY_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "history_conditioned_policy_train_summary.csv"
)
DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "player_cluster_weights_learn.parquet"
)
DEFAULT_LEARN_ANALYSIS_CSV = (
    REPO_ROOT / "benchmark_statistical" / "data" / "processed_data" / "df_analysis_learn.csv"
)
DEFAULT_LEARN_ROUNDS_CSV = (
    REPO_ROOT / "benchmark_statistical" / "data" / "raw_data" / "learning_wave" / "player-rounds.csv"
)


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), float(self.value), dtype=float)


@dataclass
class ConstantBinaryClassifier:
    positive_probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        prob = float(max(0.0, min(1.0, self.positive_probability)))
        out = np.zeros((len(frame), 2), dtype=float)
        out[:, 0] = 1.0 - prob
        out[:, 1] = prob
        return out

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


@dataclass
class PlayerVisibleStats:
    rounds_observed: int = 0
    summaries_visible_rounds: int = 0
    punish_ids_visible_rounds: int = 0
    reward_ids_visible_rounds: int = 0

    own_contrib_sum: float = 0.0
    own_payoff_norm_sum: float = 0.0
    own_punish_received_sum: float = 0.0
    own_reward_received_sum: float = 0.0
    own_punish_spent_sum: float = 0.0
    own_reward_spent_sum: float = 0.0
    own_punish_target_rank_sum: float = 0.0
    own_reward_target_rank_sum: float = 0.0
    group_mean_contrib_sum: float = 0.0
    group_std_contrib_sum: float = 0.0
    peer_mean_contrib_sum: float = 0.0
    peer_std_contrib_sum: float = 0.0
    visible_peer_payoff_norm_sum: float = 0.0
    visible_peer_punish_received_sum: float = 0.0
    visible_peer_reward_received_sum: float = 0.0
    visible_peer_punish_spent_sum: float = 0.0
    visible_peer_reward_spent_sum: float = 0.0
    visible_num_punishers_sum: float = 0.0
    visible_num_rewarders_sum: float = 0.0
    visible_mean_punisher_contrib_sum: float = 0.0
    visible_mean_rewarder_contrib_sum: float = 0.0

    last_own_contrib_rate: Optional[float] = None
    last_own_payoff_norm: Optional[float] = None
    last_own_punish_received_units: Optional[float] = None
    last_own_reward_received_units: Optional[float] = None
    last_own_punish_spent_units: Optional[float] = None
    last_own_reward_spent_units: Optional[float] = None
    last_own_punish_target_rank: Optional[float] = None
    last_own_reward_target_rank: Optional[float] = None
    last_group_mean_contrib_rate: Optional[float] = None
    last_group_std_contrib_rate: Optional[float] = None
    last_peer_mean_contrib_rate: Optional[float] = None
    last_peer_std_contrib_rate: Optional[float] = None
    last_visible_peer_payoff_norm: Optional[float] = None
    last_visible_peer_punish_received: Optional[float] = None
    last_visible_peer_reward_received: Optional[float] = None
    last_visible_peer_punish_spent: Optional[float] = None
    last_visible_peer_reward_spent: Optional[float] = None
    last_visible_num_punishers: Optional[float] = None
    last_visible_num_rewarders: Optional[float] = None
    last_visible_mean_punisher_contrib: Optional[float] = None
    last_visible_mean_rewarder_contrib: Optional[float] = None


@dataclass
class HistoryConditionedGameState:
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    cluster_by_player: Dict[str, int]
    stats_by_player: Dict[str, PlayerVisibleStats] = field(default_factory=dict)


@dataclass(frozen=True)
class HistoryConditionedArchetypePolicyConfig:
    artifacts_root: str | None = None
    rebuild_model: bool = False


def _mean_or_nan(total: float, count: int) -> float:
    return float(total / count) if int(count) > 0 else float("nan")


def _atomic_joblib_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f"{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _prepare_cluster_assignments(cluster_weights_path: Path) -> pd.DataFrame:
    cluster_weights = pd.read_parquet(cluster_weights_path).copy()
    prob_columns = [
        column
        for column in cluster_weights.columns
        if column.startswith("cluster_") and column.endswith("_prob")
    ]
    if not prob_columns:
        raise ValueError(f"No cluster probability columns found in {cluster_weights_path}")
    cluster_weights["hard_cluster_id"] = (
        cluster_weights[prob_columns].to_numpy().argmax(axis=1) + 1
    )
    return cluster_weights.rename(columns={"game_id": "gameId", "player_id": "playerId"})[
        ["gameId", "playerId", "hard_cluster_id"]
    ].copy()


def _normalized_rank(target_value: float, peer_values: Sequence[float]) -> float:
    if not peer_values:
        return 0.5
    arr = np.asarray(peer_values, dtype=float)
    less = float(np.sum(arr < target_value))
    equal = float(np.sum(arr == target_value))
    return float((less + 0.5 * equal) / max(len(arr), 1))


def _weighted_target_rank_mean(
    *,
    focal_player_id: str,
    target_units_by_player: Mapping[str, int],
    contributions_by_player: Mapping[str, float],
) -> float:
    peer_values = [
        float(value)
        for player_id, value in contributions_by_player.items()
        if str(player_id) != str(focal_player_id)
    ]
    if not peer_values:
        return float("nan")
    weighted_ranks: List[float] = []
    for target_player_id, units in target_units_by_player.items():
        if str(target_player_id) == str(focal_player_id):
            continue
        if str(target_player_id) not in contributions_by_player or int(units) <= 0:
            continue
        rank = _normalized_rank(float(contributions_by_player[str(target_player_id)]), peer_values)
        weighted_ranks.extend([rank] * int(units))
    if not weighted_ranks:
        return float("nan")
    return float(np.mean(weighted_ranks))


def _visible_env_features(env: Mapping[str, Any], round_idx: int) -> Dict[str, Any]:
    show_n_rounds = as_bool(env.get("CONFIG_showNRounds", False))
    num_rounds = float(env.get("CONFIG_numRounds", 0) or 0)
    visible_total_rounds = num_rounds if show_n_rounds else np.nan
    visible_rounds_remaining = (
        max(num_rounds - float(round_idx), 0.0) if show_n_rounds and num_rounds > 0 else np.nan
    )
    visible_round_progress = (
        float(round_idx) / float(num_rounds)
        if show_n_rounds and num_rounds > 0
        else np.nan
    )
    return {
        "round_index": float(round_idx),
        "rounds_elapsed": float(max(int(round_idx) - 1, 0)),
        "visible_total_rounds": visible_total_rounds,
        "visible_rounds_remaining": visible_rounds_remaining,
        "visible_round_progress": visible_round_progress,
        "CONFIG_playerCount": float(env.get("CONFIG_playerCount", np.nan)),
        "CONFIG_showNRounds": float(show_n_rounds),
        "CONFIG_allOrNothing": float(as_bool(env.get("CONFIG_allOrNothing", False))),
        "CONFIG_chat": float(as_bool(env.get("CONFIG_chat", False))),
        "CONFIG_defaultContribProp": float(env.get("CONFIG_defaultContribProp", np.nan)),
        "CONFIG_punishmentExists": float(as_bool(env.get("CONFIG_punishmentExists", False))),
        "CONFIG_punishmentCost": float(env.get("CONFIG_punishmentCost", np.nan)),
        "CONFIG_punishmentMagnitude": float(env.get("CONFIG_punishmentMagnitude", np.nan)),
        "CONFIG_rewardExists": float(as_bool(env.get("CONFIG_rewardExists", False))),
        "CONFIG_rewardCost": float(env.get("CONFIG_rewardCost", np.nan)),
        "CONFIG_rewardMagnitude": float(env.get("CONFIG_rewardMagnitude", np.nan)),
        "CONFIG_showOtherSummaries": float(as_bool(env.get("CONFIG_showOtherSummaries", False))),
        "CONFIG_showPunishmentId": float(as_bool(env.get("CONFIG_showPunishmentId", False))),
        "CONFIG_showRewardId": float(as_bool(env.get("CONFIG_showRewardId", False))),
        "CONFIG_MPCR": float(env.get("CONFIG_MPCR", np.nan)),
        "CONFIG_endowment": float(env.get("CONFIG_endowment", np.nan)),
        "CONFIG_multiplier": float(env.get("CONFIG_multiplier", np.nan)),
        "CONFIG_punishmentTech": str(env.get("CONFIG_punishmentTech", "NA") or "NA"),
        "CONFIG_rewardTech": str(env.get("CONFIG_rewardTech", "NA") or "NA"),
    }


def _history_feature_row(
    *,
    env: Mapping[str, Any],
    cluster_id: int,
    stats: PlayerVisibleStats,
    round_idx: int,
) -> Dict[str, Any]:
    rounds = int(stats.rounds_observed)
    summary_rounds = int(stats.summaries_visible_rounds)
    punish_id_rounds = int(stats.punish_ids_visible_rounds)
    reward_id_rounds = int(stats.reward_ids_visible_rounds)
    row = _visible_env_features(env, round_idx)
    row.update(
        {
            "cluster_id": f"cluster_{int(cluster_id)}",
            "history_rounds_observed": float(rounds),
            "own_prev_contrib_rate": stats.last_own_contrib_rate,
            "own_mean_contrib_rate": _mean_or_nan(stats.own_contrib_sum, rounds),
            "own_prev_payoff_norm": stats.last_own_payoff_norm,
            "own_mean_payoff_norm": _mean_or_nan(stats.own_payoff_norm_sum, rounds),
            "own_prev_punish_received_units": stats.last_own_punish_received_units,
            "own_mean_punish_received_units": _mean_or_nan(stats.own_punish_received_sum, rounds),
            "own_prev_reward_received_units": stats.last_own_reward_received_units,
            "own_mean_reward_received_units": _mean_or_nan(stats.own_reward_received_sum, rounds),
            "own_prev_punish_spent_units": stats.last_own_punish_spent_units,
            "own_mean_punish_spent_units": _mean_or_nan(stats.own_punish_spent_sum, rounds),
            "own_prev_reward_spent_units": stats.last_own_reward_spent_units,
            "own_mean_reward_spent_units": _mean_or_nan(stats.own_reward_spent_sum, rounds),
            "own_prev_punish_target_rank": stats.last_own_punish_target_rank,
            "own_mean_punish_target_rank": _mean_or_nan(stats.own_punish_target_rank_sum, rounds),
            "own_prev_reward_target_rank": stats.last_own_reward_target_rank,
            "own_mean_reward_target_rank": _mean_or_nan(stats.own_reward_target_rank_sum, rounds),
            "group_prev_mean_contrib_rate": stats.last_group_mean_contrib_rate,
            "group_mean_contrib_rate": _mean_or_nan(stats.group_mean_contrib_sum, rounds),
            "group_prev_std_contrib_rate": stats.last_group_std_contrib_rate,
            "group_mean_std_contrib_rate": _mean_or_nan(stats.group_std_contrib_sum, rounds),
            "peer_prev_mean_contrib_rate": stats.last_peer_mean_contrib_rate,
            "peer_mean_contrib_rate": _mean_or_nan(stats.peer_mean_contrib_sum, rounds),
            "peer_prev_std_contrib_rate": stats.last_peer_std_contrib_rate,
            "peer_mean_std_contrib_rate": _mean_or_nan(stats.peer_std_contrib_sum, rounds),
            "visible_peer_prev_payoff_norm": stats.last_visible_peer_payoff_norm,
            "visible_peer_mean_payoff_norm": _mean_or_nan(
                stats.visible_peer_payoff_norm_sum,
                summary_rounds,
            ),
            "visible_peer_prev_punish_received_units": stats.last_visible_peer_punish_received,
            "visible_peer_mean_punish_received_units": _mean_or_nan(
                stats.visible_peer_punish_received_sum,
                summary_rounds,
            ),
            "visible_peer_prev_reward_received_units": stats.last_visible_peer_reward_received,
            "visible_peer_mean_reward_received_units": _mean_or_nan(
                stats.visible_peer_reward_received_sum,
                summary_rounds,
            ),
            "visible_peer_prev_punish_spent_units": stats.last_visible_peer_punish_spent,
            "visible_peer_mean_punish_spent_units": _mean_or_nan(
                stats.visible_peer_punish_spent_sum,
                summary_rounds,
            ),
            "visible_peer_prev_reward_spent_units": stats.last_visible_peer_reward_spent,
            "visible_peer_mean_reward_spent_units": _mean_or_nan(
                stats.visible_peer_reward_spent_sum,
                summary_rounds,
            ),
            "visible_prev_num_punishers": stats.last_visible_num_punishers,
            "visible_mean_num_punishers": _mean_or_nan(
                stats.visible_num_punishers_sum,
                punish_id_rounds,
            ),
            "visible_prev_num_rewarders": stats.last_visible_num_rewarders,
            "visible_mean_num_rewarders": _mean_or_nan(
                stats.visible_num_rewarders_sum,
                reward_id_rounds,
            ),
            "visible_prev_mean_punisher_contrib": stats.last_visible_mean_punisher_contrib,
            "visible_mean_punisher_contrib": _mean_or_nan(
                stats.visible_mean_punisher_contrib_sum,
                punish_id_rounds,
            ),
            "visible_prev_mean_rewarder_contrib": stats.last_visible_mean_rewarder_contrib,
            "visible_mean_rewarder_contrib": _mean_or_nan(
                stats.visible_mean_rewarder_contrib_sum,
                reward_id_rounds,
            ),
        }
    )
    return row


def _action_feature_row(
    *,
    env: Mapping[str, Any],
    cluster_id: int,
    stats: PlayerVisibleStats,
    round_idx: int,
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
) -> Dict[str, Any]:
    row = _history_feature_row(env=env, cluster_id=cluster_id, stats=stats, round_idx=round_idx)
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    own_value = float(contributions_by_player.get(str(focal_player_id), 0.0))
    peer_items = [
        (str(player_id), float(value))
        for player_id, value in contributions_by_player.items()
        if str(player_id) != str(focal_player_id)
    ]
    peer_values = [value for _, value in peer_items]
    group_values = [float(value) for value in contributions_by_player.values()]
    own_rank = _normalized_rank(own_value, group_values)
    peer_mean = float(np.mean(peer_values)) if peer_values else float("nan")
    peer_std = float(np.std(peer_values)) if peer_values else float("nan")
    row.update(
        {
            "current_own_contrib_rate": own_value / max(endowment, 1.0),
            "current_peer_mean_contrib_rate": peer_mean / max(endowment, 1.0)
            if peer_values
            else float("nan"),
            "current_peer_std_contrib_rate": peer_std / max(endowment, 1.0)
            if peer_values
            else float("nan"),
            "current_group_mean_contrib_rate": float(np.mean(group_values)) / max(endowment, 1.0)
            if group_values
            else float("nan"),
            "current_group_std_contrib_rate": float(np.std(group_values)) / max(endowment, 1.0)
            if group_values
            else float("nan"),
            "current_own_minus_peer_mean_rate": (own_value - peer_mean) / max(endowment, 1.0)
            if peer_values
            else float("nan"),
            "current_own_rank": own_rank,
            "current_share_peers_below": float(
                sum(1 for _, value in peer_items if value < own_value) / max(len(peer_items), 1)
            )
            if peer_items
            else float("nan"),
            "current_share_peers_above": float(
                sum(1 for _, value in peer_items if value > own_value) / max(len(peer_items), 1)
            )
            if peer_items
            else float("nan"),
        }
    )
    return row


def _prepare_model_matrix(frame: pd.DataFrame, feature_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    matrix = frame.copy()
    for column in list(matrix.columns):
        series = matrix[column]
        if series.dtype == bool:
            matrix[column] = series.astype(float)
    object_columns = [column for column in matrix.columns if matrix[column].dtype == object]
    if object_columns:
        matrix = pd.get_dummies(matrix, columns=object_columns, dummy_na=True, dtype=float)
    for column in list(matrix.columns):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    if feature_columns is None:
        return matrix.sort_index(axis=1)
    out = matrix.copy()
    for column in feature_columns:
        if column not in out.columns:
            out[column] = 0.0
    out = out[list(feature_columns)].copy()
    return out


def _fit_binary_classifier(features: pd.DataFrame, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    positive_rate = float(clean_target.mean()) if len(clean_target) else 0.0
    if clean_target.nunique() < 2:
        return ConstantBinaryClassifier(positive_probability=positive_rate)
    model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=0,
    )
    model.fit(features, clean_target)
    return model


def _fit_regressor(features: pd.DataFrame, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce")
    if clean_target.notna().sum() == 0:
        return ConstantRegressor(value=0.0)
    pair = features.loc[clean_target.notna()].copy()
    y = clean_target.loc[clean_target.notna()].astype(float)
    if len(pair) < 2 or y.nunique() < 2:
        return ConstantRegressor(value=float(y.mean()))
    model = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=0,
    )
    model.fit(pair, y)
    return model


def _build_residual_store(
    *,
    actual: pd.Series,
    predicted: np.ndarray,
    cluster_ids: pd.Series,
) -> Dict[str, List[float]]:
    residuals = pd.to_numeric(actual, errors="coerce") - pd.Series(predicted, index=actual.index, dtype=float)
    store: Dict[str, List[float]] = {
        "global": residuals.dropna().astype(float).tolist(),
    }
    for cluster_key, group in residuals.groupby(cluster_ids.astype(str)):
        values = group.dropna().astype(float).tolist()
        if values:
            store[str(cluster_key)] = values
    return store


def _sample_residual(
    rng: np.random.Generator,
    residual_store: Mapping[str, Sequence[float]],
    cluster_id: int,
) -> float:
    values = residual_store.get(str(cluster_id))
    if not values:
        values = residual_store.get("global", [])
    if not values:
        return 0.0
    index = int(rng.integers(0, len(values)))
    return float(values[index])


def _safe_prob(model: object, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    value = float(model.predict(features)[0])
    return float(max(0.0, min(1.0, value)))


def _hard_cluster_id(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 1


def _round_payoff_by_player(round_rows: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for _, row in round_rows.iterrows():
        player_id = str(row.get("playerId"))
        try:
            out[player_id] = float(row.get("data.roundPayoff"))
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


def _peer_metric_mean(
    values_by_player: Mapping[str, float],
    focal_player_id: str,
) -> float:
    peer_values = [
        float(value)
        for player_id, value in values_by_player.items()
        if str(player_id) != str(focal_player_id)
    ]
    return float(np.mean(peer_values)) if peer_values else float("nan")


def _peer_metric_std(
    values_by_player: Mapping[str, float],
    focal_player_id: str,
) -> float:
    peer_values = [
        float(value)
        for player_id, value in values_by_player.items()
        if str(player_id) != str(focal_player_id)
    ]
    return float(np.std(peer_values)) if peer_values else float("nan")


def _update_visible_stats(
    *,
    stats: PlayerVisibleStats,
    env: Mapping[str, Any],
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
) -> None:
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    focal_player_id = str(focal_player_id)
    own_contrib = float(contributions_by_player.get(focal_player_id, 0))
    own_payoff = float(payoff_by_player.get(focal_player_id, 0.0))
    inbound_punish_units = float(
        sum(int(targets.get(focal_player_id, 0)) for targets in punish_by_player.values())
    )
    inbound_reward_units = float(
        sum(int(targets.get(focal_player_id, 0)) for targets in reward_by_player.values())
    )
    own_punish_spent = float(sum(int(value) for value in punish_by_player.get(focal_player_id, {}).values()))
    own_reward_spent = float(sum(int(value) for value in reward_by_player.get(focal_player_id, {}).values()))
    punish_target_rank = _weighted_target_rank_mean(
        focal_player_id=focal_player_id,
        target_units_by_player=punish_by_player.get(focal_player_id, {}),
        contributions_by_player=contributions_by_player,
    )
    reward_target_rank = _weighted_target_rank_mean(
        focal_player_id=focal_player_id,
        target_units_by_player=reward_by_player.get(focal_player_id, {}),
        contributions_by_player=contributions_by_player,
    )

    contrib_rate = own_contrib / max(endowment, 1.0)
    payoff_norm = own_payoff / max(endowment, 1.0)
    group_values = [float(value) / max(endowment, 1.0) for value in contributions_by_player.values()]

    stats.rounds_observed += 1
    stats.own_contrib_sum += contrib_rate
    stats.own_payoff_norm_sum += payoff_norm
    stats.own_punish_received_sum += inbound_punish_units
    stats.own_reward_received_sum += inbound_reward_units
    stats.own_punish_spent_sum += own_punish_spent
    stats.own_reward_spent_sum += own_reward_spent
    if not pd.isna(punish_target_rank):
        stats.own_punish_target_rank_sum += float(punish_target_rank)
    if not pd.isna(reward_target_rank):
        stats.own_reward_target_rank_sum += float(reward_target_rank)
    stats.group_mean_contrib_sum += float(np.mean(group_values)) if group_values else 0.0
    stats.group_std_contrib_sum += float(np.std(group_values)) if group_values else 0.0
    stats.peer_mean_contrib_sum += _peer_metric_mean(
        {player_id: float(value) / max(endowment, 1.0) for player_id, value in contributions_by_player.items()},
        focal_player_id,
    )
    stats.peer_std_contrib_sum += _peer_metric_std(
        {player_id: float(value) / max(endowment, 1.0) for player_id, value in contributions_by_player.items()},
        focal_player_id,
    )

    stats.last_own_contrib_rate = contrib_rate
    stats.last_own_payoff_norm = payoff_norm
    stats.last_own_punish_received_units = inbound_punish_units
    stats.last_own_reward_received_units = inbound_reward_units
    stats.last_own_punish_spent_units = own_punish_spent
    stats.last_own_reward_spent_units = own_reward_spent
    stats.last_own_punish_target_rank = None if pd.isna(punish_target_rank) else float(punish_target_rank)
    stats.last_own_reward_target_rank = None if pd.isna(reward_target_rank) else float(reward_target_rank)
    stats.last_group_mean_contrib_rate = float(np.mean(group_values)) if group_values else float("nan")
    stats.last_group_std_contrib_rate = float(np.std(group_values)) if group_values else float("nan")
    stats.last_peer_mean_contrib_rate = _peer_metric_mean(
        {player_id: float(value) / max(endowment, 1.0) for player_id, value in contributions_by_player.items()},
        focal_player_id,
    )
    stats.last_peer_std_contrib_rate = _peer_metric_std(
        {player_id: float(value) / max(endowment, 1.0) for player_id, value in contributions_by_player.items()},
        focal_player_id,
    )

    if as_bool(env.get("CONFIG_showOtherSummaries", False)):
        stats.summaries_visible_rounds += 1
        peer_payoffs = {
            str(player_id): float(value) / max(endowment, 1.0)
            for player_id, value in payoff_by_player.items()
        }
        punish_received = {
            str(player_id): float(sum(int(targets.get(str(player_id), 0)) for targets in punish_by_player.values()))
            for player_id in contributions_by_player
        }
        reward_received = {
            str(player_id): float(sum(int(targets.get(str(player_id), 0)) for targets in reward_by_player.values()))
            for player_id in contributions_by_player
        }
        punish_spent = {
            str(player_id): float(sum(int(value) for value in punish_by_player.get(str(player_id), {}).values()))
            for player_id in contributions_by_player
        }
        reward_spent = {
            str(player_id): float(sum(int(value) for value in reward_by_player.get(str(player_id), {}).values()))
            for player_id in contributions_by_player
        }
        peer_payoff_mean = _peer_metric_mean(peer_payoffs, focal_player_id)
        peer_punish_received_mean = _peer_metric_mean(punish_received, focal_player_id)
        peer_reward_received_mean = _peer_metric_mean(reward_received, focal_player_id)
        peer_punish_spent_mean = _peer_metric_mean(punish_spent, focal_player_id)
        peer_reward_spent_mean = _peer_metric_mean(reward_spent, focal_player_id)

        stats.visible_peer_payoff_norm_sum += 0.0 if pd.isna(peer_payoff_mean) else float(peer_payoff_mean)
        stats.visible_peer_punish_received_sum += 0.0 if pd.isna(peer_punish_received_mean) else float(peer_punish_received_mean)
        stats.visible_peer_reward_received_sum += 0.0 if pd.isna(peer_reward_received_mean) else float(peer_reward_received_mean)
        stats.visible_peer_punish_spent_sum += 0.0 if pd.isna(peer_punish_spent_mean) else float(peer_punish_spent_mean)
        stats.visible_peer_reward_spent_sum += 0.0 if pd.isna(peer_reward_spent_mean) else float(peer_reward_spent_mean)
        stats.last_visible_peer_payoff_norm = None if pd.isna(peer_payoff_mean) else float(peer_payoff_mean)
        stats.last_visible_peer_punish_received = None if pd.isna(peer_punish_received_mean) else float(peer_punish_received_mean)
        stats.last_visible_peer_reward_received = None if pd.isna(peer_reward_received_mean) else float(peer_reward_received_mean)
        stats.last_visible_peer_punish_spent = None if pd.isna(peer_punish_spent_mean) else float(peer_punish_spent_mean)
        stats.last_visible_peer_reward_spent = None if pd.isna(peer_reward_spent_mean) else float(peer_reward_spent_mean)

    if as_bool(env.get("CONFIG_showPunishmentId", False)):
        stats.punish_ids_visible_rounds += 1
        punishers = [
            str(source_player_id)
            for source_player_id, targets in punish_by_player.items()
            if int(targets.get(focal_player_id, 0)) > 0
        ]
        punisher_contribs = [float(contributions_by_player[player_id]) / max(endowment, 1.0) for player_id in punishers]
        visible_num_punishers = float(len(punishers))
        visible_mean_punisher_contrib = float(np.mean(punisher_contribs)) if punisher_contribs else float("nan")
        stats.visible_num_punishers_sum += visible_num_punishers
        stats.visible_mean_punisher_contrib_sum += (
            0.0 if pd.isna(visible_mean_punisher_contrib) else float(visible_mean_punisher_contrib)
        )
        stats.last_visible_num_punishers = visible_num_punishers
        stats.last_visible_mean_punisher_contrib = (
            None if pd.isna(visible_mean_punisher_contrib) else float(visible_mean_punisher_contrib)
        )

    if as_bool(env.get("CONFIG_showRewardId", False)):
        stats.reward_ids_visible_rounds += 1
        rewarders = [
            str(source_player_id)
            for source_player_id, targets in reward_by_player.items()
            if int(targets.get(focal_player_id, 0)) > 0
        ]
        rewarder_contribs = [float(contributions_by_player[player_id]) / max(endowment, 1.0) for player_id in rewarders]
        visible_num_rewarders = float(len(rewarders))
        visible_mean_rewarder_contrib = float(np.mean(rewarder_contribs)) if rewarder_contribs else float("nan")
        stats.visible_num_rewarders_sum += visible_num_rewarders
        stats.visible_mean_rewarder_contrib_sum += (
            0.0 if pd.isna(visible_mean_rewarder_contrib) else float(visible_mean_rewarder_contrib)
        )
        stats.last_visible_num_rewarders = visible_num_rewarders
        stats.last_visible_mean_rewarder_contrib = (
            None if pd.isna(visible_mean_rewarder_contrib) else float(visible_mean_rewarder_contrib)
        )


def build_history_training_datasets(
    *,
    contribution_output_path: Path = DEFAULT_HISTORY_CONTRIBUTION_DATASET_PATH,
    action_output_path: Path = DEFAULT_HISTORY_ACTION_DATASET_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    analysis = pd.read_csv(learn_analysis_csv).copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    env_lookup = (
        analysis.drop_duplicates(subset=["gameId"], keep="first")
        .set_index("gameId")
        .to_dict(orient="index")
    )

    rounds = pd.read_csv(learn_rounds_csv)
    rounds = _build_round_index(rounds)
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)

    cluster_assignments = _prepare_cluster_assignments(learn_cluster_weights_path)
    cluster_assignments["gameId"] = cluster_assignments["gameId"].astype(str)
    cluster_assignments["playerId"] = cluster_assignments["playerId"].astype(str)

    merged = rounds.merge(
        cluster_assignments,
        on=["gameId", "playerId"],
        how="left",
        validate="many_to_one",
    )
    merged["hard_cluster_id"] = pd.to_numeric(merged["hard_cluster_id"], errors="coerce").fillna(1).astype(int)

    contribution_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for game_id, game_df in merged.groupby("gameId", sort=True):
        env = env_lookup.get(str(game_id))
        if not env:
            continue
        env = dict(env)
        player_ids = list(dict.fromkeys(game_df.sort_values(["roundIndex", "playerId"])["playerId"].astype(str).tolist()))
        cluster_by_player = (
            game_df[["playerId", "hard_cluster_id"]]
            .drop_duplicates(subset=["playerId"], keep="first")
            .set_index("playerId")["hard_cluster_id"]
            .astype(int)
            .to_dict()
        )
        stats_by_player = {player_id: PlayerVisibleStats() for player_id in player_ids}

        for round_idx in sorted(int(value) for value in game_df["roundIndex"].dropna().unique().tolist() if int(value) > 0):
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

            for player_id in player_ids:
                cluster_id = _hard_cluster_id(cluster_by_player.get(player_id, 1))
                history_row = _history_feature_row(
                    env=env,
                    cluster_id=cluster_id,
                    stats=stats_by_player[player_id],
                    round_idx=int(round_idx),
                )
                contribution_rows.append(
                    {
                        "gameId": str(game_id),
                        "playerId": player_id,
                        "roundIndex": int(round_idx),
                        **history_row,
                        "target_contribution_units": int(contributions_by_player.get(player_id, 0)),
                        "target_contribution_rate": float(
                            int(contributions_by_player.get(player_id, 0))
                            / max(float(env.get("CONFIG_endowment", 20) or 20), 1.0)
                        ),
                    }
                )

                action_row = _action_feature_row(
                    env=env,
                    cluster_id=cluster_id,
                    stats=stats_by_player[player_id],
                    round_idx=int(round_idx),
                    focal_player_id=player_id,
                    contributions_by_player=contributions_by_player,
                )
                punish_targets = punish_by_player.get(player_id, {})
                reward_targets = reward_by_player.get(player_id, {})
                action_rows.append(
                    {
                        "gameId": str(game_id),
                        "playerId": player_id,
                        "roundIndex": int(round_idx),
                        **action_row,
                        "target_any_punish": int(sum(int(value) for value in punish_targets.values()) > 0),
                        "target_any_reward": int(sum(int(value) for value in reward_targets.values()) > 0),
                        "target_punish_units_total": int(sum(int(value) for value in punish_targets.values())),
                        "target_reward_units_total": int(sum(int(value) for value in reward_targets.values())),
                        "target_punish_target_count": int(sum(1 for value in punish_targets.values() if int(value) > 0)),
                        "target_reward_target_count": int(sum(1 for value in reward_targets.values() if int(value) > 0)),
                        "target_punish_target_rank": _weighted_target_rank_mean(
                            focal_player_id=player_id,
                            target_units_by_player=punish_targets,
                            contributions_by_player=contributions_by_player,
                        ),
                        "target_reward_target_rank": _weighted_target_rank_mean(
                            focal_player_id=player_id,
                            target_units_by_player=reward_targets,
                            contributions_by_player=contributions_by_player,
                        ),
                    }
                )

            for player_id in player_ids:
                _update_visible_stats(
                    stats=stats_by_player[player_id],
                    env=env,
                    focal_player_id=player_id,
                    contributions_by_player=contributions_by_player,
                    punish_by_player=punish_by_player,
                    reward_by_player=reward_by_player,
                    payoff_by_player=payoff_by_player,
                )

    contribution_df = pd.DataFrame(contribution_rows)
    action_df = pd.DataFrame(action_rows)
    contribution_output_path.parent.mkdir(parents=True, exist_ok=True)
    action_output_path.parent.mkdir(parents=True, exist_ok=True)
    contribution_df.to_parquet(contribution_output_path, index=False)
    action_df.to_parquet(action_output_path, index=False)
    return contribution_df, action_df


def train_history_conditioned_policy(
    *,
    output_model_path: Path = DEFAULT_HISTORY_POLICY_MODEL_PATH,
    contribution_dataset_path: Path = DEFAULT_HISTORY_CONTRIBUTION_DATASET_PATH,
    action_dataset_path: Path = DEFAULT_HISTORY_ACTION_DATASET_PATH,
    summary_output_path: Path = DEFAULT_HISTORY_TRAIN_SUMMARY_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
) -> Dict[str, Any]:
    contribution_df, action_df = build_history_training_datasets(
        contribution_output_path=contribution_dataset_path,
        action_output_path=action_dataset_path,
        learn_cluster_weights_path=learn_cluster_weights_path,
        learn_analysis_csv=learn_analysis_csv,
        learn_rounds_csv=learn_rounds_csv,
    )
    if contribution_df.empty or action_df.empty:
        raise ValueError("History-conditioned training datasets are empty.")

    contribution_feature_columns = [
        column
        for column in contribution_df.columns
        if column
        not in {
            "gameId",
            "playerId",
            "roundIndex",
            "target_contribution_units",
            "target_contribution_rate",
        }
    ]
    action_feature_columns = [
        column
        for column in action_df.columns
        if not column.startswith("target_") and column not in {"gameId", "playerId", "roundIndex"}
    ]

    contribution_x = _prepare_model_matrix(contribution_df[contribution_feature_columns])
    action_x = _prepare_model_matrix(action_df[action_feature_columns])
    contribution_feature_names = list(contribution_x.columns)
    action_feature_names = list(action_x.columns)

    aon_mask = pd.to_numeric(contribution_df["CONFIG_allOrNothing"], errors="coerce").fillna(0).astype(int) == 1
    continuous_mask = ~aon_mask

    continuous_model = _fit_regressor(
        contribution_x.loc[continuous_mask],
        contribution_df.loc[continuous_mask, "target_contribution_rate"],
    )
    continuous_predictions = continuous_model.predict(contribution_x.loc[continuous_mask])
    continuous_residuals = _build_residual_store(
        actual=contribution_df.loc[continuous_mask, "target_contribution_rate"],
        predicted=continuous_predictions,
        cluster_ids=contribution_df.loc[continuous_mask, "cluster_id"],
    )
    aon_model = _fit_binary_classifier(
        contribution_x.loc[aon_mask],
        (pd.to_numeric(contribution_df.loc[aon_mask, "target_contribution_rate"], errors="coerce") >= 0.5).astype(int),
    )

    punish_enabled_mask = pd.to_numeric(action_df["CONFIG_punishmentExists"], errors="coerce").fillna(0).astype(int) == 1
    reward_enabled_mask = pd.to_numeric(action_df["CONFIG_rewardExists"], errors="coerce").fillna(0).astype(int) == 1

    punish_any_model = _fit_binary_classifier(
        action_x.loc[punish_enabled_mask],
        action_df.loc[punish_enabled_mask, "target_any_punish"],
    )
    reward_any_model = _fit_binary_classifier(
        action_x.loc[reward_enabled_mask],
        action_df.loc[reward_enabled_mask, "target_any_reward"],
    )

    punish_positive_mask = punish_enabled_mask & (
        pd.to_numeric(action_df["target_any_punish"], errors="coerce").fillna(0).astype(int) == 1
    )
    reward_positive_mask = reward_enabled_mask & (
        pd.to_numeric(action_df["target_any_reward"], errors="coerce").fillna(0).astype(int) == 1
    )

    punish_units_model = _fit_regressor(
        action_x.loc[punish_positive_mask],
        action_df.loc[punish_positive_mask, "target_punish_units_total"],
    )
    punish_units_predictions = punish_units_model.predict(action_x.loc[punish_positive_mask])
    punish_units_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_units_total"],
        predicted=punish_units_predictions,
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )
    punish_target_count_model = _fit_regressor(
        action_x.loc[punish_positive_mask],
        action_df.loc[punish_positive_mask, "target_punish_target_count"],
    )
    punish_target_count_predictions = punish_target_count_model.predict(action_x.loc[punish_positive_mask])
    punish_target_count_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_target_count"],
        predicted=punish_target_count_predictions,
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )
    punish_orientation_model = _fit_regressor(
        action_x.loc[punish_positive_mask],
        action_df.loc[punish_positive_mask, "target_punish_target_rank"],
    )
    punish_orientation_predictions = punish_orientation_model.predict(action_x.loc[punish_positive_mask])
    punish_orientation_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_target_rank"],
        predicted=punish_orientation_predictions,
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )

    reward_units_model = _fit_regressor(
        action_x.loc[reward_positive_mask],
        action_df.loc[reward_positive_mask, "target_reward_units_total"],
    )
    reward_units_predictions = reward_units_model.predict(action_x.loc[reward_positive_mask])
    reward_units_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_units_total"],
        predicted=reward_units_predictions,
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )
    reward_target_count_model = _fit_regressor(
        action_x.loc[reward_positive_mask],
        action_df.loc[reward_positive_mask, "target_reward_target_count"],
    )
    reward_target_count_predictions = reward_target_count_model.predict(action_x.loc[reward_positive_mask])
    reward_target_count_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_target_count"],
        predicted=reward_target_count_predictions,
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )
    reward_orientation_model = _fit_regressor(
        action_x.loc[reward_positive_mask],
        action_df.loc[reward_positive_mask, "target_reward_target_rank"],
    )
    reward_orientation_predictions = reward_orientation_model.predict(action_x.loc[reward_positive_mask])
    reward_orientation_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_target_rank"],
        predicted=reward_orientation_predictions,
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )

    payload = {
        "version": 1,
        "contribution_feature_names": contribution_feature_names,
        "action_feature_names": action_feature_names,
        "continuous_contribution_model": continuous_model,
        "aon_contribution_model": aon_model,
        "continuous_contribution_residuals": continuous_residuals,
        "punish_any_model": punish_any_model,
        "punish_units_model": punish_units_model,
        "punish_units_residuals": punish_units_residuals,
        "punish_target_count_model": punish_target_count_model,
        "punish_target_count_residuals": punish_target_count_residuals,
        "punish_orientation_model": punish_orientation_model,
        "punish_orientation_residuals": punish_orientation_residuals,
        "reward_any_model": reward_any_model,
        "reward_units_model": reward_units_model,
        "reward_units_residuals": reward_units_residuals,
        "reward_target_count_model": reward_target_count_model,
        "reward_target_count_residuals": reward_target_count_residuals,
        "reward_orientation_model": reward_orientation_model,
        "reward_orientation_residuals": reward_orientation_residuals,
        "learn_cluster_weights_path": str(learn_cluster_weights_path),
        "learn_analysis_csv": str(learn_analysis_csv),
        "learn_rounds_csv": str(learn_rounds_csv),
    }
    _atomic_joblib_dump(payload, output_model_path)

    summary_df = pd.DataFrame(
        [
            {
                "dataset": "contribution",
                "n_rows": int(len(contribution_df)),
                "n_all_or_nothing_rows": int(aon_mask.sum()),
                "n_continuous_rows": int(continuous_mask.sum()),
                "mean_target_contribution_rate": float(
                    pd.to_numeric(contribution_df["target_contribution_rate"], errors="coerce").mean()
                ),
            },
            {
                "dataset": "punish_action",
                "n_rows": int(punish_enabled_mask.sum()),
                "n_positive_rows": int(punish_positive_mask.sum()),
                "positive_rate": float(
                    pd.to_numeric(action_df.loc[punish_enabled_mask, "target_any_punish"], errors="coerce").mean()
                )
                if int(punish_enabled_mask.sum()) > 0
                else float("nan"),
                "mean_units_if_positive": float(
                    pd.to_numeric(
                        action_df.loc[punish_positive_mask, "target_punish_units_total"],
                        errors="coerce",
                    ).mean()
                )
                if int(punish_positive_mask.sum()) > 0
                else float("nan"),
            },
            {
                "dataset": "reward_action",
                "n_rows": int(reward_enabled_mask.sum()),
                "n_positive_rows": int(reward_positive_mask.sum()),
                "positive_rate": float(
                    pd.to_numeric(action_df.loc[reward_enabled_mask, "target_any_reward"], errors="coerce").mean()
                )
                if int(reward_enabled_mask.sum()) > 0
                else float("nan"),
                "mean_units_if_positive": float(
                    pd.to_numeric(
                        action_df.loc[reward_positive_mask, "target_reward_units_total"],
                        errors="coerce",
                    ).mean()
                )
                if int(reward_positive_mask.sum()) > 0
                else float("nan"),
            },
        ]
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    return payload


class HistoryConditionedArchetypePolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor,
        model_payload: Dict[str, Any],
    ) -> None:
        self.env_model = env_model
        self.model_payload = model_payload
        self.contribution_feature_names = list(model_payload["contribution_feature_names"])
        self.action_feature_names = list(model_payload["action_feature_names"])

    @classmethod
    def from_config(
        cls,
        config: HistoryConditionedArchetypePolicyConfig,
    ) -> "HistoryConditionedArchetypePolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        env_model_path = artifacts_root / "models" / "dirichlet_env_model.pkl"
        model_path = artifacts_root / "models" / "history_conditioned_policy.pkl"
        if not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        if config.rebuild_model or not model_path.exists():
            train_history_conditioned_policy(output_model_path=model_path)
        env_model = DirichletEnvRegressor.load(env_model_path)
        model_payload = joblib.load(model_path)
        return cls(env_model=env_model, model_payload=model_payload)

    def predict_cluster_distribution(self, env: Mapping[str, Any]) -> List[float]:
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
        rng: np.random.Generator | np.random.RandomState | Any,
    ) -> HistoryConditionedGameState:
        distribution = self.predict_cluster_distribution(env)
        cluster_ids = list(range(1, len(distribution) + 1))
        sampled = list(np.random.default_rng(int(rng.integers(0, 2**32 - 1))).choice(cluster_ids, size=len(player_ids), p=distribution))
        avatar_lookup = {str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids}
        return HistoryConditionedGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player=avatar_lookup,
            player_by_avatar={avatar: player_id for player_id, avatar in avatar_lookup.items()},
            cluster_by_player={str(player_id): int(cluster_id) for player_id, cluster_id in zip(player_ids, sampled)},
            stats_by_player={str(player_id): PlayerVisibleStats() for player_id in player_ids},
        )

    def _predict_continuous_contribution_rate(
        self,
        *,
        feature_row: Dict[str, Any],
        cluster_id: int,
        rng: np.random.Generator,
    ) -> float:
        features = _prepare_model_matrix(pd.DataFrame([feature_row]), self.contribution_feature_names)
        base = float(self.model_payload["continuous_contribution_model"].predict(features)[0])
        residual = _sample_residual(rng, self.model_payload["continuous_contribution_residuals"], cluster_id)
        return float(max(0.0, min(1.0, base + residual)))

    def sample_contributions_for_round(
        self,
        *,
        game_state: HistoryConditionedGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        out: Dict[str, int] = {}
        for player_id in game_state.player_ids:
            cluster_id = int(game_state.cluster_by_player[player_id])
            feature_row = _history_feature_row(
                env=game_state.env,
                cluster_id=cluster_id,
                stats=game_state.stats_by_player[player_id],
                round_idx=int(round_idx),
            )
            features = _prepare_model_matrix(pd.DataFrame([feature_row]), self.contribution_feature_names)
            if as_bool(game_state.env.get("CONFIG_allOrNothing", False)):
                prob_all = _safe_prob(self.model_payload["aon_contribution_model"], features)
                out[player_id] = endowment if float(rng.random()) < prob_all else 0
            else:
                sampled_rate = self._predict_continuous_contribution_rate(
                    feature_row=feature_row,
                    cluster_id=cluster_id,
                    rng=rng,
                )
                out[player_id] = int(max(0, min(endowment, round(sampled_rate * endowment))))
        return out

    def _select_ranked_targets(
        self,
        *,
        focal_player_id: str,
        contributions_by_player: Mapping[str, int],
        target_rank: float,
        target_count: int,
        units_total: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        peer_items = [
            (str(player_id), float(value))
            for player_id, value in contributions_by_player.items()
            if str(player_id) != str(focal_player_id)
        ]
        if not peer_items:
            return {}
        order = list(range(len(peer_items)))
        rng.shuffle(order)
        peer_items = [peer_items[index] for index in order]
        peer_values = [value for _, value in peer_items]
        ranked = sorted(
            peer_items,
            key=lambda item: abs(_normalized_rank(float(item[1]), peer_values) - float(target_rank)),
        )
        chosen = ranked[: max(1, min(int(target_count), len(ranked)))]
        allocations = [0] * len(chosen)
        for index in range(int(max(units_total, len(chosen)))):
            allocations[index % len(chosen)] += 1
        return {
            str(player_id): int(units)
            for (player_id, _), units in zip(chosen, allocations)
            if int(units) > 0
        }

    def _sample_mechanism_actions(
        self,
        *,
        mechanism: str,
        game_state: HistoryConditionedGameState,
        player_id: str,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        cluster_id = int(game_state.cluster_by_player[player_id])
        feature_row = _action_feature_row(
            env=game_state.env,
            cluster_id=cluster_id,
            stats=game_state.stats_by_player[player_id],
            round_idx=int(round_idx),
            focal_player_id=player_id,
            contributions_by_player=contributions_by_player,
        )
        features = _prepare_model_matrix(pd.DataFrame([feature_row]), self.action_feature_names)
        if mechanism == "punish":
            any_model = self.model_payload["punish_any_model"]
            units_model = self.model_payload["punish_units_model"]
            units_residuals = self.model_payload["punish_units_residuals"]
            count_model = self.model_payload["punish_target_count_model"]
            count_residuals = self.model_payload["punish_target_count_residuals"]
            orientation_model = self.model_payload["punish_orientation_model"]
            orientation_residuals = self.model_payload["punish_orientation_residuals"]
            cost = float(game_state.env.get("CONFIG_punishmentCost", 1) or 1)
        else:
            any_model = self.model_payload["reward_any_model"]
            units_model = self.model_payload["reward_units_model"]
            units_residuals = self.model_payload["reward_units_residuals"]
            count_model = self.model_payload["reward_target_count_model"]
            count_residuals = self.model_payload["reward_target_count_residuals"]
            orientation_model = self.model_payload["reward_orientation_model"]
            orientation_residuals = self.model_payload["reward_orientation_residuals"]
            cost = float(game_state.env.get("CONFIG_rewardCost", 1) or 1)

        if float(rng.random()) >= _safe_prob(any_model, features):
            return {}

        max_units = max(
            int(np.floor(float(game_state.env.get("CONFIG_endowment", 20) or 20) / max(cost, 1.0))),
            1,
        )
        raw_units = float(units_model.predict(features)[0]) + _sample_residual(rng, units_residuals, cluster_id)
        raw_count = float(count_model.predict(features)[0]) + _sample_residual(rng, count_residuals, cluster_id)
        raw_orientation = float(orientation_model.predict(features)[0]) + _sample_residual(
            rng,
            orientation_residuals,
            cluster_id,
        )
        units_total = int(max(1, min(max_units, round(raw_units))))
        target_count = int(max(1, round(raw_count)))
        target_rank = float(max(0.0, min(1.0, raw_orientation)))
        return self._select_ranked_targets(
            focal_player_id=player_id,
            contributions_by_player=contributions_by_player,
            target_rank=target_rank,
            target_count=target_count,
            units_total=max(units_total, target_count),
            rng=rng,
        )

    def sample_actions_for_round(
        self,
        *,
        game_state: HistoryConditionedGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}
        for player_id in game_state.player_ids:
            punish_out[player_id] = (
                self._sample_mechanism_actions(
                    mechanism="punish",
                    game_state=game_state,
                    player_id=player_id,
                    contributions_by_player=contributions_by_player,
                    round_idx=round_idx,
                    rng=rng,
                )
                if as_bool(game_state.env.get("CONFIG_punishmentExists", False))
                else {}
            )
            reward_out[player_id] = (
                self._sample_mechanism_actions(
                    mechanism="reward",
                    game_state=game_state,
                    player_id=player_id,
                    contributions_by_player=contributions_by_player,
                    round_idx=round_idx,
                    rng=rng,
                )
                if as_bool(game_state.env.get("CONFIG_rewardExists", False))
                else {}
            )
        return {"punish": punish_out, "reward": reward_out}

    def record_round(
        self,
        *,
        game_state: HistoryConditionedGameState,
        contributions_by_player: Mapping[str, int],
        punish_by_player: Mapping[str, Mapping[str, int]],
        reward_by_player: Mapping[str, Mapping[str, int]],
        payoff_by_player: Mapping[str, float],
    ) -> None:
        for player_id in game_state.player_ids:
            _update_visible_stats(
                stats=game_state.stats_by_player[player_id],
                env=game_state.env,
                focal_player_id=player_id,
                contributions_by_player=contributions_by_player,
                punish_by_player=punish_by_player,
                reward_by_player=reward_by_player,
                payoff_by_player=payoff_by_player,
            )

    def record_actual_round(
        self,
        *,
        game_state: HistoryConditionedGameState,
        round_rows: pd.DataFrame,
    ) -> None:
        if round_rows is None or round_rows.empty:
            return
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
        )
