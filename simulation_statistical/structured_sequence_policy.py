from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier, SGDRegressor

from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import _build_round_index, as_bool
from simulation_statistical.history_conditioned_policy import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_LEARN_ANALYSIS_CSV,
    DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    DEFAULT_LEARN_ROUNDS_CSV,
    _action_dicts_by_player,
    _atomic_joblib_dump,
    _build_residual_store,
    _contrib_by_player,
    _hard_cluster_id,
    _normalized_rank,
    _prepare_cluster_assignments,
    _round_payoff_by_player,
    _sample_residual,
    _visible_env_features,
)


DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "exact_sequence_policy.pkl"
DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_POLICY_MODEL_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "models" / "exact_sequence_no_cluster_policy.pkl"
)
DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "exact_sequence_contribution_train.parquet"
)
DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "exact_sequence_action_train.parquet"
)
DEFAULT_EXACT_SEQUENCE_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "exact_sequence_policy_train_summary.csv"
)
DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "exact_sequence_no_cluster_policy_train_summary.csv"
)
DEFAULT_ENV_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "dirichlet_env_model.pkl"
DEFAULT_VAL_GAME_CLUSTER_DISTRIBUTION_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "game_cluster_distribution_val.parquet"
)
DEFAULT_VAL_PLAYER_GAME_TABLE_CLEAN_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "player_game_table_val_clean.parquet"
)

ACTION_LABEL_NONE = 0
ACTION_LABEL_PUNISH = 1
ACTION_LABEL_REWARD = 2
ACTION_CLASSES = np.array(
    [ACTION_LABEL_NONE, ACTION_LABEL_PUNISH, ACTION_LABEL_REWARD],
    dtype=int,
)
CONTINUOUS_CONTRIBUTION_BIN_RATES = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
CONTINUOUS_CONTRIBUTION_BIN_CLASSES = np.arange(len(CONTINUOUS_CONTRIBUTION_BIN_RATES), dtype=int)
DEFAULT_CLUSTER_CALIBRATION_STRENGTH = 0.5
DEFAULT_CLUSTER_CALIBRATION_PRIOR_WEIGHT = 24.0


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, features: Any) -> np.ndarray:
        n_rows = int(features.shape[0]) if hasattr(features, "shape") else len(features)
        return np.full(n_rows, float(self.value), dtype=float)


@dataclass
class ConstantBinaryClassifier:
    positive_probability: float

    def predict_proba(self, features: Any) -> np.ndarray:
        n_rows = int(features.shape[0]) if hasattr(features, "shape") else len(features)
        prob = float(max(0.0, min(1.0, self.positive_probability)))
        out = np.zeros((n_rows, 2), dtype=float)
        out[:, 0] = 1.0 - prob
        out[:, 1] = prob
        return out

    def predict(self, features: Any) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


@dataclass
class ConstantMulticlassClassifier:
    class_probabilities: Sequence[float]
    classes: Sequence[int] | None = None

    def __post_init__(self) -> None:
        probs = np.asarray(list(self.class_probabilities), dtype=float)
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError("ConstantMulticlassClassifier expects a non-empty 1D probability vector.")
        if self.classes is None:
            classes = np.arange(probs.size, dtype=int)
        else:
            classes = np.asarray(list(self.classes), dtype=int)
        if classes.shape != (probs.size,):
            raise ValueError("ConstantMulticlassClassifier classes must match class_probabilities length.")
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            probs = np.zeros_like(probs, dtype=float)
            probs[0] = 1.0
        else:
            probs = probs / total
        self.class_probabilities = probs.tolist()
        self.classes_ = classes.copy()

    def predict_proba(self, features: Any) -> np.ndarray:
        n_rows = int(features.shape[0]) if hasattr(features, "shape") else len(features)
        return np.tile(np.asarray(self.class_probabilities, dtype=float), (n_rows, 1))

    def predict(self, features: Any) -> np.ndarray:
        probs = self.predict_proba(features)
        return self.classes_[np.argmax(probs, axis=1)]


@dataclass(frozen=True)
class ExactSequenceArchetypePolicyConfig:
    artifacts_root: str | None = None
    rebuild_model: bool = False
    use_cluster: bool = True
    model_path: str | None = None
    cluster_source: str = "env_model"


@dataclass
class StructuredRoundState:
    round_idx: int
    contributions_by_player: Dict[str, int]
    punish_by_player: Dict[str, Dict[str, int]]
    reward_by_player: Dict[str, Dict[str, int]]
    payoff_by_player: Dict[str, float]


@dataclass
class ExactSequenceGameState:
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    cluster_by_player: Dict[str, int]
    history_rounds: List[StructuredRoundState] = field(default_factory=list)


def _default_exact_sequence_model_path(*, use_cluster: bool) -> Path:
    return (
        DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH
        if use_cluster
        else DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_POLICY_MODEL_PATH
    )


def _default_exact_sequence_summary_path(*, use_cluster: bool) -> Path:
    return (
        DEFAULT_EXACT_SEQUENCE_TRAIN_SUMMARY_PATH
        if use_cluster
        else DEFAULT_EXACT_SEQUENCE_NO_CLUSTER_TRAIN_SUMMARY_PATH
    )


def _sanitize_feature_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        name = str(key)
        if isinstance(value, str):
            out[name] = value if value else "NA"
            continue
        if value is None:
            out[name] = 0.0
            continue
        try:
            if pd.isna(value):
                out[name] = 0.0
                continue
        except Exception:
            pass
        if isinstance(value, (bool, np.bool_)):
            out[name] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[name] = float(value)
        else:
            out[name] = str(value)
    return out


def _fit_binary_classifier(features: sparse.csr_matrix, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    positive_rate = float(clean_target.mean()) if len(clean_target) else 0.0
    if clean_target.nunique() < 2:
        return ConstantBinaryClassifier(positive_probability=positive_rate)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=4000,
        tol=1e-3,
        random_state=0,
        average=True,
    )
    model.fit(features, clean_target)
    return model


def _fit_regressor(features: sparse.csr_matrix, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce")
    keep = clean_target.notna().to_numpy()
    if int(np.sum(keep)) == 0:
        return ConstantRegressor(value=0.0)
    pair = features[keep]
    y = clean_target.loc[clean_target.notna()].astype(float)
    if pair.shape[0] < 2 or y.nunique() < 2:
        return ConstantRegressor(value=float(y.mean()))
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        max_iter=4000,
        tol=1e-3,
        random_state=0,
        average=True,
    )
    model.fit(pair, y)
    return model


def _predict_probability(model: object, features: sparse.csr_matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _predict_multiclass_probabilities(
    model: object,
    features: sparse.csr_matrix,
    expected_classes: np.ndarray,
) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        labels = np.asarray(model.predict(features), dtype=int)
        out = np.zeros((len(labels), len(expected_classes)), dtype=float)
        for row_index, label in enumerate(labels):
            matches = np.where(expected_classes == int(label))[0]
            if len(matches) > 0:
                out[row_index, int(matches[0])] = 1.0
        return out

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        probs = np.asarray(model.predict_proba(features), dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    model_classes = np.asarray(getattr(model, "classes_", expected_classes), dtype=int)
    if np.array_equal(model_classes, expected_classes):
        return probs

    aligned = np.zeros((probs.shape[0], len(expected_classes)), dtype=float)
    for class_index, label in enumerate(expected_classes):
        matches = np.where(model_classes == int(label))[0]
        if len(matches) > 0:
            aligned[:, class_index] = probs[:, int(matches[0])]
    row_sums = aligned.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return aligned / safe


def _predict_action_probabilities(model: object, features: sparse.csr_matrix) -> np.ndarray:
    return _predict_multiclass_probabilities(model, features, ACTION_CLASSES)


def _continuous_rate_to_bin_label(rate: float) -> int:
    clipped = float(max(0.0, min(1.0, rate)))
    return int(np.argmin(np.abs(CONTINUOUS_CONTRIBUTION_BIN_RATES - clipped)))


def _cluster_key(cluster_id: int | str) -> str:
    text = str(cluster_id)
    if text.startswith("cluster_"):
        return text
    return f"cluster_{int(float(cluster_id))}"


def _normalize_probability_vector(values: Sequence[float]) -> np.ndarray:
    probs = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("Expected a non-empty 1D probability vector.")
    if total <= 0.0:
        return np.full(probs.size, 1.0 / probs.size, dtype=float)
    return probs / total


def _safe_logit(probability: float) -> float:
    clipped = float(max(1e-6, min(1.0 - 1e-6, probability)))
    return float(np.log(clipped / (1.0 - clipped)))


def _safe_sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _shrunk_binary_rate(
    *,
    positive_count: int,
    total_count: int,
    global_rate: float,
    prior_weight: float,
) -> float:
    if total_count <= 0:
        return float(global_rate)
    return float((float(positive_count) + float(prior_weight) * float(global_rate)) / (float(total_count) + float(prior_weight)))


def _shrunk_multiclass_probs(
    *,
    counts: Sequence[float],
    global_probs: Sequence[float],
    prior_weight: float,
) -> np.ndarray:
    count_array = np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    global_array = _normalize_probability_vector(global_probs)
    total_count = float(count_array.sum())
    if total_count <= 0.0:
        return global_array
    adjusted = count_array + float(prior_weight) * global_array
    return _normalize_probability_vector(adjusted)


def _action_label(punish_units: int, reward_units: int) -> int:
    punish_positive = int(punish_units) > 0
    reward_positive = int(reward_units) > 0
    if punish_positive and reward_positive:
        raise ValueError("A focal-target pair cannot be both punished and rewarded in the same round.")
    if punish_positive:
        return ACTION_LABEL_PUNISH
    if reward_positive:
        return ACTION_LABEL_REWARD
    return ACTION_LABEL_NONE


def _relative_peer_ids(player_ids: Sequence[str], focal_player_id: str) -> List[str]:
    focal = str(focal_player_id)
    return [str(player_id) for player_id in player_ids if str(player_id) != focal]


def _player_inbound_units(
    actions_by_player: Mapping[str, Mapping[str, int]],
    target_player_id: str,
) -> int:
    target = str(target_player_id)
    return int(
        sum(int(targets.get(target, 0)) for targets in actions_by_player.values())
    )


def _sum_units(values: Mapping[str, int]) -> int:
    return int(sum(int(value) for value in values.values()))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float)))


def _round_redistributed_share(
    env: Mapping[str, Any],
    contributions_by_player: Mapping[str, int],
) -> float:
    try:
        multiplier = float(env.get("CONFIG_multiplier", 0) or 0)
    except Exception:
        multiplier = 0.0
    n_players = max(len(contributions_by_player), 1)
    return float(multiplier * float(sum(int(value) for value in contributions_by_player.values())) / n_players)


def _history_feature_row(
    *,
    env: Mapping[str, Any],
    cluster_id: int | None,
    player_ids: Sequence[str],
    focal_player_id: str,
    history_rounds: Sequence[StructuredRoundState],
    round_idx: int,
) -> Dict[str, Any]:
    focal = str(focal_player_id)
    peers = _relative_peer_ids(player_ids, focal)
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    show_other = as_bool(env.get("CONFIG_showOtherSummaries", False))
    show_punish_id = as_bool(env.get("CONFIG_showPunishmentId", False))
    show_reward_id = as_bool(env.get("CONFIG_showRewardId", False))

    row = _visible_env_features(env, round_idx)
    row["CONFIG_punishmentTech"] = pd.to_numeric(
        pd.Series([env.get("CONFIG_punishmentTech", np.nan)]),
        errors="coerce",
    ).iloc[0]
    row["CONFIG_rewardTech"] = pd.to_numeric(
        pd.Series([env.get("CONFIG_rewardTech", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if cluster_id is not None:
        row["cluster_id"] = float(cluster_id)
    row["history_rounds_observed"] = float(len(history_rounds))
    row["peer_count"] = float(len(peers))

    for slot_index, _ in enumerate(peers, start=1):
        row[f"peer{slot_index}_exists"] = 1.0

    for lag_index, round_state in enumerate(reversed(history_rounds), start=1):
        contrib = round_state.contributions_by_player
        punish = round_state.punish_by_player
        reward = round_state.reward_by_player
        payoff = round_state.payoff_by_player

        own_contrib = float(contrib.get(focal, 0.0))
        own_payoff = float(payoff.get(focal, 0.0) or 0.0)
        own_punish_spent = float(_sum_units(punish.get(focal, {})))
        own_reward_spent = float(_sum_units(reward.get(focal, {})))
        own_punish_received = float(_player_inbound_units(punish, focal))
        own_reward_received = float(_player_inbound_units(reward, focal))
        peer_contrib_rates = [float(contrib.get(peer_id, 0.0)) / max(endowment, 1.0) for peer_id in peers]

        row[f"lag{lag_index}_observed"] = 1.0
        row[f"lag{lag_index}_own_contrib_rate"] = own_contrib / max(endowment, 1.0)
        row[f"lag{lag_index}_own_payoff_norm"] = own_payoff / max(endowment, 1.0)
        row[f"lag{lag_index}_own_punish_spent_units"] = own_punish_spent
        row[f"lag{lag_index}_own_reward_spent_units"] = own_reward_spent
        row[f"lag{lag_index}_own_punish_received_units"] = own_punish_received
        row[f"lag{lag_index}_own_reward_received_units"] = own_reward_received
        row[f"lag{lag_index}_peer_mean_contrib_rate"] = _safe_mean(peer_contrib_rates)
        row[f"lag{lag_index}_peer_std_contrib_rate"] = _safe_std(peer_contrib_rates)
        row[f"lag{lag_index}_redistributed_share_norm"] = _round_redistributed_share(env, contrib) / max(endowment, 1.0)

        for slot_index, peer_id in enumerate(peers, start=1):
            peer = str(peer_id)
            peer_payoff = float(payoff.get(peer, 0.0) or 0.0)
            row[f"lag{lag_index}_peer{slot_index}_contrib_rate"] = float(contrib.get(peer, 0.0)) / max(endowment, 1.0)
            row[f"lag{lag_index}_own_punish_to_peer{slot_index}_units"] = float(punish.get(focal, {}).get(peer, 0))
            row[f"lag{lag_index}_own_reward_to_peer{slot_index}_units"] = float(reward.get(focal, {}).get(peer, 0))

            if show_punish_id:
                row[f"lag{lag_index}_peer{slot_index}_punish_to_own_units"] = float(
                    punish.get(peer, {}).get(focal, 0)
                )
            if show_reward_id:
                row[f"lag{lag_index}_peer{slot_index}_reward_to_own_units"] = float(
                    reward.get(peer, {}).get(focal, 0)
                )
            if show_other:
                row[f"lag{lag_index}_peer{slot_index}_payoff_norm"] = peer_payoff / max(endowment, 1.0)
                row[f"lag{lag_index}_peer{slot_index}_punish_spent_units"] = float(_sum_units(punish.get(peer, {})))
                row[f"lag{lag_index}_peer{slot_index}_reward_spent_units"] = float(_sum_units(reward.get(peer, {})))
                row[f"lag{lag_index}_peer{slot_index}_punish_received_units"] = float(_player_inbound_units(punish, peer))
                row[f"lag{lag_index}_peer{slot_index}_reward_received_units"] = float(_player_inbound_units(reward, peer))

    return row


def _contribution_feature_row(
    *,
    env: Mapping[str, Any],
    cluster_id: int | None,
    player_ids: Sequence[str],
    focal_player_id: str,
    history_rounds: Sequence[StructuredRoundState],
    round_idx: int,
) -> Dict[str, Any]:
    return _history_feature_row(
        env=env,
        cluster_id=cluster_id,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        history_rounds=history_rounds,
        round_idx=round_idx,
    )


def _action_edge_feature_row(
    *,
    env: Mapping[str, Any],
    cluster_id: int | None,
    player_ids: Sequence[str],
    focal_player_id: str,
    target_player_id: str,
    history_rounds: Sequence[StructuredRoundState],
    round_idx: int,
    current_contributions_by_player: Mapping[str, int],
    base_history_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    focal = str(focal_player_id)
    target = str(target_player_id)
    peers = _relative_peer_ids(player_ids, focal)
    peer_slot_by_player = {peer_id: slot_index for slot_index, peer_id in enumerate(peers, start=1)}
    target_slot = int(peer_slot_by_player[target])
    endowment = float(env.get("CONFIG_endowment", 20) or 20)

    row = (
        dict(base_history_row)
        if base_history_row is not None
        else _history_feature_row(
            env=env,
            cluster_id=cluster_id,
            player_ids=player_ids,
            focal_player_id=focal,
            history_rounds=history_rounds,
            round_idx=round_idx,
        )
    )
    own_value = float(current_contributions_by_player.get(focal, 0.0))
    peer_values = [
        float(current_contributions_by_player.get(peer_id, 0.0)) / max(endowment, 1.0)
        for peer_id in peers
    ]
    row["candidate_peer_slot"] = float(target_slot)
    row["current_own_contrib_rate"] = own_value / max(endowment, 1.0)
    row["current_peer_mean_contrib_rate"] = _safe_mean(peer_values)
    row["current_peer_std_contrib_rate"] = _safe_std(peer_values)

    for slot_index, peer_id in enumerate(peers, start=1):
        row[f"current_peer{slot_index}_contrib_rate"] = float(
            current_contributions_by_player.get(peer_id, 0.0)
        ) / max(endowment, 1.0)

    target_value = float(current_contributions_by_player.get(target, 0.0))
    other_peer_values = [
        float(current_contributions_by_player.get(peer_id, 0.0)) / max(endowment, 1.0)
        for peer_id in peers
        if peer_id != target
    ]
    target_value_rate = target_value / max(endowment, 1.0)
    row["current_target_contrib_rate"] = target_value_rate
    row["current_target_minus_own_contrib_rate"] = target_value_rate - row["current_own_contrib_rate"]
    row["current_target_rank_among_peers"] = _normalized_rank(target_value_rate, peer_values)
    row["current_other_peers_mean_contrib_rate"] = _safe_mean(other_peer_values)
    row["current_other_peers_std_contrib_rate"] = _safe_std(other_peer_values)

    show_other = as_bool(env.get("CONFIG_showOtherSummaries", False))
    show_punish_id = as_bool(env.get("CONFIG_showPunishmentId", False))
    show_reward_id = as_bool(env.get("CONFIG_showRewardId", False))

    for lag_index, round_state in enumerate(reversed(history_rounds), start=1):
        contrib = round_state.contributions_by_player
        punish = round_state.punish_by_player
        reward = round_state.reward_by_player
        payoff = round_state.payoff_by_player
        target_payoff = float(payoff.get(target, 0.0) or 0.0)

        row[f"lag{lag_index}_target_contrib_rate"] = float(contrib.get(target, 0.0)) / max(endowment, 1.0)
        row[f"lag{lag_index}_own_punish_to_target_units"] = float(punish.get(focal, {}).get(target, 0))
        row[f"lag{lag_index}_own_reward_to_target_units"] = float(reward.get(focal, {}).get(target, 0))
        if show_punish_id:
            row[f"lag{lag_index}_target_punish_to_own_units"] = float(punish.get(target, {}).get(focal, 0))
        if show_reward_id:
            row[f"lag{lag_index}_target_reward_to_own_units"] = float(reward.get(target, {}).get(focal, 0))
        if show_other:
            row[f"lag{lag_index}_target_payoff_norm"] = target_payoff / max(endowment, 1.0)
            row[f"lag{lag_index}_target_punish_spent_units"] = float(_sum_units(punish.get(target, {})))
            row[f"lag{lag_index}_target_reward_spent_units"] = float(_sum_units(reward.get(target, {})))
            row[f"lag{lag_index}_target_punish_received_units"] = float(_player_inbound_units(punish, target))
            row[f"lag{lag_index}_target_reward_received_units"] = float(_player_inbound_units(reward, target))

    return row


def _build_round_state(
    *,
    round_idx: int,
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
) -> StructuredRoundState:
    return StructuredRoundState(
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


def build_exact_sequence_training_datasets(
    *,
    contribution_output_path: Path = DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH,
    action_output_path: Path = DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
    use_cluster: bool = True,
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

    if use_cluster:
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
    else:
        merged = rounds.copy()
        merged["hard_cluster_id"] = 0

    contribution_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

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
        history_rounds: List[StructuredRoundState] = []

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

            for player_id in player_ids:
                cluster_id = (
                    _hard_cluster_id(cluster_by_player.get(player_id, 1))
                    if use_cluster
                    else None
                )
                base_history_row = _contribution_feature_row(
                    env=env,
                    cluster_id=cluster_id,
                    player_ids=player_ids,
                    focal_player_id=player_id,
                    history_rounds=history_rounds,
                    round_idx=int(round_idx),
                )
                contribution_rows.append(
                    {
                        "gameId": str(game_id),
                        "playerId": str(player_id),
                        "roundIndex": int(round_idx),
                        **base_history_row,
                        "target_contribution_units": int(contributions_by_player.get(player_id, 0)),
                        "target_contribution_rate": float(
                            int(contributions_by_player.get(player_id, 0))
                            / max(float(env.get("CONFIG_endowment", 20) or 20), 1.0)
                        ),
                    }
                )

                punish_enabled = as_bool(env.get("CONFIG_punishmentExists", False))
                reward_enabled = as_bool(env.get("CONFIG_rewardExists", False))
                if punish_enabled or reward_enabled:
                    for target_player_id in _relative_peer_ids(player_ids, player_id):
                        punish_units = int(
                            punish_by_player.get(str(player_id), {}).get(str(target_player_id), 0)
                        )
                        reward_units = int(
                            reward_by_player.get(str(player_id), {}).get(str(target_player_id), 0)
                        )
                        action_label = _action_label(punish_units, reward_units)
                        action_rows.append(
                            {
                                "gameId": str(game_id),
                                "playerId": str(player_id),
                                "targetPlayerId": str(target_player_id),
                                "roundIndex": int(round_idx),
                                **_action_edge_feature_row(
                                    env=env,
                                    cluster_id=cluster_id,
                                    player_ids=player_ids,
                                    focal_player_id=player_id,
                                    target_player_id=target_player_id,
                                    history_rounds=history_rounds,
                                    round_idx=int(round_idx),
                                    current_contributions_by_player=contributions_by_player,
                                    base_history_row=base_history_row,
                                ),
                                "target_action_label": int(action_label),
                                "target_action_name": (
                                    "punish"
                                    if action_label == ACTION_LABEL_PUNISH
                                    else "reward"
                                    if action_label == ACTION_LABEL_REWARD
                                    else "none"
                                ),
                            }
                        )

            history_rounds.append(
                _build_round_state(
                    round_idx=int(round_idx),
                    contributions_by_player=contributions_by_player,
                    punish_by_player=punish_by_player,
                    reward_by_player=reward_by_player,
                    payoff_by_player=payoff_by_player,
                )
            )

    contribution_df = pd.DataFrame(contribution_rows)
    action_df = pd.DataFrame(action_rows)
    contribution_output_path.parent.mkdir(parents=True, exist_ok=True)
    action_output_path.parent.mkdir(parents=True, exist_ok=True)
    contribution_df.to_parquet(contribution_output_path, index=False)
    action_df.to_parquet(action_output_path, index=False)
    return contribution_df, action_df


def _iter_training_batches(
    *,
    learn_cluster_weights_path: Path,
    learn_analysis_csv: Path,
    learn_rounds_csv: Path,
    batch_size: int = 512,
    use_cluster: bool = True,
):
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

    if use_cluster:
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
    else:
        merged = rounds.copy()
        merged["hard_cluster_id"] = 0

    contribution_rows: List[Dict[str, Any]] = []
    contribution_targets: List[float] = []
    contribution_aon: List[int] = []
    contribution_cluster_ids: List[str] = []

    action_rows: List[Dict[str, Any]] = []
    action_labels: List[int] = []
    action_cluster_ids: List[str] = []

    def _emit_if_ready(force: bool = False):
        nonlocal contribution_rows, contribution_targets, contribution_aon, contribution_cluster_ids
        nonlocal action_rows, action_labels, action_cluster_ids
        if not force and len(contribution_rows) < batch_size and len(action_rows) < batch_size:
            return None
        payload = {
            "contribution_rows": contribution_rows,
            "contribution_targets": contribution_targets,
            "contribution_aon": contribution_aon,
            "contribution_cluster_ids": contribution_cluster_ids,
            "action_rows": action_rows,
            "action_labels": action_labels,
            "action_cluster_ids": action_cluster_ids,
        }
        contribution_rows = []
        contribution_targets = []
        contribution_aon = []
        contribution_cluster_ids = []
        action_rows = []
        action_labels = []
        action_cluster_ids = []
        return payload

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
        history_rounds: List[StructuredRoundState] = []

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

            for player_id in player_ids:
                cluster_id = (
                    _hard_cluster_id(cluster_by_player.get(player_id, 1))
                    if use_cluster
                    else None
                )
                base_history_row = _contribution_feature_row(
                    env=env,
                    cluster_id=cluster_id,
                    player_ids=player_ids,
                    focal_player_id=player_id,
                    history_rounds=history_rounds,
                    round_idx=int(round_idx),
                )
                contribution_rows.append(base_history_row)
                contribution_targets.append(
                    float(
                        int(contributions_by_player.get(player_id, 0))
                        / max(float(env.get("CONFIG_endowment", 20) or 20), 1.0)
                    )
                )
                contribution_aon.append(int(as_bool(env.get("CONFIG_allOrNothing", False))))
                contribution_cluster_ids.append(
                    f"cluster_{int(cluster_id)}" if cluster_id is not None else "none"
                )

                punish_enabled = as_bool(env.get("CONFIG_punishmentExists", False))
                reward_enabled = as_bool(env.get("CONFIG_rewardExists", False))
                if punish_enabled or reward_enabled:
                    for target_player_id in _relative_peer_ids(player_ids, player_id):
                        punish_units = int(
                            punish_by_player.get(str(player_id), {}).get(str(target_player_id), 0)
                        )
                        reward_units = int(
                            reward_by_player.get(str(player_id), {}).get(str(target_player_id), 0)
                        )
                        action_rows.append(
                            _action_edge_feature_row(
                                env=env,
                                cluster_id=cluster_id,
                                player_ids=player_ids,
                                focal_player_id=player_id,
                                target_player_id=target_player_id,
                                history_rounds=history_rounds,
                                round_idx=int(round_idx),
                                current_contributions_by_player=contributions_by_player,
                                base_history_row=base_history_row,
                            )
                        )
                        action_labels.append(_action_label(punish_units, reward_units))
                        action_cluster_ids.append(
                            f"cluster_{int(cluster_id)}" if cluster_id is not None else "none"
                        )

                emitted = _emit_if_ready(force=False)
                if emitted is not None:
                    yield emitted

            history_rounds.append(
                _build_round_state(
                    round_idx=int(round_idx),
                    contributions_by_player=contributions_by_player,
                    punish_by_player=punish_by_player,
                    reward_by_player=reward_by_player,
                    payoff_by_player=payoff_by_player,
                )
            )

    emitted = _emit_if_ready(force=True)
    if emitted is not None:
        yield emitted


def train_exact_sequence_policy(
    *,
    output_model_path: Path | None = None,
    contribution_dataset_path: Path = DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH,
    action_dataset_path: Path = DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH,
    summary_output_path: Path | None = None,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
    use_cluster: bool = True,
) -> Dict[str, Any]:
    if output_model_path is None:
        output_model_path = _default_exact_sequence_model_path(use_cluster=bool(use_cluster))
    if summary_output_path is None:
        summary_output_path = _default_exact_sequence_summary_path(use_cluster=bool(use_cluster))
    _ = contribution_dataset_path
    _ = action_dataset_path

    contribution_feature_names: set[str] = set()
    action_feature_names: set[str] = set()
    contrib_rows_count = 0
    contrib_aon_count = 0
    contrib_cont_count = 0
    contrib_target_sum = 0.0
    aon_positive_count = 0
    aon_cluster_positive_counts: Dict[str, int] = {}
    aon_cluster_total_counts: Dict[str, int] = {}
    continuous_bin_counts = np.zeros(len(CONTINUOUS_CONTRIBUTION_BIN_CLASSES), dtype=int)
    continuous_cluster_bin_counts: Dict[str, np.ndarray] = {}
    action_rows_count = 0
    action_none_count = 0
    action_punish_count = 0
    action_reward_count = 0

    for batch in _iter_training_batches(
        learn_cluster_weights_path=learn_cluster_weights_path,
        learn_analysis_csv=learn_analysis_csv,
        learn_rounds_csv=learn_rounds_csv,
        batch_size=512,
        use_cluster=bool(use_cluster),
    ):
        for row in batch["contribution_rows"]:
            contribution_feature_names.update(row.keys())
        for row in batch["action_rows"]:
            action_feature_names.update(row.keys())

        contrib_rows_count += int(len(batch["contribution_targets"]))
        contrib_aon_count += int(np.sum(np.asarray(batch["contribution_aon"], dtype=int) == 1))
        contrib_cont_count += int(np.sum(np.asarray(batch["contribution_aon"], dtype=int) == 0))
        contrib_targets = np.asarray(batch["contribution_targets"], dtype=float)
        contrib_cluster_ids = [str(cluster_id) for cluster_id in batch["contribution_cluster_ids"]]
        contrib_target_sum += float(np.sum(contrib_targets))
        aon_mask = np.asarray(batch["contribution_aon"], dtype=int) == 1
        cont_mask = ~aon_mask
        if bool(use_cluster) and int(np.sum(aon_mask)) > 0:
            aon_targets = (contrib_targets[aon_mask] >= 0.5).astype(int)
            aon_positive_count += int(np.sum(aon_targets))
            for cluster_id, label in zip(np.asarray(contrib_cluster_ids, dtype=object)[aon_mask], aon_targets):
                key = _cluster_key(str(cluster_id))
                aon_cluster_total_counts[key] = int(aon_cluster_total_counts.get(key, 0)) + 1
                aon_cluster_positive_counts[key] = int(aon_cluster_positive_counts.get(key, 0)) + int(label)
        elif int(np.sum(aon_mask)) > 0:
            aon_positive_count += int(np.sum((contrib_targets[aon_mask] >= 0.5).astype(int)))
        if int(np.sum(cont_mask)) > 0:
            cont_labels = np.asarray(
                [_continuous_rate_to_bin_label(value) for value in contrib_targets[cont_mask]],
                dtype=int,
            )
            continuous_bin_counts += np.bincount(
                cont_labels,
                minlength=len(CONTINUOUS_CONTRIBUTION_BIN_CLASSES),
            )
            if bool(use_cluster):
                for cluster_id, label in zip(np.asarray(contrib_cluster_ids, dtype=object)[cont_mask], cont_labels):
                    key = _cluster_key(str(cluster_id))
                    if key not in continuous_cluster_bin_counts:
                        continuous_cluster_bin_counts[key] = np.zeros(
                            len(CONTINUOUS_CONTRIBUTION_BIN_CLASSES),
                            dtype=float,
                        )
                    continuous_cluster_bin_counts[key][int(label)] += 1.0

        labels = np.asarray(batch["action_labels"], dtype=int)
        action_rows_count += int(len(labels))
        action_none_count += int(np.sum(labels == ACTION_LABEL_NONE))
        action_punish_count += int(np.sum(labels == ACTION_LABEL_PUNISH))
        action_reward_count += int(np.sum(labels == ACTION_LABEL_REWARD))

    if contrib_rows_count == 0:
        raise ValueError("Structured exact-sequence contribution training data is empty.")
    if action_rows_count == 0:
        raise ValueError("Structured exact-sequence action training data is empty.")

    contribution_feature_columns = sorted(contribution_feature_names)
    action_feature_columns = sorted(action_feature_names)

    aon_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=0,
        average=True,
    )
    continuous_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=0,
        average=True,
    )
    action_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=0,
        average=True,
    )

    have_aon = False
    have_continuous = False
    have_action_model = False

    for batch in _iter_training_batches(
        learn_cluster_weights_path=learn_cluster_weights_path,
        learn_analysis_csv=learn_analysis_csv,
        learn_rounds_csv=learn_rounds_csv,
        batch_size=512,
        use_cluster=bool(use_cluster),
    ):
        if batch["contribution_rows"]:
            contrib_x = _rows_to_numeric_frame(
                batch["contribution_rows"],
                contribution_feature_columns,
            )
            contrib_targets = np.asarray(batch["contribution_targets"], dtype=float)
            aon_mask = np.asarray(batch["contribution_aon"], dtype=int) == 1
            cont_mask = ~aon_mask

            if int(np.sum(aon_mask)) > 0:
                aon_y = (contrib_targets[aon_mask] >= 0.5).astype(int)
                if not have_aon:
                    aon_model.partial_fit(contrib_x.loc[aon_mask], aon_y, classes=np.array([0, 1], dtype=int))
                    have_aon = True
                else:
                    aon_model.partial_fit(contrib_x.loc[aon_mask], aon_y)
            if int(np.sum(cont_mask)) > 0:
                cont_y = np.asarray(
                    [_continuous_rate_to_bin_label(value) for value in contrib_targets[cont_mask]],
                    dtype=int,
                )
                if not have_continuous:
                    continuous_model.partial_fit(
                        contrib_x.loc[cont_mask],
                        cont_y,
                        classes=CONTINUOUS_CONTRIBUTION_BIN_CLASSES,
                    )
                    have_continuous = True
                else:
                    continuous_model.partial_fit(contrib_x.loc[cont_mask], cont_y)

        if batch["action_rows"]:
            action_x = _rows_to_numeric_frame(
                batch["action_rows"],
                action_feature_columns,
            )
            action_y = np.asarray(batch["action_labels"], dtype=int)
            if not have_action_model:
                action_model.partial_fit(action_x, action_y, classes=ACTION_CLASSES)
                have_action_model = True
            else:
                action_model.partial_fit(action_x, action_y)

    if have_aon:
        aon_model_out: object = aon_model
    else:
        aon_model_out = ConstantBinaryClassifier(positive_probability=0.5)

    if have_continuous:
        continuous_model_out: object = continuous_model
    else:
        default_bin = _continuous_rate_to_bin_label(float(contrib_target_sum / max(contrib_rows_count, 1)))
        default_probs = np.zeros(len(CONTINUOUS_CONTRIBUTION_BIN_CLASSES), dtype=float)
        default_probs[int(default_bin)] = 1.0
        continuous_model_out = ConstantMulticlassClassifier(
            class_probabilities=default_probs.tolist(),
            classes=CONTINUOUS_CONTRIBUTION_BIN_CLASSES,
        )

    class_probabilities = [
        float(action_none_count / max(action_rows_count, 1)),
        float(action_punish_count / max(action_rows_count, 1)),
        float(action_reward_count / max(action_rows_count, 1)),
    ]
    if have_action_model:
        action_model_out: object = action_model
    else:
        action_model_out = ConstantMulticlassClassifier(
            class_probabilities=class_probabilities,
            classes=ACTION_CLASSES,
        )

    continuous_bin_rates = [
        float(count / max(int(contrib_cont_count), 1))
        for count in continuous_bin_counts.tolist()
    ]
    aon_global_rate = float((float(aon_positive_count) + 1.0) / (float(contrib_aon_count) + 2.0))
    aon_cluster_rates = {
        key: _shrunk_binary_rate(
            positive_count=int(aon_cluster_positive_counts.get(key, 0)),
            total_count=int(aon_cluster_total_counts.get(key, 0)),
            global_rate=aon_global_rate,
            prior_weight=DEFAULT_CLUSTER_CALIBRATION_PRIOR_WEIGHT,
        )
        for key in sorted(aon_cluster_total_counts)
    } if bool(use_cluster) else {}
    continuous_global_probs = _normalize_probability_vector(continuous_bin_counts.astype(float) + 1.0)
    continuous_cluster_probs = {
        key: _shrunk_multiclass_probs(
            counts=counts,
            global_probs=continuous_global_probs,
            prior_weight=DEFAULT_CLUSTER_CALIBRATION_PRIOR_WEIGHT,
        ).tolist()
        for key, counts in sorted(continuous_cluster_bin_counts.items())
    } if bool(use_cluster) else {}

    payload = {
        "version": 8,
        "use_cluster": bool(use_cluster),
        "contribution_feature_columns": contribution_feature_columns,
        "action_feature_columns": action_feature_columns,
        "aon_contribution_model": aon_model_out,
        "aon_global_positive_rate": aon_global_rate,
        "aon_cluster_positive_rate": aon_cluster_rates,
        "continuous_contribution_model": continuous_model_out,
        "continuous_contribution_mode": "binned_classifier",
        "continuous_contribution_bin_rates": CONTINUOUS_CONTRIBUTION_BIN_RATES.tolist(),
        "continuous_global_bin_probs": continuous_global_probs.tolist(),
        "continuous_cluster_bin_probs": continuous_cluster_probs,
        "continuous_contribution_residuals": {"global": [0.0]},
        "cluster_calibration_strength": (
            float(DEFAULT_CLUSTER_CALIBRATION_STRENGTH) if bool(use_cluster) else 0.0
        ),
        "cluster_calibration_prior_weight": (
            float(DEFAULT_CLUSTER_CALIBRATION_PRIOR_WEIGHT) if bool(use_cluster) else 0.0
        ),
        "action_model": action_model_out,
        "learn_cluster_weights_path": str(learn_cluster_weights_path),
        "learn_analysis_csv": str(learn_analysis_csv),
        "learn_rounds_csv": str(learn_rounds_csv),
    }
    _atomic_joblib_dump(payload, output_model_path)

    summary_df = pd.DataFrame(
        [
            {
                "dataset": "contribution",
                "n_rows": int(contrib_rows_count),
                "n_all_or_nothing_rows": int(contrib_aon_count),
                "n_continuous_rows": int(contrib_cont_count),
                "mean_target_contribution_rate": float(contrib_target_sum / max(contrib_rows_count, 1)),
                "continuous_bin_rate_0": float(continuous_bin_rates[0]),
                "continuous_bin_rate_5": float(continuous_bin_rates[1]),
                "continuous_bin_rate_10": float(continuous_bin_rates[2]),
                "continuous_bin_rate_15": float(continuous_bin_rates[3]),
                "continuous_bin_rate_20": float(continuous_bin_rates[4]),
                "aon_global_positive_rate": aon_global_rate,
                "cluster_calibration_strength": (
                    float(DEFAULT_CLUSTER_CALIBRATION_STRENGTH) if bool(use_cluster) else 0.0
                ),
                "cluster_calibration_prior_weight": (
                    float(DEFAULT_CLUSTER_CALIBRATION_PRIOR_WEIGHT) if bool(use_cluster) else 0.0
                ),
                "use_cluster": 1.0 if bool(use_cluster) else 0.0,
            },
            {
                "dataset": "action_edge",
                "n_rows": int(action_rows_count),
                "none_rate": class_probabilities[0],
                "punish_rate": class_probabilities[1],
                "reward_rate": class_probabilities[2],
                "fixed_action_units": 1.0,
            },
        ]
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    return payload


def _available_action_budget_coins(
    *,
    env: Mapping[str, Any],
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
) -> float:
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    own_contrib = float(contributions_by_player.get(str(focal_player_id), 0.0))
    share = _round_redistributed_share(env, contributions_by_player)
    return float(max(endowment - own_contrib + share, 0.0))


def _rows_to_numeric_frame(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(feature_columns), dtype=np.float32)
    frame = pd.DataFrame.from_records(rows)
    frame = frame.reindex(columns=list(feature_columns), fill_value=0.0)
    return frame.fillna(0.0).astype(np.float32, copy=False)


def _load_oracle_treatment_cluster_distributions(
    *,
    game_cluster_distribution_path: Path,
    val_player_game_table_path: Path,
) -> tuple[Dict[str, List[float]], List[float]]:
    game_df = pd.read_parquet(game_cluster_distribution_path).copy()
    val_clean_df = pd.read_parquet(val_player_game_table_path).copy()
    treatment_lookup = (
        val_clean_df[["game_id", "CONFIG_treatmentName"]]
        .drop_duplicates(subset=["game_id"], keep="first")
        .rename(columns={"game_id": "game_id", "CONFIG_treatmentName": "CONFIG_treatmentName"})
    )
    merged = game_df.merge(
        treatment_lookup,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    prob_columns = [
        column
        for column in merged.columns
        if column.startswith("cluster_") and column.endswith("_prob")
    ]
    if not prob_columns:
        raise ValueError("No cluster probability columns found in oracle treatment distribution source.")
    grouped = (
        merged.groupby("CONFIG_treatmentName", as_index=False)[prob_columns]
        .mean()
        .sort_values("CONFIG_treatmentName")
        .reset_index(drop=True)
    )
    lookup = {
        str(row["CONFIG_treatmentName"]): _normalize_probability_vector(
            [row[column] for column in prob_columns]
        ).tolist()
        for _, row in grouped.iterrows()
    }
    global_distribution = _normalize_probability_vector(merged[prob_columns].mean(axis=0).tolist()).tolist()
    return lookup, global_distribution


def _masked_action_probabilities(
    probabilities: np.ndarray,
    *,
    punish_enabled: bool,
    reward_enabled: bool,
) -> np.ndarray:
    masked = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    if masked.ndim != 2 or masked.shape[1] != len(ACTION_CLASSES):
        raise ValueError("Expected action probabilities with shape (n_rows, 3).")
    if not punish_enabled:
        masked[:, ACTION_LABEL_PUNISH] = 0.0
    if not reward_enabled:
        masked[:, ACTION_LABEL_REWARD] = 0.0
    row_sums = masked.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    if np.any(zero_rows):
        masked[zero_rows, ACTION_LABEL_NONE] = 1.0
        row_sums = masked.sum(axis=1, keepdims=True)
    return masked / np.where(row_sums > 0, row_sums, 1.0)


class ExactSequenceArchetypePolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor | None,
        model_payload: Dict[str, Any],
        cluster_source: str = "env_model",
        oracle_treatment_distributions: Dict[str, List[float]] | None = None,
        oracle_global_distribution: List[float] | None = None,
    ) -> None:
        self.env_model = env_model
        self.model_payload = model_payload
        self.cluster_source = str(cluster_source or "env_model")
        self.oracle_treatment_distributions = oracle_treatment_distributions or {}
        self.oracle_global_distribution = oracle_global_distribution or []

    @classmethod
    def from_config(
        cls,
        config: ExactSequenceArchetypePolicyConfig,
    ) -> "ExactSequenceArchetypePolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        use_cluster = bool(config.use_cluster)
        cluster_source = str(config.cluster_source or "env_model")
        env_model_path = artifacts_root / "models" / "dirichlet_env_model.pkl"
        model_path = (
            Path(config.model_path).resolve()
            if config.model_path
            else (artifacts_root / _default_exact_sequence_model_path(use_cluster=use_cluster).name)
        )
        if use_cluster and cluster_source == "env_model" and not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        rebuild_model = bool(config.rebuild_model) or not model_path.exists()
        if not rebuild_model and model_path.exists():
            try:
                existing_payload = joblib.load(model_path)
            except Exception:
                rebuild_model = True
            else:
                payload_version = int(existing_payload.get("version", 0) or 0)
                if (
                    payload_version < 8
                    or "contribution_feature_columns" not in existing_payload
                    or "action_feature_columns" not in existing_payload
                    or "action_model" not in existing_payload
                    or existing_payload.get("continuous_contribution_mode") != "binned_classifier"
                    or "aon_global_positive_rate" not in existing_payload
                    or "aon_cluster_positive_rate" not in existing_payload
                    or "continuous_global_bin_probs" not in existing_payload
                    or "continuous_cluster_bin_probs" not in existing_payload
                    or bool(existing_payload.get("use_cluster", True)) != use_cluster
                ):
                    rebuild_model = True
        if rebuild_model:
            train_exact_sequence_policy(
                output_model_path=model_path,
                summary_output_path=artifacts_root / _default_exact_sequence_summary_path(use_cluster=use_cluster).name,
                use_cluster=use_cluster,
            )
        env_model = (
            DirichletEnvRegressor.load(env_model_path)
            if use_cluster and cluster_source == "env_model"
            else None
        )
        model_payload = joblib.load(model_path)
        oracle_treatment_distributions: Dict[str, List[float]] | None = None
        oracle_global_distribution: List[float] | None = None
        if use_cluster and cluster_source == "val_treatment_oracle":
            oracle_treatment_distributions, oracle_global_distribution = (
                _load_oracle_treatment_cluster_distributions(
                    game_cluster_distribution_path=artifacts_root
                    / "outputs"
                    / DEFAULT_VAL_GAME_CLUSTER_DISTRIBUTION_PATH.name,
                    val_player_game_table_path=artifacts_root
                    / "intermediate"
                    / DEFAULT_VAL_PLAYER_GAME_TABLE_CLEAN_PATH.name,
                )
            )
        return cls(
            env_model=env_model,
            model_payload=model_payload,
            cluster_source=cluster_source,
            oracle_treatment_distributions=oracle_treatment_distributions,
            oracle_global_distribution=oracle_global_distribution,
        )

    def uses_cluster(self) -> bool:
        return bool(self.model_payload.get("use_cluster", True))

    def predict_cluster_distribution(self, env: Mapping[str, Any]) -> List[float]:
        if not self.uses_cluster():
            raise RuntimeError("Cluster distribution requested for a no-cluster exact-sequence policy.")
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

    def _player_cluster_id(
        self,
        game_state: ExactSequenceGameState,
        player_id: str,
    ) -> int | None:
        if not self.uses_cluster():
            return None
        return int(game_state.cluster_by_player[str(player_id)])

    def _cluster_calibration_strength(self) -> float:
        return float(max(0.0, self.model_payload.get("cluster_calibration_strength", 0.0) or 0.0))

    def _calibrated_aon_probability(self, base_probability: float, cluster_id: int) -> float:
        clipped = float(max(1e-6, min(1.0 - 1e-6, float(base_probability))))
        strength = self._cluster_calibration_strength()
        if strength <= 0.0 or not self.uses_cluster():
            return clipped
        global_rate = float(self.model_payload.get("aon_global_positive_rate", 0.5) or 0.5)
        cluster_rates = self.model_payload.get("aon_cluster_positive_rate", {}) or {}
        cluster_rate = float(cluster_rates.get(_cluster_key(cluster_id), global_rate))
        shift = _safe_logit(cluster_rate) - _safe_logit(global_rate)
        return _safe_sigmoid(_safe_logit(clipped) + strength * shift)

    def _calibrated_continuous_probabilities(
        self,
        base_probabilities: Sequence[float],
        cluster_id: int,
    ) -> np.ndarray:
        base = _normalize_probability_vector(base_probabilities)
        strength = self._cluster_calibration_strength()
        if strength <= 0.0 or not self.uses_cluster():
            return base
        global_probs = _normalize_probability_vector(
            self.model_payload.get("continuous_global_bin_probs", base)
        )
        cluster_prob_map = self.model_payload.get("continuous_cluster_bin_probs", {}) or {}
        raw_cluster = cluster_prob_map.get(_cluster_key(cluster_id), global_probs)
        cluster_probs = _normalize_probability_vector(raw_cluster)
        if global_probs.shape != base.shape or cluster_probs.shape != base.shape:
            return base
        ratio = np.power(
            np.clip(cluster_probs, 1e-6, None) / np.clip(global_probs, 1e-6, None),
            strength,
        )
        return _normalize_probability_vector(base * ratio)

    def start_game(
        self,
        *,
        env: Mapping[str, Any],
        player_ids: Sequence[str],
        avatar_by_player: Mapping[str, str],
        rng: np.random.Generator | np.random.RandomState | Any,
    ) -> ExactSequenceGameState:
        sampled_map: Dict[str, int] = {}
        if self.uses_cluster():
            distribution = self.predict_cluster_distribution(env)
            cluster_ids = np.arange(1, len(distribution) + 1, dtype=int)
            sampled = list(rng.choice(cluster_ids, size=len(player_ids), p=distribution))
            sampled_map = {
                str(player_id): int(cluster_id)
                for player_id, cluster_id in zip(player_ids, sampled)
            }
        return ExactSequenceGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player={str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids},
            player_by_avatar={str(avatar): str(player_id) for player_id, avatar in avatar_by_player.items()},
            cluster_by_player=sampled_map,
            history_rounds=[],
        )

    def sample_contributions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        rows = [
            _contribution_feature_row(
                env=game_state.env,
                cluster_id=self._player_cluster_id(game_state, player_id),
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                history_rounds=game_state.history_rounds,
                round_idx=int(round_idx),
            )
            for player_id in game_state.player_ids
        ]
        features = _rows_to_numeric_frame(
            rows,
            self.model_payload["contribution_feature_columns"],
        )
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        out: Dict[str, int] = {}
        if as_bool(game_state.env.get("CONFIG_allOrNothing", False)):
            probs = _predict_probability(self.model_payload["aon_contribution_model"], features)
            for player_id, prob in zip(game_state.player_ids, probs):
                cluster_id = self._player_cluster_id(game_state, player_id)
                calibrated_prob = (
                    self._calibrated_aon_probability(float(prob), int(cluster_id))
                    if cluster_id is not None
                    else float(prob)
                )
                out[player_id] = endowment if float(rng.random()) < calibrated_prob else 0
            return out

        if self.model_payload.get("continuous_contribution_mode") == "binned_classifier":
            bin_rates = np.asarray(
                self.model_payload.get(
                    "continuous_contribution_bin_rates",
                    CONTINUOUS_CONTRIBUTION_BIN_RATES.tolist(),
                ),
                dtype=float,
            )
            bin_classes = np.arange(len(bin_rates), dtype=int)
            probabilities = _predict_multiclass_probabilities(
                self.model_payload["continuous_contribution_model"],
                features,
                bin_classes,
            )
            for player_id, row_probs in zip(game_state.player_ids, probabilities):
                cluster_id = self._player_cluster_id(game_state, player_id)
                probs = (
                    self._calibrated_continuous_probabilities(row_probs, int(cluster_id))
                    if cluster_id is not None
                    else _normalize_probability_vector(row_probs)
                )
                sampled_bin = int(rng.choice(bin_classes, p=probs))
                sampled_rate = float(bin_rates[sampled_bin])
                out[player_id] = int(max(0, min(endowment, round(sampled_rate * endowment))))
            return out

        predictions = np.asarray(
            self.model_payload["continuous_contribution_model"].predict(features),
            dtype=float,
        )
        for player_id, predicted_rate in zip(game_state.player_ids, predictions):
            cluster_id = self._player_cluster_id(game_state, player_id)
            residual = _sample_residual(
                rng,
                self.model_payload["continuous_contribution_residuals"],
                0 if cluster_id is None else int(cluster_id),
            )
            sampled_rate = float(max(0.0, min(1.0, float(predicted_rate) + residual)))
            out[player_id] = int(max(0, min(endowment, round(sampled_rate * endowment))))
        return out

    def _sample_joint_actions_for_player(
        self,
        *,
        game_state: ExactSequenceGameState,
        player_id: str,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        peer_ids = _relative_peer_ids(game_state.player_ids, player_id)
        if not peer_ids:
            return {}, {}

        cluster_id = self._player_cluster_id(game_state, player_id)
        base_history_row = _contribution_feature_row(
            env=game_state.env,
            cluster_id=cluster_id,
            player_ids=game_state.player_ids,
            focal_player_id=player_id,
            history_rounds=game_state.history_rounds,
            round_idx=int(round_idx),
        )
        rows = [
            _action_edge_feature_row(
                env=game_state.env,
                cluster_id=cluster_id,
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                target_player_id=target_player_id,
                history_rounds=game_state.history_rounds,
                round_idx=int(round_idx),
                current_contributions_by_player=contributions_by_player,
                base_history_row=base_history_row,
            )
            for target_player_id in peer_ids
        ]
        features = _rows_to_numeric_frame(
            rows,
            self.model_payload["action_feature_columns"],
        )
        punish_enabled = as_bool(game_state.env.get("CONFIG_punishmentExists", False))
        reward_enabled = as_bool(game_state.env.get("CONFIG_rewardExists", False))
        probabilities = _masked_action_probabilities(
            _predict_action_probabilities(self.model_payload["action_model"], features),
            punish_enabled=punish_enabled,
            reward_enabled=reward_enabled,
        )
        available_coins = _available_action_budget_coins(
            env=game_state.env,
            focal_player_id=player_id,
            contributions_by_player=contributions_by_player,
        )
        punish_cost = float(game_state.env.get("CONFIG_punishmentCost", 1) or 1)
        reward_cost = float(game_state.env.get("CONFIG_rewardCost", 1) or 1)

        sampled_actions: List[Tuple[str, int, float, float]] = []
        total_cost = 0.0
        for row_index, target_player_id in enumerate(peer_ids):
            label = int(rng.choice(ACTION_CLASSES, p=probabilities[row_index]))
            if label == ACTION_LABEL_NONE:
                continue
            action_cost = punish_cost if label == ACTION_LABEL_PUNISH else reward_cost
            confidence = float(probabilities[row_index, label])
            sampled_actions.append((str(target_player_id), label, action_cost, confidence))
            total_cost += float(action_cost)

        if total_cost > float(available_coins) + 1e-9:
            sampled_actions.sort(key=lambda item: item[3])
            kept_actions: List[Tuple[str, int, float, float]] = []
            running_cost = float(total_cost)
            for action in sampled_actions:
                if running_cost <= float(available_coins) + 1e-9:
                    kept_actions.append(action)
                    continue
                running_cost -= float(action[2])
            if running_cost > float(available_coins) + 1e-9:
                kept_actions = []
                running_cost = 0.0
                for action in sorted(sampled_actions, key=lambda item: item[3], reverse=True):
                    next_cost = running_cost + float(action[2])
                    if next_cost <= float(available_coins) + 1e-9:
                        kept_actions.append(action)
                        running_cost = next_cost
            sampled_actions = kept_actions

        punish_allocations: Dict[str, int] = {}
        reward_allocations: Dict[str, int] = {}
        for target_player_id, label, _, _ in sampled_actions:
            if int(label) == ACTION_LABEL_PUNISH:
                punish_allocations[str(target_player_id)] = 1
            elif int(label) == ACTION_LABEL_REWARD:
                reward_allocations[str(target_player_id)] = 1
        return punish_allocations, reward_allocations

    def sample_actions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}

        for player_id in game_state.player_ids:
            punish_allocations, reward_allocations = self._sample_joint_actions_for_player(
                game_state=game_state,
                player_id=player_id,
                contributions_by_player=contributions_by_player,
                round_idx=int(round_idx),
                rng=rng,
            )
            punish_out[player_id] = punish_allocations
            reward_out[player_id] = reward_allocations
        return {"punish": punish_out, "reward": reward_out}

    def record_round(
        self,
        *,
        game_state: ExactSequenceGameState,
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
        game_state: ExactSequenceGameState,
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
