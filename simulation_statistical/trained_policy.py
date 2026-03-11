from __future__ import annotations

import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

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
DEFAULT_CLUSTER_BEHAVIOR_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "cluster_behavior_model.pkl"
DEFAULT_VAL_GAME_CLUSTER_DISTRIBUTION_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "game_cluster_distribution_val.parquet"
)
DEFAULT_VAL_PLAYER_GAME_TABLE_CLEAN_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "player_game_table_val_clean.parquet"
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


def _round_phase(round_index: int, num_rounds: int) -> str:
    total = max(int(num_rounds), 1)
    progress = float(max(int(round_index) - 1, 0)) / float(max(total - 1, 1))
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "mid"
    return "late"


def _contribution_keys(cluster_id: int, phase: str, all_or_nothing: bool) -> list[tuple]:
    aon = int(bool(all_or_nothing))
    return [
        ("cluster_phase_aon", cluster_id, phase, aon),
        ("cluster_aon", cluster_id, aon),
        ("cluster_phase", cluster_id, phase),
        ("cluster", cluster_id),
        ("global_aon", aon),
        ("global",),
    ]


def _action_keys(cluster_id: int, phase: str) -> list[tuple]:
    return [
        ("cluster_phase", cluster_id, phase),
        ("cluster", cluster_id),
        ("global",),
    ]


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


def _safe_choice(rng: random.Random, values: Sequence[Any], default: Any) -> Any:
    if not values:
        return default
    return values[rng.randrange(len(values))]


def _normalized_rank(target_value: float, peer_values: Sequence[float]) -> float:
    if not peer_values:
        return 0.5
    arr = np.asarray(peer_values, dtype=float)
    less = float(np.sum(arr < target_value))
    equal = float(np.sum(arr == target_value))
    return float((less + 0.5 * equal) / max(len(arr), 1))


def _distribute_units(units_total: int, target_count: int) -> list[int]:
    units_total = max(int(units_total), max(int(target_count), 1))
    target_count = max(int(target_count), 1)
    allocations = [0] * target_count
    for index in range(units_total):
        allocations[index % target_count] += 1
    return allocations


def _load_cluster_behavior_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def _normalize_probability_vector(values: Sequence[float]) -> list[float]:
    probs = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("Expected a non-empty 1D probability vector.")
    total = float(probs.sum())
    if total <= 0.0:
        return (np.full(probs.size, 1.0 / probs.size, dtype=float)).tolist()
    return (probs / total).tolist()


def _load_oracle_treatment_cluster_distributions(
    *,
    artifacts_root: Path,
) -> tuple[Dict[str, list[float]], list[float]]:
    game_df = pd.read_parquet(artifacts_root / "outputs" / DEFAULT_VAL_GAME_CLUSTER_DISTRIBUTION_PATH.name).copy()
    val_clean_df = pd.read_parquet(
        artifacts_root / "intermediate" / DEFAULT_VAL_PLAYER_GAME_TABLE_CLEAN_PATH.name
    ).copy()
    treatment_lookup = (
        val_clean_df[["game_id", "CONFIG_treatmentName"]]
        .drop_duplicates(subset=["game_id"], keep="first")
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
        )
        for _, row in grouped.iterrows()
    }
    global_distribution = _normalize_probability_vector(merged[prob_columns].mean(axis=0).tolist())
    return lookup, global_distribution


def build_cluster_behavior_model(
    *,
    output_path: Path = DEFAULT_CLUSTER_BEHAVIOR_MODEL_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
) -> dict[str, Any]:
    cluster_weights = pd.read_parquet(learn_cluster_weights_path).copy()
    prob_columns = [column for column in cluster_weights.columns if column.startswith("cluster_") and column.endswith("_prob")]
    if not prob_columns:
        raise ValueError(f"No cluster probability columns found in {learn_cluster_weights_path}")
    cluster_weights["hard_cluster_id"] = cluster_weights[prob_columns].to_numpy().argmax(axis=1) + 1

    rounds = pd.read_csv(learn_rounds_csv)
    rounds = _build_round_index(rounds)
    analysis = pd.read_csv(learn_analysis_csv)
    keep_env_cols = [
        "gameId",
        "CONFIG_numRounds",
        "CONFIG_endowment",
        "CONFIG_allOrNothing",
        "CONFIG_punishmentExists",
        "CONFIG_rewardExists",
    ]
    for column in keep_env_cols:
        if column not in analysis.columns:
            raise ValueError(f"Missing {column} in {learn_analysis_csv}")

    merged = (
        rounds.merge(
            cluster_weights.rename(columns={"game_id": "gameId", "player_id": "playerId"})[
                ["gameId", "playerId", "hard_cluster_id"]
            ],
            on=["gameId", "playerId"],
            how="inner",
            validate="many_to_one",
        )
        .merge(
            analysis[keep_env_cols].drop_duplicates("gameId"),
            on="gameId",
            how="left",
            validate="many_to_one",
        )
    )
    if merged.empty:
        raise ValueError("No learning-wave player-round rows remained after joining cluster assignments.")

    merged["round_phase"] = merged.apply(
        lambda row: _round_phase(
            int(row.get("roundIndex", 1) or 1),
            int(row.get("CONFIG_numRounds", 1) or 1),
        ),
        axis=1,
    )
    merged["endowment"] = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(20.0).clip(lower=1.0)
    merged["all_or_nothing"] = merged["CONFIG_allOrNothing"].map(as_bool)
    merged["punishment_enabled"] = merged["CONFIG_punishmentExists"].map(as_bool)
    merged["reward_enabled"] = merged["CONFIG_rewardExists"].map(as_bool)
    merged["contribution_value"] = pd.to_numeric(merged["data.contribution"], errors="coerce").fillna(0.0)
    merged["contribution_prop"] = (merged["contribution_value"] / merged["endowment"]).clip(lower=0.0, upper=1.0)
    merged["punished_dict"] = merged["data.punished"].map(parse_dict)
    merged["rewarded_dict"] = merged["data.rewarded"].map(parse_dict)
    merged["punish_units_total"] = merged["punished_dict"].map(lambda value: int(sum(int(v) for v in value.values())))
    merged["reward_units_total"] = merged["rewarded_dict"].map(lambda value: int(sum(int(v) for v in value.values())))
    merged["punish_target_count"] = merged["punished_dict"].map(lambda value: int(sum(1 for v in value.values() if int(v) > 0)))
    merged["reward_target_count"] = merged["rewarded_dict"].map(lambda value: int(sum(1 for v in value.values() if int(v) > 0)))

    contribution_samples: Dict[tuple, list[float]] = {}
    punish_rate_counts: Dict[tuple, list[int]] = {}
    reward_rate_counts: Dict[tuple, list[int]] = {}
    punish_unit_samples: Dict[tuple, list[int]] = {}
    reward_unit_samples: Dict[tuple, list[int]] = {}
    punish_target_samples: Dict[tuple, list[int]] = {}
    reward_target_samples: Dict[tuple, list[int]] = {}
    punish_orientation_samples: Dict[tuple, list[float]] = {}
    reward_orientation_samples: Dict[tuple, list[float]] = {}

    round_lookup = {
        (str(game_id), int(round_index)): group.copy()
        for (game_id, round_index), group in merged.groupby(["gameId", "roundIndex"], sort=False)
    }

    for row in merged.itertuples(index=False):
        cluster_id = int(row.hard_cluster_id)
        phase = str(row.round_phase)
        contribution_keys = _contribution_keys(cluster_id, phase, bool(row.all_or_nothing))
        _add_sample(contribution_samples, contribution_keys, float(row.contribution_prop))

        action_keys = _action_keys(cluster_id, phase)
        if bool(row.punishment_enabled):
            punish_positive = int(row.punish_units_total) > 0
            _add_rate_count(punish_rate_counts, action_keys, punish_positive)
            if punish_positive:
                _add_sample(punish_unit_samples, action_keys, int(row.punish_units_total))
                _add_sample(punish_target_samples, action_keys, int(row.punish_target_count))
        if bool(row.reward_enabled):
            reward_positive = int(row.reward_units_total) > 0
            _add_rate_count(reward_rate_counts, action_keys, reward_positive)
            if reward_positive:
                _add_sample(reward_unit_samples, action_keys, int(row.reward_units_total))
                _add_sample(reward_target_samples, action_keys, int(row.reward_target_count))

    for row in merged.itertuples(index=False):
        round_group = round_lookup.get((str(row.gameId), int(row.roundIndex)))
        if round_group is None or round_group.empty:
            continue
        peer_rows = round_group[round_group["playerId"].astype(str) != str(row.playerId)]
        if peer_rows.empty:
            continue
        peer_contrib_by_pid = {
            str(peer_row["playerId"]): float(peer_row["contribution_value"])
            for _, peer_row in peer_rows.iterrows()
        }
        peer_values = list(peer_contrib_by_pid.values())
        action_keys = _action_keys(int(row.hard_cluster_id), str(row.round_phase))

        if int(row.punish_units_total) > 0:
            weighted_ranks: list[float] = []
            for target_pid, units in (row.punished_dict or {}).items():
                if str(target_pid) not in peer_contrib_by_pid or int(units) <= 0:
                    continue
                rank = _normalized_rank(peer_contrib_by_pid[str(target_pid)], peer_values)
                weighted_ranks.extend([rank] * int(units))
            if weighted_ranks:
                _add_sample(punish_orientation_samples, action_keys, float(np.mean(weighted_ranks)))

        if int(row.reward_units_total) > 0:
            weighted_ranks = []
            for target_pid, units in (row.rewarded_dict or {}).items():
                if str(target_pid) not in peer_contrib_by_pid or int(units) <= 0:
                    continue
                rank = _normalized_rank(peer_contrib_by_pid[str(target_pid)], peer_values)
                weighted_ranks.extend([rank] * int(units))
            if weighted_ranks:
                _add_sample(reward_orientation_samples, action_keys, float(np.mean(weighted_ranks)))

    behavior_model = {
        "version": 1,
        "n_clusters": int(len(prob_columns)),
        "cluster_ids": [index + 1 for index in range(len(prob_columns))],
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
    return behavior_model


@dataclass(frozen=True)
class ArchetypeClusterPolicyConfig:
    artifacts_root: str | None = None
    rebuild_behavior_model: bool = False
    cluster_source: str = "env_model"


class ArchetypeClusterPolicyRuntime:
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
        self.n_clusters = int(behavior_model["n_clusters"])
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
    def from_config(cls, config: ArchetypeClusterPolicyConfig) -> "ArchetypeClusterPolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        cluster_source = str(config.cluster_source or "env_model")
        env_model_path = artifacts_root / "models" / "dirichlet_env_model.pkl"
        behavior_model_path = artifacts_root / "models" / "cluster_behavior_model.pkl"
        if cluster_source == "env_model" and not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        if config.rebuild_behavior_model or not behavior_model_path.exists():
            build_cluster_behavior_model(output_path=behavior_model_path)
        env_model = DirichletEnvRegressor.load(env_model_path) if cluster_source == "env_model" else None
        behavior_model = _load_cluster_behavior_model(behavior_model_path)
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

    def assign_clusters_for_game(
        self,
        env: Mapping[str, Any],
        player_ids: Sequence[str],
        rng: random.Random,
    ) -> Dict[str, int]:
        distribution = self.predict_cluster_distribution(env)
        sampled = rng.choices(self.cluster_ids, weights=distribution, k=len(player_ids))
        return {player_id: int(cluster_id) for player_id, cluster_id in zip(player_ids, sampled)}

    def _lookup_samples(self, store: Dict[tuple, list], keys: Sequence[tuple]) -> list:
        for key in keys:
            values = store.get(key)
            if values:
                return values
        return []

    def _lookup_rate(self, store: Dict[tuple, list[int]], keys: Sequence[tuple], default: float) -> float:
        for key in keys:
            counts = store.get(key)
            if counts and counts[1] > 0:
                return float(counts[0]) / float(counts[1])
        return default

    def sample_contribution(
        self,
        *,
        cluster_id: int,
        env: Mapping[str, Any],
        round_idx: int,
        rng: random.Random,
    ) -> int:
        endowment = int(env.get("CONFIG_endowment", 20) or 20)
        all_or_nothing = as_bool(env.get("CONFIG_allOrNothing", False))
        phase = _round_phase(int(round_idx), int(env.get("CONFIG_numRounds", 1) or 1))
        sample_pool = self._lookup_samples(
            self.contribution_samples,
            _contribution_keys(int(cluster_id), phase, all_or_nothing),
        )
        sampled_prop = float(_safe_choice(rng, sample_pool, 0.5))
        sampled_prop = max(0.0, min(1.0, sampled_prop))
        if all_or_nothing:
            return int(endowment if sampled_prop >= 0.5 else 0)
        return int(max(0, min(endowment, round(sampled_prop * endowment))))

    def _sample_ranked_targets(
        self,
        *,
        target_rank: float,
        target_count: int,
        units_total: int,
        peer_contributions: Dict[str, int],
        rng: random.Random,
    ) -> Dict[str, int]:
        if not peer_contributions:
            return {}
        peers = list(peer_contributions.items())
        rng.shuffle(peers)
        peer_values = [float(value) for _, value in peers]
        ranked = sorted(
            peers,
            key=lambda item: abs(_normalized_rank(float(item[1]), peer_values) - target_rank),
        )
        chosen = ranked[: max(1, min(int(target_count), len(ranked)))]
        allocations = _distribute_units(units_total=units_total, target_count=len(chosen))
        return {player_avatar: int(units) for (player_avatar, _), units in zip(chosen, allocations) if int(units) > 0}

    def _sample_action_dict(
        self,
        *,
        mechanism: str,
        cluster_id: int,
        env: Mapping[str, Any],
        round_idx: int,
        peer_contributions: Dict[str, int],
        rng: random.Random,
    ) -> Dict[str, int]:
        phase = _round_phase(int(round_idx), int(env.get("CONFIG_numRounds", 1) or 1))
        action_keys = _action_keys(int(cluster_id), phase)
        if mechanism == "punish":
            rate = self._lookup_rate(self.punish_rate_counts, action_keys, default=0.0)
            unit_pool = self._lookup_samples(self.punish_unit_samples, action_keys)
            target_pool = self._lookup_samples(self.punish_target_samples, action_keys)
            orientation_pool = self._lookup_samples(self.punish_orientation_samples, action_keys)
            cost = float(env.get("CONFIG_punishmentCost", 1) or 1)
            default_rank = 0.0
        else:
            rate = self._lookup_rate(self.reward_rate_counts, action_keys, default=0.0)
            unit_pool = self._lookup_samples(self.reward_unit_samples, action_keys)
            target_pool = self._lookup_samples(self.reward_target_samples, action_keys)
            orientation_pool = self._lookup_samples(self.reward_orientation_samples, action_keys)
            cost = float(env.get("CONFIG_rewardCost", 1) or 1)
            default_rank = 1.0

        if rng.random() >= max(0.0, min(1.0, rate)):
            return {}

        target_count = int(max(1, _safe_choice(rng, target_pool, 1)))
        max_units = max(int(math.floor((float(env.get("CONFIG_endowment", 20) or 20)) / max(cost, 1.0))), 1)
        units_total = int(max(1, min(max_units, _safe_choice(rng, unit_pool, 1))))
        target_rank = float(_safe_choice(rng, orientation_pool, default_rank))
        target_rank = max(0.0, min(1.0, target_rank))
        return self._sample_ranked_targets(
            target_rank=target_rank,
            target_count=target_count,
            units_total=units_total,
            peer_contributions=peer_contributions,
            rng=rng,
        )

    def sample_game_actions(
        self,
        *,
        cluster_by_player: Mapping[str, int],
        env: Mapping[str, Any],
        avatar_by_player: Mapping[str, str],
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: random.Random,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        contributions_by_avatar = {
            avatar_by_player[player_id]: int(contributions_by_player[player_id])
            for player_id in contributions_by_player
        }
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}
        for player_id, cluster_id in cluster_by_player.items():
            focal_avatar = avatar_by_player[player_id]
            peer_contributions = {
                avatar: contribution
                for avatar, contribution in contributions_by_avatar.items()
                if avatar != focal_avatar
            }
            punish_out[focal_avatar] = (
                self._sample_action_dict(
                    mechanism="punish",
                    cluster_id=int(cluster_id),
                    env=env,
                    round_idx=round_idx,
                    peer_contributions=peer_contributions,
                    rng=rng,
                )
                if as_bool(env.get("CONFIG_punishmentExists", False))
                else {}
            )
            reward_out[focal_avatar] = (
                self._sample_action_dict(
                    mechanism="reward",
                    cluster_id=int(cluster_id),
                    env=env,
                    round_idx=round_idx,
                    peer_contributions=peer_contributions,
                    rng=rng,
                )
                if as_bool(env.get("CONFIG_rewardExists", False))
                else {}
            )
        return {"punish": punish_out, "reward": reward_out}
