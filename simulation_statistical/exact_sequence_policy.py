from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import MaxAbsScaler

from Macro_simulation_eval.prompt_builder import redist_line, round_open
from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import _build_round_index, as_bool, json_compact
from simulation_statistical.history_conditioned_policy import (
    ConstantBinaryClassifier,
    ConstantRegressor,
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
    _safe_prob,
    _sample_residual,
    _visible_env_features,
    _weighted_target_rank_mean,
)


DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "exact_sequence_policy.pkl"
DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "exact_sequence_contribution_train.parquet"
)
DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "intermediate" / "exact_sequence_action_train.parquet"
)
DEFAULT_EXACT_SEQUENCE_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "exact_sequence_policy_train_summary.csv"
)


@dataclass(frozen=True)
class ExactSequenceArchetypePolicyConfig:
    artifacts_root: str | None = None
    rebuild_model: bool = False


@dataclass
class ExactSequenceGameState:
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    cluster_by_player: Dict[str, int]
    visible_name_by_player: Dict[str, str]
    transcript_lines_by_player: Dict[str, List[str]] = field(default_factory=dict)


def _fit_binary_classifier(features: sparse.csr_matrix, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    positive_rate = float(clean_target.mean()) if len(clean_target) else 0.0
    if clean_target.nunique() < 2:
        return ConstantBinaryClassifier(positive_probability=positive_rate)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=3000,
        tol=1e-3,
        random_state=0,
        average=True,
    )
    model.fit(features, clean_target)
    return model


def _fit_regressor(features: sparse.csr_matrix, target: pd.Series) -> object:
    clean_target = pd.to_numeric(target, errors="coerce")
    if clean_target.notna().sum() == 0:
        return ConstantRegressor(value=0.0)
    keep = clean_target.notna().to_numpy()
    pair = features[keep]
    y = clean_target.loc[clean_target.notna()].astype(float)
    if pair.shape[0] < 2 or y.nunique() < 2:
        return ConstantRegressor(value=float(y.mean()))
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        max_iter=3000,
        tol=1e-3,
        random_state=0,
        average=True,
    )
    model.fit(pair, y)
    return model


def _visible_name_map(player_ids: Sequence[str]) -> Dict[str, str]:
    return {str(player_id): f"PLAYER_{index}" for index, player_id in enumerate(player_ids, start=1)}


def _initial_transcript_lines(visible_name: str) -> List[str]:
    return [
        "# GAME STARTS",
        f"Your avatar is {visible_name}.",
    ]


def _base_structured_features(env: Mapping[str, Any], cluster_id: int, round_idx: int) -> Dict[str, Any]:
    row = _visible_env_features(env, round_idx)
    row["cluster_id"] = f"cluster_{int(cluster_id)}"
    return row


def _action_structured_features(
    *,
    env: Mapping[str, Any],
    cluster_id: int,
    round_idx: int,
    focal_player_id: str,
    contributions_by_player: Mapping[str, int],
) -> Dict[str, Any]:
    row = _base_structured_features(env, cluster_id, round_idx)
    endowment = float(env.get("CONFIG_endowment", 20) or 20)
    own_value = float(contributions_by_player.get(str(focal_player_id), 0.0))
    peer_values = [
        float(value)
        for player_id, value in contributions_by_player.items()
        if str(player_id) != str(focal_player_id)
    ]
    group_values = [float(value) for value in contributions_by_player.values()]
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
            "current_own_rank": _normalized_rank(own_value, group_values),
        }
    )
    return row


def _peer_contributions_csv(
    *,
    player_ids: Sequence[str],
    focal_player_id: str,
    visible_name_by_player: Mapping[str, str],
    contributions_by_player: Mapping[str, int],
) -> str:
    peer_ids = [str(player_id) for player_id in player_ids if str(player_id) != str(focal_player_id)]
    parts = [
        f"{visible_name_by_player[player_id]}={int(contributions_by_player.get(player_id, 0))}"
        for player_id in peer_ids
    ]
    return ",".join(parts)


def _pre_action_round_lines(
    *,
    env: Mapping[str, Any],
    round_idx: int,
    player_ids: Sequence[str],
    focal_player_id: str,
    visible_name_by_player: Mapping[str, str],
    contributions_by_player: Mapping[str, int],
) -> List[str]:
    total_contrib = int(sum(int(value) for value in contributions_by_player.values()))
    try:
        multiplied = float(env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
    except Exception:
        multiplied = float("nan")
    own_contribution = int(contributions_by_player.get(str(focal_player_id), 0))
    peers_csv = _peer_contributions_csv(
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        visible_name_by_player=visible_name_by_player,
        contributions_by_player=contributions_by_player,
    )
    return [
        round_open(dict(env), int(round_idx)),
        f'<CONTRIB v="{own_contribution}"/>',
        redist_line(total_contrib, multiplied, len(player_ids)),
        f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>",
    ]


def _action_line(
    *,
    env: Mapping[str, Any],
    focal_player_id: str,
    visible_name_by_player: Mapping[str, str],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
) -> Optional[str]:
    reward_on = as_bool(env.get("CONFIG_rewardExists", False))
    punish_on = as_bool(env.get("CONFIG_punishmentExists", False))
    own_punish = {
        visible_name_by_player[str(target_player_id)]: int(units)
        for target_player_id, units in punish_by_player.get(str(focal_player_id), {}).items()
        if int(units) > 0
    }
    own_reward = {
        visible_name_by_player[str(target_player_id)]: int(units)
        for target_player_id, units in reward_by_player.get(str(focal_player_id), {}).items()
        if int(units) > 0
    }
    if reward_on and not punish_on:
        return f"<REWARD>{json_compact(own_reward)}</REWARD>"
    if punish_on and not reward_on:
        return f"<PUNISHMENT>{json_compact(own_punish)}</PUNISHMENT>"
    if reward_on or punish_on:
        return f'<ACTIONS punish="{json_compact(own_punish)}" reward="{json_compact(own_reward)}"/>'
    return None


def _round_summary_payload(
    *,
    env: Mapping[str, Any],
    player_ids: Sequence[str],
    focal_player_id: str,
    visible_name_by_player: Mapping[str, str],
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
) -> Dict[str, Any]:
    endowment = int(env.get("CONFIG_endowment", 0) or 0)
    punish_on = as_bool(env.get("CONFIG_punishmentExists", False))
    reward_on = as_bool(env.get("CONFIG_rewardExists", False))
    show_other = as_bool(env.get("CONFIG_showOtherSummaries", False))
    try:
        multiplied = float(env.get("CONFIG_multiplier", 0) or 0) * float(
            sum(int(value) for value in contributions_by_player.values())
        )
    except Exception:
        multiplied = 0.0
    share = float(multiplied) / max(len(player_ids), 1)

    inbound_reward_units = {
        str(player_id): int(
            sum(int(targets.get(str(player_id), 0)) for targets in reward_by_player.values())
        )
        for player_id in player_ids
    }
    inbound_punish_units = {
        str(player_id): int(
            sum(int(targets.get(str(player_id), 0)) for targets in punish_by_player.values())
        )
        for player_id in player_ids
    }

    def _player_summary(player_id: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if punish_on:
            out["coins_spent_on_punish"] = int(
                sum(int(value) for value in punish_by_player.get(str(player_id), {}).values())
                * int(env.get("CONFIG_punishmentCost", 0) or 0)
            )
            out["coins_deducted_from_you" if player_id == focal_player_id else "coins_deducted_from_them"] = int(
                inbound_punish_units[str(player_id)] * int(env.get("CONFIG_punishmentMagnitude", 0) or 0)
            )
        if reward_on:
            out["coins_spent_on_reward"] = int(
                sum(int(value) for value in reward_by_player.get(str(player_id), {}).values())
                * int(env.get("CONFIG_rewardCost", 0) or 0)
            )
            out["coins_rewarded_to_you" if player_id == focal_player_id else "coins_rewarded_to_them"] = int(
                inbound_reward_units[str(player_id)] * int(env.get("CONFIG_rewardMagnitude", 0) or 0)
            )
        private_kept = int(endowment - int(contributions_by_player.get(str(player_id), 0)))
        payoff_value = payoff_by_player.get(str(player_id), 0.0)
        try:
            payoff_float = float(payoff_value)
        except Exception:
            payoff_float = float("nan")
        if pd.isna(payoff_float):
            payoff_float = (
                float(private_kept)
                + float(share)
                - float(out.get("coins_spent_on_punish", 0))
                - float(out.get("coins_spent_on_reward", 0))
                - float(out.get("coins_deducted_from_you", out.get("coins_deducted_from_them", 0)))
                + float(out.get("coins_rewarded_to_you", out.get("coins_rewarded_to_them", 0)))
            )
        out["payoff"] = int(round(float(payoff_float)))
        return out

    summary = {
        f'{visible_name_by_player[str(focal_player_id)]} (YOU)': _player_summary(str(focal_player_id))
    }
    if show_other:
        for other_player_id in player_ids:
            other_player_id = str(other_player_id)
            if other_player_id == str(focal_player_id):
                continue
            summary[visible_name_by_player[other_player_id]] = _player_summary(other_player_id)
    return summary


def _completed_round_lines(
    *,
    env: Mapping[str, Any],
    round_idx: int,
    player_ids: Sequence[str],
    focal_player_id: str,
    visible_name_by_player: Mapping[str, str],
    contributions_by_player: Mapping[str, int],
    punish_by_player: Mapping[str, Mapping[str, int]],
    reward_by_player: Mapping[str, Mapping[str, int]],
    payoff_by_player: Mapping[str, float],
) -> List[str]:
    lines = _pre_action_round_lines(
        env=env,
        round_idx=round_idx,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        visible_name_by_player=visible_name_by_player,
        contributions_by_player=contributions_by_player,
    )
    action_line = _action_line(
        env=env,
        focal_player_id=focal_player_id,
        visible_name_by_player=visible_name_by_player,
        punish_by_player=punish_by_player,
        reward_by_player=reward_by_player,
    )
    if action_line:
        lines.append(action_line)
    if as_bool(env.get("CONFIG_showPunishmentId", False)) and as_bool(env.get("CONFIG_punishmentExists", False)):
        punishers = {
            visible_name_by_player[str(source_player_id)]: int(targets.get(str(focal_player_id), 0))
            for source_player_id, targets in punish_by_player.items()
            if int(targets.get(str(focal_player_id), 0)) > 0
        }
        lines.append(f"<PUNISHED_BY>{json_compact(punishers)}</PUNISHED_BY>")
    if as_bool(env.get("CONFIG_showRewardId", False)) and as_bool(env.get("CONFIG_rewardExists", False)):
        rewarders = {
            visible_name_by_player[str(source_player_id)]: int(targets.get(str(focal_player_id), 0))
            for source_player_id, targets in reward_by_player.items()
            if int(targets.get(str(focal_player_id), 0)) > 0
        }
        lines.append(f"<REWARDED_BY>{json_compact(rewarders)}</REWARDED_BY>")
    summary = _round_summary_payload(
        env=env,
        player_ids=player_ids,
        focal_player_id=focal_player_id,
        visible_name_by_player=visible_name_by_player,
        contributions_by_player=contributions_by_player,
        punish_by_player=punish_by_player,
        reward_by_player=reward_by_player,
        payoff_by_player=payoff_by_player,
    )
    lines.append(f"<ROUND_SUMMARY>{json_compact(summary)}</ROUND_SUMMARY>")
    return lines


def _sanitize_struct_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[str(key)] = value if value else "NA"
            continue
        if value is None:
            out[str(key)] = 0.0
            continue
        try:
            if pd.isna(value):
                out[str(key)] = 0.0
                continue
        except Exception:
            pass
        if isinstance(value, (bool, np.bool_)):
            out[str(key)] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = str(value)
    return out


def _fit_feature_stack(
    *,
    texts: Sequence[str],
    structured_rows: Sequence[Mapping[str, Any]],
) -> Tuple[TfidfVectorizer, DictVectorizer, MaxAbsScaler, sparse.csr_matrix]:
    text_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    struct_vectorizer = DictVectorizer(sparse=True)
    text_x = text_vectorizer.fit_transform([str(text or "") for text in texts])
    struct_x = struct_vectorizer.fit_transform([_sanitize_struct_row(row) for row in structured_rows])
    combined = sparse.hstack([text_x, struct_x], format="csr")
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(combined)
    return text_vectorizer, struct_vectorizer, scaler, scaled.tocsr()


def _transform_feature_stack(
    *,
    texts: Sequence[str],
    structured_rows: Sequence[Mapping[str, Any]],
    text_vectorizer: TfidfVectorizer,
    struct_vectorizer: DictVectorizer,
    scaler: MaxAbsScaler,
) -> sparse.csr_matrix:
    text_x = text_vectorizer.transform([str(text or "") for text in texts])
    struct_x = struct_vectorizer.transform([_sanitize_struct_row(row) for row in structured_rows])
    combined = sparse.hstack([text_x, struct_x], format="csr")
    return scaler.transform(combined).tocsr()


def build_exact_sequence_training_datasets(
    *,
    contribution_output_path: Path = DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH,
    action_output_path: Path = DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH,
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
        player_ids = list(
            dict.fromkeys(
                game_df.sort_values(["roundIndex", "playerId"])["playerId"].astype(str).tolist()
            )
        )
        if not player_ids:
            continue
        visible_name_by_player = _visible_name_map(player_ids)
        cluster_by_player = (
            game_df[["playerId", "hard_cluster_id"]]
            .drop_duplicates(subset=["playerId"], keep="first")
            .set_index("playerId")["hard_cluster_id"]
            .astype(int)
            .to_dict()
        )
        transcript_lines_by_player = {
            player_id: _initial_transcript_lines(visible_name_by_player[player_id])
            for player_id in player_ids
        }

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
                cluster_id = _hard_cluster_id(cluster_by_player.get(player_id, 1))
                contribution_rows.append(
                    {
                        "gameId": str(game_id),
                        "playerId": str(player_id),
                        "roundIndex": int(round_idx),
                        "history_text": "\n".join(
                            transcript_lines_by_player[player_id]
                            + [round_open(env, int(round_idx))]
                        ),
                        **_base_structured_features(env, cluster_id, int(round_idx)),
                        "target_contribution_units": int(contributions_by_player.get(player_id, 0)),
                        "target_contribution_rate": float(
                            int(contributions_by_player.get(player_id, 0))
                            / max(float(env.get("CONFIG_endowment", 20) or 20), 1.0)
                        ),
                    }
                )
                action_rows.append(
                    {
                        "gameId": str(game_id),
                        "playerId": str(player_id),
                        "roundIndex": int(round_idx),
                        "history_text": "\n".join(
                            transcript_lines_by_player[player_id]
                            + _pre_action_round_lines(
                                env=env,
                                round_idx=int(round_idx),
                                player_ids=player_ids,
                                focal_player_id=player_id,
                                visible_name_by_player=visible_name_by_player,
                                contributions_by_player=contributions_by_player,
                            )
                        ),
                        **_action_structured_features(
                            env=env,
                            cluster_id=cluster_id,
                            round_idx=int(round_idx),
                            focal_player_id=player_id,
                            contributions_by_player=contributions_by_player,
                        ),
                        "target_any_punish": int(
                            sum(int(value) for value in punish_by_player.get(player_id, {}).values()) > 0
                        ),
                        "target_any_reward": int(
                            sum(int(value) for value in reward_by_player.get(player_id, {}).values()) > 0
                        ),
                        "target_punish_units_total": int(
                            sum(int(value) for value in punish_by_player.get(player_id, {}).values())
                        ),
                        "target_reward_units_total": int(
                            sum(int(value) for value in reward_by_player.get(player_id, {}).values())
                        ),
                        "target_punish_target_count": int(
                            sum(1 for value in punish_by_player.get(player_id, {}).values() if int(value) > 0)
                        ),
                        "target_reward_target_count": int(
                            sum(1 for value in reward_by_player.get(player_id, {}).values() if int(value) > 0)
                        ),
                        "target_punish_target_rank": _weighted_target_rank_mean(
                            focal_player_id=player_id,
                            target_units_by_player=punish_by_player.get(player_id, {}),
                            contributions_by_player=contributions_by_player,
                        ),
                        "target_reward_target_rank": _weighted_target_rank_mean(
                            focal_player_id=player_id,
                            target_units_by_player=reward_by_player.get(player_id, {}),
                            contributions_by_player=contributions_by_player,
                        ),
                    }
                )

            for player_id in player_ids:
                transcript_lines_by_player[player_id].extend(
                    _completed_round_lines(
                        env=env,
                        round_idx=int(round_idx),
                        player_ids=player_ids,
                        focal_player_id=player_id,
                        visible_name_by_player=visible_name_by_player,
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


def train_exact_sequence_policy(
    *,
    output_model_path: Path = DEFAULT_EXACT_SEQUENCE_POLICY_MODEL_PATH,
    contribution_dataset_path: Path = DEFAULT_EXACT_SEQUENCE_CONTRIBUTION_DATASET_PATH,
    action_dataset_path: Path = DEFAULT_EXACT_SEQUENCE_ACTION_DATASET_PATH,
    summary_output_path: Path = DEFAULT_EXACT_SEQUENCE_TRAIN_SUMMARY_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
) -> Dict[str, Any]:
    contribution_df, action_df = build_exact_sequence_training_datasets(
        contribution_output_path=contribution_dataset_path,
        action_output_path=action_dataset_path,
        learn_cluster_weights_path=learn_cluster_weights_path,
        learn_analysis_csv=learn_analysis_csv,
        learn_rounds_csv=learn_rounds_csv,
    )
    if contribution_df.empty or action_df.empty:
        raise ValueError("Exact-sequence training datasets are empty.")

    contribution_struct_cols = [
        column
        for column in contribution_df.columns
        if column
        not in {
            "gameId",
            "playerId",
            "roundIndex",
            "history_text",
            "target_contribution_units",
            "target_contribution_rate",
        }
    ]
    action_struct_cols = [
        column
        for column in action_df.columns
        if column
        not in {
            "gameId",
            "playerId",
            "roundIndex",
            "history_text",
            "target_any_punish",
            "target_any_reward",
            "target_punish_units_total",
            "target_reward_units_total",
            "target_punish_target_count",
            "target_reward_target_count",
            "target_punish_target_rank",
            "target_reward_target_rank",
        }
    ]

    contribution_text_vectorizer, contribution_struct_vectorizer, contribution_scaler, contribution_x = _fit_feature_stack(
        texts=contribution_df["history_text"].fillna("").astype(str).tolist(),
        structured_rows=contribution_df[contribution_struct_cols].to_dict(orient="records"),
    )
    action_text_vectorizer, action_struct_vectorizer, action_scaler, action_x = _fit_feature_stack(
        texts=action_df["history_text"].fillna("").astype(str).tolist(),
        structured_rows=action_df[action_struct_cols].to_dict(orient="records"),
    )

    aon_mask = pd.to_numeric(contribution_df["CONFIG_allOrNothing"], errors="coerce").fillna(0).astype(int) == 1
    continuous_mask = ~aon_mask

    continuous_model = _fit_regressor(
        contribution_x[continuous_mask.to_numpy()],
        contribution_df.loc[continuous_mask, "target_contribution_rate"],
    )
    continuous_predictions = continuous_model.predict(contribution_x[continuous_mask.to_numpy()])
    continuous_residuals = _build_residual_store(
        actual=contribution_df.loc[continuous_mask, "target_contribution_rate"],
        predicted=np.asarray(continuous_predictions, dtype=float),
        cluster_ids=contribution_df.loc[continuous_mask, "cluster_id"],
    )
    aon_model = _fit_binary_classifier(
        contribution_x[aon_mask.to_numpy()],
        (pd.to_numeric(contribution_df.loc[aon_mask, "target_contribution_rate"], errors="coerce") >= 0.5).astype(int),
    )

    punish_enabled_mask = pd.to_numeric(action_df["CONFIG_punishmentExists"], errors="coerce").fillna(0).astype(int) == 1
    reward_enabled_mask = pd.to_numeric(action_df["CONFIG_rewardExists"], errors="coerce").fillna(0).astype(int) == 1

    punish_any_model = _fit_binary_classifier(
        action_x[punish_enabled_mask.to_numpy()],
        action_df.loc[punish_enabled_mask, "target_any_punish"],
    )
    reward_any_model = _fit_binary_classifier(
        action_x[reward_enabled_mask.to_numpy()],
        action_df.loc[reward_enabled_mask, "target_any_reward"],
    )

    punish_positive_mask = punish_enabled_mask & (
        pd.to_numeric(action_df["target_any_punish"], errors="coerce").fillna(0).astype(int) == 1
    )
    reward_positive_mask = reward_enabled_mask & (
        pd.to_numeric(action_df["target_any_reward"], errors="coerce").fillna(0).astype(int) == 1
    )

    punish_units_model = _fit_regressor(
        action_x[punish_positive_mask.to_numpy()],
        action_df.loc[punish_positive_mask, "target_punish_units_total"],
    )
    punish_units_predictions = punish_units_model.predict(action_x[punish_positive_mask.to_numpy()])
    punish_units_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_units_total"],
        predicted=np.asarray(punish_units_predictions, dtype=float),
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )
    punish_target_count_model = _fit_regressor(
        action_x[punish_positive_mask.to_numpy()],
        action_df.loc[punish_positive_mask, "target_punish_target_count"],
    )
    punish_target_count_predictions = punish_target_count_model.predict(action_x[punish_positive_mask.to_numpy()])
    punish_target_count_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_target_count"],
        predicted=np.asarray(punish_target_count_predictions, dtype=float),
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )
    punish_orientation_model = _fit_regressor(
        action_x[punish_positive_mask.to_numpy()],
        action_df.loc[punish_positive_mask, "target_punish_target_rank"],
    )
    punish_orientation_predictions = punish_orientation_model.predict(action_x[punish_positive_mask.to_numpy()])
    punish_orientation_residuals = _build_residual_store(
        actual=action_df.loc[punish_positive_mask, "target_punish_target_rank"],
        predicted=np.asarray(punish_orientation_predictions, dtype=float),
        cluster_ids=action_df.loc[punish_positive_mask, "cluster_id"],
    )

    reward_units_model = _fit_regressor(
        action_x[reward_positive_mask.to_numpy()],
        action_df.loc[reward_positive_mask, "target_reward_units_total"],
    )
    reward_units_predictions = reward_units_model.predict(action_x[reward_positive_mask.to_numpy()])
    reward_units_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_units_total"],
        predicted=np.asarray(reward_units_predictions, dtype=float),
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )
    reward_target_count_model = _fit_regressor(
        action_x[reward_positive_mask.to_numpy()],
        action_df.loc[reward_positive_mask, "target_reward_target_count"],
    )
    reward_target_count_predictions = reward_target_count_model.predict(action_x[reward_positive_mask.to_numpy()])
    reward_target_count_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_target_count"],
        predicted=np.asarray(reward_target_count_predictions, dtype=float),
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )
    reward_orientation_model = _fit_regressor(
        action_x[reward_positive_mask.to_numpy()],
        action_df.loc[reward_positive_mask, "target_reward_target_rank"],
    )
    reward_orientation_predictions = reward_orientation_model.predict(action_x[reward_positive_mask.to_numpy()])
    reward_orientation_residuals = _build_residual_store(
        actual=action_df.loc[reward_positive_mask, "target_reward_target_rank"],
        predicted=np.asarray(reward_orientation_predictions, dtype=float),
        cluster_ids=action_df.loc[reward_positive_mask, "cluster_id"],
    )

    payload = {
        "version": 1,
        "contribution_text_vectorizer": contribution_text_vectorizer,
        "contribution_struct_vectorizer": contribution_struct_vectorizer,
        "contribution_scaler": contribution_scaler,
        "action_text_vectorizer": action_text_vectorizer,
        "action_struct_vectorizer": action_struct_vectorizer,
        "action_scaler": action_scaler,
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
                "mean_history_chars": float(contribution_df["history_text"].fillna("").astype(str).str.len().mean()),
                "n_all_or_nothing_rows": int(aon_mask.sum()),
                "n_continuous_rows": int(continuous_mask.sum()),
                "mean_target_contribution_rate": float(
                    pd.to_numeric(contribution_df["target_contribution_rate"], errors="coerce").mean()
                ),
            },
            {
                "dataset": "punish_action",
                "n_rows": int(punish_enabled_mask.sum()),
                "mean_history_chars": float(
                    action_df.loc[punish_enabled_mask, "history_text"].fillna("").astype(str).str.len().mean()
                )
                if int(punish_enabled_mask.sum()) > 0
                else float("nan"),
                "n_positive_rows": int(punish_positive_mask.sum()),
                "positive_rate": float(
                    pd.to_numeric(action_df.loc[punish_enabled_mask, "target_any_punish"], errors="coerce").mean()
                )
                if int(punish_enabled_mask.sum()) > 0
                else float("nan"),
                "mean_units_if_positive": float(
                    pd.to_numeric(action_df.loc[punish_positive_mask, "target_punish_units_total"], errors="coerce").mean()
                )
                if int(punish_positive_mask.sum()) > 0
                else float("nan"),
            },
            {
                "dataset": "reward_action",
                "n_rows": int(reward_enabled_mask.sum()),
                "mean_history_chars": float(
                    action_df.loc[reward_enabled_mask, "history_text"].fillna("").astype(str).str.len().mean()
                )
                if int(reward_enabled_mask.sum()) > 0
                else float("nan"),
                "n_positive_rows": int(reward_positive_mask.sum()),
                "positive_rate": float(
                    pd.to_numeric(action_df.loc[reward_enabled_mask, "target_any_reward"], errors="coerce").mean()
                )
                if int(reward_enabled_mask.sum()) > 0
                else float("nan"),
                "mean_units_if_positive": float(
                    pd.to_numeric(action_df.loc[reward_positive_mask, "target_reward_units_total"], errors="coerce").mean()
                )
                if int(reward_positive_mask.sum()) > 0
                else float("nan"),
            },
        ]
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    return payload


class ExactSequenceArchetypePolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor,
        model_payload: Dict[str, Any],
    ) -> None:
        self.env_model = env_model
        self.model_payload = model_payload

    @classmethod
    def from_config(
        cls,
        config: ExactSequenceArchetypePolicyConfig,
    ) -> "ExactSequenceArchetypePolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        env_model_path = artifacts_root / "models" / "dirichlet_env_model.pkl"
        model_path = artifacts_root / "models" / "exact_sequence_policy.pkl"
        if not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        if config.rebuild_model or not model_path.exists():
            train_exact_sequence_policy(output_model_path=model_path)
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
    ) -> ExactSequenceGameState:
        distribution = self.predict_cluster_distribution(env)
        cluster_ids = np.arange(1, len(distribution) + 1, dtype=int)
        sampled = list(rng.choice(cluster_ids, size=len(player_ids), p=distribution))
        visible_name_by_player = _visible_name_map(player_ids)
        return ExactSequenceGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player={str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids},
            player_by_avatar={str(avatar): str(player_id) for player_id, avatar in avatar_by_player.items()},
            cluster_by_player={str(player_id): int(cluster_id) for player_id, cluster_id in zip(player_ids, sampled)},
            visible_name_by_player=visible_name_by_player,
            transcript_lines_by_player={
                str(player_id): _initial_transcript_lines(visible_name_by_player[str(player_id)])
                for player_id in player_ids
            },
        )

    def _transform_contribution_inputs(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_idx: int,
        player_ids: Sequence[str],
    ) -> sparse.csr_matrix:
        texts = [
            "\n".join(game_state.transcript_lines_by_player[str(player_id)] + [round_open(game_state.env, int(round_idx))])
            for player_id in player_ids
        ]
        structured = [
            _base_structured_features(
                game_state.env,
                int(game_state.cluster_by_player[str(player_id)]),
                int(round_idx),
            )
            for player_id in player_ids
        ]
        return _transform_feature_stack(
            texts=texts,
            structured_rows=structured,
            text_vectorizer=self.model_payload["contribution_text_vectorizer"],
            struct_vectorizer=self.model_payload["contribution_struct_vectorizer"],
            scaler=self.model_payload["contribution_scaler"],
        )

    def sample_contributions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        player_ids = list(game_state.player_ids)
        features = self._transform_contribution_inputs(
            game_state=game_state,
            round_idx=int(round_idx),
            player_ids=player_ids,
        )
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        out: Dict[str, int] = {}
        if as_bool(game_state.env.get("CONFIG_allOrNothing", False)):
            probs = self.model_payload["aon_contribution_model"].predict_proba(features)[:, 1]
            for player_id, prob in zip(player_ids, probs):
                out[str(player_id)] = endowment if float(rng.random()) < float(prob) else 0
            return out

        base = np.asarray(self.model_payload["continuous_contribution_model"].predict(features), dtype=float)
        for player_id, predicted_rate in zip(player_ids, base):
            cluster_id = int(game_state.cluster_by_player[str(player_id)])
            residual = _sample_residual(
                rng,
                self.model_payload["continuous_contribution_residuals"],
                cluster_id,
            )
            sampled_rate = float(max(0.0, min(1.0, float(predicted_rate) + residual)))
            out[str(player_id)] = int(max(0, min(endowment, round(sampled_rate * endowment))))
        return out

    def _transform_action_inputs(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_idx: int,
        contributions_by_player: Mapping[str, int],
        player_ids: Sequence[str],
    ) -> sparse.csr_matrix:
        texts = [
            "\n".join(
                game_state.transcript_lines_by_player[str(player_id)]
                + _pre_action_round_lines(
                    env=game_state.env,
                    round_idx=int(round_idx),
                    player_ids=game_state.player_ids,
                    focal_player_id=str(player_id),
                    visible_name_by_player=game_state.visible_name_by_player,
                    contributions_by_player=contributions_by_player,
                )
            )
            for player_id in player_ids
        ]
        structured = [
            _action_structured_features(
                env=game_state.env,
                cluster_id=int(game_state.cluster_by_player[str(player_id)]),
                round_idx=int(round_idx),
                focal_player_id=str(player_id),
                contributions_by_player=contributions_by_player,
            )
            for player_id in player_ids
        ]
        return _transform_feature_stack(
            texts=texts,
            structured_rows=structured,
            text_vectorizer=self.model_payload["action_text_vectorizer"],
            struct_vectorizer=self.model_payload["action_struct_vectorizer"],
            scaler=self.model_payload["action_scaler"],
        )

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
        order = np.arange(len(peer_items))
        rng.shuffle(order)
        shuffled = [peer_items[int(index)] for index in order]
        peer_values = [value for _, value in shuffled]
        ranked = sorted(
            shuffled,
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
        features: sparse.csr_matrix,
        player_ids: Sequence[str],
        contributions_by_player: Mapping[str, int],
        game_state: ExactSequenceGameState,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, int]]:
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

        any_probs = any_model.predict_proba(features)[:, 1]
        raw_units = np.asarray(units_model.predict(features), dtype=float)
        raw_count = np.asarray(count_model.predict(features), dtype=float)
        raw_orientation = np.asarray(orientation_model.predict(features), dtype=float)
        max_units = max(
            int(np.floor(float(game_state.env.get("CONFIG_endowment", 20) or 20) / max(cost, 1.0))),
            1,
        )
        out: Dict[str, Dict[str, int]] = {}
        for index, player_id in enumerate(player_ids):
            if float(rng.random()) >= float(any_probs[index]):
                out[str(player_id)] = {}
                continue
            cluster_id = int(game_state.cluster_by_player[str(player_id)])
            units_total = int(
                max(
                    1,
                    min(
                        max_units,
                        round(float(raw_units[index]) + _sample_residual(rng, units_residuals, cluster_id)),
                    ),
                )
            )
            target_count = int(
                max(
                    1,
                    round(float(raw_count[index]) + _sample_residual(rng, count_residuals, cluster_id)),
                )
            )
            target_rank = float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(raw_orientation[index])
                        + _sample_residual(rng, orientation_residuals, cluster_id),
                    ),
                )
            )
            out[str(player_id)] = self._select_ranked_targets(
                focal_player_id=str(player_id),
                contributions_by_player=contributions_by_player,
                target_rank=target_rank,
                target_count=target_count,
                units_total=max(units_total, target_count),
                rng=rng,
            )
        return out

    def sample_actions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        player_ids = list(game_state.player_ids)
        features = self._transform_action_inputs(
            game_state=game_state,
            round_idx=int(round_idx),
            contributions_by_player=contributions_by_player,
            player_ids=player_ids,
        )
        punish_out = (
            self._sample_mechanism_actions(
                mechanism="punish",
                features=features,
                player_ids=player_ids,
                contributions_by_player=contributions_by_player,
                game_state=game_state,
                rng=rng,
            )
            if as_bool(game_state.env.get("CONFIG_punishmentExists", False))
            else {str(player_id): {} for player_id in player_ids}
        )
        reward_out = (
            self._sample_mechanism_actions(
                mechanism="reward",
                features=features,
                player_ids=player_ids,
                contributions_by_player=contributions_by_player,
                game_state=game_state,
                rng=rng,
            )
            if as_bool(game_state.env.get("CONFIG_rewardExists", False))
            else {str(player_id): {} for player_id in player_ids}
        )
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
            completed_rounds = [
                sum(1 for line in lines if str(line).startswith("<ROUND "))
                for lines in game_state.transcript_lines_by_player.values()
            ]
            round_idx = max(completed_rounds, default=0) + 1
        for player_id in game_state.player_ids:
            game_state.transcript_lines_by_player[str(player_id)].extend(
                _completed_round_lines(
                    env=game_state.env,
                    round_idx=int(round_idx),
                    player_ids=game_state.player_ids,
                    focal_player_id=str(player_id),
                    visible_name_by_player=game_state.visible_name_by_player,
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
