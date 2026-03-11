from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import joblib
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

from simulation_statistical.algorithmic_latent.inference.build_state_table import (  # noqa: E402
    DEFAULT_STATE_TABLE_ROOT,
)
from simulation_statistical.algorithmic_latent.inference.fit_family_policies import (  # noqa: E402
    DEFAULT_MODEL_OUTPUT_PATH,
    _build_records,
)


DEFAULT_PLAYER_OUTPUT_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "learning_wave_player_family_posteriors.parquet"
)
DEFAULT_TREATMENT_OUTPUT_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "learning_wave_treatment_family_mixture.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "learning_wave_player_family_posteriors_summary.json"
)
EPS = 1e-12


PLAYER_GROUP_COLUMNS: tuple[str, ...] = (
    "wave",
    "gameId",
    "gameName",
    "CONFIG_treatmentName",
    "playerId",
    "playerAvatar",
)


def _safe_log(value: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.clip(value, EPS, 1.0))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / np.clip(denom, EPS, None)


def _score_rows_for_family(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    model_bundle: Mapping[str, Any],
    target_column: str,
) -> np.ndarray:
    records = _build_records(frame, feature_names)
    pipeline = model_bundle
    probabilities = pipeline.predict_proba(records)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums, 0.0)
    if np.any(zero_rows):
        probabilities[zero_rows[:, 0], :] = 1.0 / max(probabilities.shape[1], 1)
        row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, EPS, None)
    classes = [str(value) for value in pipeline.named_steps["model"].classes_.tolist()]
    class_index = {label: idx for idx, label in enumerate(classes)}

    target_labels = frame[target_column].astype(str).tolist()
    out = np.full(len(target_labels), fill_value=math.log(EPS), dtype=float)
    for row_idx, target_label in enumerate(target_labels):
        class_idx = class_index.get(str(target_label))
        if class_idx is None:
            continue
        out[row_idx] = float(_safe_log(probabilities[row_idx, class_idx]))
    return out


def _contribution_stage_scores(
    *,
    contribution_df: pd.DataFrame,
    family_payload: Mapping[str, Any],
) -> pd.DataFrame:
    frame = contribution_df.copy()
    frame["family_contrib_logp"] = _score_rows_for_family(
        frame=frame,
        feature_names=family_payload["contribution_features"],
        model_bundle=family_payload["contribution_model"],
        target_column="actual_contribution_bin5",
    )
    grouped = (
        frame.groupby(list(PLAYER_GROUP_COLUMNS), dropna=False)
        .agg(
            contribution_logp_mean=("family_contrib_logp", "mean"),
            contribution_logp_sum=("family_contrib_logp", "sum"),
            contribution_round_count=("row_id", "count"),
        )
        .reset_index()
    )
    return grouped


def _action_stage_scores(
    *,
    action_df: pd.DataFrame,
    family_payload: Mapping[str, Any],
) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame(columns=list(PLAYER_GROUP_COLUMNS) + ["action_logp_mean", "action_logp_sum", "action_round_count"])

    frame = action_df.copy()
    frame["family_action_logp"] = _score_rows_for_family(
        frame=frame,
        feature_names=family_payload["action_features"],
        model_bundle=family_payload["action_model"],
        target_column="observed_action_label",
    )
    round_group_columns = list(PLAYER_GROUP_COLUMNS) + ["roundIndex"]
    round_scores = (
        frame.groupby(round_group_columns, dropna=False)
        .agg(
            action_round_logp_mean=("family_action_logp", "mean"),
            action_edge_count=("row_id", "count"),
        )
        .reset_index()
    )
    grouped = (
        round_scores.groupby(list(PLAYER_GROUP_COLUMNS), dropna=False)
        .agg(
            action_logp_mean=("action_round_logp_mean", "mean"),
            action_logp_sum=("action_round_logp_mean", "sum"),
            action_round_count=("roundIndex", "count"),
            action_edge_count=("action_edge_count", "sum"),
        )
        .reset_index()
    )
    return grouped


def infer_player_posteriors(
    *,
    contribution_stage_path: Path,
    action_stage_path: Path,
    family_model_path: Path = DEFAULT_MODEL_OUTPUT_PATH,
    player_output_path: Path = DEFAULT_PLAYER_OUTPUT_PATH,
    treatment_output_path: Path = DEFAULT_TREATMENT_OUTPUT_PATH,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    contribution_stage_weight: float = 1.0,
    action_stage_weight: float = 1.0,
) -> Dict[str, Any]:
    contribution_df = pd.read_parquet(contribution_stage_path)
    action_df = pd.read_parquet(action_stage_path) if action_stage_path.exists() else pd.DataFrame()
    bundle = joblib.load(family_model_path)

    family_names = [str(name) for name in bundle["family_order"]]
    merged: pd.DataFrame | None = None
    family_stage_columns: Dict[str, Dict[str, str]] = {}

    for family_name in family_names:
        family_payload = bundle["families"][family_name]
        contrib_scores = _contribution_stage_scores(
            contribution_df=contribution_df,
            family_payload=family_payload,
        )
        action_scores = _action_stage_scores(
            action_df=action_df,
            family_payload=family_payload,
        )
        family_scores = pd.merge(
            contrib_scores,
            action_scores,
            on=list(PLAYER_GROUP_COLUMNS),
            how="left",
        )
        family_scores[f"{family_name}__contribution_logp_mean"] = family_scores["contribution_logp_mean"]
        family_scores[f"{family_name}__contribution_logp_sum"] = family_scores["contribution_logp_sum"]
        family_scores[f"{family_name}__contribution_round_count"] = family_scores["contribution_round_count"]
        family_scores[f"{family_name}__action_logp_mean"] = family_scores["action_logp_mean"]
        family_scores[f"{family_name}__action_logp_sum"] = family_scores["action_logp_sum"]
        family_scores[f"{family_name}__action_round_count"] = family_scores["action_round_count"]
        family_scores[f"{family_name}__action_edge_count"] = family_scores["action_edge_count"]
        keep_columns = list(PLAYER_GROUP_COLUMNS) + [
            f"{family_name}__contribution_logp_mean",
            f"{family_name}__contribution_logp_sum",
            f"{family_name}__contribution_round_count",
            f"{family_name}__action_logp_mean",
            f"{family_name}__action_logp_sum",
            f"{family_name}__action_round_count",
            f"{family_name}__action_edge_count",
        ]
        family_scores = family_scores.loc[:, keep_columns]
        if merged is None:
            merged = family_scores
        else:
            merged = pd.merge(
                merged,
                family_scores,
                on=list(PLAYER_GROUP_COLUMNS),
                how="outer",
            )
        family_stage_columns[family_name] = {
            "contribution_logp_mean": f"{family_name}__contribution_logp_mean",
            "action_logp_mean": f"{family_name}__action_logp_mean",
        }

    if merged is None:
        raise ValueError("No family scores were computed.")

    merged = merged.copy()
    score_matrix = []
    for family_name in family_names:
        contrib_col = family_stage_columns[family_name]["contribution_logp_mean"]
        action_col = family_stage_columns[family_name]["action_logp_mean"]
        contrib_values = merged[contrib_col].to_numpy(dtype=float)
        action_values = merged[action_col].to_numpy(dtype=float)
        contrib_available = ~np.isnan(contrib_values)
        action_available = ~np.isnan(action_values)
        contrib_component = np.where(contrib_available, contribution_stage_weight * contrib_values, 0.0)
        action_component = np.where(action_available, action_stage_weight * action_values, 0.0)
        total_weight = (
            contribution_stage_weight * contrib_available.astype(float)
            + action_stage_weight * action_available.astype(float)
        )
        total_weight = np.where(total_weight <= 0.0, 1.0, total_weight)
        score = (contrib_component + action_component) / total_weight
        merged[f"{family_name}__posterior_score"] = score
        score_matrix.append(score)
    score_array = np.column_stack(score_matrix)
    posterior_probs = _softmax(score_array)

    for idx, family_name in enumerate(family_names):
        merged[f"{family_name}__posterior_prob"] = posterior_probs[:, idx]

    merged["posterior_entropy"] = -np.sum(
        posterior_probs * np.log(np.clip(posterior_probs, EPS, 1.0)),
        axis=1,
    )
    top_family_idx = np.argmax(posterior_probs, axis=1)
    merged["top_family"] = [family_names[idx] for idx in top_family_idx]
    merged["top_family_prob"] = posterior_probs[np.arange(len(merged)), top_family_idx]
    merged["n_families"] = int(len(family_names))

    player_output_path.parent.mkdir(parents=True, exist_ok=True)
    treatment_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(player_output_path, index=False)

    treatment_mixture = (
        merged.groupby(["wave", "CONFIG_treatmentName"], dropna=False)
        .agg(
            n_players=("playerId", "nunique"),
            avg_posterior_entropy=("posterior_entropy", "mean"),
            **{
                f"{family_name}__mixture": (f"{family_name}__posterior_prob", "mean")
                for family_name in family_names
            },
        )
        .reset_index()
    )
    treatment_mixture.to_csv(treatment_output_path, index=False)

    summary = {
        "version": 1,
        "family_model_path": str(family_model_path),
        "contribution_stage_path": str(contribution_stage_path),
        "action_stage_path": str(action_stage_path),
        "player_output_path": str(player_output_path),
        "treatment_output_path": str(treatment_output_path),
        "n_players": int(len(merged)),
        "n_treatments": int(treatment_mixture["CONFIG_treatmentName"].nunique(dropna=True)),
        "family_order": family_names,
        "contribution_stage_weight": float(contribution_stage_weight),
        "action_stage_weight": float(action_stage_weight),
        "mean_posterior_entropy": float(merged["posterior_entropy"].mean()),
        "mean_top_family_prob": float(merged["top_family_prob"].mean()),
        "top_family_counts": {
            str(key): int(value)
            for key, value in merged["top_family"].value_counts(dropna=False).to_dict().items()
        },
    }
    with open(summary_output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer player-level posterior weights over algorithmic families from state tables."
    )
    parser.add_argument(
        "--contribution_stage_path",
        type=str,
        default=str(DEFAULT_STATE_TABLE_ROOT / "learning_wave_contribution_stage.parquet"),
    )
    parser.add_argument(
        "--action_stage_path",
        type=str,
        default=str(DEFAULT_STATE_TABLE_ROOT / "learning_wave_action_stage.parquet"),
    )
    parser.add_argument(
        "--family_model_path",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT_PATH),
    )
    parser.add_argument(
        "--player_output_path",
        type=str,
        default=str(DEFAULT_PLAYER_OUTPUT_PATH),
    )
    parser.add_argument(
        "--treatment_output_path",
        type=str,
        default=str(DEFAULT_TREATMENT_OUTPUT_PATH),
    )
    parser.add_argument(
        "--summary_output_path",
        type=str,
        default=str(DEFAULT_SUMMARY_OUTPUT_PATH),
    )
    parser.add_argument("--contribution_stage_weight", type=float, default=1.0)
    parser.add_argument("--action_stage_weight", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = infer_player_posteriors(
        contribution_stage_path=Path(args.contribution_stage_path).resolve(),
        action_stage_path=Path(args.action_stage_path).resolve(),
        family_model_path=Path(args.family_model_path).resolve(),
        player_output_path=Path(args.player_output_path).resolve(),
        treatment_output_path=Path(args.treatment_output_path).resolve(),
        summary_output_path=Path(args.summary_output_path).resolve(),
        contribution_stage_weight=float(args.contribution_stage_weight),
        action_stage_weight=float(args.action_stage_weight),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
