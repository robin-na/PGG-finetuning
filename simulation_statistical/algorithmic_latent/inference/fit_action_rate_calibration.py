from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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

import joblib  # noqa: E402

from simulation_statistical.algorithmic_latent.inference.fit_family_policies import (  # noqa: E402
    DEFAULT_MODEL_OUTPUT_PATH as DEFAULT_FAMILY_POLICY_MODEL_PATH,
    _build_records,
)
from simulation_statistical.algorithmic_latent.inference.infer_player_posteriors import (  # noqa: E402
    DEFAULT_PLAYER_OUTPUT_PATH,
)
from simulation_statistical.algorithmic_latent.inference.build_state_table import (  # noqa: E402
    DEFAULT_STATE_TABLE_ROOT,
)


DEFAULT_OUTPUT_PATH = ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "action_rate_calibration.json"
DEFAULT_SUMMARY_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "action_rate_calibration_summary.csv"
)
EPS = 1e-12


GROUP_COLUMNS: tuple[str, ...] = (
    "wave",
    "gameId",
    "playerId",
    "roundIndex",
)


def _normalize_probs(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums, 0.0)
    if np.any(zero_rows):
        matrix[zero_rows[:, 0], :] = 1.0 / max(matrix.shape[1], 1)
        row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(row_sums, EPS, None)


def _predict_family_action_probs(
    *,
    frame: pd.DataFrame,
    family_payload: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    records = _build_records(frame, family_payload["action_features"])
    probs = family_payload["action_model"].predict_proba(records)
    probs = _normalize_probs(probs)
    classes = [str(value) for value in family_payload["action_model"].named_steps["model"].classes_.tolist()]
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    none = probs[:, class_to_index.get("none", 0)] if "none" in class_to_index else np.zeros(len(frame), dtype=float)
    punish = (
        probs[:, class_to_index.get("punish", 0)]
        if "punish" in class_to_index
        else np.zeros(len(frame), dtype=float)
    )
    reward = (
        probs[:, class_to_index.get("reward", 0)]
        if "reward" in class_to_index
        else np.zeros(len(frame), dtype=float)
    )
    return {
        "none": np.asarray(none, dtype=float),
        "punish": np.asarray(punish, dtype=float),
        "reward": np.asarray(reward, dtype=float),
    }


def _build_group_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(list(GROUP_COLUMNS), dropna=False)
        .agg(
            actual_any_punish=("observed_any_punish", "max"),
            actual_any_reward=("observed_any_reward", "max"),
            punish_enabled=("CONFIG_punishmentExists", "max"),
            reward_enabled=("CONFIG_rewardExists", "max"),
            n_edges=("row_id", "count"),
        )
        .reset_index()
    )
    grouped["punish_enabled"] = grouped["punish_enabled"].astype(bool)
    grouped["reward_enabled"] = grouped["reward_enabled"].astype(bool)
    return grouped


def _group_any_rates(group_ids: np.ndarray, probs: np.ndarray, n_groups: int) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0 - 1e-9)
    log_no = np.log1p(-probs)
    summed = np.bincount(group_ids, weights=log_no, minlength=n_groups)
    return 1.0 - np.exp(summed)


def _apply_scales(
    none: np.ndarray,
    punish: np.ndarray,
    reward: np.ndarray,
    punish_scale: float,
    reward_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted_none = np.asarray(none, dtype=float)
    adjusted_punish = np.asarray(punish, dtype=float) * float(punish_scale)
    adjusted_reward = np.asarray(reward, dtype=float) * float(reward_scale)
    denom = adjusted_none + adjusted_punish + adjusted_reward
    denom = np.clip(denom, EPS, None)
    return adjusted_punish / denom, adjusted_reward / denom


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)[mask]))


def _solve_scale_for_target(
    *,
    none: np.ndarray,
    punish: np.ndarray,
    reward: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    mechanism: str,
    target_rate: float,
    active_group_mask: np.ndarray,
    fixed_other_scale: float,
    initial_scale: float,
) -> float:
    if target_rate <= 0.0:
        return 0.0
    if not np.any(active_group_mask):
        return float(initial_scale)

    def predicted_rate(scale: float) -> float:
        if mechanism == "punish":
            adjusted_punish, adjusted_reward = _apply_scales(
                none,
                punish,
                reward,
                punish_scale=scale,
                reward_scale=fixed_other_scale,
            )
            row_any = _group_any_rates(group_ids, adjusted_punish, n_groups)
        else:
            adjusted_punish, adjusted_reward = _apply_scales(
                none,
                punish,
                reward,
                punish_scale=fixed_other_scale,
                reward_scale=scale,
            )
            row_any = _group_any_rates(group_ids, adjusted_reward, n_groups)
        return _mean_masked(row_any, active_group_mask)

    low = 0.0
    high = max(float(initial_scale), 1.0)
    high_value = predicted_rate(high)
    while high_value < target_rate and high < 1024.0:
        high *= 2.0
        high_value = predicted_rate(high)

    for _ in range(40):
        mid = 0.5 * (low + high)
        value = predicted_rate(mid)
        if value < target_rate:
            low = mid
        else:
            high = mid
    return float(high)


def fit_action_rate_calibration(
    *,
    action_stage_path: Path = DEFAULT_STATE_TABLE_ROOT / "learning_wave_action_stage.parquet",
    family_model_path: Path = DEFAULT_FAMILY_POLICY_MODEL_PATH,
    player_posteriors_path: Path = DEFAULT_PLAYER_OUTPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_output_path: Path = DEFAULT_SUMMARY_PATH,
) -> Dict[str, Any]:
    action_df = pd.read_parquet(action_stage_path).copy()
    player_posteriors = pd.read_parquet(player_posteriors_path).copy()
    family_bundle = joblib.load(family_model_path)
    family_names = [str(name) for name in family_bundle["family_order"]]

    posterior_columns = [f"{family_name}__posterior_prob" for family_name in family_names]
    missing = [column for column in posterior_columns if column not in player_posteriors.columns]
    if missing:
        raise ValueError(f"Missing posterior columns in player posterior table: {missing}")

    merge_columns = ["wave", "gameId", "playerId"]
    posterior_subset = player_posteriors.loc[:, merge_columns + posterior_columns].drop_duplicates(
        subset=merge_columns,
        keep="first",
    )
    action_df = action_df.merge(
        posterior_subset,
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )
    for column in posterior_columns:
        action_df[column] = pd.to_numeric(action_df[column], errors="coerce").fillna(0.0)

    mixture_none = np.zeros(len(action_df), dtype=float)
    mixture_punish = np.zeros(len(action_df), dtype=float)
    mixture_reward = np.zeros(len(action_df), dtype=float)

    for family_name in family_names:
        family_payload = family_bundle["families"][family_name]
        family_probs = _predict_family_action_probs(frame=action_df, family_payload=family_payload)
        weights = action_df[f"{family_name}__posterior_prob"].to_numpy(dtype=float)
        mixture_none += weights * family_probs["none"]
        mixture_punish += weights * family_probs["punish"]
        mixture_reward += weights * family_probs["reward"]

    mixture_matrix = np.column_stack([mixture_none, mixture_punish, mixture_reward])
    mixture_matrix = _normalize_probs(mixture_matrix)
    mixture_none = mixture_matrix[:, 0]
    mixture_punish = mixture_matrix[:, 1]
    mixture_reward = mixture_matrix[:, 2]

    group_frame = _build_group_metadata(action_df)
    group_key_df = group_frame.loc[:, list(GROUP_COLUMNS)].copy()
    group_key_df["group_id"] = np.arange(len(group_key_df), dtype=int)
    action_df = action_df.merge(group_key_df, on=list(GROUP_COLUMNS), how="left", validate="many_to_one")
    group_ids = action_df["group_id"].to_numpy(dtype=int)
    n_groups = len(group_frame)

    actual_punish_rate = _mean_masked(
        group_frame["actual_any_punish"].to_numpy(dtype=float),
        group_frame["punish_enabled"].to_numpy(dtype=bool),
    )
    actual_reward_rate = _mean_masked(
        group_frame["actual_any_reward"].to_numpy(dtype=float),
        group_frame["reward_enabled"].to_numpy(dtype=bool),
    )

    punish_scale = 1.0
    reward_scale = 1.0
    punish_enabled_mask = group_frame["punish_enabled"].to_numpy(dtype=bool)
    reward_enabled_mask = group_frame["reward_enabled"].to_numpy(dtype=bool)

    for _ in range(6):
        punish_scale = _solve_scale_for_target(
            none=mixture_none,
            punish=mixture_punish,
            reward=mixture_reward,
            group_ids=group_ids,
            n_groups=n_groups,
            mechanism="punish",
            target_rate=actual_punish_rate,
            active_group_mask=punish_enabled_mask,
            fixed_other_scale=reward_scale,
            initial_scale=punish_scale,
        )
        reward_scale = _solve_scale_for_target(
            none=mixture_none,
            punish=mixture_punish,
            reward=mixture_reward,
            group_ids=group_ids,
            n_groups=n_groups,
            mechanism="reward",
            target_rate=actual_reward_rate,
            active_group_mask=reward_enabled_mask,
            fixed_other_scale=punish_scale,
            initial_scale=reward_scale,
        )

    before_row_punish = _group_any_rates(group_ids, mixture_punish, n_groups)
    before_row_reward = _group_any_rates(group_ids, mixture_reward, n_groups)
    adjusted_punish, adjusted_reward = _apply_scales(
        mixture_none,
        mixture_punish,
        mixture_reward,
        punish_scale=punish_scale,
        reward_scale=reward_scale,
    )
    after_row_punish = _group_any_rates(group_ids, adjusted_punish, n_groups)
    after_row_reward = _group_any_rates(group_ids, adjusted_reward, n_groups)

    summary_rows = [
        {
            "mechanism": "punish",
            "actual_row_rate": float(actual_punish_rate),
            "predicted_row_rate_before": _mean_masked(before_row_punish, punish_enabled_mask),
            "predicted_row_rate_after": _mean_masked(after_row_punish, punish_enabled_mask),
            "scale": float(punish_scale),
        },
        {
            "mechanism": "reward",
            "actual_row_rate": float(actual_reward_rate),
            "predicted_row_rate_before": _mean_masked(before_row_reward, reward_enabled_mask),
            "predicted_row_rate_after": _mean_masked(after_row_reward, reward_enabled_mask),
            "scale": float(reward_scale),
        },
    ]

    artifact = {
        "version": 1,
        "action_stage_path": str(action_stage_path),
        "family_model_path": str(family_model_path),
        "player_posteriors_path": str(player_posteriors_path),
        "global": {
            "punish_scale": float(punish_scale),
            "reward_scale": float(reward_scale),
        },
        "summary_rows": summary_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    pd.DataFrame(summary_rows).to_csv(summary_output_path, index=False)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a post-hoc punishment/reward row-rate calibration for the algorithmic-latent simulator."
    )
    parser.add_argument(
        "--action_stage_path",
        type=str,
        default=str(DEFAULT_STATE_TABLE_ROOT / "learning_wave_action_stage.parquet"),
    )
    parser.add_argument("--family_model_path", type=str, default=str(DEFAULT_FAMILY_POLICY_MODEL_PATH))
    parser.add_argument("--player_posteriors_path", type=str, default=str(DEFAULT_PLAYER_OUTPUT_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary_output_path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifact = fit_action_rate_calibration(
        action_stage_path=Path(args.action_stage_path).resolve(),
        family_model_path=Path(args.family_model_path).resolve(),
        player_posteriors_path=Path(args.player_posteriors_path).resolve(),
        output_path=Path(args.output_path).resolve(),
        summary_output_path=Path(args.summary_output_path).resolve(),
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
