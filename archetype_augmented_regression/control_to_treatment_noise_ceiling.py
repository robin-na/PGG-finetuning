#!/usr/bin/env python3
"""Estimate control->treatment noise ceiling from df_analysis tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_LEARN = Path("benchmark/data/processed_data/df_analysis_learn.csv")
DEFAULT_VAL = Path("benchmark/data/processed_data/df_analysis_val.csv")
DEFAULT_OUTDIR = Path("reports/archetype_augmented_regression/noise_ceiling_control_to_treatment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute control->treatment noise-ceiling style metrics from grouped CONFIG_treatmentName pairs."
        )
    )
    parser.add_argument("--learn-csv", type=Path, default=DEFAULT_LEARN)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--target", type=str, default="itt_efficiency")
    parser.add_argument("--valid-col", type=str, default="valid_number_of_starting_players")
    parser.add_argument("--treatment-col", type=str, default="CONFIG_treatmentName")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def r2_with_mean(y: np.ndarray, yhat: np.ndarray, baseline_mean: float) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_base = float(np.sum((y - baseline_mean) ** 2))
    if ss_base <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_base


def extract_pairs(
    df: pd.DataFrame,
    target: str,
    treatment_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    tn = tmp[treatment_col].astype(str)
    tmp["arm"] = tn.str.extract(r"_(C|T)$", expand=False)
    tmp["config_base"] = tn.str.replace(r"_(C|T)$", "", regex=True)
    tmp = tmp[tmp["arm"].isin(["C", "T"])].copy()

    arm_stats = (
        tmp.groupby(["config_base", "arm"], as_index=False)
        .agg(
            mean_target=(target, "mean"),
            std_target=(target, "std"),
            n_games=(target, "size"),
        )
    )
    arm_stats["se_target"] = arm_stats["std_target"] / np.sqrt(arm_stats["n_games"])

    pivot = arm_stats.pivot(index="config_base", columns="arm")
    pivot.columns = [f"{metric}_{arm}" for metric, arm in pivot.columns]
    pairs = pivot.reset_index()
    pairs = pairs.dropna(subset=["mean_target_C", "mean_target_T"]).copy()
    pairs = pairs.sort_values("config_base").reset_index(drop=True)
    return pairs, arm_stats.sort_values(["config_base", "arm"]).reset_index(drop=True)


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2_train_mean: float
    r2_test_mean: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2_oos_train_mean": self.r2_train_mean,
            "r2_test_mean": self.r2_test_mean,
        }


def eval_fixed_baseline(y: np.ndarray, yhat: np.ndarray, train_mean: float) -> Metrics:
    return Metrics(
        rmse=rmse(y, yhat),
        mae=mae(y, yhat),
        r2_train_mean=r2_with_mean(y, yhat, train_mean),
        r2_test_mean=r2_with_mean(y, yhat, float(y.mean())),
    )


def loocv_linear_metrics(x: np.ndarray, y: np.ndarray) -> Metrics:
    n = len(y)
    if n < 3:
        return Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
    preds = np.zeros(n, dtype=float)
    baseline = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xtr = x[mask].reshape(-1, 1)
        ytr = y[mask]
        model = LinearRegression().fit(xtr, ytr)
        preds[i] = float(model.predict(x[i].reshape(1, -1))[0])
        baseline[i] = float(ytr.mean())
    ss_res = float(np.sum((y - preds) ** 2))
    ss_base = float(np.sum((y - baseline) ** 2))
    r2_oos = 1.0 - ss_res / ss_base if ss_base > 0 else float("nan")
    return Metrics(
        rmse=rmse(y, preds),
        mae=mae(y, preds),
        r2_train_mean=r2_oos,
        r2_test_mean=r2_with_mean(y, preds, float(y.mean())),
    )


def loocv_identity_metrics(x: np.ndarray, y: np.ndarray) -> Metrics:
    n = len(y)
    if n < 2:
        return Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
    preds = x.copy()
    baseline = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        baseline[i] = float(y[mask].mean())
    ss_res = float(np.sum((y - preds) ** 2))
    ss_base = float(np.sum((y - baseline) ** 2))
    r2_oos = 1.0 - ss_res / ss_base if ss_base > 0 else float("nan")
    return Metrics(
        rmse=rmse(y, preds),
        mae=mae(y, preds),
        r2_train_mean=r2_oos,
        r2_test_mean=r2_with_mean(y, preds, float(y.mean())),
    )


def fit_linear_train_eval_test(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    model = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
    yhat = model.predict(x_test.reshape(-1, 1))
    met = eval_fixed_baseline(y_test, yhat, train_mean=float(y_train.mean()))
    return {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        **met.as_dict(),
    }


def evaluate_wave(pairs: pd.DataFrame) -> dict[str, Any]:
    x = pairs["mean_target_C"].to_numpy(dtype=float)
    y = pairs["mean_target_T"].to_numpy(dtype=float)
    n = int(len(pairs))

    # In-sample identity yhat=x
    in_id = eval_fixed_baseline(y, x, train_mean=float(y.mean())).as_dict()

    # In-sample linear
    if n >= 2:
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        yhat_lr = lr.predict(x.reshape(-1, 1))
        in_lr = {
            "coef": float(lr.coef_[0]),
            "intercept": float(lr.intercept_),
            **eval_fixed_baseline(y, yhat_lr, train_mean=float(y.mean())).as_dict(),
        }
    else:
        in_lr = {"coef": float("nan"), "intercept": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2_oos_train_mean": float("nan"), "r2_test_mean": float("nan")}

    loocv_lr = loocv_linear_metrics(x, y).as_dict()
    loocv_id = loocv_identity_metrics(x, y).as_dict()
    return {
        "n_paired_configs": n,
        "identity_in_sample": in_id,
        "linear_in_sample": in_lr,
        "identity_loocv": loocv_id,
        "linear_loocv": loocv_lr,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    learn = pd.read_csv(args.learn_csv)
    val = pd.read_csv(args.val_csv)

    learn["valid_bool"] = to_bool(learn[args.valid_col])
    val["valid_bool"] = to_bool(val[args.valid_col])

    learn_valid = learn[learn["valid_bool"] == True].copy()
    val_valid = val[val["valid_bool"] == True].copy()

    learn_pairs, learn_arm_stats = extract_pairs(learn_valid, args.target, args.treatment_col)
    val_pairs, val_arm_stats = extract_pairs(val_valid, args.target, args.treatment_col)

    # Cross-wave OOS: train on learn pairs, test on val pairs
    xtr = learn_pairs["mean_target_C"].to_numpy(dtype=float)
    ytr = learn_pairs["mean_target_T"].to_numpy(dtype=float)
    xte = val_pairs["mean_target_C"].to_numpy(dtype=float)
    yte = val_pairs["mean_target_T"].to_numpy(dtype=float)
    cross_linear = fit_linear_train_eval_test(xtr, ytr, xte, yte)
    cross_identity = eval_fixed_baseline(yte, xte, train_mean=float(ytr.mean())).as_dict()

    summary = {
        "inputs": {
            "learn_csv": str(args.learn_csv),
            "val_csv": str(args.val_csv),
            "target": args.target,
            "valid_col": args.valid_col,
            "treatment_col": args.treatment_col,
        },
        "valid_number_of_starting_players": {
            "learn_rows_total": int(len(learn)),
            "learn_rows_valid_true": int(len(learn_valid)),
            "learn_all_true": bool(len(learn_valid) == len(learn)),
            "val_rows_total": int(len(val)),
            "val_rows_valid_true": int(len(val_valid)),
            "val_all_true": bool(len(val_valid) == len(val)),
        },
        "paired_config_counts": {
            "learn_paired_configs": int(len(learn_pairs)),
            "val_paired_configs": int(len(val_pairs)),
        },
        "within_wave": {
            "learn": evaluate_wave(learn_pairs),
            "val": evaluate_wave(val_pairs),
        },
        "cross_wave_oos_learn_to_val": {
            "identity_yhat_equals_control_mean": cross_identity,
            "linear_fit_yT_on_xC": cross_linear,
        },
    }

    # Save detailed tables for reproducibility.
    learn_pairs.to_csv(args.output_dir / "learn_pairs_control_to_treatment.csv", index=False)
    val_pairs.to_csv(args.output_dir / "val_pairs_control_to_treatment.csv", index=False)
    learn_arm_stats.to_csv(args.output_dir / "learn_arm_stats.csv", index=False)
    val_arm_stats.to_csv(args.output_dir / "val_arm_stats.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print compact summary.
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()

