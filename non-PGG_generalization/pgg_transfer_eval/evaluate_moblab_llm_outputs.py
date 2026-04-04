#!/usr/bin/env python3
"""Evaluate MobLab LLM prediction outputs and compare against statistical baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluate_batch_results import extract_content, parse_json_object  # type: ignore
from moblab_llm_utils import DEFAULT_OUTPUT_ROOT, load_jsonl, measure_family, stat_baseline_reference


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "evals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    ranked = joined.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def r2_from_arrays(gold: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((gold - gold.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1 - np.sum((gold - pred) ** 2) / denom)


def parse_prediction_share(obj: Dict[str, Any]) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    pred = obj.get("prediction")
    if isinstance(pred, dict):
        value = pred.get("share_percent")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None
    return None


def parse_prediction_trajectory(obj: Dict[str, Any]) -> Optional[List[float]]:
    if not isinstance(obj, dict):
        return None
    pred = obj.get("prediction")
    if not isinstance(pred, dict):
        return None
    values = pred.get("future_round_share_percents")
    if not isinstance(values, list) or not values:
        return None
    parsed: List[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except Exception:
            return None
    return parsed


def metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (baseline, task_type, prediction_mode, target_measure), group in df.groupby(
        ["baseline", "task_type", "prediction_mode", "target_measure"], sort=False
    ):
        if task_type == "task2" and prediction_mode == "trajectory":
            gold_flat: List[float] = []
            pred_flat: List[float] = []
            seq_maes: List[float] = []
            valid_instances = 0
            for row in group.itertuples():
                gold_seq = list(row.gold_future_rounds_share_percent or [])
                pred_seq = list(row.predicted_future_rounds_share_percent or [])
                if len(gold_seq) != len(pred_seq) or not gold_seq:
                    continue
                gold_arr = np.asarray(gold_seq, dtype=float)
                pred_arr = np.asarray(pred_seq, dtype=float)
                gold_flat.extend(gold_arr.tolist())
                pred_flat.extend(pred_arr.tolist())
                seq_maes.append(float(np.mean(np.abs(gold_arr - pred_arr))))
                valid_instances += 1
            if not gold_flat:
                continue
            gold_np = np.asarray(gold_flat, dtype=float)
            pred_np = np.asarray(pred_flat, dtype=float)
            rows.append(
                {
                    "baseline": baseline,
                    "task_type": task_type,
                    "prediction_mode": prediction_mode,
                    "target_measure": target_measure,
                    "n": int(valid_instances),
                    "n_values": int(len(gold_np)),
                    "mae": float(np.mean(np.abs(gold_np - pred_np))),
                    "rmse": float(np.sqrt(np.mean((gold_np - pred_np) ** 2))),
                    "r2": r2_from_arrays(gold_np, pred_np),
                    "spearman": spearman(pd.Series(gold_np), pd.Series(pred_np)),
                    "mean_gold": float(gold_np.mean()),
                    "mean_pred": float(pred_np.mean()),
                    "sequence_mae": float(np.mean(seq_maes)),
                }
            )
            continue

        gold = group["gold_share_percent"].astype(float)
        pred = group["predicted_share_percent"].astype(float)
        rows.append(
            {
                "baseline": baseline,
                "task_type": task_type,
                "prediction_mode": prediction_mode,
                "target_measure": target_measure,
                "n": int(len(group)),
                "n_values": int(len(group)),
                "mae": float((gold - pred).abs().mean()),
                "rmse": float(np.sqrt(np.mean((gold - pred) ** 2))),
                "r2": r2_from_arrays(gold.to_numpy(dtype=float), pred.to_numpy(dtype=float)),
                "spearman": spearman(gold, pred),
                "mean_gold": float(gold.mean()),
                "mean_pred": float(pred.mean()),
                "sequence_mae": float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "baseline",
                "task_type",
                "prediction_mode",
                "target_measure",
                "n",
                "n_values",
                "mae",
                "rmse",
                "r2",
                "spearman",
                "mean_gold",
                "mean_pred",
                "sequence_mae",
            ]
        )
    return pd.DataFrame(rows).sort_values(["task_type", "prediction_mode", "target_measure", "baseline"]).reset_index(drop=True)


def build_task_baseline_reference(tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    task_df = pd.DataFrame(tasks)
    if task_df.empty:
        return pd.DataFrame(rows)

    task1_ref = stat_baseline_reference("task1").rename(
        columns={"cv_r2": "stat_r2", "cv_mae": "stat_mae", "pred_spearman": "stat_spearman"}
    )
    task1_ref = task1_ref[["target_measure", "stat_r2", "stat_mae", "stat_spearman"]].copy()
    task1_ref["stat_mae"] = task1_ref["stat_mae"].astype(float) * 100.0
    task1_ref["task_type"] = "task1"
    task1_ref["prediction_mode"] = "scalar"
    rows.extend(task1_ref.to_dict(orient="records"))

    task2_ref = stat_baseline_reference("task2").rename(
        columns={"measure": "target_measure", "persistence_r2": "stat_r2", "persistence_mae": "stat_mae", "persistence_spearman": "stat_spearman"}
    )
    task2_ref = task2_ref[["target_measure", "stat_r2", "stat_mae", "stat_spearman"]].copy()
    task2_ref["stat_mae"] = task2_ref["stat_mae"].astype(float) * 100.0
    task2_ref["task_type"] = "task2"
    task2_ref["prediction_mode"] = "future_mean"
    rows.extend(task2_ref.to_dict(orient="records"))

    trajectory_tasks = task_df[(task_df["task_type"] == "task2") & (task_df["prediction_mode"] == "trajectory")].copy()
    for target_measure, group in trajectory_tasks.groupby("target_measure", sort=False):
        gold_flat: List[float] = []
        pred_flat: List[float] = []
        for row in group.itertuples():
            gold_seq = list(row.gold_future_rounds_share_percent or [])
            pred_seq = list(row.persistence_future_rounds_share_percent or [])
            if len(gold_seq) != len(pred_seq) or not gold_seq:
                continue
            gold_flat.extend(float(v) for v in gold_seq)
            pred_flat.extend(float(v) for v in pred_seq)
        if not gold_flat:
            continue
        gold_np = np.asarray(gold_flat, dtype=float)
        pred_np = np.asarray(pred_flat, dtype=float)
        rows.append(
            {
                "task_type": "task2",
                "prediction_mode": "trajectory",
                "target_measure": target_measure,
                "stat_r2": r2_from_arrays(gold_np, pred_np),
                "stat_mae": float(np.mean(np.abs(gold_np - pred_np))),
                "stat_spearman": spearman(pd.Series(gold_np), pd.Series(pred_np)),
            }
        )
    return pd.DataFrame(rows)


def attach_stat_references(metrics_df: pd.DataFrame, tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    ref = build_task_baseline_reference(tasks)
    merged = metrics_df.merge(ref, on=["task_type", "prediction_mode", "target_measure"], how="left")
    return merged


def build_summary_md(scored_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    lines = [
        "# MobLab LLM Evaluation",
        "",
        "## Setup",
        "",
        "- Predictions are evaluated on share percent (0-100).",
        "- Task 1 compares LLM predictions against the non-LLM first-round cross-game baseline.",
        "- Task 2 `future_mean` compares LLM predictions against the non-LLM `k=1 persistence` baseline on future means.",
        "- Task 2 `trajectory` compares LLM predictions against the non-LLM `k=1 persistence` baseline on flattened future rounds.",
        "",
        "## Aggregate Results",
        "",
        "| Baseline | Task | Mode | Target | N | Values | MAE | RMSE | R^2 | Spearman | Sequence MAE | Statistical baseline MAE | Statistical baseline R^2 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_df.itertuples():
        lines.append(
            f"| {row.baseline} | {row.task_type} | {row.prediction_mode} | {row.target_measure} | {row.n} | {row.n_values} | "
            f"{row.mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} | {row.spearman:.3f} | "
            f"{'' if pd.isna(row.sequence_mae) else f'{row.sequence_mae:.3f}'} | "
            f"{row.stat_mae:.3f} | {row.stat_r2:.3f} |"
        )

    failures = scored_df[scored_df["parse_ok"] == False]
    lines.extend(["", "## Parse Failures", "", f"- {len(failures)} of {len(scored_df)} predictions could not be parsed into the expected JSON prediction schema.", ""])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_jsonl(args.manifest_jsonl)
    manifest_by_id = {str(row["custom_id"]): row for row in manifest_rows}
    task_rows = load_jsonl(args.tasks_jsonl)
    tasks_by_id = {str(row["instance_id"]): row for row in task_rows}

    scored_rows: List[Dict[str, Any]] = []
    for response_row in load_jsonl(args.responses_jsonl):
        custom_id = str(response_row.get("custom_id", ""))
        manifest = manifest_by_id.get(custom_id)
        if manifest is None:
            continue
        task = tasks_by_id.get(str(manifest["instance_id"]))
        if task is None:
            continue
        content, error, _ = extract_content(response_row)
        parsed_obj = parse_json_object(content) if content else None
        prediction_mode = str(task.get("prediction_mode", "scalar"))
        pred = parse_prediction_share(parsed_obj) if parsed_obj and prediction_mode != "trajectory" else None
        pred_seq = parse_prediction_trajectory(parsed_obj) if parsed_obj and prediction_mode == "trajectory" else None
        parse_ok = pred is not None if prediction_mode != "trajectory" else pred_seq is not None and len(pred_seq) == int(task.get("n_future_rounds", 0))
        scored_rows.append(
            {
                "custom_id": custom_id,
                "baseline": manifest["baseline"],
                "task_type": manifest["task_type"],
                "prediction_mode": prediction_mode,
                "target_measure": manifest["target_measure"],
                "target_family": measure_family(manifest["target_measure"]),
                "gold_share_percent": float(task.get("gold_share_percent", task.get("gold_future_share_percent"))),
                "predicted_share_percent": pred,
                "gold_future_rounds_share_percent": task.get("gold_future_rounds_share_percent"),
                "predicted_future_rounds_share_percent": pred_seq,
                "persistence_share_percent": task.get("persistence_share_percent"),
                "persistence_future_rounds_share_percent": task.get("persistence_future_rounds_share_percent"),
                "parse_ok": parse_ok,
                "parse_error": error,
                "raw_content": content,
            }
        )

    scored_df = pd.DataFrame(scored_rows)
    scored_df.to_csv(args.output_dir / "scored_predictions.csv", index=False)

    valid_df = scored_df[scored_df["parse_ok"] == True].copy()
    metrics_df = metric_table(valid_df)
    metrics_df = attach_stat_references(metrics_df, task_rows)
    metrics_df.to_csv(args.output_dir / "metrics_by_target.csv", index=False)

    summary_md = build_summary_md(scored_df, metrics_df)
    (args.output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
