#!/usr/bin/env python3
"""Evaluate LLM predictions against ground truth wave1_3 responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats

from config import (
    HF_CONFIG_WAVE_SPLIT,
    HF_DATASET_NAME,
    OUTPUT_ROOT,
    QUESTION_CATALOG_JSON,
)


def load_ground_truth() -> pd.DataFrame:
    """Load wave1_3 human responses from HuggingFace and extract economic question answers."""
    ds = load_dataset(HF_DATASET_NAME, HF_CONFIG_WAVE_SPLIT)["data"]

    catalog = json.load(QUESTION_CATALOG_JSON.open("r", encoding="utf-8"))
    econ_questions = [
        q for q in catalog
        if "Economic preferences" in q.get("BlockName", "")
        and not q.get("is_descriptive", False)
        and q["QuestionType"] in ("MC", "Matrix")
    ]

    # Build QID → question mapping
    qid_to_q = {q["QuestionID"]: q for q in econ_questions}

    # Extract answers from persona JSON for each participant
    rows = []
    for example in ds:
        pid = str(example["pid"])
        persona = json.loads(example["wave1_3_persona_json"])

        answers = {}
        for block in persona:
            for q in block.get("Questions", []):
                qid = q.get("QuestionID")
                if qid not in qid_to_q:
                    continue
                q_def = qid_to_q[qid]
                q_answers = q.get("Answers", {})

                if q_def["QuestionType"] == "MC":
                    pos = q_answers.get("SelectedByPosition")
                    if pos is not None:
                        answers[qid] = int(pos)

                elif q_def["QuestionType"] == "Matrix":
                    csv_cols = q_def.get("csv_columns", [])
                    selected = q_answers.get("SelectedByPosition", [])
                    if isinstance(selected, list):
                        for i, val in enumerate(selected):
                            if i < len(csv_cols) and val is not None:
                                answers[csv_cols[i]] = int(val)

        row = {"pid": pid}
        row.update(answers)
        rows.append(row)

    return pd.DataFrame(rows)


def load_predictions(predictions_jsonl: Path) -> pd.DataFrame:
    """Load LLM predictions and pivot to wide format."""
    rows = []
    with predictions_jsonl.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    if not rows:
        return pd.DataFrame()

    # Load question catalog for column mapping
    catalog = json.load(QUESTION_CATALOG_JSON.open("r", encoding="utf-8"))
    qid_to_q = {q["QuestionID"]: q for q in catalog}

    # Expand predictions into per-column values
    expanded = []
    for r in rows:
        pid = str(r["pid"])
        qid = r["question_id"]
        q = qid_to_q.get(qid)
        if not q:
            continue

        parsed = r.get("parsed_answer")
        if parsed is None:
            continue

        if q["QuestionType"] == "MC":
            expanded.append({"pid": pid, "col": qid, "value": parsed})

        elif q["QuestionType"] == "Matrix":
            csv_cols = q.get("csv_columns", [])
            if isinstance(parsed, list):
                for i, val in enumerate(parsed):
                    if i < len(csv_cols):
                        expanded.append({"pid": pid, "col": csv_cols[i], "value": val})

    if not expanded:
        return pd.DataFrame()

    df = pd.DataFrame(expanded)
    pivot = df.pivot_table(index="pid", columns="col", values="value", aggfunc="first")
    pivot = pivot.reset_index()
    return pivot


def compute_metrics(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    questions: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute per-question and overall metrics."""
    # Merge on pid
    merged = gt.merge(pred, on="pid", suffixes=("_gt", "_pred"))

    # Get all columns we can evaluate
    gt_cols = [c for c in gt.columns if c != "pid"]
    pred_cols = [c for c in pred.columns if c != "pid"]
    shared_cols = sorted(set(gt_cols) & set(pred_cols))

    per_question = []
    for col in shared_cols:
        col_gt_name = f"{col}_gt" if f"{col}_gt" in merged.columns else col
        col_pred_name = f"{col}_pred" if f"{col}_pred" in merged.columns else col

        if col_gt_name not in merged.columns or col_pred_name not in merged.columns:
            continue

        gt_vals = pd.to_numeric(merged[col_gt_name], errors="coerce")
        pred_vals = pd.to_numeric(merged[col_pred_name], errors="coerce")

        valid = gt_vals.notna() & pred_vals.notna()
        if valid.sum() < 3:
            continue

        gt_v = gt_vals[valid].values
        pred_v = pred_vals[valid].values

        # Exact match accuracy
        accuracy = float(np.mean(gt_v == pred_v))

        # MAD
        mad = float(np.mean(np.abs(gt_v - pred_v)))

        # Correlation
        if np.std(gt_v) > 0 and np.std(pred_v) > 0:
            corr, p_val = stats.pearsonr(gt_v, pred_v)
        else:
            corr, p_val = float("nan"), float("nan")

        per_question.append({
            "column": col,
            "n_valid": int(valid.sum()),
            "accuracy": accuracy,
            "mad": mad,
            "correlation": float(corr),
            "p_value": float(p_val),
            "gt_mean": float(np.mean(gt_v)),
            "pred_mean": float(np.mean(pred_v)),
            "gt_std": float(np.std(gt_v)),
            "pred_std": float(np.std(pred_v)),
        })

    pq_df = pd.DataFrame(per_question)

    # Overall summary
    overall = {}
    if not pq_df.empty:
        overall = {
            "n_questions_evaluated": len(pq_df),
            "mean_accuracy": float(pq_df["accuracy"].mean()),
            "mean_mad": float(pq_df["mad"].mean()),
            "mean_correlation": float(pq_df["correlation"].dropna().mean()),
            "median_correlation": float(pq_df["correlation"].dropna().median()),
            "n_significant_correlations": int((pq_df["p_value"] < 0.05).sum()),
            "n_participants": len(merged),
        }

    return pq_df, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[
            OUTPUT_ROOT / "predictions" / "predictions_archetype.jsonl",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "evaluation",
    )
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth from HuggingFace...")
    gt = load_ground_truth()
    print(f"  Ground truth: {len(gt)} participants, {len(gt.columns)-1} columns")

    # Load question catalog
    catalog = json.load(QUESTION_CATALOG_JSON.open("r", encoding="utf-8"))
    questions = [
        q for q in catalog
        if "Economic preferences" in q.get("BlockName", "")
        and not q.get("is_descriptive", False)
        and q["QuestionType"] in ("MC", "Matrix")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for pred_path in args.predictions:
        if not pred_path.exists():
            print(f"Skipping {pred_path} (not found)")
            continue

        mode = pred_path.stem.replace("predictions_", "")
        print(f"\nEvaluating: {mode} ({pred_path})")

        pred = load_predictions(pred_path)
        if pred.empty:
            print("  No predictions found.")
            continue
        print(f"  Predictions: {len(pred)} participants, {len(pred.columns)-1} columns")

        pq_df, overall = compute_metrics(gt, pred, questions)
        overall["mode"] = mode

        # Save per-question results
        pq_path = args.output_dir / f"per_question_{mode}.csv"
        pq_df.to_csv(pq_path, index=False)
        print(f"  Per-question metrics → {pq_path}")

        # Save overall summary
        summary_path = args.output_dir / f"summary_{mode}.json"
        with summary_path.open("w") as f:
            json.dump(overall, f, indent=2)
            f.write("\n")
        print(f"  Summary → {summary_path}")

        # Print highlights
        print(f"  Overall: accuracy={overall.get('mean_accuracy', 0):.3f}, "
              f"MAD={overall.get('mean_mad', 0):.3f}, "
              f"correlation={overall.get('mean_correlation', 0):.3f}")

        all_summaries.append(overall)

    # Comparison table if multiple modes
    if len(all_summaries) > 1:
        comp_df = pd.DataFrame(all_summaries)
        comp_path = args.output_dir / "comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparison table → {comp_path}")
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
