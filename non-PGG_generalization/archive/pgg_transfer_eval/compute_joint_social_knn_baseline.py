#!/usr/bin/env python3
"""Compute a kNN human-neighbor baseline using the same structured Twin inputs as the tuned prompt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from build_joint_social_baseline_batch import (
    INVENTORY_CSV,
    MENTAL_ACCOUNTING_QIDS,
    RISK_GAIN_QIDS,
    RISK_LOSS_QIDS,
    STRUCTURED_Q232_ROWS,
    STRUCTURED_Q233_ROWS,
    STRUCTURED_Q236_ROWS,
    STRUCTURED_Q238_ROWS,
    STRUCTURED_Q25_ROWS,
    STRUCTURED_Q27_ROWS,
    STRUCTURED_Q29_ROWS,
    TIME_PREFERENCE_QIDS,
    build_matrix_answer_map,
    build_source_by_ref,
    find_question_map,
    load_inventory,
    load_question_catalog,
    normalize_whitespace,
    ref_for_parts,
    ref_for_row,
    select_allowed_and_target_refs,
)
from compute_joint_social_comparison_baselines import (
    DEFAULT_MANIFEST,
    TASK_ORDER,
    TRUST_RETURN_RECEIVED,
    ULTIMATUM_RECEIVER_AMOUNT,
    build_question_metadata,
    extract_mc_answer,
    infer_ultimatum_threshold,
    load_jsonl,
    load_wave_split,
    mean_ci_95,
    normalized_accuracy,
    round_to_int,
    trust_option_from_return_amount,
    trust_return_amount_from_option,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_baseline"
    / "comparison_baselines"
    / "knn_allowed_input"
)
GENEROUS_QIDS = ["QID117", "QID224", "QID231"]


def add_named_matrix_features(
    feature_map: Dict[str, float],
    question: Optional[Dict[str, Any]],
    named_rows: Sequence[Tuple[str, str]],
    prefix: str,
) -> None:
    answer_map = build_matrix_answer_map(question)
    if not answer_map:
        return
    for short_name, row_text in named_rows:
        value = answer_map.get(normalize_whitespace(row_text))
        if value is None:
            continue
        feature_map[f"{prefix}__{short_name}"] = float(value)


def add_mc_feature(
    feature_map: Dict[str, float],
    question: Optional[Dict[str, Any]],
    prefix: str,
) -> None:
    if not question or question.get("QuestionType") != "MC":
        return
    value = extract_mc_answer(question)
    if value is None:
        return
    feature_map[prefix] = float(value)


def add_switch_summary_features(
    feature_map: Dict[str, float],
    question: Optional[Dict[str, Any]],
    prefix: str,
) -> None:
    if not question or question.get("QuestionType") != "Matrix":
        return
    values = question.get("Answers", {}).get("SelectedByPosition", []) or []
    clean = [int(v) for v in values if v is not None]
    if not clean:
        return
    n_rows = len(clean)
    right_count = sum(1 for v in clean if v == 2)
    first_right = next((idx for idx, v in enumerate(clean, start=1) if v == 2), None)
    if first_right is None:
        first_right_norm = (n_rows + 1) / n_rows
    else:
        first_right_norm = first_right / n_rows
    feature_map[f"{prefix}__share_right"] = right_count / n_rows
    feature_map[f"{prefix}__first_right_norm"] = first_right_norm


def build_structured_feature_map(ref_to_question: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    qid_to_question = {
        str(question.get("QuestionID", "")): question
        for question in ref_to_question.values()
    }
    features: Dict[str, float] = {}

    add_named_matrix_features(features, qid_to_question.get("QID25"), STRUCTURED_Q25_ROWS, "QID25")
    add_named_matrix_features(features, qid_to_question.get("QID27"), STRUCTURED_Q27_ROWS, "QID27")
    add_named_matrix_features(features, qid_to_question.get("QID29"), STRUCTURED_Q29_ROWS, "QID29")
    add_named_matrix_features(features, qid_to_question.get("QID232"), STRUCTURED_Q232_ROWS, "QID232")
    add_named_matrix_features(features, qid_to_question.get("QID233"), STRUCTURED_Q233_ROWS, "QID233")
    add_named_matrix_features(features, qid_to_question.get("QID236"), STRUCTURED_Q236_ROWS, "QID236")
    add_named_matrix_features(features, qid_to_question.get("QID238"), STRUCTURED_Q238_ROWS, "QID238")

    for qid in MENTAL_ACCOUNTING_QIDS:
        add_mc_feature(features, qid_to_question.get(qid), qid)
    for qid in TIME_PREFERENCE_QIDS:
        add_switch_summary_features(features, qid_to_question.get(qid), qid)
    for qid in RISK_GAIN_QIDS:
        add_switch_summary_features(features, qid_to_question.get(qid), qid)
    for qid in RISK_LOSS_QIDS:
        add_switch_summary_features(features, qid_to_question.get(qid), qid)

    return features


def build_qid_task_maps(manifest_example: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    target_qids = list(manifest_example["target_question_ids"])
    qid_to_task: Dict[str, str] = {}
    mapping = manifest_example.get("target_family_to_qids", {})
    if isinstance(mapping, dict):
        for task_name, qids in mapping.items():
            if not isinstance(qids, list):
                continue
            for qid in qids:
                qid_to_task[str(qid)] = str(task_name)
    return target_qids, qid_to_task


def build_target_lookup(
    target_rows: Sequence[Dict[str, str]],
    qid_to_task: Dict[str, str],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in target_rows:
        qid = row["question_id"]
        if qid not in qid_to_task:
            continue
        lookup[qid] = ref_for_row(row)
    return lookup


def build_dataset_records(
    ds,
    feature_pool_pids: Optional[set[str]],
    target_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for example in ds:
        pid = str(example["pid"])
        if feature_pool_pids is not None and pid not in feature_pool_pids:
            continue
        ref_to_question = find_question_map(example)
        feature_map = build_structured_feature_map(ref_to_question)
        if not feature_map:
            continue

        target_answers: Dict[str, int] = {}
        for qid, ref in target_lookup.items():
            question = ref_to_question.get(ref)
            if question is None:
                continue
            answer = extract_mc_answer(question)
            if answer is None:
                continue
            target_answers[qid] = answer

        if not target_answers:
            continue

        records.append(
            {
                "pid": pid,
                "features": feature_map,
                "target_answers": target_answers,
            }
        )
    return records


def build_feature_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = {"pid": record["pid"]}
        row.update(record["features"])
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("pid").sort_index()
    frame = frame.astype(float)
    frame = frame.dropna(axis=1, how="all")
    frame = frame.fillna(frame.mean())
    return frame


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=0)
    stds = frame.std(axis=0, ddof=0).replace(0, 1.0)
    return (frame - means) / stds


def compute_neighbor_lists(
    feature_frame: pd.DataFrame,
    eval_pids: Sequence[str],
    k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    pids = list(feature_frame.index)
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
    matrix = feature_frame.to_numpy(dtype=float)
    neighbors_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for pid in eval_pids:
        if pid not in pid_to_idx:
            neighbors_by_pid[pid] = []
            continue
        idx = pid_to_idx[pid]
        diff = matrix - matrix[idx]
        dist = np.linalg.norm(diff, axis=1)
        mask = np.arange(len(pids)) != idx
        candidate_idxs = np.where(mask)[0]
        candidate_idxs = candidate_idxs[np.argsort(dist[candidate_idxs])]
        top_idxs = candidate_idxs[:k]
        neighbor_rows: List[Dict[str, Any]] = []
        for nbr_idx in top_idxs:
            distance = float(dist[nbr_idx])
            weight = 1.0 / (distance + 1e-6)
            neighbor_rows.append(
                {
                    "pid": pids[nbr_idx],
                    "distance": distance,
                    "weight": weight,
                }
            )
        total_weight = sum(row["weight"] for row in neighbor_rows)
        if total_weight > 0:
            for row in neighbor_rows:
                row["weight_normalized"] = row["weight"] / total_weight
        neighbors_by_pid[pid] = neighbor_rows
    return neighbors_by_pid


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total = float(sum(weights))
    if total <= 0:
        return None
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def predict_generosity_from_neighbors(
    target_qid: str,
    neighbors: Sequence[Dict[str, Any]],
    answers_by_pid: Dict[str, Dict[str, int]],
) -> Optional[int]:
    values: List[float] = []
    weights: List[float] = []
    for neighbor in neighbors:
        answer = answers_by_pid.get(neighbor["pid"], {}).get(target_qid)
        if answer is None:
            continue
        values.append(float(answer))
        weights.append(float(neighbor["weight"]))
    mean = weighted_mean(values, weights)
    if mean is None:
        return None
    return round_to_int(mean, 1, 6)


def predict_trust_return_from_neighbors(
    target_qid: str,
    neighbors: Sequence[Dict[str, Any]],
    answers_by_pid: Dict[str, Dict[str, int]],
) -> Optional[int]:
    received = TRUST_RETURN_RECEIVED[target_qid]
    rates: List[float] = []
    weights: List[float] = []
    for neighbor in neighbors:
        answer = answers_by_pid.get(neighbor["pid"], {}).get(target_qid)
        if answer is None:
            continue
        returned = trust_return_amount_from_option(target_qid, answer)
        rates.append(returned / received if received else 0.0)
        weights.append(float(neighbor["weight"]))
    mean_rate = weighted_mean(rates, weights)
    if mean_rate is None:
        return None
    predicted_return = round_to_int(mean_rate * received, 0, received)
    return trust_option_from_return_amount(target_qid, predicted_return)


def predict_ultimatum_receiver_from_neighbors(
    target_qid: str,
    neighbors: Sequence[Dict[str, Any]],
    answers_by_pid: Dict[str, Dict[str, int]],
) -> Optional[int]:
    thresholds: List[float] = []
    weights: List[float] = []
    for neighbor in neighbors:
        answers = answers_by_pid.get(neighbor["pid"], {})
        observed = {
            qid: answers[qid]
            for qid in ULTIMATUM_RECEIVER_AMOUNT
            if qid in answers
        }
        if not observed:
            continue
        thresholds.append(float(infer_ultimatum_threshold(observed)))
        weights.append(float(neighbor["weight"]))
    threshold = weighted_mean(thresholds, weights)
    if threshold is None:
        return None
    receiver_amt = ULTIMATUM_RECEIVER_AMOUNT[target_qid]
    return 1 if receiver_amt >= threshold else 2


def predict_from_neighbors(
    qid: str,
    neighbors: Sequence[Dict[str, Any]],
    answers_by_pid: Dict[str, Dict[str, int]],
) -> Optional[int]:
    if qid in GENEROUS_QIDS:
        return predict_generosity_from_neighbors(qid, neighbors, answers_by_pid)
    if qid in TRUST_RETURN_RECEIVED:
        return predict_trust_return_from_neighbors(qid, neighbors, answers_by_pid)
    if qid in ULTIMATUM_RECEIVER_AMOUNT:
        return predict_ultimatum_receiver_from_neighbors(qid, neighbors, answers_by_pid)
    return None


def summarize_by_question(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["baseline_name", "task_name", "question_id"], as_index=False)
        .agg(
            n=("pid", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_exact_match_rate=("exact_match_rate", "mean"),
        )
        .sort_values(["baseline_name", "task_name", "question_id"])
    )


def summarize_by_task(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    request_level = (
        df.groupby(["baseline_name", "pid", "task_name"], as_index=False)
        .agg(
            task_accuracy=("accuracy", "mean"),
            task_exact_match_rate=("exact_match_rate", "mean"),
            n_questions=("question_id", "size"),
        )
    )
    rows: List[Dict[str, Any]] = []
    for (baseline_name, task_name), part in request_level.groupby(["baseline_name", "task_name"]):
        mean_acc, ci_low, ci_high = mean_ci_95(part["task_accuracy"])
        rows.append(
            {
                "baseline_name": baseline_name,
                "task_name": task_name,
                "n_requests": int(part["pid"].nunique()),
                "mean_task_accuracy": mean_acc,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_task_exact_match_rate": float(part["task_exact_match_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["baseline_name", "task_name"])


def summarize_overall(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    request_level = (
        df.groupby(["baseline_name", "pid", "task_name"], as_index=False)
        .agg(task_accuracy=("accuracy", "mean"))
    )
    out: Dict[str, Any] = {}
    for baseline_name, part in request_level.groupby("baseline_name"):
        mean_acc, ci_low, ci_high = mean_ci_95(part["task_accuracy"])
        out[str(baseline_name)] = {
            "n_participant_tasks": int(len(part)),
            "mean_task_accuracy": mean_acc,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a leave-one-out kNN human-neighbor baseline using the tuned structured Twin inputs."
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--neighbor-source",
        choices=["manifest", "dataset"],
        default="manifest",
        help="Whether to search neighbors only within the manifest subset or across the full dataset.",
    )
    parser.add_argument("--k", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    manifest_rows = load_jsonl(args.manifest_jsonl)
    if not manifest_rows:
        raise ValueError(f"No manifest rows found in {args.manifest_jsonl}")
    manifest_example = manifest_rows[0]
    eval_pids = [str(row["pid"]) for row in manifest_rows if row.get("pid") is not None]
    eval_pid_set = set(eval_pids)

    catalog_rows = load_question_catalog()
    inventory_rows = load_inventory(INVENTORY_CSV)
    source_by_ref = build_source_by_ref(catalog_rows)
    _, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    metadata_by_ref, _ = build_question_metadata(catalog_rows, inventory_rows)
    target_qids, qid_to_task = build_qid_task_maps(manifest_example)
    target_lookup = build_target_lookup(target_rows, qid_to_task)

    if args.neighbor_source == "manifest":
        feature_pool_pids: Optional[set[str]] = eval_pid_set
    else:
        feature_pool_pids = None

    ds = load_wave_split()
    records = build_dataset_records(
        ds=ds,
        feature_pool_pids=feature_pool_pids,
        target_lookup=target_lookup,
    )
    if not records:
        raise ValueError("No dataset records with structured features were built.")

    answers_by_pid = {record["pid"]: record["target_answers"] for record in records}
    feature_frame = build_feature_frame(records)
    feature_frame = standardize_frame(feature_frame)
    neighbors_by_pid = compute_neighbor_lists(feature_frame, eval_pids, args.k)

    raw_rows: List[Dict[str, Any]] = []
    neighbor_trace_rows: List[Dict[str, Any]] = []
    for pid in eval_pids:
        answers = answers_by_pid.get(pid, {})
        if not answers:
            continue
        neighbors = neighbors_by_pid.get(pid, [])
        neighbor_trace_rows.append(
            {
                "pid": pid,
                "neighbor_source": args.neighbor_source,
                "k": args.k,
                "neighbors": neighbors,
            }
        )
        for qid in target_qids:
            task_name = qid_to_task.get(qid, "unknown")
            truth = answers.get(qid)
            if truth is None:
                continue
            option_count = metadata_by_ref[target_lookup[qid]]["option_count"]
            pred = predict_from_neighbors(qid, neighbors, answers_by_pid)
            raw_rows.append(
                {
                    "pid": pid,
                    "task_name": task_name,
                    "question_id": qid,
                    "ground_truth": truth,
                    "option_count": option_count,
                    "baseline_name": "knn_allowed_input",
                    "predicted": pred,
                    "accuracy": normalized_accuracy(pred, truth, option_count),
                    "exact_match_rate": float(pred == truth) if pred is not None else float("nan"),
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    question_df = summarize_by_question(raw_df)
    task_df = summarize_by_task(raw_df)
    overall = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "neighbor_source": args.neighbor_source,
        "k": args.k,
        "n_rows": int(len(raw_df)),
        "n_feature_pool_pids": int(len(feature_frame)),
        "n_eval_pids": int(len(eval_pids)),
        "overall": summarize_overall(raw_df),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "knn_allowed_input_raw.csv"
    question_path = args.output_dir / "knn_allowed_input_per_question.csv"
    task_path = args.output_dir / "knn_allowed_input_task_summary.csv"
    overall_path = args.output_dir / "knn_allowed_input_overall.json"
    neighbors_path = args.output_dir / "knn_allowed_input_neighbors.jsonl"

    raw_df.to_csv(raw_path, index=False)
    question_df.to_csv(question_path, index=False)
    task_df.to_csv(task_path, index=False)
    overall_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    with neighbors_path.open("w", encoding="utf-8") as f:
        for row in neighbor_trace_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {raw_path}")
    print(f"Wrote {question_path}")
    print(f"Wrote {task_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {neighbors_path}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
