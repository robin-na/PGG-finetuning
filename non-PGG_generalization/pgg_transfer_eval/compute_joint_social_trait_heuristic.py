#!/usr/bin/env python3
"""Compute a deterministic trait-index baseline using only the participant's own allowed Twin inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from build_joint_social_baseline_batch import (
    INVENTORY_CSV,
    build_source_by_ref,
    load_inventory,
    load_question_catalog,
    ref_for_parts,
    select_allowed_and_target_refs,
)
from compute_joint_social_comparison_baselines import (
    DEFAULT_MANIFEST,
    TRUST_RETURN_RECEIVED,
    ULTIMATUM_RECEIVER_AMOUNT,
    build_question_metadata,
    mean_ci_95,
    normalized_accuracy,
)
from compute_joint_social_knn_baseline import (
    build_dataset_records,
    build_qid_task_maps,
    build_structured_feature_map,
    build_target_lookup,
    load_jsonl,
    load_wave_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_baseline"
    / "comparison_baselines"
    / "trait_heuristic"
)
GENEROUS_QIDS = ["QID117", "QID224", "QID231"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_to_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def centered_likert(value: Optional[float]) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return (float(value) - 3.0) / 2.0


def ratio_feature(value: Optional[float]) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return clamp(float(value), 0.0, 1.0) * 2.0 - 1.0


def first_switch_feature(value: Optional[float]) -> float:
    if value is None or math.isnan(value):
        return 0.0
    value = float(value)
    if value > 1.0:
        return -1.0
    return 1.0 - 2.0 * value


def get_centered(feature_map: Dict[str, float], key: str) -> float:
    return centered_likert(feature_map.get(key))


def get_ratio(feature_map: Dict[str, float], key: str) -> float:
    return ratio_feature(feature_map.get(key))


def get_first_switch(feature_map: Dict[str, float], key: str) -> float:
    return first_switch_feature(feature_map.get(key))


def mean(values: List[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def build_trait_indices(feature_map: Dict[str, float]) -> Dict[str, float]:
    prosociality = mean(
        [
            get_centered(feature_map, "QID25__helpful_unselfish"),
            get_centered(feature_map, "QID25__considerate_kind"),
            get_centered(feature_map, "QID25__likes_to_cooperate"),
            get_centered(feature_map, "QID25__forgiving"),
            get_centered(feature_map, "QID25__generally_trusting"),
            -get_centered(feature_map, "QID25__cold_aloof"),
            -get_centered(feature_map, "QID25__starts_quarrels"),
            -get_centered(feature_map, "QID25__rude_to_others"),
            get_centered(feature_map, "QID29__altruism_value"),
            get_centered(feature_map, "QID29__compassion_value"),
            get_centered(feature_map, "QID29__equality_value"),
            get_centered(feature_map, "QID29__harmony_value"),
            get_centered(feature_map, "QID29__honesty_value"),
            get_centered(feature_map, "QID29__humility_value"),
            get_centered(feature_map, "QID29__trust_value"),
            get_centered(feature_map, "QID232__feel_friend_sadness"),
            get_centered(feature_map, "QID232__understand_friend_happiness"),
            get_centered(feature_map, "QID232__caught_up_in_others_feelings"),
            get_centered(feature_map, "QID232__understand_when_down"),
            get_centered(feature_map, "QID232__infer_feelings_before_told"),
            get_centered(feature_map, "QID232__realize_friend_angry"),
            -get_centered(feature_map, "QID233__depend_on_self"),
            -get_centered(feature_map, "QID233__winning_is_everything"),
            -get_centered(feature_map, "QID233__competition_is_natural"),
            get_centered(feature_map, "QID233__coworker_prize_pride"),
            get_centered(feature_map, "QID233__coworker_wellbeing"),
            get_centered(feature_map, "QID233__pleasure_with_others"),
            get_centered(feature_map, "QID233__cooperate_feels_good"),
        ]
    )

    fairness_enforcement = mean(
        [
            get_centered(feature_map, "QID29__equality_value"),
            get_centered(feature_map, "QID29__honesty_value"),
            get_centered(feature_map, "QID29__politeness_value"),
            get_centered(feature_map, "QID29__trust_value"),
            get_centered(feature_map, "QID27__resentful_when_denied"),
            get_centered(feature_map, "QID27__get_even"),
            get_centered(feature_map, "QID27__jealous_of_others"),
            get_centered(feature_map, "QID27__irritated_by_favors"),
            -get_centered(feature_map, "QID27__always_courteous"),
            -get_centered(feature_map, "QID25__forgiving"),
            get_centered(feature_map, "QID25__starts_quarrels"),
            get_centered(feature_map, "QID25__rude_to_others"),
            get_centered(feature_map, "QID29__power_value"),
            get_centered(feature_map, "QID29__superiority_value"),
        ]
    )

    impatience = mean([get_ratio(feature_map, f"{qid}__share_right") for qid in ["QID84", "QID244", "QID245", "QID246", "QID247", "QID248"]])
    impatience_switch = mean([get_first_switch(feature_map, f"{qid}__first_right_norm") for qid in ["QID84", "QID244", "QID245", "QID246", "QID247", "QID248"]])
    risk_caution = mean([get_ratio(feature_map, f"{qid}__share_right") for qid in ["QID250", "QID251", "QID252", "QID276", "QID277", "QID278", "QID279"]])
    uncertainty_aversion = mean(
        [
            get_centered(feature_map, "QID238__dislike_uncertainty"),
            get_centered(feature_map, "QID238__prefer_order"),
            get_centered(feature_map, "QID238__relieved_after_decision"),
            get_centered(feature_map, "QID238__reach_solution_quickly"),
            get_centered(feature_map, "QID238__dislike_unpredictable"),
        ]
    )
    self_interest = mean(
        [
            get_centered(feature_map, "QID29__power_value"),
            get_centered(feature_map, "QID29__superiority_value"),
            get_centered(feature_map, "QID27__took_advantage"),
            get_centered(feature_map, "QID233__depend_on_self"),
            get_centered(feature_map, "QID233__winning_is_everything"),
            get_centered(feature_map, "QID233__competition_is_natural"),
            -get_centered(feature_map, "QID29__altruism_value"),
            -get_centered(feature_map, "QID29__compassion_value"),
        ]
    )

    caution_self_interest = mean([impatience, impatience_switch, risk_caution, uncertainty_aversion, self_interest])
    trustingness = mean(
        [
            get_centered(feature_map, "QID25__generally_trusting"),
            get_centered(feature_map, "QID29__trust_value"),
            get_centered(feature_map, "QID29__harmony_value"),
            get_centered(feature_map, "QID29__honesty_value"),
        ]
    )

    return {
        "prosociality": prosociality,
        "fairness_enforcement": fairness_enforcement,
        "caution_self_interest": caution_self_interest,
        "trustingness": trustingness,
    }


def predict_qid117(indices: Dict[str, float]) -> int:
    score = 4.1 - 1.2 * indices["prosociality"] - 0.5 * indices["trustingness"] + 0.7 * indices["caution_self_interest"]
    return round_to_int(score, 1, 6)


def predict_trust_return_rate(indices: Dict[str, float]) -> float:
    rate = 0.47 + 0.16 * indices["prosociality"] + 0.12 * indices["trustingness"] - 0.08 * indices["caution_self_interest"]
    return clamp(rate, 0.05, 0.95)


def predict_trust_qid(qid: str, indices: Dict[str, float]) -> int:
    received = TRUST_RETURN_RECEIVED[qid]
    base_rate = predict_trust_return_rate(indices)
    amount_adjust = 0.03 * ((received - 9.0) / 6.0)
    fairness_adjust = 0.04 * indices["fairness_enforcement"]
    rate = clamp(base_rate + amount_adjust + fairness_adjust, 0.05, 0.95)
    returned = round_to_int(rate * received, 0, received)
    return (received + 1) - returned


def predict_qid224(indices: Dict[str, float]) -> int:
    score = 4.2 - 0.9 * indices["prosociality"] - 0.4 * indices["fairness_enforcement"] + 0.6 * indices["caution_self_interest"]
    return round_to_int(score, 1, 6)


def predict_ultimatum_receiver(qid: str, indices: Dict[str, float]) -> int:
    receiver_amt = ULTIMATUM_RECEIVER_AMOUNT[qid]
    threshold = 1.7 + 1.0 * indices["fairness_enforcement"] - 0.8 * indices["prosociality"] - 0.5 * indices["caution_self_interest"]
    threshold = clamp(threshold, 0.0, 5.0)
    return 1 if receiver_amt >= threshold else 2


def predict_qid231(indices: Dict[str, float]) -> int:
    score = 4.35 - 0.9 * indices["prosociality"] - 0.3 * indices["fairness_enforcement"] + 0.8 * indices["caution_self_interest"]
    return round_to_int(score, 1, 6)


def predict_question(qid: str, indices: Dict[str, float]) -> Optional[int]:
    if qid == "QID117":
        return predict_qid117(indices)
    if qid in TRUST_RETURN_RECEIVED:
        return predict_trust_qid(qid, indices)
    if qid == "QID224":
        return predict_qid224(indices)
    if qid in ULTIMATUM_RECEIVER_AMOUNT:
        return predict_ultimatum_receiver(qid, indices)
    if qid == "QID231":
        return predict_qid231(indices)
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
        description="Compute a deterministic trait-index heuristic baseline using only allowed Twin inputs."
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = load_jsonl(args.manifest_jsonl)
    if not manifest_rows:
        raise ValueError(f"No manifest rows found in {args.manifest_jsonl}")
    manifest_example = manifest_rows[0]
    eval_pids = {str(row["pid"]) for row in manifest_rows if row.get("pid") is not None}

    catalog_rows = load_question_catalog()
    inventory_rows = load_inventory(INVENTORY_CSV)
    source_by_ref = build_source_by_ref(catalog_rows)
    _, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    metadata_by_ref, _ = build_question_metadata(catalog_rows, inventory_rows)
    target_qids, qid_to_task = build_qid_task_maps(manifest_example)
    target_lookup = build_target_lookup(target_rows, qid_to_task)

    ds = load_wave_split()
    records = build_dataset_records(ds=ds, feature_pool_pids=eval_pids, target_lookup=target_lookup)
    if not records:
        raise ValueError("No records found for trait heuristic baseline.")

    raw_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    for record in records:
        pid = record["pid"]
        answers = record["target_answers"]
        indices = build_trait_indices(record["features"])
        index_rows.append({"pid": pid, **indices})
        for qid in target_qids:
            truth = answers.get(qid)
            if truth is None:
                continue
            pred = predict_question(qid, indices)
            option_count = metadata_by_ref[target_lookup[qid]]["option_count"]
            raw_rows.append(
                {
                    "pid": pid,
                    "task_name": qid_to_task.get(qid, "unknown"),
                    "question_id": qid,
                    "ground_truth": truth,
                    "option_count": option_count,
                    "baseline_name": "trait_index_heuristic",
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
        "n_rows": int(len(raw_df)),
        "n_eval_pids": int(len(index_rows)),
        "overall": summarize_overall(raw_df),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "trait_index_heuristic_raw.csv"
    question_path = args.output_dir / "trait_index_heuristic_per_question.csv"
    task_path = args.output_dir / "trait_index_heuristic_task_summary.csv"
    overall_path = args.output_dir / "trait_index_heuristic_overall.json"
    index_path = args.output_dir / "trait_index_heuristic_indices.csv"

    raw_df.to_csv(raw_path, index=False)
    question_df.to_csv(question_path, index=False)
    task_df.to_csv(task_path, index=False)
    pd.DataFrame(index_rows).sort_values("pid").to_csv(index_path, index=False)
    overall_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {question_path}")
    print(f"Wrote {task_path}")
    print(f"Wrote {index_path}")
    print(f"Wrote {overall_path}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
