#!/usr/bin/env python3
"""Compute random and human-consistency comparison baselines for the joint social benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from scipy import stats


REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
CONFIG = "wave_split"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_baseline"
    / "manifest_joint_social_baseline.jsonl"
)
LOCAL_WAVE_SPLIT = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "wave_split_dataset"
)
QUESTION_CATALOG_PATH = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "question_catalog.json"
)
INVENTORY_CSV = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "task_grounding"
    / "twin_question_inventory.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_baseline"
    / "comparison_baselines"
)

TRUST_RETURN_RECEIVED = {
    "QID118": 15,
    "QID119": 12,
    "QID120": 9,
    "QID121": 6,
    "QID122": 3,
}
ULTIMATUM_RECEIVER_AMOUNT = {
    "QID225": 5,
    "QID226": 4,
    "QID227": 3,
    "QID228": 2,
    "QID229": 1,
    "QID230": 0,
}
GENEROUS_QIDS = ["QID117", "QID224", "QID231"]
TASK_ORDER = ["trust", "ultimatum", "dictator"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_question_catalog(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inventory(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_wave_split():
    try:
        return load_dataset(REPO_ID, CONFIG)["data"]
    except Exception:
        if not LOCAL_WAVE_SPLIT.exists():
            raise
        return load_from_disk(str(LOCAL_WAVE_SPLIT))["data"]


def ref_for_parts(block_name: str, question_id: str) -> str:
    return f"{block_name}::{question_id}"


def option_count_for_question(question: Dict[str, Any]) -> Optional[int]:
    qtype = question.get("QuestionType")
    if qtype == "MC":
        options = question.get("Options", []) or []
        return len(options) if options else None
    if qtype == "Matrix":
        columns = question.get("Columns", []) or []
        return len(columns) if columns else None
    return None


def build_question_metadata(
    catalog_rows: Sequence[Dict[str, Any]],
    inventory_rows: Sequence[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], str]]:
    metadata_by_ref: Dict[str, Dict[str, Any]] = {}
    for question in catalog_rows:
        ref = ref_for_parts(str(question.get("BlockName", "")), str(question.get("QuestionID", "")))
        metadata_by_ref[ref] = {
            "block_name": str(question.get("BlockName", "")),
            "question_id": str(question.get("QuestionID", "")),
            "question_type": str(question.get("QuestionType", "")),
            "option_count": option_count_for_question(question),
        }

    ref_by_family_qid: Dict[Tuple[str, str], str] = {}
    for row in inventory_rows:
        ref_by_family_qid[(row["family"], row["question_id"])] = ref_for_parts(
            row["block_name"], row["question_id"]
        )
    return metadata_by_ref, ref_by_family_qid


def normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and not np.isnan(value):
        return int(round(value))
    return None


def find_question_map(example: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    blocks = json.loads(example["wave1_3_persona_json"])
    ref_to_question: Dict[str, Dict[str, Any]] = {}
    for block in blocks:
        block_name = block.get("BlockName", "")
        for question in block.get("Questions", []):
            ref = ref_for_parts(block_name, question.get("QuestionID", ""))
            ref_to_question[ref] = question
    return ref_to_question


def extract_mc_answer(question: Dict[str, Any]) -> Optional[int]:
    answers = question.get("Answers", {})
    value = answers.get("SelectedByPosition")
    if isinstance(value, list):
        if not value:
            return None
        value = next((item for item in value if item is not None), None)
    return normalize_int(value)


def question_range_from_option_count(option_count: Optional[int]) -> Optional[float]:
    if option_count is None:
        return None
    if option_count <= 1:
        return 0.0
    return float(option_count - 1)


def normalized_accuracy(pred: Optional[int], truth: Optional[int], option_count: Optional[int]) -> float:
    if pred is None or truth is None or option_count is None:
        return float("nan")
    q_range = question_range_from_option_count(option_count)
    if q_range is None:
        return float("nan")
    if q_range == 0:
        return 1.0 if pred == truth else 0.0
    score = 1.0 - (abs(pred - truth) / q_range)
    return float(max(0.0, min(1.0, score)))


def mean_ci_95(values: Iterable[float]) -> Tuple[float, float, float]:
    clean = pd.Series(list(values)).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, mean, mean
    sem = stats.sem(clean, nan_policy="omit")
    if not np.isfinite(sem) or sem == 0:
        return mean, mean, mean
    low, high = stats.t.interval(0.95, len(clean) - 1, loc=mean, scale=sem)
    return mean, float(low), float(high)


def random_expected_accuracy(truth: int, option_count: int) -> float:
    if option_count <= 0:
        return float("nan")
    scores = [
        normalized_accuracy(pred=pred, truth=truth, option_count=option_count)
        for pred in range(1, option_count + 1)
    ]
    return float(np.mean(scores))


def round_to_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def predict_generosity_from_other_games(answers: Dict[str, int], target_qid: str) -> Optional[int]:
    others = [answers[qid] for qid in GENEROUS_QIDS if qid != target_qid and qid in answers]
    if not others:
        return None
    return round_to_int(float(np.mean(others)), 1, 6)


def trust_return_amount_from_option(qid: str, option_index: int) -> int:
    received = TRUST_RETURN_RECEIVED[qid]
    return (received + 1) - option_index


def trust_option_from_return_amount(qid: str, returned: int) -> int:
    received = TRUST_RETURN_RECEIVED[qid]
    returned = max(0, min(received, returned))
    return (received + 1) - returned


def predict_trust_reciprocity(answers: Dict[str, int], target_qid: str) -> Optional[int]:
    rates: List[float] = []
    for qid, received in TRUST_RETURN_RECEIVED.items():
        if qid == target_qid:
            continue
        option = answers.get(qid)
        if option is None:
            continue
        returned = trust_return_amount_from_option(qid, option)
        rates.append(returned / received if received else 0.0)
    if not rates:
        return None
    target_received = TRUST_RETURN_RECEIVED[target_qid]
    predicted_return = round_to_int(float(np.mean(rates) * target_received), 0, target_received)
    return trust_option_from_return_amount(target_qid, predicted_return)


def infer_ultimatum_threshold(observed: Dict[str, int]) -> int:
    best_thresholds: List[int] = []
    best_error: Optional[int] = None
    for threshold in range(0, 7):
        errors = 0
        for qid, answer in observed.items():
            receiver_amt = ULTIMATUM_RECEIVER_AMOUNT[qid]
            pred = 1 if receiver_amt >= threshold else 2
            if pred != answer:
                errors += 1
        if best_error is None or errors < best_error:
            best_error = errors
            best_thresholds = [threshold]
        elif errors == best_error:
            best_thresholds.append(threshold)
    if not best_thresholds:
        return 6
    return int(round(float(np.mean(best_thresholds))))


def predict_ultimatum_acceptance(answers: Dict[str, int], target_qid: str) -> Optional[int]:
    observed = {
        qid: answers[qid]
        for qid in ULTIMATUM_RECEIVER_AMOUNT
        if qid != target_qid and qid in answers
    }
    if not observed:
        return None
    threshold = infer_ultimatum_threshold(observed)
    receiver_amt = ULTIMATUM_RECEIVER_AMOUNT[target_qid]
    return 1 if receiver_amt >= threshold else 2


def task_name_for_qid(manifest_example: Dict[str, Any], qid: str) -> str:
    mapping = manifest_example.get("target_family_to_qids", {})
    if isinstance(mapping, dict):
        for family, qids in mapping.items():
            if isinstance(qids, list) and qid in qids:
                return str(family)
    return "unknown"


def build_ground_truth_rows(
    ds,
    manifest_example: Dict[str, Any],
    ref_by_family_qid: Dict[Tuple[str, str], str],
    metadata_by_ref: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_qids = manifest_example["target_question_ids"]
    rows: List[Dict[str, Any]] = []

    for example in ds:
        pid = str(example["pid"])
        ref_to_question = find_question_map(example)
        answers_by_qid: Dict[str, int] = {}
        option_count_by_qid: Dict[str, Optional[int]] = {}
        for qid in target_qids:
            family = task_name_for_qid(manifest_example, qid)
            ref = ref_by_family_qid.get((family, qid))
            if not ref:
                continue
            question = ref_to_question.get(ref)
            if question is None:
                continue
            answer = extract_mc_answer(question)
            if answer is None:
                continue
            answers_by_qid[qid] = answer
            option_count_by_qid[qid] = metadata_by_ref.get(ref, {}).get("option_count")

        for qid, truth in answers_by_qid.items():
            task_name = task_name_for_qid(manifest_example, qid)
            option_count = option_count_by_qid[qid]
            rows.append(
                {
                    "pid": pid,
                    "task_name": task_name,
                    "question_id": qid,
                    "ground_truth": truth,
                    "option_count": option_count,
                    "all_social_answers": answers_by_qid,
                }
            )
    return rows


def add_random_baseline(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    random_rows: List[Dict[str, Any]] = []
    for row in rows:
        truth = row["ground_truth"]
        option_count = row["option_count"]
        acc = random_expected_accuracy(truth, option_count)
        random_rows.append(
            {
                "pid": row["pid"],
                "task_name": row["task_name"],
                "question_id": row["question_id"],
                "ground_truth": truth,
                "option_count": option_count,
                "baseline_name": "random_uniform_expected",
                "predicted": None,
                "accuracy": acc,
                "exact_match_rate": 1.0 / option_count if option_count else float("nan"),
            }
        )
    return pd.DataFrame(random_rows)


def add_human_proxy_baseline(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    proxy_rows: List[Dict[str, Any]] = []
    for row in rows:
        qid = row["question_id"]
        answers = row["all_social_answers"]
        truth = row["ground_truth"]
        option_count = row["option_count"]

        pred: Optional[int]
        if qid in GENEROUS_QIDS:
            pred = predict_generosity_from_other_games(answers, qid)
        elif qid in TRUST_RETURN_RECEIVED:
            pred = predict_trust_reciprocity(answers, qid)
        elif qid in ULTIMATUM_RECEIVER_AMOUNT:
            pred = predict_ultimatum_acceptance(answers, qid)
        else:
            pred = None

        proxy_rows.append(
            {
                "pid": row["pid"],
                "task_name": row["task_name"],
                "question_id": qid,
                "ground_truth": truth,
                "option_count": option_count,
                "baseline_name": "human_consistency_proxy",
                "predicted": pred,
                "accuracy": normalized_accuracy(pred, truth, option_count),
                "exact_match_rate": float(pred == truth) if pred is not None else float("nan"),
            }
        )
    return pd.DataFrame(proxy_rows)


def summarize_by_question(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["baseline_name", "task_name", "question_id"], as_index=False)
        .agg(
            n=("pid", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_exact_match_rate=("exact_match_rate", "mean"),
        )
        .sort_values(["baseline_name", "task_name", "question_id"])
    )
    return out


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
    by_baseline: Dict[str, Any] = {}
    for baseline_name, part in request_level.groupby("baseline_name"):
        mean_acc, ci_low, ci_high = mean_ci_95(part["task_accuracy"])
        by_baseline[str(baseline_name)] = {
            "n_participant_tasks": int(len(part)),
            "mean_task_accuracy": mean_acc,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return by_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute random and human-consistency comparison baselines for the joint social benchmark."
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.manifest_jsonl)
    if not manifest_rows:
        raise ValueError(f"No manifest rows found in {args.manifest_jsonl}")
    manifest_example = manifest_rows[0]

    catalog_rows = load_question_catalog(QUESTION_CATALOG_PATH)
    inventory_rows = load_inventory(INVENTORY_CSV)
    metadata_by_ref, ref_by_family_qid = build_question_metadata(catalog_rows, inventory_rows)
    ds = load_wave_split()

    ground_truth_rows = build_ground_truth_rows(
        ds=ds,
        manifest_example=manifest_example,
        ref_by_family_qid=ref_by_family_qid,
        metadata_by_ref=metadata_by_ref,
    )

    random_df = add_random_baseline(ground_truth_rows)
    proxy_df = add_human_proxy_baseline(ground_truth_rows)
    combined = pd.concat([random_df, proxy_df], ignore_index=True)

    question_df = summarize_by_question(combined)
    task_df = summarize_by_task(combined)
    overall = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "n_rows": int(len(combined)),
        "overall": summarize_overall(combined),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "comparison_baselines_raw.csv"
    question_path = args.output_dir / "comparison_baselines_per_question.csv"
    task_path = args.output_dir / "comparison_baselines_task_summary.csv"
    overall_path = args.output_dir / "comparison_baselines_overall.json"

    combined.to_csv(raw_path, index=False)
    question_df.to_csv(question_path, index=False)
    task_df.to_csv(task_path, index=False)
    overall_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {question_path}")
    print(f"Wrote {task_path}")
    print(f"Wrote {overall_path}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
