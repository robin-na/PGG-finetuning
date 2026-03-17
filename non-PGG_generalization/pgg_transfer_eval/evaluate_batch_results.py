#!/usr/bin/env python3
"""Convert OpenAI Batch output JSONL into scored evals for the PGG-transfer benchmark."""

from __future__ import annotations

import argparse
import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], str], Dict[str, str]]:
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
    unique_ref_by_qid: Dict[str, str] = {}
    qid_counts: Dict[str, int] = {}
    for row in inventory_rows:
        ref = ref_for_parts(row["block_name"], row["question_id"])
        key = (row["family"], row["question_id"])
        ref_by_family_qid[key] = ref
        qid_counts[row["question_id"]] = qid_counts.get(row["question_id"], 0) + 1
        unique_ref_by_qid[row["question_id"]] = ref

    unique_ref_by_qid = {
        qid: ref
        for qid, ref in unique_ref_by_qid.items()
        if qid_counts.get(qid, 0) == 1
    }
    return metadata_by_ref, ref_by_family_qid, unique_ref_by_qid


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def extract_balanced_json_object(text: str, key: str) -> Optional[Dict[str, Any]]:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\{{', re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None

    brace_start = match.end() - 1
    depth = 0
    in_string = False
    escaped = False

    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


def recover_partial_json_object(text: str) -> Optional[Dict[str, Any]]:
    answers = extract_balanced_json_object(text, "answers")
    if answers is None:
        return None

    recovered: Dict[str, Any] = {"answers": answers}
    reasoning = extract_balanced_json_object(text, "reasoning")
    if reasoning is not None:
        recovered["reasoning"] = reasoning
    return recovered


def extract_content(response_row: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    error = response_row.get("error")
    if error:
        return None, {"type": "batch_error", "detail": error}, None

    response = response_row.get("response", {})
    if response.get("status_code") != 200:
        return (
            None,
            {
                "type": "http_error",
                "status_code": response.get("status_code"),
                "body": response.get("body"),
            },
            None,
        )

    body = response.get("body", {})
    if isinstance(body, dict) and body.get("object") == "response":
        traces: List[Dict[str, Any]] = []
        output_items = body.get("output", []) or []
        text_parts: List[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "file_search_call":
                traces.append(item)
                continue
            if item_type == "message":
                for content_part in item.get("content", []) or []:
                    if isinstance(content_part, dict) and content_part.get("type") == "output_text":
                        text = content_part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts), None, traces or None
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text, None, traces or None
        return None, {"type": "missing_responses_output_text", "body": body}, traces or None

    choices = body.get("choices", [])
    if not choices:
        return None, {"type": "missing_choices", "body": body}, None

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content, None, None
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "\n".join(text_parts), None, None
    return None, {"type": "missing_content", "message": message}, None


def normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not np.isnan(value):
        return int(round(value))
    if isinstance(value, str):
        nums = re.findall(r"-?\d+", value)
        if nums:
            return int(nums[0])
    return None


def extract_answers(obj: Dict[str, Any], expected_qids: Iterable[str]) -> Dict[str, Optional[int]]:
    if "answers" in obj and isinstance(obj["answers"], dict):
        raw_answers = obj["answers"]
    else:
        raw_answers = obj
    answers: Dict[str, Optional[int]] = {}
    for qid in expected_qids:
        answers[qid] = normalize_int(raw_answers.get(qid))
    return answers


def question_family_from_manifest(manifest: Dict[str, Any], qid: str) -> str:
    mapping = manifest.get("target_family_to_qids")
    if isinstance(mapping, dict):
        for family, qids in mapping.items():
            if isinstance(qids, list) and qid in qids:
                return str(family)
    return str(manifest.get("target_family", "unknown"))


def question_ref_from_manifest(
    manifest: Dict[str, Any],
    qid: str,
    ref_by_family_qid: Dict[Tuple[str, str], str],
    unique_ref_by_qid: Dict[str, str],
) -> Optional[str]:
    refs = manifest.get("excluded_target_refs")
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str) and ref.endswith(f"::{qid}"):
                return ref
    family = question_family_from_manifest(manifest, qid)
    ref = ref_by_family_qid.get((family, qid))
    if ref:
        return ref
    return unique_ref_by_qid.get(qid)


def question_range_from_option_count(option_count: Optional[int]) -> Optional[float]:
    if option_count is None:
        return None
    if option_count <= 1:
        return 0.0
    return float(option_count - 1)


def normalized_accuracy(
    pred: Optional[int],
    truth: Optional[int],
    option_count: Optional[int],
) -> float:
    if pred is None or truth is None or option_count is None:
        return float("nan")
    q_range = question_range_from_option_count(option_count)
    if q_range is None:
        return float("nan")
    if q_range == 0:
        return 1.0 if pred == truth else 0.0
    score = 1.0 - (abs(pred - truth) / q_range)
    return float(max(0.0, min(1.0, score)))


def mean_ci_95(values: pd.Series) -> Tuple[float, float, float]:
    clean = pd.Series(values).dropna().astype(float)
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


def summarize_request_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(
        ["custom_id", "pid", "target_family", "condition", "model"], as_index=False
    ).agg(
        n_questions=("question_id", "size"),
        n_answered=("predicted", lambda s: int(pd.Series(s).notna().sum())),
        block_accuracy=("normalized_accuracy", "mean"),
        block_exact_match_accuracy=("exact_match", "mean"),
        block_mad=("abs_error", "mean"),
    )
    grouped["block_all_exact_match"] = grouped["block_exact_match_accuracy"] == 1.0
    return grouped


def per_question_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (task_name, qid), part in df.groupby(["task_name", "question_id"]):
        valid = part.dropna(subset=["predicted", "ground_truth"])
        if valid.empty:
            continue
        gt = valid["ground_truth"].astype(float).to_numpy()
        pred = valid["predicted"].astype(float).to_numpy()
        corr = float("nan")
        p_value = float("nan")
        if len(valid) >= 3 and np.std(gt) > 0 and np.std(pred) > 0:
            corr, p_value = stats.pearsonr(gt, pred)
        rows.append(
            {
                "task_name": task_name,
                "question_id": qid,
                "n": len(valid),
                "accuracy": float(valid["normalized_accuracy"].mean()),
                "exact_match_rate": float(valid["exact_match"].mean()),
                "mad": float(np.mean(np.abs(gt - pred))),
                "correlation": float(corr),
                "p_value": float(p_value),
                "gt_mean": float(np.mean(gt)),
                "pred_mean": float(np.mean(pred)),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_name", "question_id"])


def summarize_task_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(
        ["custom_id", "pid", "task_name", "condition", "model"], as_index=False
    ).agg(
        n_questions=("question_id", "size"),
        n_answered=("predicted", lambda s: int(pd.Series(s).notna().sum())),
        task_accuracy=("normalized_accuracy", "mean"),
        task_exact_match_accuracy=("exact_match", "mean"),
        task_mad=("abs_error", "mean"),
    )
    grouped["task_all_exact_match"] = grouped["task_exact_match_accuracy"] == 1.0
    return grouped


def task_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (task_name, condition, model), part in task_df.groupby(["task_name", "condition", "model"]):
        mean_acc, ci_low, ci_high = mean_ci_95(part["task_accuracy"])
        rows.append(
            {
                "task_name": task_name,
                "condition": condition,
                "model": model,
                "n_requests": int(part["custom_id"].nunique()),
                "mean_task_accuracy": mean_acc,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_task_exact_match_accuracy": float(part["task_exact_match_accuracy"].mean()),
                "task_all_exact_match_rate": float(part["task_all_exact_match"].mean()),
                "mean_task_mad": float(part["task_mad"].mean()),
                "mean_questions_answered": float(part["n_answered"].mean()),
            }
        )
    out = (
        pd.DataFrame(rows)
        .sort_values(["task_name", "condition", "model"])
        .reset_index(drop=True)
    )
    return out


def family_summary(request_df: pd.DataFrame) -> pd.DataFrame:
    if request_df.empty:
        return pd.DataFrame()
    out = (
        request_df.groupby(["target_family", "condition", "model"], as_index=False)
        .agg(
            n_requests=("custom_id", "size"),
            mean_block_accuracy=("block_accuracy", "mean"),
            mean_block_exact_match_accuracy=("block_exact_match_accuracy", "mean"),
            mean_block_mad=("block_mad", "mean"),
            block_all_exact_match_rate=("block_all_exact_match", "mean"),
            mean_questions_answered=("n_answered", "mean"),
        )
        .sort_values(["target_family", "condition", "model"])
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI Batch outputs for the PGG-transfer benchmark.")
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.responses_jsonl.resolve().parent / "evals"

    manifest_rows = load_jsonl(args.manifest_jsonl)
    response_rows = load_jsonl(args.responses_jsonl)
    manifest_by_id = {row["custom_id"]: row for row in manifest_rows}
    catalog_rows = load_question_catalog(QUESTION_CATALOG_PATH)
    inventory_rows = load_inventory(INVENTORY_CSV)
    metadata_by_ref, ref_by_family_qid, unique_ref_by_qid = build_question_metadata(
        catalog_rows=catalog_rows,
        inventory_rows=inventory_rows,
    )

    parsed_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    retrieval_trace_rows: List[Dict[str, Any]] = []
    usage_prompt_tokens = 0
    usage_completion_tokens = 0

    for response_row in response_rows:
        custom_id = response_row.get("custom_id")
        manifest = manifest_by_id.get(custom_id)
        if not manifest:
            error_rows.append(
                {
                    "custom_id": custom_id,
                    "error_type": "missing_manifest",
                    "raw_row": response_row,
                }
            )
            continue

        content, error, retrieval_traces = extract_content(response_row)
        response_body = response_row.get("response", {}).get("body", {})
        usage = response_body.get("usage", {}) if isinstance(response_body, dict) else {}
        usage_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        usage_completion_tokens += int(usage.get("completion_tokens", 0) or 0)

        if error:
            error_rows.append(
                {
                    "custom_id": custom_id,
                    "pid": manifest["pid"],
                    "target_family": manifest["target_family"],
                    "error_type": error.get("type"),
                    "detail": error,
                }
            )
            continue

        if retrieval_traces:
            for trace in retrieval_traces:
                retrieval_trace_rows.append(
                    {
                        "custom_id": custom_id,
                        "pid": manifest["pid"],
                        "target_family": manifest["target_family"],
                        "queries": trace.get("queries"),
                        "status": trace.get("status"),
                        "results": trace.get("results"),
                    }
                )

        parsed_obj = parse_json_object(content or "")
        recovered_from_partial = False
        if parsed_obj is None:
            parsed_obj = recover_partial_json_object(content or "")
            recovered_from_partial = parsed_obj is not None
        if parsed_obj is None:
            error_rows.append(
                {
                    "custom_id": custom_id,
                    "pid": manifest["pid"],
                    "target_family": manifest["target_family"],
                    "error_type": "invalid_json",
                    "raw_content": content,
                }
            )
            continue

        expected_qids = manifest["target_question_ids"]
        pred_answers = extract_answers(parsed_obj, expected_qids)
        reasoning = parsed_obj.get("reasoning") if isinstance(parsed_obj.get("reasoning"), dict) else {}

        for qid in expected_qids:
            truth = normalize_int(manifest["ground_truth_answers"].get(qid))
            pred = pred_answers.get(qid)
            task_name = question_family_from_manifest(manifest, qid)
            question_ref = question_ref_from_manifest(
                manifest=manifest,
                qid=qid,
                ref_by_family_qid=ref_by_family_qid,
                unique_ref_by_qid=unique_ref_by_qid,
            )
            question_meta = metadata_by_ref.get(question_ref or "", {})
            option_count = question_meta.get("option_count")
            score = normalized_accuracy(pred, truth, option_count)
            parsed_rows.append(
                {
                    "custom_id": custom_id,
                    "pid": manifest["pid"],
                    "target_family": manifest["target_family"],
                    "task_name": task_name,
                    "condition": manifest["condition"],
                    "model": manifest["model"],
                    "question_id": qid,
                    "question_ref": question_ref,
                    "option_count": option_count,
                    "question_range": question_range_from_option_count(option_count),
                    "ground_truth": truth,
                    "predicted": pred,
                    "exact_match": float(pred == truth) if pred is not None and truth is not None else np.nan,
                    "abs_error": abs(pred - truth) if pred is not None and truth is not None else np.nan,
                    "normalized_accuracy": score,
                    "reasoning": reasoning.get(qid),
                    "recovered_from_partial_json": recovered_from_partial,
                    "raw_content": content,
                }
            )

    parsed_df = pd.DataFrame(parsed_rows)
    request_df = summarize_request_level(parsed_df)
    task_request_df = summarize_task_level(parsed_df)
    question_df = per_question_metrics(parsed_df)
    task_df = task_summary(task_request_df)
    family_df = family_summary(request_df)

    overall_summary = {
        "responses_jsonl": str(args.responses_jsonl),
        "manifest_jsonl": str(args.manifest_jsonl),
        "n_manifest_requests": len(manifest_rows),
        "n_response_rows": len(response_rows),
        "n_successful_requests": int(request_df["custom_id"].nunique()) if not request_df.empty else 0,
        "n_error_rows": len(error_rows),
        "mean_block_accuracy": float(request_df["block_accuracy"].mean()) if not request_df.empty else 0.0,
        "mean_block_exact_match_accuracy": float(request_df["block_exact_match_accuracy"].mean()) if not request_df.empty else 0.0,
        "mean_block_mad": float(request_df["block_mad"].mean()) if not request_df.empty else 0.0,
        "block_all_exact_match_rate": float(request_df["block_all_exact_match"].mean()) if not request_df.empty else 0.0,
        "mean_task_accuracy": float(task_request_df["task_accuracy"].mean()) if not task_request_df.empty else 0.0,
        "usage_prompt_tokens": usage_prompt_tokens,
        "usage_completion_tokens": usage_completion_tokens,
        "retrieval_trace_rows": len(retrieval_trace_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = args.output_dir / "parsed_predictions.jsonl"
    errors_path = args.output_dir / "batch_errors.jsonl"
    retrieval_trace_path = args.output_dir / "retrieval_traces.jsonl"
    request_path = args.output_dir / "request_level_metrics.csv"
    task_request_path = args.output_dir / "task_request_metrics.csv"
    question_path = args.output_dir / "per_question_metrics.csv"
    task_path = args.output_dir / "task_summary.csv"
    family_path = args.output_dir / "family_summary.csv"
    overall_path = args.output_dir / "overall_summary.json"

    with parsed_path.open("w", encoding="utf-8") as f:
        for row in parsed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with errors_path.open("w", encoding="utf-8") as f:
        for row in error_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with retrieval_trace_path.open("w", encoding="utf-8") as f:
        for row in retrieval_trace_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    request_df.to_csv(request_path, index=False)
    task_request_df.to_csv(task_request_path, index=False)
    question_df.to_csv(question_path, index=False)
    task_df.to_csv(task_path, index=False)
    family_df.to_csv(family_path, index=False)
    overall_path.write_text(json.dumps(overall_summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {parsed_path}")
    print(f"Wrote {errors_path}")
    print(f"Wrote {retrieval_trace_path}")
    print(f"Wrote {request_path}")
    print(f"Wrote {task_request_path}")
    print(f"Wrote {question_path}")
    print(f"Wrote {task_path}")
    print(f"Wrote {family_path}")
    print(f"Wrote {overall_path}")
    print(json.dumps(overall_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
