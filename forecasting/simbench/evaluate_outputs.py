from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    jensen_shannon_divergence,
    load_parsed_outputs_df,
    load_request_manifest_df,
    modal_label,
    normalize_distribution,
    read_jsonl,
    simbench_score,
    shannon_entropy,
    total_variation_distance,
    uniform_baseline_tvd,
    write_csv,
    write_json,
)


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
    if frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _parse_question_manifest(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return json.loads(text)


def _coerce_answers_object(raw: Any) -> dict[str, dict[str, float]]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, float) and pd.isna(raw):
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    loaded = json.loads(text)
    return loaded if isinstance(loaded, dict) else {}


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _serialize_for_csv(frame: pd.DataFrame, object_columns: list[str]) -> pd.DataFrame:
    serializable = frame.copy()
    for column in object_columns:
        if column in serializable.columns:
            serializable[column] = serializable[column].map(_json_cell)
    return serializable


def _mean_distribution(
    distributions: list[dict[str, float]],
    option_labels: list[str],
) -> dict[str, float]:
    if not distributions:
        return {}
    averaged = {
        label: float(sum(float(dist.get(label, 0.0)) for dist in distributions) / len(distributions))
        for label in option_labels
    }
    return normalize_distribution(averaged)


def _expand_request_question_predictions(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for request_row in merged_df.to_dict(orient="records"):
        question_manifest = _parse_question_manifest(request_row.get("question_manifest_json"))
        parsed_answers = _coerce_answers_object(request_row.get("parsed_answers"))
        response_received = bool(request_row.get("response_received", False))
        request_parse_success = bool(request_row.get("parse_success", False))

        for item in question_manifest:
            question_id = str(item["question_id"])
            simbench_row_id = str(item["simbench_row_id"])
            option_labels = [str(label) for label in item.get("option_labels", [])]
            predicted_distribution = parsed_answers.get(question_id)
            prediction_available = isinstance(predicted_distribution, dict) and bool(predicted_distribution)

            rows.append(
                {
                    "custom_id": str(request_row["custom_id"]),
                    "run_name": str(request_row["run_name"]),
                    "model": str(request_row["model"]),
                    "variant": str(request_row["variant"]),
                    "simbench_split": str(request_row["simbench_split"]),
                    "context_id": str(request_row["context_id"]),
                    "dataset_name": str(request_row["dataset_name"]),
                    "sample_index": int(request_row["sample_index"]),
                    "twin_pid": "" if pd.isna(request_row["twin_pid"]) else str(request_row["twin_pid"]),
                    "question_id": question_id,
                    "simbench_row_id": simbench_row_id,
                    "option_labels_json": json.dumps(option_labels, ensure_ascii=False),
                    "response_received": response_received,
                    "request_parse_success": request_parse_success,
                    "prediction_available": prediction_available,
                    "predicted_distribution": predicted_distribution if prediction_available else None,
                    "explanation": "" if pd.isna(request_row.get("explanation")) else str(request_row.get("explanation", "")),
                    "parse_errors": request_row.get("parse_errors"),
                    "validation_errors": request_row.get("validation_errors"),
                    "question_validation_errors": request_row.get("question_validation_errors"),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SimBench batched distribution predictions, including Twin-panel aggregation."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--gold-targets-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.run_name:
        metadata_dir = args.forecasting_root / "metadata" / args.run_name
        args.parsed_output_jsonl = args.parsed_output_jsonl or (metadata_dir / "parsed_output.jsonl")
        args.request_manifest_csv = args.request_manifest_csv or (metadata_dir / "request_manifest.csv")
        args.gold_targets_jsonl = args.gold_targets_jsonl or (metadata_dir / "gold_targets.jsonl")
        args.output_dir = args.output_dir or (args.forecasting_root / "results" / f"{args.run_name}__gold_eval")

    if (
        args.parsed_output_jsonl is None
        or args.request_manifest_csv is None
        or args.gold_targets_jsonl is None
        or args.output_dir is None
    ):
        raise ValueError(
            "Provide either --run-name or all of --parsed-output-jsonl, --request-manifest-csv, --gold-targets-jsonl, and --output-dir."
        )

    parsed_df = load_parsed_outputs_df(args.parsed_output_jsonl)
    manifest_df = load_request_manifest_df(args.request_manifest_csv)
    gold_rows = read_jsonl(args.gold_targets_jsonl)
    gold_df = pd.DataFrame(gold_rows)
    if gold_df.empty:
        raise ValueError(f"No gold targets found in {args.gold_targets_jsonl}")

    if parsed_df.empty:
        parsed_subset = pd.DataFrame(
            columns=[
                "custom_id",
                "parse_success",
                "parse_errors",
                "validation_errors",
                "question_validation_errors",
                "text",
                "json_text",
                "explanation",
                "parsed_answers",
                "expected_question_count",
                "valid_question_count",
            ]
        )
    else:
        parsed_subset = parsed_df[
            [
                "custom_id",
                "parse_success",
                "parse_errors",
                "validation_errors",
                "question_validation_errors",
                "text",
                "json_text",
                "explanation",
                "parsed_answers",
                "expected_question_count",
                "valid_question_count",
            ]
        ]

    merged_df = manifest_df.merge(parsed_subset, on="custom_id", how="left")
    completed_custom_ids = set(parsed_df["custom_id"].astype(str).tolist()) if not parsed_df.empty else set()
    merged_df["response_received"] = merged_df["custom_id"].astype(str).isin(completed_custom_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_level_csv = _serialize_for_csv(
        merged_df,
        [
            "parse_errors",
            "validation_errors",
            "question_validation_errors",
            "parsed_answers",
            "question_manifest_json",
        ],
    )
    write_csv(args.output_dir / "request_level_predictions.csv", request_level_csv)

    request_question_df = _expand_request_question_predictions(merged_df)
    request_question_csv = _serialize_for_csv(
        request_question_df,
        ["predicted_distribution", "parse_errors", "validation_errors", "question_validation_errors"],
    )
    write_csv(args.output_dir / "request_question_predictions.csv", request_question_csv)

    request_questions_by_row = {
        str(row_id): group.copy()
        for row_id, group in request_question_df.groupby("simbench_row_id", sort=False)
    }

    row_eval_rows: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        simbench_row_id = str(gold_row["simbench_row_id"])
        request_group = request_questions_by_row.get(simbench_row_id, pd.DataFrame())
        option_labels = [str(label) for label in gold_row["option_labels"]]
        gold_distribution = normalize_distribution(gold_row["gold_distribution"])

        expected_request_count = int(len(request_group))
        completed_group = request_group[request_group["response_received"].astype(bool)].copy()
        completed_request_count = int(len(completed_group))
        full_success_group = request_group[request_group["request_parse_success"].astype(bool)].copy()
        full_request_success_count = int(len(full_success_group))
        valid_prediction_group = request_group[request_group["prediction_available"].astype(bool)].copy()
        valid_prediction_count = int(len(valid_prediction_group))

        predicted_distribution = _mean_distribution(
            [dict(value) for value in valid_prediction_group["predicted_distribution"].tolist()],
            option_labels,
        )
        predicted_modal = modal_label(predicted_distribution, option_labels) if predicted_distribution else None
        gold_modal = modal_label(gold_distribution, option_labels)
        evaluated = bool(predicted_distribution)
        baseline_tvd = uniform_baseline_tvd(gold_distribution, option_labels)

        row_eval_rows.append(
            {
                "simbench_row_id": simbench_row_id,
                "simbench_split": str(gold_row["simbench_split"]),
                "dataset_name": str(gold_row["dataset_name"]),
                "group_size": int(gold_row["group_size"]),
                "option_count": int(len(option_labels)),
                "expected_request_count": expected_request_count,
                "completed_request_count": completed_request_count,
                "missing_request_count": int(expected_request_count - completed_request_count),
                "full_request_success_count": full_request_success_count,
                "full_request_failure_count": int(expected_request_count - full_request_success_count),
                "valid_prediction_count": valid_prediction_count,
                "invalid_or_missing_prediction_count": int(expected_request_count - valid_prediction_count),
                "completion_rate": (
                    float(completed_request_count / expected_request_count)
                    if expected_request_count
                    else float("nan")
                ),
                "parse_success_rate": (
                    float(full_request_success_count / expected_request_count)
                    if expected_request_count
                    else float("nan")
                ),
                "full_request_success_rate": (
                    float(full_request_success_count / expected_request_count)
                    if expected_request_count
                    else float("nan")
                ),
                "valid_prediction_rate": (
                    float(valid_prediction_count / expected_request_count)
                    if expected_request_count
                    else float("nan")
                ),
                "evaluated": evaluated,
                "predicted_distribution_json": json.dumps(predicted_distribution, ensure_ascii=False),
                "gold_distribution_json": json.dumps(gold_distribution, ensure_ascii=False),
                "predicted_modal_label": predicted_modal,
                "gold_modal_label": gold_modal,
                "modal_match": (
                    int(predicted_modal == gold_modal) if predicted_modal is not None and gold_modal is not None else None
                ),
                "predicted_entropy": (
                    shannon_entropy(predicted_distribution, option_labels) if evaluated else float("nan")
                ),
                "gold_entropy": shannon_entropy(gold_distribution, option_labels),
                "uniform_baseline_tvd": baseline_tvd,
                "tvd": (
                    total_variation_distance(predicted_distribution, gold_distribution, option_labels)
                    if evaluated
                    else float("nan")
                ),
                "jsd": (
                    jensen_shannon_divergence(predicted_distribution, gold_distribution, option_labels)
                    if evaluated
                    else float("nan")
                ),
                "simbench_score_raw": (
                    simbench_score(predicted_distribution, gold_distribution, option_labels, scale=1.0)
                    if evaluated
                    else float("nan")
                ),
                "simbench_score": (
                    simbench_score(predicted_distribution, gold_distribution, option_labels, scale=100.0)
                    if evaluated
                    else float("nan")
                ),
            }
        )

    row_eval_df = pd.DataFrame(row_eval_rows)
    write_csv(args.output_dir / "row_level_evaluation.csv", row_eval_df)

    dataset_summary_rows: list[dict[str, Any]] = []
    evaluated_rows = row_eval_df[row_eval_df["evaluated"].astype(bool)].copy()
    for dataset_name, group in row_eval_df.groupby("dataset_name", sort=True):
        evaluated_group = group[group["evaluated"].astype(bool)].copy()
        dataset_summary_rows.append(
            {
                "dataset_name": dataset_name,
                "n_rows": int(len(group)),
                "evaluated_row_count": int(len(evaluated_group)),
                "row_evaluable_rate": float(len(evaluated_group) / len(group)) if len(group) else float("nan"),
                "mean_completion_rate": _safe_mean(group["completion_rate"]),
                "mean_parse_success_rate": _safe_mean(group["parse_success_rate"]),
                "mean_full_request_success_rate": _safe_mean(group["full_request_success_rate"]),
                "mean_valid_prediction_rate": _safe_mean(group["valid_prediction_rate"]),
                "mean_tvd": _safe_mean(evaluated_group["tvd"]),
                "median_tvd": (
                    float(pd.to_numeric(evaluated_group["tvd"], errors="coerce").median())
                    if not evaluated_group.empty
                    else float("nan")
                ),
                "weighted_mean_tvd_by_group_size": _weighted_mean(evaluated_group, "tvd", "group_size"),
                "mean_simbench_score": _safe_mean(evaluated_group["simbench_score"]),
                "median_simbench_score": (
                    float(pd.to_numeric(evaluated_group["simbench_score"], errors="coerce").median())
                    if not evaluated_group.empty
                    else float("nan")
                ),
                "weighted_mean_simbench_score_by_group_size": _weighted_mean(
                    evaluated_group, "simbench_score", "group_size"
                ),
                "mean_jsd": _safe_mean(evaluated_group["jsd"]),
                "modal_match_rate": _safe_mean(evaluated_group["modal_match"]),
            }
        )

    dataset_summary_df = pd.DataFrame(dataset_summary_rows)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_summary_df)

    manifest_summary: dict[str, Any] = {}
    for column in ["run_name", "model", "variant", "simbench_split"]:
        if column in manifest_df.columns:
            values = sorted({str(value) for value in manifest_df[column].dropna().astype(str).tolist()})
            manifest_summary[column] = values[0] if len(values) == 1 else values

    overall_summary = {
        **manifest_summary,
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "gold_targets_jsonl": str(args.gold_targets_jsonl),
        "output_dir": str(args.output_dir),
        "num_request_rows_expected": int(len(manifest_df)),
        "num_request_rows_completed": int(len(parsed_df)),
        "request_completion_rate": float(len(parsed_df) / len(manifest_df)) if len(manifest_df) else float("nan"),
        "request_parse_success_rate_observed": (
            _safe_mean(parsed_df["parse_success"]) if not parsed_df.empty else float("nan")
        ),
        "request_with_any_valid_answers_rate_observed": (
            float(
                pd.to_numeric(parsed_df["valid_question_count"], errors="coerce").gt(0).mean()
            )
            if not parsed_df.empty
            else float("nan")
        ),
        "num_simbench_rows": int(len(row_eval_df)),
        "evaluated_row_count": int(len(evaluated_rows)),
        "row_evaluable_rate": float(len(evaluated_rows) / len(row_eval_df)) if len(row_eval_df) else float("nan"),
        "mean_completion_rate": _safe_mean(row_eval_df["completion_rate"]),
        "mean_parse_success_rate": _safe_mean(row_eval_df["parse_success_rate"]),
        "mean_full_request_success_rate": _safe_mean(row_eval_df["full_request_success_rate"]),
        "mean_valid_prediction_rate": _safe_mean(row_eval_df["valid_prediction_rate"]),
        "mean_tvd": _safe_mean(evaluated_rows["tvd"]),
        "median_tvd": (
            float(pd.to_numeric(evaluated_rows["tvd"], errors="coerce").median())
            if not evaluated_rows.empty
            else float("nan")
        ),
        "weighted_mean_tvd_by_group_size": _weighted_mean(evaluated_rows, "tvd", "group_size"),
        "mean_simbench_score": _safe_mean(evaluated_rows["simbench_score"]),
        "median_simbench_score": (
            float(pd.to_numeric(evaluated_rows["simbench_score"], errors="coerce").median())
            if not evaluated_rows.empty
            else float("nan")
        ),
        "weighted_mean_simbench_score_by_group_size": _weighted_mean(
            evaluated_rows, "simbench_score", "group_size"
        ),
        "mean_jsd": _safe_mean(evaluated_rows["jsd"]),
        "modal_match_rate": _safe_mean(evaluated_rows["modal_match"]),
    }
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "primary_metric_family": "distribution_distance_after_batched_prediction_aggregation",
            "primary_metrics": ["simbench_score", "tvd", "jsd"],
            "secondary_metrics": [
                "modal_match_rate",
                "completion_rate",
                "valid_prediction_rate",
                "full_request_success_rate",
            ],
            "aggregation_note": (
                "Each request may contain multiple question-level distributions. "
                "Question-level predictions are expanded first; Twin runs then average those distributions "
                "across sample_index within each simbench_row_id."
            ),
        },
    )


if __name__ == "__main__":
    main()
