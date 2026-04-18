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
    shannon_entropy,
    total_variation_distance,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SimBench individual-level predictions after aggregating over sampled Twin profiles."
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

    merged_df = manifest_df.merge(
        parsed_df[
            [
                "custom_id",
                "parse_success",
                "parse_errors",
                "validation_errors",
                "text",
                "parsed_label",
            ]
        ]
        if not parsed_df.empty
        else pd.DataFrame(columns=["custom_id", "parse_success", "parse_errors", "validation_errors", "text", "parsed_label"]),
        on="custom_id",
        how="left",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "request_level_predictions.csv", merged_df)

    manifest_by_row = {
        str(row_id): group.copy()
        for row_id, group in manifest_df.groupby("simbench_row_id", sort=False)
    }
    merged_by_row = {
        str(row_id): group.copy()
        for row_id, group in merged_df.groupby("simbench_row_id", sort=False)
    }

    row_eval_rows: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        simbench_row_id = str(gold_row["simbench_row_id"])
        manifest_group = manifest_by_row.get(simbench_row_id, pd.DataFrame())
        generated_group = merged_by_row.get(simbench_row_id, pd.DataFrame())
        option_labels = [str(label) for label in gold_row["option_labels"]]
        gold_distribution = normalize_distribution(gold_row["gold_distribution"])

        completed_count = int(len(generated_group))
        parse_success_group = generated_group[generated_group["parse_success"].fillna(False).astype(bool)].copy()
        parse_success_count = int(len(parse_success_group))
        predicted_counts = {
            label: int((parse_success_group["parsed_label"] == label).sum()) for label in option_labels
        }
        predicted_distribution = normalize_distribution(predicted_counts)
        predicted_modal = modal_label(predicted_distribution, option_labels)
        gold_modal = modal_label(gold_distribution, option_labels)
        evaluated = parse_success_count > 0

        row_eval_rows.append(
            {
                "simbench_row_id": simbench_row_id,
                "simbench_split": str(gold_row["simbench_split"]),
                "dataset_name": str(gold_row["dataset_name"]),
                "group_size": int(gold_row["group_size"]),
                "option_count": int(len(option_labels)),
                "expected_request_count": int(len(manifest_group)),
                "completed_request_count": completed_count,
                "missing_request_count": int(len(manifest_group) - len(generated_group)),
                "parse_success_count": parse_success_count,
                "parse_failure_count": int(len(generated_group) - parse_success_count),
                "completion_rate": (
                    float(completed_count / len(manifest_group)) if len(manifest_group) else float("nan")
                ),
                "parse_success_rate": (
                    float(parse_success_count / len(manifest_group)) if len(manifest_group) else float("nan")
                ),
                "evaluated": evaluated,
                "predicted_distribution_json": json.dumps(predicted_distribution, ensure_ascii=False),
                "gold_distribution_json": json.dumps(gold_distribution, ensure_ascii=False),
                "predicted_counts_json": json.dumps(predicted_counts, ensure_ascii=False),
                "predicted_modal_label": predicted_modal,
                "gold_modal_label": gold_modal,
                "modal_match": (
                    int(predicted_modal == gold_modal) if predicted_modal is not None and gold_modal is not None else None
                ),
                "predicted_entropy": (
                    shannon_entropy(predicted_distribution, option_labels) if evaluated else float("nan")
                ),
                "gold_entropy": shannon_entropy(gold_distribution, option_labels),
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
                "mean_tvd": _safe_mean(evaluated_group["tvd"]),
                "median_tvd": (
                    float(pd.to_numeric(evaluated_group["tvd"], errors="coerce").median())
                    if not evaluated_group.empty
                    else float("nan")
                ),
                "weighted_mean_tvd_by_group_size": _weighted_mean(evaluated_group, "tvd", "group_size"),
                "mean_jsd": _safe_mean(evaluated_group["jsd"]),
                "modal_match_rate": _safe_mean(evaluated_group["modal_match"]),
            }
        )

    dataset_summary_df = pd.DataFrame(dataset_summary_rows)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_summary_df)

    manifest_summary = {}
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
        "num_simbench_rows": int(len(row_eval_df)),
        "evaluated_row_count": int(len(evaluated_rows)),
        "row_evaluable_rate": float(len(evaluated_rows) / len(row_eval_df)) if len(row_eval_df) else float("nan"),
        "mean_completion_rate": _safe_mean(row_eval_df["completion_rate"]),
        "mean_parse_success_rate": _safe_mean(row_eval_df["parse_success_rate"]),
        "mean_tvd": _safe_mean(evaluated_rows["tvd"]),
        "median_tvd": (
            float(pd.to_numeric(evaluated_rows["tvd"], errors="coerce").median())
            if not evaluated_rows.empty
            else float("nan")
        ),
        "weighted_mean_tvd_by_group_size": _weighted_mean(evaluated_rows, "tvd", "group_size"),
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
            "primary_metric_family": "distribution_distance_after_individual_aggregation",
            "primary_metrics": ["tvd", "jsd"],
            "secondary_metrics": ["modal_match_rate", "completion_rate", "parse_success_rate"],
        },
    )


if __name__ == "__main__":
    main()
