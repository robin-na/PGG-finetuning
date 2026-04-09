from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    ROLE_A_CHECK,
    ROLE_A_TIME,
    ROLE_B_HIDDEN_CHECK,
    ROLE_B_HIDDEN_TIME,
    ROLE_B_OBSERVABLE_CHECK,
    ROLE_B_OBSERVABLE_TIME,
    build_human_records_df,
    build_generated_records_df,
    load_parsed_outputs_df,
    load_request_manifest_df,
    write_csv,
    write_json,
)


def _row_metrics(schema_type: str, predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exact_match": int(predicted == gold),
    }

    if schema_type == ROLE_A_CHECK:
        metrics["check_match"] = int(predicted["check"] == gold["check"])
        metrics["act_match"] = int(predicted["act"] == gold["act"])
        metrics["return_abs_error"] = abs(int(predicted["return_pct"]) - int(gold["return_pct"]))
        metrics["joint_pattern_match"] = int(
            predicted["check"] == gold["check"] and predicted["act"] == gold["act"]
        )
        return metrics

    if schema_type == ROLE_A_TIME:
        metrics["decision_time_bucket_match"] = int(
            predicted["decision_time_bucket"] == gold["decision_time_bucket"]
        )
        metrics["act_match"] = int(predicted["act"] == gold["act"])
        metrics["return_abs_error"] = abs(int(predicted["return_pct"]) - int(gold["return_pct"]))
        metrics["joint_pattern_match"] = int(
            predicted["decision_time_bucket"] == gold["decision_time_bucket"]
            and predicted["act"] == gold["act"]
        )
        return metrics

    if schema_type == ROLE_B_OBSERVABLE_CHECK:
        fields = [
            "send_if_act_without_check",
            "send_if_act_after_check",
            "send_if_no_act_without_check",
            "send_if_no_act_after_check",
        ]
    elif schema_type == ROLE_B_OBSERVABLE_TIME:
        fields = [
            "send_if_act_fast",
            "send_if_no_act_fast",
            "send_if_act_slow",
            "send_if_no_act_slow",
        ]
    elif schema_type in {ROLE_B_HIDDEN_CHECK, ROLE_B_HIDDEN_TIME}:
        fields = ["send_if_act", "send_if_no_act"]
    else:
        raise ValueError(f"Unsupported schema_type: {schema_type}")

    field_errors = []
    for field_name in fields:
        error = abs(int(predicted[field_name]) - int(gold[field_name]))
        metrics[f"{field_name}_abs_error"] = error
        field_errors.append(error)
    metrics["mean_send_abs_error"] = float(sum(field_errors) / len(field_errors))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parsed two-stage outputs against the sampled human gold targets."
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
    gold_df = build_human_records_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_df = build_generated_records_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )

    gold_lookup = gold_df.set_index("custom_id")
    row_eval_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_df[manifest_df["custom_id"] == custom_id]
        schema_type = str(manifest_row.iloc[0]["schema_type"]) if not manifest_row.empty else ""
        parse_success = bool(row.get("parse_success"))
        eval_row = {
            "custom_id": custom_id,
            "schema_type": schema_type,
            "parse_success": parse_success,
            "evaluated": False,
        }
        if not parse_success or custom_id not in gold_lookup.index:
            row_eval_rows.append(eval_row)
            continue
        predicted = row.get("parsed_target") or {}
        gold = gold_lookup.loc[custom_id, "gold_target"]
        eval_row.update(_row_metrics(schema_type, predicted, gold))
        eval_row["evaluated"] = True
        row_eval_rows.append(eval_row)

    row_eval_df = pd.DataFrame(row_eval_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "row_level_evaluation.csv", row_eval_df)

    evaluated = row_eval_df[row_eval_df["evaluated"].astype(bool)].copy()
    schema_summary_rows: list[dict[str, Any]] = []
    if not evaluated.empty:
        for schema_type, group in evaluated.groupby("schema_type", sort=True):
            row: dict[str, Any] = {
                "schema_type": schema_type,
                "n_rows": int(len(group)),
                "exact_match_rate": float(group["exact_match"].mean()),
            }
            for field_name in [
                "return_abs_error",
                "mean_send_abs_error",
                "check_match",
                "act_match",
                "decision_time_bucket_match",
                "joint_pattern_match",
            ]:
                if field_name in group.columns:
                    values = group[field_name].dropna().astype(float)
                    if not values.empty:
                        row[f"mean_{field_name}"] = float(values.mean())
            schema_summary_rows.append(row)
    schema_summary_df = pd.DataFrame(schema_summary_rows)
    write_csv(args.output_dir / "schema_summary.csv", schema_summary_df)

    parse_success_rate = (
        float(parsed_df["parse_success"].astype(bool).mean()) if not parsed_df.empty else float("nan")
    )
    overall_summary = {
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "gold_targets_jsonl": str(args.gold_targets_jsonl),
        "note": "Row-level exact-match and absolute-error metrics are secondary diagnostics. Use the treatment-level distribution-distance analysis as the primary benchmark.",
        "generated_parse_success_rate": parse_success_rate,
        "generated_parsed_count": int(len(generated_df)),
        "human_gold_count": int(len(gold_df)),
        "evaluated_row_count": int(len(evaluated)),
        "exact_match_rate": (
            float(evaluated["exact_match"].mean()) if not evaluated.empty else float("nan")
        ),
    }
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "primary_metric_family": "distribution_distance",
            "secondary_metric_family": "row_level_exact_or_abs_error",
        },
    )


if __name__ == "__main__":
    main()
