from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DELEGATION_FIELDS,
    NUMERIC_SCENARIO_FIELDS,
    SCENARIO_FIELDS,
    build_generated_sessions_df,
    build_human_sessions_df,
    load_parsed_outputs_df,
    load_request_manifest_df,
    write_csv,
    write_json,
)


def _row_metrics(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exact_match": int(predicted == gold),
    }

    delegation_matches: list[int] = []
    for field_name in DELEGATION_FIELDS:
        match = int(predicted.get(field_name) == gold.get(field_name))
        metrics[f"{field_name}_match"] = match
        delegation_matches.append(match)
    metrics["delegation_match_rate"] = float(sum(delegation_matches) / len(delegation_matches))

    predicted_scenarios = predicted.get("scenario_outputs") or []
    gold_scenarios = gold.get("scenario_outputs") or []
    scenario_exact_matches: list[int] = []
    field_matches: list[int] = []
    numeric_abs_errors: list[float] = []

    for pred_scenario, gold_scenario in zip(predicted_scenarios, gold_scenarios):
        scenario_exact_matches.append(int(pred_scenario == gold_scenario))
        for field_name in SCENARIO_FIELDS:
            pred_value = pred_scenario.get(field_name)
            gold_value = gold_scenario.get(field_name)
            field_matches.append(int(pred_value == gold_value))
            if field_name in NUMERIC_SCENARIO_FIELDS and pred_value is not None and gold_value is not None:
                numeric_abs_errors.append(abs(int(pred_value) - int(gold_value)))

    metrics["scenario_exact_match_rate"] = (
        float(sum(scenario_exact_matches) / len(scenario_exact_matches)) if scenario_exact_matches else float("nan")
    )
    metrics["scenario_field_match_rate"] = (
        float(sum(field_matches) / len(field_matches)) if field_matches else float("nan")
    )
    metrics["numeric_direct_abs_error_mean"] = (
        float(sum(numeric_abs_errors) / len(numeric_abs_errors)) if numeric_abs_errors else float("nan")
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parsed multi-game outputs against the sampled human gold targets."
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
    gold_df = build_human_sessions_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_df = build_generated_sessions_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )

    gold_lookup = gold_df.set_index("custom_id")
    row_eval_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_df[manifest_df["custom_id"] == custom_id]
        treatment_name = str(manifest_row.iloc[0]["treatment_name"]) if not manifest_row.empty else ""
        parse_success = bool(row.get("parse_success"))
        eval_row = {
            "custom_id": custom_id,
            "treatment_name": treatment_name,
            "parse_success": parse_success,
            "evaluated": False,
        }
        if not parse_success or custom_id not in gold_lookup.index:
            row_eval_rows.append(eval_row)
            continue
        predicted = row.get("parsed_target") or {}
        gold = gold_lookup.loc[custom_id, "gold_target"]
        eval_row.update(_row_metrics(predicted, gold))
        eval_row["evaluated"] = True
        row_eval_rows.append(eval_row)

    row_eval_df = pd.DataFrame(row_eval_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "row_level_evaluation.csv", row_eval_df)

    evaluated = row_eval_df[row_eval_df["evaluated"].astype(bool)].copy()
    treatment_summary_rows: list[dict[str, Any]] = []
    if not evaluated.empty:
        for treatment_name, group in evaluated.groupby("treatment_name", sort=True):
            treatment_summary_rows.append(
                {
                    "treatment_name": treatment_name,
                    "n_rows": int(len(group)),
                    "exact_match_rate": float(group["exact_match"].mean()),
                    "delegation_match_rate": float(group["delegation_match_rate"].mean()),
                    "scenario_exact_match_rate": float(group["scenario_exact_match_rate"].mean()),
                    "scenario_field_match_rate": float(group["scenario_field_match_rate"].mean()),
                    "numeric_direct_abs_error_mean": float(group["numeric_direct_abs_error_mean"].mean()),
                }
            )
    treatment_summary_df = pd.DataFrame(treatment_summary_rows)
    write_csv(args.output_dir / "treatment_summary.csv", treatment_summary_df)

    parse_success_rate = (
        float(parsed_df["parse_success"].astype(bool).mean()) if not parsed_df.empty else float("nan")
    )
    overall_summary = {
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "gold_targets_jsonl": str(args.gold_targets_jsonl),
        "note": "Row-level exact-match and field-match metrics are secondary diagnostics. Use the treatment-level distribution-distance analysis as the primary benchmark.",
        "generated_parse_success_rate": parse_success_rate,
        "generated_parsed_count": int(len(generated_df)),
        "human_gold_count": int(len(gold_df)),
        "evaluated_row_count": int(len(evaluated)),
        "exact_match_rate": float(evaluated["exact_match"].mean()) if not evaluated.empty else float("nan"),
        "delegation_match_rate": (
            float(evaluated["delegation_match_rate"].mean()) if not evaluated.empty else float("nan")
        ),
        "scenario_exact_match_rate": (
            float(evaluated["scenario_exact_match_rate"].mean()) if not evaluated.empty else float("nan")
        ),
        "scenario_field_match_rate": (
            float(evaluated["scenario_field_match_rate"].mean()) if not evaluated.empty else float("nan")
        ),
        "numeric_direct_abs_error_mean": (
            float(evaluated["numeric_direct_abs_error_mean"].mean()) if not evaluated.empty else float("nan")
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
            "secondary_metric_family": "row_level_exact_or_match_rate",
        },
    )


if __name__ == "__main__":
    main()
