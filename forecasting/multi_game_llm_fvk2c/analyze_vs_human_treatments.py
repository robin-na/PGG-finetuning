from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_utils import (
    build_primary_distribution_summary,
    compute_metric_tables,
    sort_overall_metric_summary,
)
from common import (
    build_generated_scenarios_df,
    build_generated_sessions_df,
    build_human_scenarios_df,
    build_human_sessions_df,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generated multi-game outputs to human distributions within each treatment and nested scenario cell."
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
        args.output_dir = args.output_dir or (
            args.forecasting_root / "results" / f"{args.run_name}__vs_human_treatments"
        )

    if (
        args.parsed_output_jsonl is None
        or args.request_manifest_csv is None
        or args.gold_targets_jsonl is None
        or args.output_dir is None
    ):
        raise ValueError(
            "Provide either --run-name or all of --parsed-output-jsonl, --request-manifest-csv, --gold-targets-jsonl, and --output-dir."
        )

    human_sessions = build_human_sessions_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_sessions = build_generated_sessions_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    human_scenarios = build_human_scenarios_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_scenarios = build_generated_scenarios_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )

    mean_df, dist_df, overall_df = compute_metric_tables(
        generated_sessions=generated_sessions,
        human_sessions=human_sessions,
        generated_scenarios=generated_scenarios,
        human_scenarios=human_scenarios,
    )
    overall_df = sort_overall_metric_summary(overall_df)
    primary_dist_df = build_primary_distribution_summary(overall_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "generated_sessions.csv", generated_sessions)
    write_csv(args.output_dir / "human_sessions.csv", human_sessions)
    write_csv(args.output_dir / "generated_scenarios.csv", generated_scenarios)
    write_csv(args.output_dir / "human_scenarios.csv", human_scenarios)
    write_csv(args.output_dir / "mean_alignment.csv", mean_df)
    write_csv(args.output_dir / "distribution_distance.csv", dist_df)
    write_csv(args.output_dir / "primary_distribution_summary.csv", primary_dist_df)
    write_csv(args.output_dir / "overall_metric_summary.csv", overall_df)

    if not mean_df.empty:
        mean_summary = (
            mean_df.groupby(["metric_scope", "metric"], as_index=False)
            .agg(
                n_groups=("treatment_name", "nunique"),
                mean_abs_error=("abs_error", "mean"),
                median_abs_error=("abs_error", "median"),
            )
        )
    else:
        mean_summary = pd.DataFrame()

    if not dist_df.empty:
        dist_summary = (
            dist_df.groupby(["metric_family", "metric", "distance_kind"], as_index=False)
            .agg(
                n_groups=("treatment_name", "nunique"),
                mean_score=("score", "mean"),
                median_score=("score", "median"),
            )
        )
        dist_summary = sort_overall_metric_summary(
            dist_summary.rename(
                columns={
                    "mean_score": "mean_value",
                    "median_score": "median_value",
                }
            )
        ).rename(
            columns={
                "mean_value": "mean_score",
                "median_value": "median_score",
            }
        )
    else:
        dist_summary = pd.DataFrame()

    write_csv(args.output_dir / "mean_alignment_summary.csv", mean_summary)
    write_csv(args.output_dir / "distribution_distance_summary.csv", dist_summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "generated_session_count": int(len(generated_sessions)),
            "human_session_count": int(len(human_sessions)),
            "generated_scenario_count": int(len(generated_scenarios)),
            "human_scenario_count": int(len(human_scenarios)),
            "primary_metric_families": [
                "delegation_distribution_distance",
                "scenario_state_distribution_distance",
                "scenario_direct_value_distance",
            ],
        },
    )


if __name__ == "__main__":
    main()
