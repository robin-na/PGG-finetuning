from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.multi_game_llm_fvk2c.common import (
    build_generated_scenarios_df,
    build_generated_sessions_df,
    build_human_scenarios_df,
    build_human_sessions_df,
    write_csv,
    write_json,
)
from forecasting.multi_game_llm_fvk2c.exploratory.payoff_utils import (
    compute_payoff_alignment_tables,
    simulate_expected_payoffs,
    sort_payoff_distance_rows,
    sort_payoff_overall_summary,
)


def _build_relative_error_summary(relative_compare: pd.DataFrame) -> pd.DataFrame:
    if relative_compare.empty:
        return pd.DataFrame()

    summaries: list[pd.DataFrame] = []

    by_game = (
        relative_compare.groupby(["role_slice", "game"], as_index=False, observed=False)
        .agg(
            n_case_groups=("case_group", "nunique"),
            mean_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "mean"),
            median_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "median"),
        )
    )
    by_game["summary_scope"] = "by_game"
    summaries.append(by_game)

    if "Benchmark" in set(relative_compare["case_group"].astype(str)):
        non_benchmark = relative_compare[relative_compare["case_group"].astype(str) != "Benchmark"].copy()
        if not non_benchmark.empty:
            nb_summary = (
                non_benchmark.groupby(["role_slice", "game"], as_index=False, observed=False)
                .agg(
                    n_case_groups=("case_group", "nunique"),
                    mean_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "mean"),
                    median_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "median"),
                )
            )
            nb_summary["summary_scope"] = "by_game_non_benchmark"
            summaries.append(nb_summary)

    overall = (
        relative_compare.groupby(["role_slice"], as_index=False, observed=False)
        .agg(
            n_case_groups=("case_group", "nunique"),
            mean_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "mean"),
            median_abs_error_relative_payoff_diff_pct=("abs_error_relative_payoff_diff_pct", "median"),
        )
    )
    overall["game"] = "ALL"
    overall["summary_scope"] = "overall"
    summaries.append(overall)

    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate expected payoffs for the multi-game delegation benchmark and compare generated vs human payoff distributions."
    )
    parser.add_argument("--forecasting-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--gold-targets-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run_name:
        metadata_dir = args.forecasting_root / "metadata" / args.run_name
        args.parsed_output_jsonl = args.parsed_output_jsonl or (metadata_dir / "parsed_output.jsonl")
        args.request_manifest_csv = args.request_manifest_csv or (metadata_dir / "request_manifest.csv")
        args.gold_targets_jsonl = args.gold_targets_jsonl or (metadata_dir / "gold_targets.jsonl")
        args.output_dir = args.output_dir or (
            args.forecasting_root / "results" / f"{args.run_name}__payoff_analysis"
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

    repo_root = REPO_ROOT

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

    generated_expected = simulate_expected_payoffs(
        focal_sessions=generated_sessions,
        focal_scenarios=generated_scenarios,
        counterpart_sessions=human_sessions,
        counterpart_scenarios=human_scenarios,
        repo_root=repo_root,
        seed=args.seed,
    )
    human_expected = simulate_expected_payoffs(
        focal_sessions=human_sessions,
        focal_scenarios=human_scenarios,
        counterpart_sessions=human_sessions,
        counterpart_scenarios=human_scenarios,
        repo_root=repo_root,
        seed=args.seed,
    )

    dist_df, overall_df, relative_df, relative_compare = compute_payoff_alignment_tables(
        generated_expected=generated_expected,
        human_expected=human_expected,
    )
    dist_df = sort_payoff_distance_rows(dist_df)
    overall_df = sort_payoff_overall_summary(overall_df)
    relative_error_summary = _build_relative_error_summary(relative_compare)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "generated_expected_payoffs.csv", generated_expected)
    write_csv(args.output_dir / "human_expected_payoffs.csv", human_expected)
    write_csv(args.output_dir / "payoff_distribution_distance.csv", dist_df)
    write_csv(args.output_dir / "payoff_primary_summary.csv", overall_df)
    write_csv(args.output_dir / "relative_payoff_difference.csv", relative_df)
    write_csv(args.output_dir / "relative_payoff_difference_comparison.csv", relative_compare)
    write_csv(args.output_dir / "relative_payoff_difference_error_summary.csv", relative_error_summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "repo_root": str(repo_root),
            "seed": int(args.seed),
            "generated_session_count": int(len(generated_sessions)),
            "human_session_count": int(len(human_sessions)),
            "generated_expected_payoff_rows": int(len(generated_expected)),
            "human_expected_payoff_rows": int(len(human_expected)),
            "case_groups": ["Benchmark", "AIR", "TD", "OD"],
            "role_slices": ["both", "no", "yes"],
        },
    )


if __name__ == "__main__":
    main()
