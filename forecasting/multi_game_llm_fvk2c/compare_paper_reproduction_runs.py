from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import write_csv, write_json


def _pretty_run_label(run_name: str) -> str:
    mapping = {
        "baseline_gpt_5_mini": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
        "baseline_gpt_5_1": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_1": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_1": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
    }
    return mapping.get(run_name, run_name.replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paper-reproduction metrics across multi-game forecasting runs."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    default_output_name = "__vs__".join(args.runs) + "__paper_reproduction_compare"
    output_dir = args.output_dir or (args.forecasting_root / "results" / default_output_name)

    summary_rows: list[dict[str, object]] = []
    table1_parts: list[pd.DataFrame] = []
    figure1_parts: list[pd.DataFrame] = []
    for run_name in args.runs:
        result_dir = args.forecasting_root / "results" / f"{run_name}__paper_reproduction"
        summary = json.loads((result_dir / "summary.json").read_text())
        summary["run_name"] = run_name
        summary["run_label"] = _pretty_run_label(run_name)
        summary_rows.append(summary)

        table1_df = pd.read_csv(result_dir / "table1_reproduction_comparison.csv")
        table1_df["run_name"] = run_name
        table1_df["run_label"] = _pretty_run_label(run_name)
        table1_parts.append(table1_df)

        figure1_df = pd.read_csv(result_dir / "figure1_reproduction_comparison.csv")
        figure1_df["run_name"] = run_name
        figure1_df["run_label"] = _pretty_run_label(run_name)
        figure1_parts.append(figure1_df)

    summary_df = pd.DataFrame(summary_rows)
    table1_df = pd.concat(table1_parts, ignore_index=True)
    figure1_df = pd.concat(figure1_parts, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "headline_reproduction_summary.csv", summary_df)
    write_csv(output_dir / "table1_run_comparison.csv", table1_df)
    write_csv(output_dir / "figure1_run_comparison.csv", figure1_df)
    write_json(
        output_dir / "manifest.json",
        {
            "runs": args.runs,
            "output_dir": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
