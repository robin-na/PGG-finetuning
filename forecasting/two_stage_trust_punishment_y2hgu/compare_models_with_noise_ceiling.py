from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    PRIMARY_METRIC_FAMILY,
    build_primary_distribution_summary,
    compute_treatment_metric_tables,
    sort_overall_metric_summary,
)
from common import build_generated_records_df, build_human_records_df, write_csv, write_json


def _bootstrap_noise_ceiling(
    *,
    human_records: pd.DataFrame,
    generated_records: pd.DataFrame,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    generated_counts = (
        generated_records.groupby("treatment_name", sort=True)
        .size()
        .rename("generated_n")
        .reset_index()
    )
    generated_count_map = {
        str(row["treatment_name"]): int(row["generated_n"])
        for row in generated_counts.to_dict(orient="records")
    }

    bootstrap_rows: list[dict[str, object]] = []
    for bootstrap_iter in range(bootstrap_iters):
        pseudo_generated_parts: list[pd.DataFrame] = []
        pseudo_human_parts: list[pd.DataFrame] = []
        for treatment_name, human_group in human_records.groupby("treatment_name", sort=True):
            treatment_name = str(treatment_name)
            n_generated = generated_count_map.get(treatment_name, 0)
            if n_generated <= 0:
                continue
            pseudo_generated_parts.append(
                human_group.sample(n=n_generated, replace=True, random_state=int(rng.integers(2**31 - 1)))
            )
            pseudo_human_parts.append(
                human_group.sample(
                    n=len(human_group),
                    replace=True,
                    random_state=int(rng.integers(2**31 - 1)),
                )
            )
        if not pseudo_generated_parts or not pseudo_human_parts:
            continue
        pseudo_generated = pd.concat(pseudo_generated_parts, ignore_index=True)
        pseudo_human = pd.concat(pseudo_human_parts, ignore_index=True)
        _, _, overall_df = compute_treatment_metric_tables(
            generated_records=pseudo_generated,
            human_records=pseudo_human,
        )
        for row in overall_df.to_dict(orient="records"):
            bootstrap_rows.append(
                {
                    "bootstrap_iter": bootstrap_iter,
                    **row,
                    "score": row["mean_value"],
                }
            )
    return pd.DataFrame(bootstrap_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a human-vs-human noise ceiling for the two-stage benchmark."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--gold-targets-jsonl", type=Path, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.run_name:
        metadata_dir = args.forecasting_root / "metadata" / args.run_name
        args.parsed_output_jsonl = args.parsed_output_jsonl or (metadata_dir / "parsed_output.jsonl")
        args.request_manifest_csv = args.request_manifest_csv or (metadata_dir / "request_manifest.csv")
        args.gold_targets_jsonl = args.gold_targets_jsonl or (metadata_dir / "gold_targets.jsonl")
        args.output_dir = args.output_dir or (
            args.forecasting_root / "results" / f"{args.run_name}__noise_ceiling"
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

    human_records = build_human_records_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_records = build_generated_records_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    _, _, model_overall = compute_treatment_metric_tables(
        generated_records=generated_records,
        human_records=human_records,
    )
    model_overall = sort_overall_metric_summary(model_overall)
    model_primary = build_primary_distribution_summary(model_overall)
    bootstrap_df = _bootstrap_noise_ceiling(
        human_records=human_records,
        generated_records=generated_records,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )

    if not bootstrap_df.empty:
        summary = (
            bootstrap_df.groupby(["metric_family", "metric", "distance_kind"], as_index=False)
            .agg(
                bootstrap_mean=("score", "mean"),
                bootstrap_median=("score", "median"),
                bootstrap_p05=("score", lambda s: float(np.quantile(s, 0.05))),
                bootstrap_p95=("score", lambda s: float(np.quantile(s, 0.95))),
            )
        )
    else:
        summary = pd.DataFrame()

    if not model_overall.empty and not summary.empty:
        summary = summary.merge(
            model_overall.rename(columns={"mean_value": "model_score"}),
            on=["metric_family", "metric", "distance_kind"],
            how="left",
        )
        summary = sort_overall_metric_summary(summary)
    primary_summary = build_primary_distribution_summary(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "noise_ceiling_bootstrap.csv", bootstrap_df)
    write_csv(args.output_dir / "noise_ceiling_summary.csv", summary)
    write_csv(args.output_dir / "primary_noise_ceiling_summary.csv", primary_summary)
    write_csv(args.output_dir / "model_overall_metric_summary.csv", model_overall)
    write_csv(args.output_dir / "model_primary_metric_summary.csv", model_primary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "run_name": args.run_name,
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "bootstrap_iters": args.bootstrap_iters,
            "seed": args.seed,
            "primary_metric_family": PRIMARY_METRIC_FAMILY,
        },
    )


if __name__ == "__main__":
    main()
