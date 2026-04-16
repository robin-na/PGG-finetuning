from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    PRIMARY_METRIC_FAMILY,
    build_primary_distribution_summary,
    compute_metric_tables,
    sort_overall_metric_summary,
)
from common import (
    build_generated_game_records_df,
    build_generated_turn_records_df,
    build_human_game_records_df,
    build_human_turn_records_df,
    write_csv,
    write_json,
)


def _resample_game_and_turn_records(
    *,
    game_records: pd.DataFrame,
    turn_records: pd.DataFrame,
    n_games: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_games = game_records.sample(n=n_games, replace=True, random_state=int(rng.integers(2**31 - 1)))
    sampled_game_parts: list[pd.DataFrame] = []
    sampled_turn_parts: list[pd.DataFrame] = []
    for sample_index, game_row in enumerate(sampled_games.to_dict(orient="records")):
        custom_id = str(game_row["custom_id"])
        game_part = pd.DataFrame([{**game_row, "bootstrap_sample_index": sample_index}])
        turn_part = turn_records[turn_records["custom_id"] == custom_id].copy()
        turn_part["bootstrap_sample_index"] = sample_index
        sampled_game_parts.append(game_part)
        sampled_turn_parts.append(turn_part)
    return (
        pd.concat(sampled_game_parts, ignore_index=True) if sampled_game_parts else pd.DataFrame(),
        pd.concat(sampled_turn_parts, ignore_index=True) if sampled_turn_parts else pd.DataFrame(),
    )


def _bootstrap_noise_ceiling(
    *,
    human_game_records: pd.DataFrame,
    human_turn_records: pd.DataFrame,
    generated_game_records: pd.DataFrame,
    bootstrap_iters: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    generated_counts = (
        generated_game_records.groupby("treatment_name", sort=True)
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
        pseudo_generated_game_parts: list[pd.DataFrame] = []
        pseudo_generated_turn_parts: list[pd.DataFrame] = []
        pseudo_human_game_parts: list[pd.DataFrame] = []
        pseudo_human_turn_parts: list[pd.DataFrame] = []
        for treatment_name, human_games in human_game_records.groupby("treatment_name", sort=True):
            treatment_name = str(treatment_name)
            human_turns = human_turn_records[human_turn_records["treatment_name"] == treatment_name]
            n_generated = generated_count_map.get(treatment_name, 0)
            if n_generated <= 0:
                continue
            pseudo_generated_games, pseudo_generated_turns = _resample_game_and_turn_records(
                game_records=human_games,
                turn_records=human_turns,
                n_games=n_generated,
                rng=rng,
            )
            pseudo_human_games, pseudo_human_turns = _resample_game_and_turn_records(
                game_records=human_games,
                turn_records=human_turns,
                n_games=len(human_games),
                rng=rng,
            )
            pseudo_generated_game_parts.append(pseudo_generated_games)
            pseudo_generated_turn_parts.append(pseudo_generated_turns)
            pseudo_human_game_parts.append(pseudo_human_games)
            pseudo_human_turn_parts.append(pseudo_human_turns)

        if not pseudo_generated_game_parts or not pseudo_human_game_parts:
            continue

        pseudo_generated_games = pd.concat(pseudo_generated_game_parts, ignore_index=True)
        pseudo_generated_turns = pd.concat(pseudo_generated_turn_parts, ignore_index=True)
        pseudo_human_games = pd.concat(pseudo_human_game_parts, ignore_index=True)
        pseudo_human_turns = pd.concat(pseudo_human_turn_parts, ignore_index=True)

        _, _, overall_df = compute_metric_tables(
            generated_game_records=pseudo_generated_games,
            human_game_records=pseudo_human_games,
            generated_turn_records=pseudo_generated_turns,
            human_turn_records=pseudo_human_turns,
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
        description="Bootstrap a human-vs-human noise ceiling for the chip-bargain benchmark."
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

    human_games = build_human_game_records_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    human_turns = build_human_turn_records_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_games = build_generated_game_records_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_turns = build_generated_turn_records_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    _, _, model_overall = compute_metric_tables(
        generated_game_records=generated_games,
        human_game_records=human_games,
        generated_turn_records=generated_turns,
        human_turn_records=human_turns,
    )
    model_overall = sort_overall_metric_summary(model_overall)
    model_primary = build_primary_distribution_summary(model_overall)
    bootstrap_df = _bootstrap_noise_ceiling(
        human_game_records=human_games,
        human_turn_records=human_turns,
        generated_game_records=generated_games,
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
