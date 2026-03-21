from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_game_summary(path: Path, baseline_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if baseline_name is not None:
        df = df.copy()
        df["baseline"] = baseline_name
    return df


def _common_game_ids_by_k(game_summary_dfs: list[pd.DataFrame]) -> dict[int, set[str]]:
    common: dict[int, set[str]] = {}
    ks = sorted(set().union(*[set(df["k"].astype(int).unique()) for df in game_summary_dfs]))
    for k in ks:
        sets = [set(df[df["k"] == k]["game_id"].astype(str)) for df in game_summary_dfs]
        common[int(k)] = set.intersection(*sets) if sets else set()
    return common


def _filter_to_common_subset(df: pd.DataFrame, common_game_ids_by_k: dict[int, set[str]]) -> pd.DataFrame:
    parts = []
    for k, game_ids in common_game_ids_by_k.items():
        parts.append(df[(df["k"] == k) & (df["game_id"].astype(str).isin(game_ids))])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _summarize_from_game_summary(game_summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "num_rounds",
        "num_players",
        "future_rounds",
        "actor_rows",
        "contribution_mae",
        "contribution_rate_mae",
        "contribution_exact_rate",
        "round_payoff_mae",
        "round_coin_gen_mae",
        "round_relative_efficiency_mae",
        "round_normalized_efficiency_mae",
        "total_contribution_mae",
        "total_contribution_rate_mae",
        "actual_future_relative_efficiency",
        "predicted_future_relative_efficiency",
        "future_relative_efficiency_abs_error",
        "actual_future_normalized_efficiency",
        "predicted_future_normalized_efficiency",
        "future_normalized_efficiency_abs_error",
        "target_precision",
        "target_recall",
        "target_f1",
        "target_set_exact_rate",
        "action_exact_match_rate",
        "punish_target_precision",
        "punish_target_recall",
        "punish_target_f1",
        "reward_target_precision",
        "reward_target_recall",
        "reward_target_f1",
        "punish_any_accuracy",
        "reward_any_accuracy",
        "unit_mae_on_overlap",
        "punish_unit_mae_on_overlap",
        "reward_unit_mae_on_overlap",
        "cumulative_future_payoff_mae",
    ]
    summary = (
        game_summary_df.groupby(["baseline", "k"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["k", "baseline"])
        .reset_index(drop=True)
    )
    counts = (
        game_summary_df.groupby(["baseline", "k"], as_index=False)["game_id"]
        .nunique()
        .rename(columns={"game_id": "num_games"})
    )
    return counts.merge(summary, on=["baseline", "k"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two LLM trajectory-completion runs plus persistence on their common evaluated subset."
    )
    parser.add_argument("--gpt4-game-summary-csv", type=Path, required=True)
    parser.add_argument("--gpt5-game-summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-game-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gpt4_df = _load_game_summary(args.gpt4_game_summary_csv, baseline_name="gpt_4_1_mini")
    gpt5_df = _load_game_summary(args.gpt5_game_summary_csv, baseline_name="gpt_5_1")
    baseline_df = _load_game_summary(args.baseline_game_summary_csv)
    baseline_df = baseline_df[baseline_df["baseline"] == "persistence"].copy()

    common_game_ids_by_k = _common_game_ids_by_k([gpt4_df, gpt5_df, baseline_df])

    gpt4_common = _filter_to_common_subset(gpt4_df, common_game_ids_by_k)
    gpt5_common = _filter_to_common_subset(gpt5_df, common_game_ids_by_k)
    baseline_common = _filter_to_common_subset(baseline_df, common_game_ids_by_k)

    comparison_game_summary_df = pd.concat(
        [gpt4_common, gpt5_common, baseline_common],
        ignore_index=True,
    ).sort_values(["k", "baseline", "game_id"]).reset_index(drop=True)
    comparison_overall_df = _summarize_from_game_summary(comparison_game_summary_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_game_summary_df.to_csv(args.output_dir / "comparison_game_summary.csv", index=False)
    comparison_overall_df.to_csv(args.output_dir / "comparison_overall_summary.csv", index=False)

    manifest = {
        "gpt4_game_summary_csv": str(args.gpt4_game_summary_csv),
        "gpt5_game_summary_csv": str(args.gpt5_game_summary_csv),
        "baseline_game_summary_csv": str(args.baseline_game_summary_csv),
        "common_games_by_k": {str(k): sorted(game_ids) for k, game_ids in common_game_ids_by_k.items()},
        "common_game_counts_by_k": {str(k): len(game_ids) for k, game_ids in common_game_ids_by_k.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(comparison_overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
