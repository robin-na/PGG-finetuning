#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLUSTER_JSONL = Path(
    "Persona/archetype_retrieval/learning_wave/CONTRIBUTION/contribution_clustered.jsonl"
)
DEFAULT_CLUSTER_CATALOG = Path(
    "Persona/cluster/tag_section_clusters_openai_learn/CONTRIBUTION/cluster_catalog_polished.json"
)
DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_NONEXTREME_LABELS = Path("reports/learning_nonextreme_patterns/valid_only_player_game_pattern_labels.csv")
DEFAULT_OUTPUT_DIR = Path("reports/learning_contribution_persona_stability")

DEFAULT_BINARY_CONFIGS = [
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_showRewardId",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate contribution-archetype clusters to observed contribution trajectory "
            "patterns, and quantify how those patterns shift across binary config values."
        )
    )
    parser.add_argument("--cluster-jsonl", type=Path, default=DEFAULT_CLUSTER_JSONL)
    parser.add_argument("--cluster-catalog", type=Path, default=DEFAULT_CLUSTER_CATALOG)
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--nonextreme-labels", type=Path, default=DEFAULT_NONEXTREME_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_BINARY_CONFIGS))
    return parser.parse_args()


def load_cluster_titles(path: Path) -> dict[int, str]:
    with path.open() as handle:
        payload = json.load(handle)
    titles: dict[int, str] = {}
    for row in payload:
        titles[int(row["cluster_id"])] = str(row["cluster_title"])
    return titles


def build_pattern_table(
    *,
    extreme_flags_path: Path,
    nonextreme_labels_path: Path,
    config_cols: list[str],
) -> pd.DataFrame:
    flags = pd.read_csv(extreme_flags_path)
    flags = flags[
        flags["valid_number_of_starting_players"].astype(bool)
        & flags["complete_round_coverage"].astype(bool)
    ].copy()

    flags["pattern"] = pd.Series(pd.NA, index=flags.index, dtype="object")
    flags.loc[flags["always_full"].astype(bool), "pattern"] = "always_full"
    flags.loc[flags["always_zero"].astype(bool), "pattern"] = "always_zero"

    nonextreme = pd.read_csv(nonextreme_labels_path)

    labels = flags[["gameId", "playerId", "pattern"] + config_cols].merge(
        nonextreme[["gameId", "playerId", "pattern"]],
        on=["gameId", "playerId"],
        how="left",
        suffixes=("_extreme", "_nonextreme"),
    )
    labels["pattern"] = labels["pattern_extreme"].fillna(labels["pattern_nonextreme"])

    labels = labels.drop(columns=["pattern_extreme", "pattern_nonextreme"])
    labels["gameId"] = labels["gameId"].astype(str)
    labels["playerId"] = labels["playerId"].astype(str)
    return labels


def pattern_share_dict(series: pd.Series) -> dict[str, float]:
    shares = series.value_counts(normalize=True)
    return {str(idx): float(value) for idx, value in shares.items()}


def summarize_cluster_purity(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster_id", sort=True):
        shares = group["pattern"].value_counts(normalize=True)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_title": str(group["cluster_title"].iloc[0]),
                "n_player_game_rows": int(len(group)),
                "n_patterns": int(group["pattern"].nunique()),
                "dominant_pattern": str(shares.index[0]),
                "dominant_share": float(shares.iloc[0]),
                "entropy_bits": float(-(shares * np.log2(shares)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def summarize_pattern_by_cluster(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_df = (
        df.groupby(["cluster_id", "cluster_title", "pattern"], as_index=False)
        .agg(n_player_game_rows=("playerId", "size"))
        .sort_values(["cluster_id", "n_player_game_rows"], ascending=[True, False])
    )
    count_df["share_within_cluster"] = (
        count_df["n_player_game_rows"]
        / count_df.groupby("cluster_id")["n_player_game_rows"].transform("sum")
    )

    wide_df = (
        count_df.pivot(
            index=["cluster_id", "cluster_title"],
            columns="pattern",
            values="share_within_cluster",
        )
        .fillna(0.0)
        .reset_index()
    )
    return count_df, wide_df


def summarize_config_shifts(df: pd.DataFrame, config_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, cluster_group in df.groupby("cluster_id", sort=True):
        cluster_title = str(cluster_group["cluster_title"].iloc[0])
        for config_col in config_cols:
            sub = cluster_group.dropna(subset=[config_col]).copy()
            if sub[config_col].nunique() != 2:
                continue

            values = sorted(sub[config_col].unique(), key=lambda value: str(value))
            value_summaries: dict[object, dict[str, object]] = {}

            for value, value_group in sub.groupby(config_col):
                shares = value_group["pattern"].value_counts(normalize=True)
                value_summaries[value] = {
                    "n_rows": int(len(value_group)),
                    "dominant_pattern": str(shares.index[0]),
                    "dominant_share": float(shares.iloc[0]),
                    "pattern_shares": {
                        str(idx): float(share) for idx, share in shares.items()
                    },
                }

            value0, value1 = values
            share0 = value_summaries[value0]["pattern_shares"]
            share1 = value_summaries[value1]["pattern_shares"]
            patterns = sorted(set(share0) | set(share1))
            tvd = 0.5 * sum(
                abs(float(share0.get(pattern, 0.0)) - float(share1.get(pattern, 0.0)))
                for pattern in patterns
            )

            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_title": cluster_title,
                    "config_feature": config_col,
                    "config_value_0": str(value0),
                    "config_value_1": str(value1),
                    "n_rows_0": int(value_summaries[value0]["n_rows"]),
                    "n_rows_1": int(value_summaries[value1]["n_rows"]),
                    "dominant_pattern_0": str(value_summaries[value0]["dominant_pattern"]),
                    "dominant_pattern_1": str(value_summaries[value1]["dominant_pattern"]),
                    "dominant_share_0": float(value_summaries[value0]["dominant_share"]),
                    "dominant_share_1": float(value_summaries[value1]["dominant_share"]),
                    "dominant_pattern_changes": bool(
                        value_summaries[value0]["dominant_pattern"]
                        != value_summaries[value1]["dominant_pattern"]
                    ),
                    "total_variation_distance": float(tvd),
                    "always_full_gap": float(share1.get("always_full", 0.0) - share0.get("always_full", 0.0)),
                    "always_zero_gap": float(share1.get("always_zero", 0.0) - share0.get("always_zero", 0.0)),
                    "high_cooperator_gap": float(share1.get("high_cooperator", 0.0) - share0.get("high_cooperator", 0.0)),
                    "binary_0_20_gap": float(share1.get("binary_0_20", 0.0) - share0.get("binary_0_20", 0.0)),
                    "variable_mid_gap": float(share1.get("variable_mid", 0.0) - share0.get("variable_mid", 0.0)),
                    "stable_mid_gap": float(share1.get("stable_mid", 0.0) - share0.get("stable_mid", 0.0)),
                    "low_contributor_gap": float(share1.get("low_contributor", 0.0) - share0.get("low_contributor", 0.0)),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["total_variation_distance", "dominant_pattern_changes", "cluster_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_cols = list(dict.fromkeys(args.configs))

    cluster_df = pd.read_json(args.cluster_jsonl, lines=True)
    cluster_df["gameId"] = cluster_df["gameId"].astype(str)
    cluster_df["playerId"] = cluster_df["playerId"].astype(str)

    cluster_titles = load_cluster_titles(args.cluster_catalog)
    cluster_df["cluster_title"] = cluster_df["cluster_id"].map(cluster_titles).fillna(cluster_df["cluster_title"])

    pattern_df = build_pattern_table(
        extreme_flags_path=args.extreme_flags,
        nonextreme_labels_path=args.nonextreme_labels,
        config_cols=config_cols,
    )

    merged = pattern_df.merge(
        cluster_df[["gameId", "playerId", "cluster_id", "cluster_title"]],
        on=["gameId", "playerId"],
        how="inner",
    )
    merged = merged.dropna(subset=["pattern"]).copy()

    cluster_purity_df = summarize_cluster_purity(merged)
    pattern_by_cluster_df, pattern_by_cluster_wide_df = summarize_pattern_by_cluster(merged)
    config_shift_df = summarize_config_shifts(merged, config_cols=config_cols)

    merged.to_csv(args.output_dir / "valid_only_complete_pattern_cluster_labels.csv", index=False)
    cluster_purity_df.to_csv(args.output_dir / "valid_only_cluster_purity_summary.csv", index=False)
    pattern_by_cluster_df.to_csv(args.output_dir / "valid_only_pattern_by_cluster.csv", index=False)
    pattern_by_cluster_wide_df.to_csv(args.output_dir / "valid_only_pattern_by_cluster_wide.csv", index=False)
    config_shift_df.to_csv(args.output_dir / "valid_only_cluster_config_shift_summary.csv", index=False)

    manifest = {
        "cluster_jsonl": str(args.cluster_jsonl),
        "cluster_catalog": str(args.cluster_catalog),
        "extreme_flags": str(args.extreme_flags),
        "nonextreme_labels": str(args.nonextreme_labels),
        "output_dir": str(args.output_dir),
        "binary_configs": config_cols,
        "notes": [
            "This analysis is type-level, not person-level: each playerId appears in exactly one game in the learning wave.",
            "Persona clusters come from the CONTRIBUTION section summaries, so they are behavior-derived rather than independent background traits.",
            "The pattern labels require complete round coverage and use always_full / always_zero plus the non-extreme motif labels from reports/learning_nonextreme_patterns.",
            "Within-cluster config shift is summarized by total variation distance between the pattern distributions under the two config values.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
