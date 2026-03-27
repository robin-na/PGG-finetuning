#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_CONFIG_CSV = Path("data/processed_data/df_analysis_learn.csv")
DEFAULT_FEATURE_MANIFEST = Path(
    "Persona/misc/tag_section_clusters_openai/analysis_regression_multinomial/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("reports/learning_player_contributions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the distribution of average contribution per player in the "
            "learning-wave human PGG data and compare it across selected CONFIG features."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--config-csv", type=Path, default=DEFAULT_CONFIG_CSV)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_features(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    features = payload.get("features_requested")
    if not isinstance(features, list) or not features:
        raise ValueError(f"No features_requested list found in {manifest_path}")
    return [str(feature) for feature in features]


def coerce_bool_like(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return value


def normalize_value(value: Any) -> Any:
    value = coerce_bool_like(value)
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return float(numeric)


def value_sort_key(value: Any) -> tuple[int, Any]:
    if pd.isna(value):
        return (3, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float, np.number)):
        return (1, float(value))
    return (2, str(value))


def value_label(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"
    return str(value)


def distribution_stats(series: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {
            "n_players": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    percentiles = np.quantile(arr, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n_players": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "p50": float(percentiles[4]),
        "p75": float(percentiles[5]),
        "p90": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "p99": float(percentiles[8]),
        "max": float(arr.max()),
    }


def build_scope_frames(player_avg_all: pd.DataFrame, player_avg_cfg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes: dict[str, pd.DataFrame] = {
        "all_raw": player_avg_all.copy(),
        "matched_all": player_avg_cfg.copy(),
    }
    if "valid_number_of_starting_players" in player_avg_cfg.columns:
        valid_mask = (
            player_avg_cfg["valid_number_of_starting_players"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        scopes["valid_only"] = player_avg_cfg.loc[valid_mask].copy()
    return scopes


def summarize_overall(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_name, df in scopes.items():
        stats = distribution_stats(df["mean_contribution"])
        stats["scope"] = scope_name
        stats["n_games"] = int(df["gameId"].nunique()) if "gameId" in df.columns else np.nan
        stats["n_player_game_rows"] = int(len(df))
        rows.append(stats)
    cols = [
        "scope",
        "n_games",
        "n_player_game_rows",
        "n_players",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    return pd.DataFrame(rows)[cols]


def summarize_by_config(
    scopes: dict[str, pd.DataFrame],
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    for scope_name, df in scopes.items():
        if scope_name == "all_raw":
            continue
        overall_arr = pd.to_numeric(df["mean_contribution"], errors="coerce").dropna().to_numpy(dtype=float)
        overall_mean = float(overall_arr.mean()) if len(overall_arr) else np.nan
        overall_ss = float(((overall_arr - overall_mean) ** 2).sum()) if len(overall_arr) else np.nan

        for feature in features:
            if feature not in df.columns:
                continue

            sub = df[[feature, "gameId", "mean_contribution"]].copy()
            sub = sub.rename(columns={feature: "config_value"})
            sub["config_value"] = sub["config_value"].map(normalize_value)
            sub = sub.dropna(subset=["config_value", "mean_contribution"]).copy()
            if sub.empty:
                continue

            feature_rows: list[dict[str, Any]] = []
            grouped_means: list[float] = []
            grouped_counts: list[int] = []
            ss_between = 0.0

            for config_value, grp in sorted(
                sub.groupby("config_value", dropna=False),
                key=lambda item: value_sort_key(item[0]),
            ):
                stats = distribution_stats(grp["mean_contribution"])
                mean_value = stats["mean"]
                count_value = int(stats["n_players"])
                grouped_means.append(mean_value)
                grouped_counts.append(count_value)
                if np.isfinite(overall_mean):
                    ss_between += count_value * (mean_value - overall_mean) ** 2

                row: dict[str, Any] = {
                    "scope": scope_name,
                    "config_feature": feature,
                    "config_value": value_label(config_value),
                    "n_games": int(grp["gameId"].nunique()),
                    "delta_vs_scope_mean": float(mean_value - overall_mean) if np.isfinite(overall_mean) else np.nan,
                }
                row.update(stats)
                feature_rows.append(row)

            if len(feature_rows) < 2:
                continue

            summary_rows.extend(feature_rows)
            max_gap = float(max(grouped_means) - min(grouped_means))
            eta_squared = float(ss_between / overall_ss) if overall_ss and overall_ss > 0 else np.nan
            effect_rows.append(
                {
                    "scope": scope_name,
                    "config_feature": feature,
                    "n_values": int(len(feature_rows)),
                    "n_players": int(sum(grouped_counts)),
                    "scope_mean_contribution": overall_mean,
                    "max_group_mean_gap": max_gap,
                    "eta_squared": eta_squared,
                    "min_group_size": int(min(grouped_counts)),
                    "max_group_size": int(max(grouped_counts)),
                }
            )

    summary_cols = [
        "scope",
        "config_feature",
        "config_value",
        "n_games",
        "n_players",
        "delta_vs_scope_mean",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    effect_cols = [
        "scope",
        "config_feature",
        "n_values",
        "n_players",
        "scope_mean_contribution",
        "max_group_mean_gap",
        "eta_squared",
        "min_group_size",
        "max_group_size",
    ]

    summary_df = pd.DataFrame(summary_rows)
    effect_df = pd.DataFrame(effect_rows)
    if not summary_df.empty:
        summary_df = summary_df[summary_cols].sort_values(
            ["scope", "config_feature", "config_value"],
            ascending=[True, True, True],
        )
    if not effect_df.empty:
        effect_df = effect_df[effect_cols].sort_values(
            ["scope", "eta_squared", "max_group_mean_gap"],
            ascending=[True, False, False],
        )
    return summary_df, effect_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args.feature_manifest)

    player_rounds = pd.read_csv(
        args.player_rounds,
        usecols=["gameId", "playerId", "data.contribution"],
    )
    player_rounds["gameId"] = player_rounds["gameId"].astype(str)
    player_rounds["playerId"] = player_rounds["playerId"].astype(str)
    player_rounds["data.contribution"] = pd.to_numeric(
        player_rounds["data.contribution"],
        errors="coerce",
    )

    player_avg_all = (
        player_rounds.dropna(subset=["data.contribution"])
        .groupby(["gameId", "playerId"], as_index=False)
        .agg(
            mean_contribution=("data.contribution", "mean"),
            n_observed_rounds=("data.contribution", "size"),
        )
    )

    config_cols = ["gameId", "valid_number_of_starting_players", "CONFIG_endowment"] + features
    config_cols = list(dict.fromkeys(config_cols))
    config_df = pd.read_csv(args.config_csv, usecols=config_cols)
    config_df["gameId"] = config_df["gameId"].astype(str)
    config_df = config_df.drop_duplicates(subset=["gameId"], keep="first").copy()

    missing_cols = [col for col in features if col not in config_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in {args.config_csv}: {missing_cols}")

    player_avg_cfg = player_avg_all.merge(config_df, on="gameId", how="left")
    player_avg_cfg["mean_contribution_rate"] = (
        player_avg_cfg["mean_contribution"]
        / pd.to_numeric(player_avg_cfg["CONFIG_endowment"], errors="coerce")
    )

    matched_mask = player_avg_cfg[features].notna().all(axis=1)
    player_avg_cfg = player_avg_cfg.loc[matched_mask].copy()

    scopes = build_scope_frames(player_avg_all, player_avg_cfg)
    overall_df = summarize_overall(scopes)
    config_summary_df, effect_df = summarize_by_config(scopes, features)

    player_avg_cfg.to_csv(args.output_dir / "player_average_contributions_matched.csv", index=False)
    overall_df.to_csv(args.output_dir / "overall_distribution_summary.csv", index=False)
    config_summary_df.to_csv(args.output_dir / "config_value_distribution_summary.csv", index=False)
    effect_df.to_csv(args.output_dir / "config_effect_ranking.csv", index=False)

    manifest = {
        "player_rounds": str(args.player_rounds),
        "config_csv": str(args.config_csv),
        "feature_manifest": str(args.feature_manifest),
        "output_dir": str(args.output_dir),
        "features_used": features,
        "n_games_raw": int(player_rounds["gameId"].nunique()),
        "n_games_player_avg_all": int(player_avg_all["gameId"].nunique()),
        "n_player_game_rows_all": int(len(player_avg_all)),
        "n_games_matched": int(player_avg_cfg["gameId"].nunique()),
        "n_player_game_rows_matched": int(len(player_avg_cfg)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote outputs to {args.output_dir}")
    print(f"Features used ({len(features)}): {', '.join(features)}")


if __name__ == "__main__":
    main()
