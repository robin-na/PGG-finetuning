#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_ROUNDS = Path("data/raw_data/learning_wave/rounds.csv")
DEFAULT_CONFIG_CSV = Path("data/processed_data/df_analysis_learn.csv")
DEFAULT_FEATURE_MANIFEST = Path(
    "Persona/misc/tag_section_clusters_openai/analysis_regression_multinomial/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("reports/learning_round_dynamics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze round-by-round contribution dynamics in the learning-wave "
            "human PGG data, including overall trends, per-game heterogeneity, "
            "and config-level differences."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--rounds-csv", type=Path, default=DEFAULT_ROUNDS)
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


def build_scope_frames(round_df_all: pd.DataFrame, round_df_cfg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes: dict[str, pd.DataFrame] = {
        "all_raw": round_df_all.copy(),
        "matched_all": round_df_cfg.copy(),
    }
    if "valid_number_of_starting_players" in round_df_cfg.columns:
        valid_mask = (
            round_df_cfg["valid_number_of_starting_players"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        scopes["valid_only"] = round_df_cfg.loc[valid_mask].copy()
    return scopes


def summarize_progress_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    sub = df.copy()
    sub = sub[sub["progress_norm"].notna()].copy()
    sub["progress_bin"] = np.minimum(
        np.floor(sub["progress_norm"].clip(lower=0.0, upper=0.999999) * n_bins).astype(int),
        n_bins - 1,
    )
    grouped = (
        sub.groupby("progress_bin", as_index=False)
        .agg(
            n_player_rounds=("contribution", "size"),
            n_games=("gameId", "nunique"),
            mean_contribution=("contribution", "mean"),
            median_contribution=("contribution", "median"),
            p25_contribution=("contribution", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.25))),
            p75_contribution=("contribution", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.75))),
            mean_progress_norm=("progress_norm", "mean"),
        )
    )
    grouped["progress_bin_start"] = grouped["progress_bin"] / n_bins
    grouped["progress_bin_end"] = (grouped["progress_bin"] + 1) / n_bins
    return grouped[
        [
            "progress_bin",
            "progress_bin_start",
            "progress_bin_end",
            "mean_progress_norm",
            "n_games",
            "n_player_rounds",
            "mean_contribution",
            "median_contribution",
            "p25_contribution",
            "p75_contribution",
        ]
    ]


def summarize_absolute_rounds(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("round_number", as_index=False)
        .agg(
            n_player_rounds=("contribution", "size"),
            n_games=("gameId", "nunique"),
            mean_contribution=("contribution", "mean"),
            median_contribution=("contribution", "median"),
        )
        .sort_values("round_number")
    )
    return grouped


def compute_game_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = [col for col in df.columns if col.startswith("CONFIG_")]
    extra_cols = [col for col in ["valid_number_of_starting_players"] if col in df.columns]

    for game_id, grp in df.groupby("gameId", sort=False):
        grp = grp.sort_values("round_index")
        x = grp["progress_norm"].to_numpy(dtype=float)
        y = grp["mean_round_contribution"].to_numpy(dtype=float)
        n_rounds_observed = int(len(grp))
        if n_rounds_observed < 2 or float(np.var(x)) == 0.0:
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept = np.polyfit(x, y, deg=1)

        first_value = float(y[0])
        last_value = float(y[-1])
        delta = last_value - first_value
        peak_round = int(grp.loc[grp["mean_round_contribution"].idxmax(), "round_number"])
        trough_round = int(grp.loc[grp["mean_round_contribution"].idxmin(), "round_number"])
        row: dict[str, Any] = {
            "gameId": game_id,
            "n_rounds_observed": n_rounds_observed,
            "first_round_contribution": first_value,
            "last_round_contribution": last_value,
            "last_minus_first": delta,
            "slope_per_unit_progress": float(slope) if np.isfinite(slope) else np.nan,
            "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            "peak_round_number": peak_round,
            "trough_round_number": trough_round,
            "mean_contribution_over_rounds": float(y.mean()),
            "std_contribution_over_rounds": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        }
        first_row = grp.iloc[0]
        for col in feature_cols + extra_cols:
            row[col] = first_row.get(col)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_game_slopes(game_slopes: pd.DataFrame) -> pd.DataFrame:
    sub = game_slopes.dropna(subset=["slope_per_unit_progress"]).copy()
    slopes = sub["slope_per_unit_progress"].to_numpy(dtype=float)
    deltas = sub["last_minus_first"].to_numpy(dtype=float)

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.quantile(arr, q)) if len(arr) else np.nan

    return pd.DataFrame(
        [
            {
                "n_games_with_slope": int(len(sub)),
                "mean_slope_per_unit_progress": float(slopes.mean()),
                "median_slope_per_unit_progress": float(np.median(slopes)),
                "std_slope_per_unit_progress": float(slopes.std(ddof=1)) if len(slopes) > 1 else 0.0,
                "p10_slope_per_unit_progress": pct(slopes, 0.10),
                "p25_slope_per_unit_progress": pct(slopes, 0.25),
                "p75_slope_per_unit_progress": pct(slopes, 0.75),
                "p90_slope_per_unit_progress": pct(slopes, 0.90),
                "share_negative_slope": float((slopes < 0).mean()),
                "share_positive_slope": float((slopes > 0).mean()),
                "mean_last_minus_first": float(deltas.mean()),
                "median_last_minus_first": float(np.median(deltas)),
                "share_last_below_first": float((deltas < 0).mean()),
                "share_last_above_first": float((deltas > 0).mean()),
            }
        ]
    )


def summarize_slope_by_config(game_slopes: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    base = game_slopes.dropna(subset=["slope_per_unit_progress"]).copy()
    overall_mean = float(base["slope_per_unit_progress"].mean())
    overall_ss = float(((base["slope_per_unit_progress"] - overall_mean) ** 2).sum())

    for feature in features:
        if feature not in base.columns:
            continue
        sub = base[[feature, "slope_per_unit_progress", "last_minus_first", "gameId"]].copy()
        sub["config_value"] = sub[feature].map(normalize_value)
        sub = sub.dropna(subset=["config_value"]).copy()
        if sub.empty:
            continue

        grouped_means: list[float] = []
        grouped_counts: list[int] = []
        ss_between = 0.0
        feature_rows: list[dict[str, Any]] = []

        for config_value, grp in sorted(
            sub.groupby("config_value", dropna=False),
            key=lambda item: value_sort_key(item[0]),
        ):
            count_value = int(len(grp))
            mean_slope = float(grp["slope_per_unit_progress"].mean())
            grouped_means.append(mean_slope)
            grouped_counts.append(count_value)
            ss_between += count_value * (mean_slope - overall_mean) ** 2
            feature_rows.append(
                {
                    "config_feature": feature,
                    "config_value": value_label(config_value),
                    "n_games": count_value,
                    "mean_slope_per_unit_progress": mean_slope,
                    "median_slope_per_unit_progress": float(grp["slope_per_unit_progress"].median()),
                    "mean_last_minus_first": float(grp["last_minus_first"].mean()),
                    "median_last_minus_first": float(grp["last_minus_first"].median()),
                    "share_negative_slope": float((grp["slope_per_unit_progress"] < 0).mean()),
                    "share_last_below_first": float((grp["last_minus_first"] < 0).mean()),
                    "delta_vs_overall_mean_slope": mean_slope - overall_mean,
                }
            )

        if len(feature_rows) < 2:
            continue

        summary_rows.extend(feature_rows)
        effect_rows.append(
            {
                "config_feature": feature,
                "n_values": int(len(feature_rows)),
                "n_games": int(sum(grouped_counts)),
                "overall_mean_slope_per_unit_progress": overall_mean,
                "max_group_mean_slope_gap": float(max(grouped_means) - min(grouped_means)),
                "eta_squared_slope": float(ss_between / overall_ss) if overall_ss > 0 else np.nan,
                "min_group_size": int(min(grouped_counts)),
                "max_group_size": int(max(grouped_counts)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    effect_df = pd.DataFrame(effect_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["config_feature", "config_value"])
    if not effect_df.empty:
        effect_df = effect_df.sort_values(
            ["eta_squared_slope", "max_group_mean_slope_gap"],
            ascending=[False, False],
        )
    return summary_df, effect_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args.feature_manifest)

    rounds = pd.read_csv(args.rounds_csv, usecols=["_id", "gameId", "index"])
    rounds = rounds.rename(columns={"_id": "roundId", "index": "round_index"})
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["roundId"] = rounds["roundId"].astype(str)
    rounds["round_index"] = pd.to_numeric(rounds["round_index"], errors="coerce")

    player_rounds = pd.read_csv(
        args.player_rounds,
        usecols=["gameId", "playerId", "roundId", "data.contribution"],
    )
    player_rounds["gameId"] = player_rounds["gameId"].astype(str)
    player_rounds["playerId"] = player_rounds["playerId"].astype(str)
    player_rounds["roundId"] = player_rounds["roundId"].astype(str)
    player_rounds["contribution"] = pd.to_numeric(player_rounds["data.contribution"], errors="coerce")
    player_rounds = player_rounds.drop(columns=["data.contribution"])
    player_rounds = player_rounds.dropna(subset=["contribution"]).copy()

    round_df_all = player_rounds.merge(
        rounds,
        on=["gameId", "roundId"],
        how="left",
        validate="many_to_one",
    )
    round_df_all["round_number"] = round_df_all["round_index"] + 1

    config_cols = ["gameId", "valid_number_of_starting_players", "CONFIG_endowment"] + features
    config_cols = list(dict.fromkeys(config_cols))
    config_df = pd.read_csv(args.config_csv, usecols=config_cols)
    config_df["gameId"] = config_df["gameId"].astype(str)
    config_df = config_df.drop_duplicates(subset=["gameId"], keep="first").copy()

    round_df_cfg = round_df_all.merge(config_df, on="gameId", how="left")
    matched_mask = round_df_cfg[features].notna().all(axis=1)
    round_df_cfg = round_df_cfg.loc[matched_mask].copy()

    scopes = build_scope_frames(round_df_all, round_df_cfg)

    manifest: dict[str, Any] = {
        "player_rounds": str(args.player_rounds),
        "rounds_csv": str(args.rounds_csv),
        "config_csv": str(args.config_csv),
        "feature_manifest": str(args.feature_manifest),
        "output_dir": str(args.output_dir),
        "features_used": features,
        "scope_counts": {},
    }

    for scope_name, df in scopes.items():
        round_max = df.groupby("gameId")["round_index"].transform("max")
        denom = round_max.where(round_max > 0, 1.0)
        df = df.copy()
        df["progress_norm"] = np.where(round_max > 0, df["round_index"] / denom, 0.0)
        df["progress_pct"] = 100.0 * df["progress_norm"]

        game_round = (
            df.groupby(["gameId", "round_index", "round_number"], as_index=False)
            .agg(
                mean_round_contribution=("contribution", "mean"),
                median_round_contribution=("contribution", "median"),
                n_players=("playerId", "nunique"),
                progress_norm=("progress_norm", "first"),
            )
        )

        config_keep = ["gameId"] + [f for f in ["valid_number_of_starting_players", *features] if f in df.columns]
        config_keep = list(dict.fromkeys(config_keep))
        game_level_cfg = df[config_keep].drop_duplicates(subset=["gameId"], keep="first")
        game_round = game_round.merge(game_level_cfg, on="gameId", how="left", validate="many_to_one")

        progress_df = summarize_progress_bins(df)
        absolute_df = summarize_absolute_rounds(df)
        game_slopes_df = compute_game_slopes(game_round)
        game_slope_summary_df = summarize_game_slopes(game_slopes_df)
        slope_by_config_df, slope_config_effect_df = summarize_slope_by_config(game_slopes_df, features)

        progress_df.to_csv(args.output_dir / f"{scope_name}_progress_bin_summary.csv", index=False)
        absolute_df.to_csv(args.output_dir / f"{scope_name}_absolute_round_summary.csv", index=False)
        game_slopes_df.to_csv(args.output_dir / f"{scope_name}_game_slopes.csv", index=False)
        game_slope_summary_df.to_csv(args.output_dir / f"{scope_name}_game_slope_summary.csv", index=False)
        slope_by_config_df.to_csv(args.output_dir / f"{scope_name}_slope_by_config_value.csv", index=False)
        slope_config_effect_df.to_csv(args.output_dir / f"{scope_name}_slope_config_effect_ranking.csv", index=False)

        manifest["scope_counts"][scope_name] = {
            "n_player_round_rows": int(len(df)),
            "n_games": int(df["gameId"].nunique()),
            "n_games_with_slopes": int(game_slopes_df["slope_per_unit_progress"].notna().sum()),
            "n_game_round_rows": int(len(game_round)),
        }

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
