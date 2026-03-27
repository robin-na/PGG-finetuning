#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
DEFAULT_OUTPUT_DIR = Path("reports/learning_action_behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize punishment and reward behavior in the learning-wave human PGG data "
            "and compare it across selected CONFIG features."
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


def parse_action_dict(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text or text == "{}" or text.lower() == "nan":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, raw in parsed.items():
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            cleaned[str(key)] = numeric
    return cleaned


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


def build_scope_frames(base_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes: dict[str, pd.DataFrame] = {
        "matched_all": base_df.copy(),
    }
    if "valid_number_of_starting_players" in base_df.columns:
        valid_mask = (
            base_df["valid_number_of_starting_players"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        scopes["valid_only"] = base_df.loc[valid_mask].copy()
    return scopes


def summarize_action_distribution(df: pd.DataFrame, units_col: str, targets_col: str) -> pd.DataFrame:
    units = pd.to_numeric(df[units_col], errors="coerce").fillna(0.0)
    targets = pd.to_numeric(df[targets_col], errors="coerce").fillna(0.0)
    any_flag = units > 0
    cond_units = units.loc[any_flag]
    return pd.DataFrame(
        [
            {
                "n_player_round_rows": int(len(df)),
                "share_any": float(any_flag.mean()) if len(df) else np.nan,
                "mean_units": float(units.mean()) if len(df) else np.nan,
                "median_units": float(units.median()) if len(df) else np.nan,
                "p95_units": float(units.quantile(0.95)) if len(df) else np.nan,
                "p99_units": float(units.quantile(0.99)) if len(df) else np.nan,
                "max_units": float(units.max()) if len(df) else np.nan,
                "mean_targets": float(targets.mean()) if len(df) else np.nan,
                "median_targets": float(targets.median()) if len(df) else np.nan,
                "p95_targets": float(targets.quantile(0.95)) if len(df) else np.nan,
                "conditional_mean_units_if_any": float(cond_units.mean()) if len(cond_units) else 0.0,
                "conditional_median_units_if_any": float(cond_units.median()) if len(cond_units) else 0.0,
            }
        ]
    )


def summarize_by_config(
    df: pd.DataFrame,
    features: list[str],
    units_col: str,
    targets_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    base = df.copy()
    base["units"] = pd.to_numeric(base[units_col], errors="coerce").fillna(0.0)
    base["targets"] = pd.to_numeric(base[targets_col], errors="coerce").fillna(0.0)
    base["any_flag"] = (base["units"] > 0).astype(float)

    overall_mean_units = float(base["units"].mean())
    overall_mean_any = float(base["any_flag"].mean())
    overall_ss_units = float(((base["units"] - overall_mean_units) ** 2).sum())

    for feature in features:
        if feature not in base.columns:
            continue
        sub = base[[feature, "units", "targets", "any_flag", "gameId"]].copy()
        sub["config_value"] = sub[feature].map(normalize_value)
        sub = sub.dropna(subset=["config_value"]).copy()
        if sub.empty:
            continue

        grouped_means: list[float] = []
        grouped_any: list[float] = []
        grouped_counts: list[int] = []
        ss_between_units = 0.0
        feature_rows: list[dict[str, Any]] = []

        for config_value, grp in sorted(
            sub.groupby("config_value", dropna=False),
            key=lambda item: value_sort_key(item[0]),
        ):
            count_value = int(len(grp))
            mean_units = float(grp["units"].mean())
            share_any = float(grp["any_flag"].mean())
            grouped_means.append(mean_units)
            grouped_any.append(share_any)
            grouped_counts.append(count_value)
            ss_between_units += count_value * (mean_units - overall_mean_units) ** 2
            cond = grp.loc[grp["units"] > 0, "units"]
            feature_rows.append(
                {
                    "config_feature": feature,
                    "config_value": value_label(config_value),
                    "n_games": int(grp["gameId"].nunique()),
                    "n_player_round_rows": count_value,
                    "share_any": share_any,
                    "mean_units": mean_units,
                    "median_units": float(grp["units"].median()),
                    "p95_units": float(grp["units"].quantile(0.95)),
                    "mean_targets": float(grp["targets"].mean()),
                    "conditional_mean_units_if_any": float(cond.mean()) if len(cond) else 0.0,
                    "delta_vs_overall_mean_units": mean_units - overall_mean_units,
                    "delta_vs_overall_share_any": share_any - overall_mean_any,
                }
            )

        if len(feature_rows) < 2:
            continue

        summary_rows.extend(feature_rows)
        effect_rows.append(
            {
                "config_feature": feature,
                "n_values": int(len(feature_rows)),
                "n_player_round_rows": int(sum(grouped_counts)),
                "overall_mean_units": overall_mean_units,
                "overall_share_any": overall_mean_any,
                "max_group_mean_units_gap": float(max(grouped_means) - min(grouped_means)),
                "max_group_share_any_gap": float(max(grouped_any) - min(grouped_any)),
                "eta_squared_units": float(ss_between_units / overall_ss_units) if overall_ss_units > 0 else np.nan,
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
            ["eta_squared_units", "max_group_share_any_gap", "max_group_mean_units_gap"],
            ascending=[False, False, False],
        )
    return summary_df, effect_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args.feature_manifest)

    rounds = pd.read_csv(
        args.player_rounds,
        usecols=[
            "gameId",
            "playerId",
            "roundId",
            "data.punished",
            "data.rewarded",
            "data.punishedBy",
            "data.rewardedBy",
        ],
    )
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["roundId"] = rounds["roundId"].astype(str)

    rounds["punish_units_given"] = rounds["data.punished"].map(parse_action_dict).map(
        lambda d: float(sum(d.values()))
    )
    rounds["reward_units_given"] = rounds["data.rewarded"].map(parse_action_dict).map(
        lambda d: float(sum(d.values()))
    )
    rounds["punish_targets_given"] = rounds["data.punished"].map(parse_action_dict).map(
        lambda d: float(len(d))
    )
    rounds["reward_targets_given"] = rounds["data.rewarded"].map(parse_action_dict).map(
        lambda d: float(len(d))
    )
    rounds["punish_units_received"] = rounds["data.punishedBy"].map(parse_action_dict).map(
        lambda d: float(sum(d.values()))
    )
    rounds["reward_units_received"] = rounds["data.rewardedBy"].map(parse_action_dict).map(
        lambda d: float(sum(d.values()))
    )

    config_cols = ["gameId", "valid_number_of_starting_players"] + features
    config_cols = list(dict.fromkeys(config_cols))
    config_df = pd.read_csv(args.config_csv, usecols=config_cols)
    config_df["gameId"] = config_df["gameId"].astype(str)
    config_df = config_df.drop_duplicates(subset=["gameId"], keep="first").copy()

    missing_cols = [col for col in features if col not in config_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in {args.config_csv}: {missing_cols}")

    matched = rounds.merge(config_df, on="gameId", how="left")
    matched = matched.loc[matched[features].notna().all(axis=1)].copy()

    scopes = build_scope_frames(matched)

    manifest: dict[str, Any] = {
        "player_rounds": str(args.player_rounds),
        "config_csv": str(args.config_csv),
        "feature_manifest": str(args.feature_manifest),
        "output_dir": str(args.output_dir),
        "features_used": features,
    }

    action_specs = [
        ("punishment", "CONFIG_punishmentExists", "punish_units_given", "punish_targets_given"),
        ("reward", "CONFIG_rewardExists", "reward_units_given", "reward_targets_given"),
    ]

    for scope_name, df in scopes.items():
        manifest[scope_name] = {}
        for action_name, enable_col, units_col, targets_col in action_specs:
            if enable_col not in df.columns:
                continue
            active = df.loc[df[enable_col].astype("boolean").fillna(False).astype(bool)].copy()
            overall_df = summarize_action_distribution(active, units_col, targets_col)
            by_config_df, effect_df = summarize_by_config(active, features, units_col, targets_col)

            overall_df.to_csv(args.output_dir / f"{scope_name}_{action_name}_overall_summary.csv", index=False)
            by_config_df.to_csv(args.output_dir / f"{scope_name}_{action_name}_config_value_summary.csv", index=False)
            effect_df.to_csv(args.output_dir / f"{scope_name}_{action_name}_config_effect_ranking.csv", index=False)

            manifest[scope_name][action_name] = {
                "n_player_round_rows": int(len(active)),
                "n_games": int(active["gameId"].nunique()),
            }

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
