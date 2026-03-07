#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.astype(str).str.strip().str.lower().map(lambda x: x in true_values).fillna(False)
    )


def parse_dict_field(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if text in {"", "{}", "None", "nan", "null"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def build_game_level_rate_targets(rows_csv: Path, analysis_csv: Path) -> pd.DataFrame:
    cfg = pd.read_csv(analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    cfg["gameId"] = cfg["gameId"].astype(str)
    keep_cols = ["gameId", "CONFIG_endowment", "CONFIG_punishmentExists", "CONFIG_rewardExists"]
    cfg = cfg[[c for c in keep_cols if c in cfg.columns]].copy()
    if "CONFIG_endowment" not in cfg.columns:
        cfg["CONFIG_endowment"] = 20.0
    if "CONFIG_punishmentExists" not in cfg.columns:
        cfg["CONFIG_punishmentExists"] = False
    if "CONFIG_rewardExists" not in cfg.columns:
        cfg["CONFIG_rewardExists"] = False

    rows = pd.read_csv(rows_csv)
    rows["gameId"] = rows["gameId"].astype(str)
    rows["contribution"] = pd.to_numeric(rows.get("data.contribution"), errors="coerce").fillna(0.0)
    rows["punished_dict"] = rows["data.punished"].map(parse_dict_field)
    rows["rewarded_dict"] = rows["data.rewarded"].map(parse_dict_field)

    merged = rows.merge(cfg, on="gameId", how="left")
    endowment = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(20.0)
    pun_exists = parse_bool_series(merged["CONFIG_punishmentExists"])
    rew_exists = parse_bool_series(merged["CONFIG_rewardExists"])

    merged["target_contrib_rate"] = np.where(endowment != 0, merged["contribution"] / endowment, 0.0)
    merged["target_punishment_rate"] = np.where(
        pun_exists, merged["punished_dict"].map(lambda d: int(bool(d))), 0
    )
    merged["target_reward_rate"] = np.where(
        rew_exists, merged["rewarded_dict"].map(lambda d: int(bool(d))), 0
    )

    return (
        merged.groupby("gameId", as_index=False)
        .agg(
            target_contrib_rate=("target_contrib_rate", "mean"),
            target_punishment_rate=("target_punishment_rate", "mean"),
            target_reward_rate=("target_reward_rate", "mean"),
        )
        .copy()
    )


def train_ols_for_target(
    learn_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
) -> tuple[float, int]:
    config_cols = [c for c in learn_df.columns if c.startswith("CONFIG_") and c in val_df.columns]
    if not config_cols:
        raise ValueError("No CONFIG_* features found in both learning and validation tables.")
    if target_col not in learn_df.columns or target_col not in val_df.columns:
        raise ValueError(f"Target column '{target_col}' missing in train/eval tables.")

    x_train = learn_df[config_cols].copy()
    y_train = pd.to_numeric(learn_df[target_col], errors="coerce")
    x_test = val_df[config_cols].copy()
    y_test = pd.to_numeric(val_df[target_col], errors="coerce")

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    x_train = x_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    x_test = x_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    num_cols = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return float(np.sqrt(mean_squared_error(y_test, pred))), int(len(y_test))


def make_grouped_rmse_plot(
    df: pd.DataFrame,
    out_path: Path,
    scope: str,
    ols_rmse_by_target: dict[str, float],
    n_games: int,
    dpi: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_order = ["no_archetype", "random_archetype", "retrieved_archetype", "oracle_archetype"]
    df = df.copy()
    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)

    metric_cols = [
        ("rmse_contrib_rate", "contribution"),
        ("rmse_normalized_efficiency", "normalized_efficiency"),
        ("rmse_punishment_rate", "punishment_rate"),
        ("rmse_reward_rate", "reward_rate"),
    ]
    metric_names = [x[1] for x in metric_cols]
    x = np.arange(len(metric_cols))
    width = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for idx, variant in enumerate(variant_order):
        row = df[df["variant"] == variant]
        if row.empty:
            vals = [np.nan] * len(metric_cols)
        else:
            vals = [float(row.iloc[0][col]) for col, _ in metric_cols]
        ax.bar(x + offsets[idx], vals, width=width, label=variant)

    # OLS CONFIG baselines (one marker per target group).
    baseline_vals = [ols_rmse_by_target.get(name, np.nan) for name in metric_names]
    ax.scatter(
        x,
        baseline_vals,
        marker="D",
        s=64,
        color="black",
        label="ols_config_baseline",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=0, ha="center")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Macro RMSE by Target ({scope}, n={n_games})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_grouped_variance_plot(
    df: pd.DataFrame,
    out_path: Path,
    scope: str,
    title_suffix: str,
    metric_specs: list[tuple[str, str, str]],
    n_games: int,
    dpi: int,
) -> None:
    """
    metric_specs entries:
      (sim_column, human_column, label)
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_order = ["no_archetype", "random_archetype", "retrieved_archetype", "oracle_archetype"]
    df = df.copy()
    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)

    labels = [x[2] for x in metric_specs]
    x = np.arange(len(metric_specs))
    width = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for idx, variant in enumerate(variant_order):
        row = df[df["variant"] == variant]
        if row.empty:
            vals = [np.nan] * len(metric_specs)
        else:
            vals = [float(row.iloc[0][sim_col]) for sim_col, _, _ in metric_specs]
        ax.bar(x + offsets[idx], vals, width=width, label=variant)

    # Human references per metric as marker points.
    human_vals = []
    for _, human_col, _ in metric_specs:
        if human_col in df.columns and not df.empty:
            human_vals.append(float(df.iloc[0][human_col]))
        else:
            human_vals.append(np.nan)
    ax.scatter(
        x,
        human_vals,
        marker="D",
        s=64,
        color="black",
        label="human",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Variance")
    ax.set_title(f"{title_suffix} ({scope}, n={n_games})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped 4-variant macro RMSE with OLS CONFIG baselines (game-level)."
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest"),
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest/macro_four_variant_partial_rmse_variance_summary.csv"),
    )
    parser.add_argument(
        "--shared_game_ids_csv",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest/macro_four_variant_shared4_game_ids.csv"),
    )
    parser.add_argument(
        "--learn_analysis_csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_learn.csv"),
    )
    parser.add_argument(
        "--val_analysis_csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument(
        "--learn_rows_csv",
        type=Path,
        default=Path("benchmark/data/raw_data/learning_wave/player-rounds.csv"),
    )
    parser.add_argument(
        "--val_rows_csv",
        type=Path,
        default=Path("benchmark/data/raw_data/validation_wave/player-rounds.csv"),
    )
    parser.add_argument("--scope", type=str, default="shared_4way")
    parser.add_argument("--eff_target_col", type=str, default="itt_relative_efficiency")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    summary_csv = args.summary_csv.resolve()
    shared_game_ids_csv = args.shared_game_ids_csv.resolve()
    learn_csv = args.learn_analysis_csv.resolve()
    val_csv = args.val_analysis_csv.resolve()
    learn_rows_csv = args.learn_rows_csv.resolve()
    val_rows_csv = args.val_rows_csv.resolve()

    summary = pd.read_csv(summary_csv)
    scoped = summary[summary["scope"] == args.scope].copy()
    if scoped.empty:
        raise ValueError(f"No rows found for scope='{args.scope}' in {summary_csv}")

    variant_order = ["no_archetype", "random_archetype", "retrieved_archetype", "oracle_archetype"]
    scoped["variant"] = pd.Categorical(scoped["variant"], categories=variant_order, ordered=True)
    scoped = scoped.sort_values("variant").reset_index(drop=True)

    shared_ids = pd.read_csv(shared_game_ids_csv)["gameId"].astype(str).tolist()
    learn_analysis = pd.read_csv(learn_csv).drop_duplicates(subset=["gameId"], keep="first")
    val_analysis = pd.read_csv(val_csv).drop_duplicates(subset=["gameId"], keep="first")
    learn_analysis["gameId"] = learn_analysis["gameId"].astype(str)
    val_analysis["gameId"] = val_analysis["gameId"].astype(str)

    learn_rates = build_game_level_rate_targets(learn_rows_csv, learn_csv)
    val_rates = build_game_level_rate_targets(val_rows_csv, val_csv)
    learn_rates["gameId"] = learn_rates["gameId"].astype(str)
    val_rates["gameId"] = val_rates["gameId"].astype(str)

    learn_table = learn_analysis.merge(learn_rates, on="gameId", how="left")
    val_table = val_analysis.merge(val_rates, on="gameId", how="left")
    val_table = val_table[val_table["gameId"].isin(set(shared_ids))].copy()

    ols_eff_rmse, n_eval_eff = train_ols_for_target(
        learn_table,
        val_table,
        target_col=args.eff_target_col,
    )
    ols_contrib_rmse, n_eval_contrib = train_ols_for_target(
        learn_table,
        val_table,
        target_col="target_contrib_rate",
    )
    ols_punish_rmse, n_eval_punish = train_ols_for_target(
        learn_table,
        val_table,
        target_col="target_punishment_rate",
    )
    ols_reward_rmse, n_eval_reward = train_ols_for_target(
        learn_table,
        val_table,
        target_col="target_reward_rate",
    )
    ols_rmse_by_target = {
        "contribution": float(ols_contrib_rmse),
        "normalized_efficiency": float(ols_eff_rmse),
        "punishment_rate": float(ols_punish_rmse),
        "reward_rate": float(ols_reward_rmse),
    }
    n_eval_by_target = {
        "contribution": int(n_eval_contrib),
        "normalized_efficiency": int(n_eval_eff),
        "punishment_rate": int(n_eval_punish),
        "reward_rate": int(n_eval_reward),
    }

    if len(set(n_eval_by_target.values())) != 1:
        # Keep going, but this should usually be identical on shared IDs.
        print(f"Warning: OLS eval counts differ by target: {n_eval_by_target}")

    # Save scoped table with OLS RMSE columns for reproducibility.
    scoped_out = scoped.copy()
    scoped_out["ols_config_rmse_contrib_rate"] = ols_rmse_by_target["contribution"]
    scoped_out["ols_config_rmse_normalized_efficiency"] = ols_rmse_by_target["normalized_efficiency"]
    scoped_out["ols_config_rmse_punishment_rate"] = ols_rmse_by_target["punishment_rate"]
    scoped_out["ols_config_rmse_reward_rate"] = ols_rmse_by_target["reward_rate"]
    scoped_out_path = analysis_dir / f"macro_four_variant_{args.scope}_with_ols_table.csv"
    scoped_out.to_csv(scoped_out_path, index=False)

    figures_dir = analysis_dir / "figures"
    rmse_fig = figures_dir / f"macro_four_variant_rmse_grouped_by_target_{args.scope}_with_ols.png"
    variance_players_fig = figures_dir / f"macro_four_variant_variance_across_players_{args.scope}.png"
    variance_games_fig = figures_dir / f"macro_four_variant_variance_across_games_{args.scope}.png"
    n_games_scope = int(scoped["n_games"].iloc[0]) if not scoped.empty else 0
    make_grouped_rmse_plot(
        scoped,
        rmse_fig,
        scope=args.scope,
        ols_rmse_by_target=ols_rmse_by_target,
        n_games=n_games_scope,
        dpi=args.dpi,
    )
    make_grouped_variance_plot(
        scoped,
        variance_players_fig,
        scope=args.scope,
        title_suffix="Within-game Player Variance",
        metric_specs=[
            (
                "mean_var_players_contrib_rate_sim",
                "mean_var_players_contrib_rate_human",
                "contribution",
            ),
            (
                "mean_var_players_punishment_rate_sim",
                "mean_var_players_punishment_rate_human",
                "punishment_rate",
            ),
            (
                "mean_var_players_reward_rate_sim",
                "mean_var_players_reward_rate_human",
                "reward_rate",
            ),
        ],
        n_games=n_games_scope,
        dpi=args.dpi,
    )
    make_grouped_variance_plot(
        scoped,
        variance_games_fig,
        scope=args.scope,
        title_suffix="Across-game Variance of Game-level Means",
        metric_specs=[
            (
                "var_across_games_contrib_rate_sim",
                "var_across_games_contrib_rate_human",
                "contribution",
            ),
            (
                "var_across_games_punishment_rate_sim",
                "var_across_games_punishment_rate_human",
                "punishment_rate",
            ),
            (
                "var_across_games_reward_rate_sim",
                "var_across_games_reward_rate_human",
                "reward_rate",
            ),
            (
                "var_across_games_normalized_efficiency_sim",
                "var_across_games_normalized_efficiency_human",
                "normalized_efficiency",
            ),
        ],
        n_games=n_games_scope,
        dpi=args.dpi,
    )

    manifest = {
        "summary_csv": str(summary_csv),
        "shared_game_ids_csv": str(shared_game_ids_csv),
        "scope": args.scope,
        "n_games_scope": n_games_scope,
        "n_eval_games_ols_by_target": n_eval_by_target,
        "ols_config_rmse_by_target": ols_rmse_by_target,
        "outputs": {
            "table": str(scoped_out_path),
            "rmse_figure": str(rmse_fig),
            "variance_across_players_figure": str(variance_players_fig),
            "variance_across_games_figure": str(variance_games_fig),
        },
    }
    manifest_path = analysis_dir / f"macro_four_variant_{args.scope}_with_ols_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {scoped_out_path}")
    print(f"Wrote {rmse_fig}")
    print(f"Wrote {variance_players_fig}")
    print(f"Wrote {variance_games_fig}")
    print(f"Wrote {manifest_path}")
    print(f"OLS CONFIG baseline RMSE by target: {ols_rmse_by_target}")
    print(f"OLS eval counts by target: {n_eval_by_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
