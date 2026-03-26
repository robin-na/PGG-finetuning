#!/usr/bin/env python3
"""
OLS Scaling Curve vs LLM Simulation.

X-axis: number of unique training configs used to train OLS
Y-axis: RMSE on validation set
OLS: scaling curve (subsample configs, retrain, repeat)
LLM: horizontal dashed lines (no training data needed)
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from plot_macro_four_variant_summary import build_game_level_rate_targets


def train_ols_rmse(
    learn_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
) -> float:
    """Train OLS on learn_df, return RMSE on val_df."""
    config_cols = [
        c for c in learn_df.columns
        if c.startswith("CONFIG_") and c in val_df.columns
    ]
    x_train = learn_df[config_cols].copy()
    y_train = pd.to_numeric(learn_df[target_col], errors="coerce")
    x_test = val_df[config_cols].copy()
    y_test = pd.to_numeric(val_df[target_col], errors="coerce")

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    x_train, y_train = x_train.loc[train_mask], y_train.loc[train_mask]
    x_test, y_test = x_test.loc[test_mask], y_test.loc[test_mask]

    if len(x_train) < 2 or len(x_test) == 0:
        return np.nan

    num_cols = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("lr", Ridge(alpha=1.0))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return float(np.sqrt(mean_squared_error(y_test, pred)))


def compute_llm_rmse(
    per_game_csv: Path,
    target_human_col: str,
    target_sim_col: str,
) -> dict[str, float]:
    """Return {variant: RMSE} from game_level_parity.csv."""
    pg = pd.read_csv(per_game_csv)
    source_to_variant = {
        "no archetype": "no_archetype",
        "oracle archetype": "oracle_archetype",
        "retrieved archetype": "retrieved_archetype",
        "random archetype": "random_archetype",
    }
    pg["variant"] = pg["source"].map(source_to_variant)
    out = {}
    for variant, grp in pg.groupby("variant"):
        sq = (grp[target_sim_col] - grp[target_human_col]) ** 2
        out[variant] = float(np.sqrt(sq.mean()))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--learn_analysis_csv", type=Path,
                   default=Path("benchmark/data/processed_data/df_analysis_learn.csv"))
    p.add_argument("--val_analysis_csv", type=Path,
                   default=Path("benchmark/data/processed_data/df_analysis_val.csv"))
    p.add_argument("--learn_rows_csv", type=Path,
                   default=Path("benchmark/data/raw_data/learning_wave/player-rounds.csv"))
    p.add_argument("--val_rows_csv", type=Path,
                   default=Path("benchmark/data/raw_data/validation_wave/player-rounds.csv"))
    p.add_argument("--per_game_csv", type=Path,
                   default=Path("reports/benchmark/macro_simulation_eval/local12b_four_variant_rollout/game_level_parity.csv"))
    p.add_argument("--eff_target_col", type=str, default="itt_relative_efficiency")
    p.add_argument("--out_dir", type=Path,
                   default=Path("reports/benchmark/macro_simulation_eval/local12b_four_variant_rollout/figures"))
    p.add_argument("--n_repeats", type=int, default=50)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Load data ---
    learn_analysis = pd.read_csv(args.learn_analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    val_analysis = pd.read_csv(args.val_analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    learn_analysis["gameId"] = learn_analysis["gameId"].astype(str)
    val_analysis["gameId"] = val_analysis["gameId"].astype(str)

    learn_rates = build_game_level_rate_targets(args.learn_rows_csv, args.learn_analysis_csv)
    val_rates = build_game_level_rate_targets(args.val_rows_csv, args.val_analysis_csv)
    learn_rates["gameId"] = learn_rates["gameId"].astype(str)
    val_rates["gameId"] = val_rates["gameId"].astype(str)

    learn_table = learn_analysis.merge(learn_rates, on="gameId", how="left")
    val_table = val_analysis.merge(val_rates, on="gameId", how="left")

    all_config_ids = learn_table["CONFIG_configId"].unique()
    n_total_configs = len(all_config_ids)
    print(f"Total unique training configs: {n_total_configs}")

    # Scaling points
    n_configs_list = [2, 5, 10, 20, 40, 80, n_total_configs]
    n_configs_list = [n for n in n_configs_list if n <= n_total_configs]

    # Target definitions: (label, ols_target_col, parity_human_col, parity_sim_col)
    targets = [
        ("Contribution Rate", "target_contrib_rate", "human_contribution_rate", "sim_contribution_rate"),
        ("Normalized Efficiency", args.eff_target_col, "human_normalized_efficiency", "sim_normalized_efficiency"),
        ("Punishment Rate", "target_punishment_rate", "human_punishment_rate", "sim_punishment_rate"),
        ("Reward Rate", "target_reward_rate", "human_reward_rate", "sim_reward_rate"),
    ]

    # --- Scaling loop ---
    rng = np.random.default_rng(42)
    # results[label] = {n_configs: [rmse_seed0, rmse_seed1, ...]}
    scaling_results: dict[str, dict[int, list[float]]] = {t[0]: {} for t in targets}

    for n_cfg in n_configs_list:
        print(f"  n_configs={n_cfg} ...", end="", flush=True)
        n_rep = 1 if n_cfg == n_total_configs else args.n_repeats

        for label, target_col, _, _ in targets:
            scaling_results[label][n_cfg] = []

        for seed in range(n_rep):
            if n_cfg == n_total_configs:
                sampled_ids = all_config_ids
            else:
                sampled_ids = rng.choice(all_config_ids, size=n_cfg, replace=False)

            learn_subset = learn_table[learn_table["CONFIG_configId"].isin(set(sampled_ids))].copy()

            for label, target_col, _, _ in targets:
                rmse = train_ols_rmse(learn_subset, val_table, target_col)
                scaling_results[label][n_cfg].append(rmse)

        print(f" done ({n_rep} repeats, ~{len(learn_subset)} games)")

    # --- LLM RMSE ---
    llm_rmses: dict[str, dict[str, float]] = {}
    for label, _, human_col, sim_col in targets:
        llm_rmses[label] = compute_llm_rmse(args.per_game_csv, human_col, sim_col)

    # --- Plot ---
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_styles = {
        "no_archetype":       ("#1f77b4", "--", "LLM: no archetype"),
        "random_archetype":   ("#d62728", "--", "LLM: random archetype"),
        "retrieved_archetype": ("#2ca02c", "-.", "LLM: retrieved archetype"),
        "oracle_archetype":   ("#ff7f0e", "-.", "LLM: oracle archetype"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (label, target_col, human_col, sim_col) in enumerate(targets):
        ax = axes[idx]
        res = scaling_results[label]

        xs = sorted(res.keys())
        means = [np.nanmean(res[n]) for n in xs]
        lo = [np.nanpercentile(res[n], 2.5) if len(res[n]) > 1 else means[i] for i, n in enumerate(xs)]
        hi = [np.nanpercentile(res[n], 97.5) if len(res[n]) > 1 else means[i] for i, n in enumerate(xs)]

        ax.plot(xs, means, "k-o", linewidth=2.5, markersize=7, label="OLS config baseline", zorder=5)
        ax.fill_between(xs, lo, hi, alpha=0.15, color="black")

        # Annotate mean games per config count
        for i, n in enumerate(xs):
            n_games = int(learn_table[learn_table["CONFIG_configId"].isin(
                rng.choice(all_config_ids, size=min(n, n_total_configs), replace=False)
            )].shape[0]) if n < n_total_configs else len(learn_table)
            ax.annotate(f"{n}", (xs[i], means[i]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color="dimgray",
                        fontweight="bold")

        # LLM horizontal lines
        for variant in ["no_archetype", "random_archetype", "retrieved_archetype", "oracle_archetype"]:
            if variant not in llm_rmses[label]:
                continue
            color, ls, vlabel = variant_styles[variant]
            ax.axhline(
                llm_rmses[label][variant],
                color=color, linestyle=ls, linewidth=2, alpha=0.85,
                label=vlabel,
            )

        # Set y-axis limits: cap at reasonable range to show the interesting region
        all_llm_vals = [v for v in llm_rmses[label].values()]
        y_max = max(max(means), max(all_llm_vals)) * 1.5
        ax.set_ylim(0, min(y_max, 0.8))

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("# unique training configs")
        ax.grid(axis="y", alpha=0.25)

    # Single legend from first subplot
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.suptitle(
        "OLS Scaling Curve vs LLM Simulation\n"
        f"(OLS: subsample from {n_total_configs} training configs, evaluate on 249 val games; "
        "LLM: 31 shared games)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out_path = args.out_dir / "ols_scaling_vs_llm.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
