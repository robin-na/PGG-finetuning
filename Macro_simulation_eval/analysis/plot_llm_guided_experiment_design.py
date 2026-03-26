#!/usr/bin/env python3
"""
LLM as Experimental Designer — Optimization framing.

Y-axis: best real outcome found so far (contribution rate / normalized efficiency)
X-axis: number of experiments run
Methods: Random, OLS-guided (cold), OLS-guided (with learning), LLM oracle, LLM retrieved
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from plot_macro_four_variant_summary import build_game_level_rate_targets


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------
def build_ols_pipeline(x_train: pd.DataFrame) -> Pipeline:
    num_cols = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
    cat_cols = [c for c in x_train.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("lr", Ridge(alpha=1.0))])


def ols_predict_remaining(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    target_col: str,
    remaining_ids: list[int],
) -> int:
    """Train OLS on train_df, predict candidates, return configId with highest predicted value."""
    config_cols = [c for c in train_df.columns if c.startswith("CONFIG_") and c in candidate_df.columns]
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    mask = y_train.notna()
    x_tr = train_df.loc[mask, config_cols]
    y_tr = y_train.loc[mask]

    if len(x_tr) < 2:
        return remaining_ids[0]

    model = build_ols_pipeline(x_tr)
    model.fit(x_tr, y_tr)

    cands = candidate_df[candidate_df["CONFIG_configId"].isin(set(remaining_ids))]
    preds = model.predict(cands[config_cols])
    best_idx = int(np.argmax(preds))
    return int(cands.iloc[best_idx]["CONFIG_configId"])


# ---------------------------------------------------------------------------
# Sequential optimization methods
# ---------------------------------------------------------------------------
def run_random_search(
    config_ids: list[int],
    gt_map: dict[int, float],
    rng: np.random.Generator,
) -> list[float]:
    """Random order → best-so-far at each step."""
    order = rng.permutation(config_ids).tolist()
    best_so_far = []
    current_best = -np.inf
    for cid in order:
        current_best = max(current_best, gt_map[cid])
        best_so_far.append(current_best)
    return best_so_far


def run_ols_guided_cold(
    config_ids: list[int],
    gt_map: dict[int, float],
    config_table: pd.DataFrame,
    target_col: str,
    rng: np.random.Generator,
) -> list[float]:
    """OLS cold start: random first 2, then greedy."""
    order = rng.permutation(config_ids).tolist()
    observed = []
    remaining = list(config_ids)
    best_so_far = []
    current_best = -np.inf

    for t in range(len(config_ids)):
        if t < 2:
            pick = order[t]
        else:
            obs_df = config_table[config_table["CONFIG_configId"].isin(set(observed))].copy()
            pick = ols_predict_remaining(obs_df, config_table, target_col, remaining)

        observed.append(pick)
        remaining = [c for c in remaining if c != pick]
        current_best = max(current_best, gt_map[pick])
        best_so_far.append(current_best)

    return best_so_far


def run_ols_guided_warm(
    config_ids: list[int],
    gt_map: dict[int, float],
    config_table: pd.DataFrame,
    learn_table: pd.DataFrame,
    target_col: str,
) -> list[float]:
    """OLS with learning wave prior. Deterministic (picks highest predicted each step)."""
    observed = []
    remaining = list(config_ids)
    best_so_far = []
    current_best = -np.inf

    for t in range(len(config_ids)):
        combined = pd.concat([learn_table] + (
            [config_table[config_table["CONFIG_configId"].isin(set(observed))]]
            if observed else []
        ), ignore_index=True)
        pick = ols_predict_remaining(combined, config_table, target_col, remaining)

        observed.append(pick)
        remaining = [c for c in remaining if c != pick]
        current_best = max(current_best, gt_map[pick])
        best_so_far.append(current_best)

    return best_so_far


def run_llm_guided(
    config_ids: list[int],
    gt_map: dict[int, float],
    llm_map: dict[int, float],
) -> list[float]:
    """LLM ranks configs by predicted value, observe in that order."""
    ranked = sorted(config_ids, key=lambda c: llm_map.get(c, 0), reverse=True)
    best_so_far = []
    current_best = -np.inf
    for cid in ranked:
        current_best = max(current_best, gt_map[cid])
        best_so_far.append(current_best)
    return best_so_far


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    p.add_argument("--parity_csv", type=Path,
                   default=Path("reports/benchmark/macro_simulation_eval/local12b_four_variant_rollout/game_level_parity.csv"))
    p.add_argument("--eff_target_col", type=str, default="itt_relative_efficiency")
    p.add_argument("--out_dir", type=Path,
                   default=Path("reports/benchmark/macro_simulation_eval/local12b_four_variant_rollout/figures"))
    p.add_argument("--n_repeats", type=int, default=500)
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

    # Per-config ground truth
    val_config_truth = val_table.groupby("CONFIG_configId", as_index=False).agg(
        gt_contrib=("target_contrib_rate", "mean"),
        gt_efficiency=(args.eff_target_col, "mean"),
    )
    all_config_ids = val_config_truth["CONFIG_configId"].tolist()
    n_configs = len(all_config_ids)
    print(f"Config universe: {n_configs} configs")

    # Config-level table (one row per config, with CONFIG features + ground truth)
    config_table = val_table.drop_duplicates(subset=["CONFIG_configId"], keep="first").copy()
    config_table = config_table.merge(val_config_truth, on="CONFIG_configId", how="left")

    # --- LLM predictions per config ---
    parity = pd.read_csv(args.parity_csv)
    parity["gameId"] = parity["gameId"].astype(str)
    parity = parity.merge(val_analysis[["gameId", "CONFIG_configId"]], on="gameId", how="left")

    variants_to_plot = {
        "oracle archetype": "LLM: oracle archetype",
        "retrieved archetype": "LLM: retrieved archetype",
    }

    targets = [
        ("Contribution Rate", "target_contrib_rate", "gt_contrib", "sim_contribution_rate"),
        ("Normalized Efficiency", args.eff_target_col, "gt_efficiency", "sim_normalized_efficiency"),
    ]

    # Build LLM prediction maps: {target: {variant: {configId: predicted_value}}}
    llm_maps: dict[str, dict[str, dict[int, float]]] = {}
    for label, _, _, sim_col in targets:
        llm_maps[label] = {}
        for source_name in variants_to_plot:
            sub = parity[parity["source"] == source_name]
            cfg_pred = sub.groupby("CONFIG_configId")[sim_col].mean().to_dict()
            llm_maps[label][source_name] = cfg_pred

    # --- Run experiments ---
    rng = np.random.default_rng(42)
    # results[target][method] = list of best-so-far curves (each curve = list of length n_configs)
    results: dict[str, dict[str, list[list[float]]]] = {}

    for label, ols_target, gt_col, sim_col in targets:
        print(f"\n=== {label} ===")
        gt_map = dict(zip(val_config_truth["CONFIG_configId"], val_config_truth[gt_col]))
        true_max = max(gt_map.values())
        print(f"  True optimum: {true_max:.4f}")

        results[label] = {}

        # 1. Random search
        print(f"  Random search ({args.n_repeats} repeats)...", end="", flush=True)
        random_curves = []
        for _ in range(args.n_repeats):
            curve = run_random_search(all_config_ids, gt_map, rng)
            random_curves.append(curve)
        results[label]["Random search"] = random_curves
        print(" done")

        # 2. OLS cold start
        print(f"  OLS cold start ({args.n_repeats} repeats)...", end="", flush=True)
        ols_cold_curves = []
        for _ in range(args.n_repeats):
            curve = run_ols_guided_cold(all_config_ids, gt_map, config_table, ols_target, rng)
            ols_cold_curves.append(curve)
        results[label]["OLS-guided (cold)"] = ols_cold_curves
        print(" done")

        # 3. LLM-guided
        for source_name, display_name in variants_to_plot.items():
            llm_map = llm_maps[label][source_name]
            curve = run_llm_guided(all_config_ids, gt_map, llm_map)
            results[label][display_name] = [curve]
            print(f"  {display_name}: step 1 = {curve[0]:.4f}, step 3 = {curve[2]:.4f}")

    # --- Plot ---
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_styles = {
        "Random search":            ("gray",    "-",  2.0),
        "OLS-guided (cold)":        ("#1f77b4", "-",  2.0),
        "LLM: oracle archetype":    ("#ff7f0e", "-",  2.5),
        "LLM: retrieved archetype": ("#2ca02c", "-",  2.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    steps = np.arange(1, n_configs + 1)

    for idx, (label, ols_target, gt_col, sim_col) in enumerate(targets):
        ax = axes[idx]
        gt_map = dict(zip(val_config_truth["CONFIG_configId"], val_config_truth[gt_col]))
        true_max = max(gt_map.values())

        # True optimum line
        ax.axhline(true_max, color="black", linestyle=":", linewidth=1.5, alpha=0.5, label="True optimum")

        for method, (color, ls, lw) in method_styles.items():
            curves = results[label].get(method, [])
            if not curves:
                continue
            arr = np.array(curves)  # (n_repeats, n_configs)
            mean = arr.mean(axis=0)
            ax.plot(steps, mean, color=color, linestyle=ls, linewidth=lw, label=method)

            if arr.shape[0] > 1:
                lo = np.percentile(arr, 2.5, axis=0)
                hi = np.percentile(arr, 97.5, axis=0)
                ax.fill_between(steps, lo, hi, alpha=0.12, color=color)

        ax.set_xlabel("# experiments run", fontsize=12)
        ax.set_ylabel(f"Best {label.lower()} found so far", fontsize=12)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlim(1, n_configs)
        ax.grid(axis="y", alpha=0.25)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    fig.suptitle(
        "LLM-Guided Experimental Design: Finding the Optimal Config\n"
        f"(Sequential search over {n_configs} candidate configs)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out_path = args.out_dir / "llm_guided_experiment_design.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
