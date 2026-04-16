#!/usr/bin/env python3
"""Analyze pure statistical baselines on MobLab first-round and multiround behavior."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from analyze_moblab_persistence_and_correlation import (
    MEASURE_LABELS,
    build_all_rounds,
    configure_plot_style,
    plot_corr_heatmap,
    summarize_sessions,
    summarize_users,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "moblab_statistical_baselines"
)
PRIOR_RICH_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "moblab_multiround_analysis"
)

FAMILY_TO_MEASURES = {
    "dictator": ["dictator"],
    "trust": ["trust_investor", "trust_banker"],
    "ultimatum": ["ultimatum_proposer", "ultimatum_responder"],
    "pg": ["pg_contribution"],
}
FAMILY_LABELS = {
    "dictator": "Dictator",
    "trust": "Trust",
    "ultimatum": "Ultimatum",
    "pg": "Public Goods",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def spearman_value(left: pd.Series, right: pd.Series) -> float:
    merged = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(merged) < 3:
        return float("nan")
    ranked = merged.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def pearson_value(left: pd.Series, right: pd.Series) -> float:
    merged = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(merged) < 3:
        return float("nan")
    return float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))


def cv_splits(n_rows: int) -> int:
    return max(3, min(5, n_rows))


def evaluate_cv_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> Tuple[np.ndarray, Dict[str, float]]:
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    usable_cols = [
        col
        for col in X.columns
        if X[col].notna().sum() >= max(5, int(len(X) * 0.02)) and X[col].dropna().nunique() > 1
    ]
    if not usable_cols:
        usable_cols = [col for col in X.columns if X[col].notna().sum() >= 2 and X[col].dropna().nunique() > 1]
    if not usable_cols:
        raise ValueError("No usable feature columns after sanitization.")
    X = X[usable_cols]
    cv = KFold(n_splits=cv_splits(len(X)), shuffle=True, random_state=seed)
    preds = cross_val_predict(model, X, y, cv=cv, method="predict")
    metrics = {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(mean_squared_error(y, preds) ** 0.5),
        "pred_spearman": spearman_value(y, pd.Series(preds, index=y.index)),
        "pred_pearson": pearson_value(y, pd.Series(preds, index=y.index)),
    }
    return preds, metrics


def linear_1d_model() -> Pipeline:
    return Pipeline([("linear", LinearRegression())])


def ridge_multifeature_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("linear", LinearRegression()),
        ]
    )


def pairwise_first_round_prediction(first_user: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    measures = [m for m in MEASURE_LABELS if m in first_user.columns]
    for source in measures:
        for target in measures:
            if source == target:
                continue
            merged = first_user[[source, target]].dropna()
            if len(merged) < 80:
                continue
            X = merged[[source]].astype(float)
            y = merged[target].astype(float)
            _, cv_metrics = evaluate_cv_model(linear_1d_model(), X, y, seed)
            pearson = pearson_value(merged[source], merged[target])
            spearman = spearman_value(merged[source], merged[target])
            rows.append(
                {
                    "source_measure": source,
                    "target_measure": target,
                    "source_label": MEASURE_LABELS[source],
                    "target_label": MEASURE_LABELS[target],
                    "n_users": int(len(merged)),
                    "source_mean": float(merged[source].mean()),
                    "target_mean": float(merged[target].mean()),
                    "target_std": float(merged[target].std(ddof=0)),
                    "pearson": pearson,
                    "pearson_sq": pearson ** 2 if pd.notna(pearson) else float("nan"),
                    "spearman": spearman,
                    "cv_r2": cv_metrics["r2"],
                    "cv_mae": cv_metrics["mae"],
                    "cv_rmse": cv_metrics["rmse"],
                    "pred_spearman": cv_metrics["pred_spearman"],
                    "pred_pearson": cv_metrics["pred_pearson"],
                }
            )
    return pd.DataFrame(rows).sort_values(["target_measure", "source_measure"]).reset_index(drop=True)


def family_first_round_prediction(first_user: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for family, cols in FAMILY_TO_MEASURES.items():
        usable_cols = [col for col in cols if col in first_user.columns]
        if not usable_cols:
            continue
        for target in [m for m in MEASURE_LABELS if m in first_user.columns]:
            if target in usable_cols:
                continue
            candidate = first_user[usable_cols + [target]].copy()
            source_mask = candidate[usable_cols].notna().any(axis=1)
            usable = candidate[source_mask & candidate[target].notna()].copy()
            if len(usable) < 120:
                continue
            X = usable[usable_cols].astype(float)
            y = usable[target].astype(float)
            _, metrics = evaluate_cv_model(ridge_multifeature_model(), X, y, seed)
            rows.append(
                {
                    "source_family": family,
                    "source_label": FAMILY_LABELS[family],
                    "target_measure": target,
                    "target_label": MEASURE_LABELS[target],
                    "n_users": int(len(usable)),
                    "n_source_features": len(usable_cols),
                    "cv_r2": metrics["r2"],
                    "cv_mae": metrics["mae"],
                    "cv_rmse": metrics["rmse"],
                    "pred_spearman": metrics["pred_spearman"],
                    "pred_pearson": metrics["pred_pearson"],
                }
            )
    return pd.DataFrame(rows).sort_values(["source_family", "target_measure"]).reset_index(drop=True)


def all_other_games_prediction(first_user: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    measures = [m for m in MEASURE_LABELS if m in first_user.columns]
    for target in measures:
        feature_cols = [col for col in measures if col != target]
        candidate = first_user[feature_cols + [target]].copy()
        source_mask = candidate[feature_cols].notna().any(axis=1)
        usable = candidate[source_mask & candidate[target].notna()].copy()
        if len(usable) < 120:
            continue
        X = usable[feature_cols].astype(float)
        y = usable[target].astype(float)
        _, metrics = evaluate_cv_model(ridge_multifeature_model(), X, y, seed)
        rows.append(
            {
                "target_measure": target,
                "target_label": MEASURE_LABELS[target],
                "n_users": int(len(usable)),
                "n_source_features": len(feature_cols),
                "cv_r2": metrics["r2"],
                "cv_mae": metrics["mae"],
                "cv_rmse": metrics["rmse"],
                "pred_spearman": metrics["pred_spearman"],
                "pred_pearson": metrics["pred_pearson"],
            }
        )
    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)


def k1_persistence_baseline(session_df: pd.DataFrame) -> pd.DataFrame:
    strict = session_df[session_df["session_rounds"] >= 2].copy()
    rows: List[Dict[str, float]] = []
    for measure, group in strict.groupby("measure", sort=False):
        y = group["future_mean"].astype(float)
        preds = group["first_round_value"].astype(float)
        rows.append(
            {
                "measure": measure,
                "label": MEASURE_LABELS[measure],
                "n_sessions": int(len(group)),
                "n_users": int(group["UserID"].nunique()),
                "median_session_rounds": float(group["session_rounds"].median()),
                "round1_mean": float(preds.mean()),
                "future_mean": float(y.mean()),
                "future_std": float(y.std(ddof=0)),
                "persistence_r2": float(r2_score(y, preds)),
                "persistence_mae": float(mean_absolute_error(y, preds)),
                "persistence_rmse": float(mean_squared_error(y, preds) ** 0.5),
                "persistence_spearman": spearman_value(y, preds),
                "persistence_pearson": pearson_value(y, preds),
            }
        )
    table = pd.DataFrame(rows)
    table["measure"] = pd.Categorical(table["measure"], categories=list(MEASURE_LABELS.keys()), ordered=True)
    return table.sort_values("measure").reset_index(drop=True)


def matrix_from_long(df: pd.DataFrame, row_col: str, col_col: str, value_col: str, row_order: Iterable[str], col_order: Iterable[str]) -> pd.DataFrame:
    matrix = df.pivot(index=row_col, columns=col_col, values=value_col)
    return matrix.reindex(index=list(row_order), columns=list(col_order))


def plot_pairwise_r2_heatmap(pairwise_df: pd.DataFrame, out_path: Path) -> None:
    measures = [m for m in MEASURE_LABELS if m in set(pairwise_df["source_measure"]).union(pairwise_df["target_measure"])]
    matrix = matrix_from_long(pairwise_df, "source_measure", "target_measure", "cv_r2", measures, measures)
    fig, ax = plt.subplots(figsize=(9.0, 7.7))
    cmap = LinearSegmentedColormap.from_list("r2map", ["#c44536", "#f2d7a1", "#fbf7ef", "#7fb3c8", "#1d3557"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=-0.05, vmax=0.12)
    ax.set_xticks(np.arange(len(measures)), labels=[MEASURE_LABELS[m] for m in measures], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(measures)), labels=[MEASURE_LABELS[m] for m in measures])
    ax.set_title("Task 1: Pairwise First-Round Statistical Prediction (CV R^2)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = "" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.82, label="5-fold CV R^2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_family_r2_heatmap(family_df: pd.DataFrame, out_path: Path) -> None:
    rows = [family for family in FAMILY_TO_MEASURES if family in set(family_df["source_family"])]
    cols = [measure for measure in MEASURE_LABELS if measure in set(family_df["target_measure"])]
    matrix = matrix_from_long(family_df, "source_family", "target_measure", "cv_r2", rows, cols)
    fig, ax = plt.subplots(figsize=(8.9, 6.4))
    cmap = LinearSegmentedColormap.from_list("familyr2", ["#c44536", "#f2d7a1", "#fbf7ef", "#7fb3c8", "#1d3557"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=-0.05, vmax=0.12)
    ax.set_xticks(np.arange(len(cols)), labels=[MEASURE_LABELS[m] for m in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)), labels=[FAMILY_LABELS[f] for f in rows])
    ax.set_title("Task 1: Family-Level First-Round Baseline (CV R^2)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = "" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.82, label="5-fold CV R^2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_all_other_bar(all_other_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = all_other_df.sort_values("cv_r2")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bars = ax.barh(plot_df["target_label"], plot_df["cv_r2"], color="#1f6f8b", alpha=0.92)
    ax.axvline(0.0, color="#1d2a3a", linewidth=1.0)
    ax.set_xlabel("5-fold CV R^2")
    ax.set_title("Task 1: Predict Each First-Round Target from All Other Games")
    for bar, value in zip(bars, plot_df["cv_r2"]):
        ax.text(value + (0.002 if value >= 0 else -0.002), bar.get_y() + bar.get_height() / 2.0, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_persistence_bar(persistence_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = persistence_df[persistence_df["n_sessions"] >= 100].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    axes[0].barh(plot_df["label"], plot_df["persistence_r2"], color="#2a9d8f", alpha=0.9)
    axes[0].axvline(0.0, color="#1d2a3a", linewidth=1.0)
    axes[0].set_title("Task 2: K=1 Persistence R^2 (n>=100)")
    axes[0].set_xlabel("R^2 against future mean")
    axes[1].barh(plot_df["label"], plot_df["persistence_mae"], color="#d17b0f", alpha=0.9)
    axes[1].set_title("Task 2: K=1 Persistence MAE (n>=100)")
    axes[1].set_xlabel("MAE against future mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_corr_vs_r2(pairwise_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    x = pairwise_df["pearson_sq"].astype(float).to_numpy()
    y = pairwise_df["cv_r2"].astype(float).to_numpy()
    ax.scatter(x, y, s=54, color="#1f6f8b", alpha=0.78, edgecolors="white", linewidth=0.6)
    upper = max(np.nanmax(x), np.nanmax(y), 0.08)
    ax.plot([0.0, upper], [0.0, upper], linestyle="--", color="#8d99ae", linewidth=1.2, label="Pearson^2 = CV R^2")
    for row in pairwise_df.itertuples():
        if row.pearson_sq >= 0.025 or row.cv_r2 >= 0.02:
            label = f"{row.source_measure} -> {row.target_measure}"
            ax.text(row.pearson_sq + 0.001, row.cv_r2 + 0.001, label, fontsize=8)
    ax.set_xlabel("Pairwise Pearson^2 on overlap users")
    ax.set_ylabel("Pairwise 5-fold CV R^2")
    ax.set_title("Why Correlation Can Look Stronger Than Prediction Power")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rich_vs_first_round(family_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    prior_df = pd.read_csv(PRIOR_RICH_DIR / "crossgame_rich_vs_scalar.csv")
    target_map = {
        "dictator": "dictator",
        "trust_1": "trust_investor",
        "trust_3": "trust_banker",
        "ultimatum_1": "ultimatum_proposer",
        "ultimatum_2": "ultimatum_responder",
    }
    prior_df["target_measure"] = prior_df["target"].map(target_map)
    merged = family_df.merge(
        prior_df[["source_family", "target_measure", "rich_r2", "baseline_r2", "n_users"]],
        on=["source_family", "target_measure"],
        how="inner",
        suffixes=("_first_round", "_rich"),
    )
    merged["rich_minus_first_round_r2"] = merged["rich_r2"] - merged["cv_r2"]

    plot_df = merged.sort_values("rich_minus_first_round_r2")
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    labels = [f"{row.source_family} -> {row.target_measure}" for row in plot_df.itertuples()]
    bars = ax.barh(labels, plot_df["rich_minus_first_round_r2"], color=["#2a9d8f" if value >= 0 else "#c44536" for value in plot_df["rich_minus_first_round_r2"]])
    ax.axvline(0.0, color="#1d2a3a", linewidth=1.0)
    ax.set_xlabel("Rich multiround R^2 - first-round family baseline R^2")
    ax.set_title("Increment from Rich Multiround Features over First-Round Family Baselines")
    for bar, value in zip(bars, plot_df["rich_minus_first_round_r2"]):
        ax.text(value + (0.002 if value >= 0 else -0.002), bar.get_y() + bar.get_height() / 2.0, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return merged


def build_summary_markdown(
    pairwise_df: pd.DataFrame,
    family_df: pd.DataFrame,
    all_other_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> str:
    other_only = pairwise_df.copy()
    other_only["same_family"] = other_only.apply(
        lambda row: any(row["source_measure"] in cols and row["target_measure"] in cols for cols in FAMILY_TO_MEASURES.values()),
        axis=1,
    )
    cross_game_only = other_only[~other_only["same_family"]].copy()
    top_corr = cross_game_only.sort_values("spearman", ascending=False).head(5)
    top_pred = cross_game_only.sort_values("cv_r2", ascending=False).head(5)
    top_all_other = all_other_df.sort_values("cv_r2", ascending=False).head(6)
    richest_gain = compare_df.sort_values("rich_minus_first_round_r2", ascending=False).head(5)
    richest_loss = compare_df.sort_values("rich_minus_first_round_r2").head(5)

    lines = [
        "# MobLab Statistical Baselines",
        "",
        "## Setup",
        "",
        "- Task 1: predict a target game's first-round action using other games' first-round actions only.",
        "- Task 2: predict the same session's future mean behavior using the session's round-1 action only (`k=1 persistence`).",
        "- Trust is restricted to the fixed scale `Total[0] == 100`.",
        "- Task 1 includes ultimatum because it is effectively a one-shot game in this dataset.",
        "- All predictive metrics are 5-fold cross-validated and use no LLM components.",
        "",
        "## Why the Earlier Correlations Can Look Larger Than Predictive Power",
        "",
        f"- Among cross-game pairs, the largest first-round Spearman is only `{top_corr.iloc[0]['spearman']:.3f}` and the largest Pearson^2 is `{cross_game_only['pearson_sq'].max():.3f}`.",
        "- A rank correlation around 0.15 to 0.20 can look visually strong in a heatmap, but it implies only a few percent of explainable variance under a simple statistical predictor.",
        "- The earlier `Cross-Game Predictive Power` plot reports out-of-fold R^2, which is much stricter than pairwise correlation.",
        "- Several apparently strong correlations come from bounded or discrete actions where rank ordering is easier than accurate level prediction.",
        "- For Task 2, ultimatum has only 37 multiround sessions, so the persistence comparison should be read off the dictator, trust, and PGG rows.",
        "",
        "## Task 1: Strongest Cross-Game First-Round Correlations",
        "",
    ]
    for row in top_corr.itertuples():
        lines.append(
            f"- {row.source_label} -> {row.target_label}: n={row.n_users}, Spearman={row.spearman:.3f}, Pearson={row.pearson:.3f}, CV R^2={row.cv_r2:.3f}"
        )
    lines.extend(
        [
            "",
            "## Task 1: Strongest Cross-Game First-Round Prediction Baselines",
            "",
        ]
    )
    for row in top_pred.itertuples():
        lines.append(
            f"- {row.source_label} -> {row.target_label}: n={row.n_users}, CV R^2={row.cv_r2:.3f}, MAE={row.cv_mae:.3f}, Spearman={row.spearman:.3f}"
        )
    lines.extend(
        [
            "",
            "## Task 1: Predicting Each Target from All Other First-Round Games",
            "",
        ]
    )
    for row in top_all_other.itertuples():
        lines.append(
            f"- {row.target_label}: n={row.n_users}, CV R^2={row.cv_r2:.3f}, MAE={row.cv_mae:.3f}, prediction Spearman={row.pred_spearman:.3f}"
        )
    lines.extend(
        [
            "",
            "## Task 2: K=1 Persistence Baseline",
            "",
            "| Measure | Sessions | Users | Persistence R^2 | Persistence MAE | Persistence Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in persistence_df.itertuples():
        lines.append(
            f"| {row.label} | {row.n_sessions} | {row.n_users} | {row.persistence_r2:.3f} | {row.persistence_mae:.3f} | {row.persistence_spearman:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rich Multiround vs First-Round Family Baselines",
            "",
        ]
    )
    for row in richest_gain.itertuples():
        lines.append(
            f"- Gain: {row.source_family} -> {row.target_measure}: first-round baseline R^2={row.cv_r2:.3f}, rich R^2={row.rich_r2:.3f}, delta={row.rich_minus_first_round_r2:+.3f}"
        )
    for row in richest_loss.itertuples():
        lines.append(
            f"- Loss: {row.source_family} -> {row.target_measure}: first-round baseline R^2={row.cv_r2:.3f}, rich R^2={row.rich_r2:.3f}, delta={row.rich_minus_first_round_r2:+.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    configure_plot_style()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rounds_df = build_all_rounds()
    session_df = summarize_sessions(rounds_df)
    first_user = summarize_users(session_df, "first_round_value")
    first_corr = first_user.corr(method="spearman")

    pairwise_df = pairwise_first_round_prediction(first_user, seed=args.seed)
    family_df = family_first_round_prediction(first_user, seed=args.seed)
    all_other_df = all_other_games_prediction(first_user, seed=args.seed)
    persistence_df = k1_persistence_baseline(session_df)
    compare_df = plot_rich_vs_first_round(family_df, plots_dir / "rich_minus_first_round_family_baseline.png")

    pairwise_df.to_csv(output_dir / "pairwise_first_round_prediction.csv", index=False)
    family_df.to_csv(output_dir / "family_first_round_prediction.csv", index=False)
    all_other_df.to_csv(output_dir / "all_other_first_round_prediction.csv", index=False)
    persistence_df.to_csv(output_dir / "k1_persistence_baseline.csv", index=False)
    compare_df.to_csv(output_dir / "rich_vs_first_round_family_baseline.csv", index=False)
    first_corr.to_csv(output_dir / "first_round_spearman_matrix.csv")

    plot_corr_heatmap(first_corr.loc[[m for m in MEASURE_LABELS if m in first_corr.index], [m for m in MEASURE_LABELS if m in first_corr.columns]], "First-Round Cross-Game Spearman Correlation", plots_dir / "first_round_corr_heatmap.png")
    plot_pairwise_r2_heatmap(pairwise_df, plots_dir / "pairwise_first_round_r2_heatmap.png")
    plot_family_r2_heatmap(family_df, plots_dir / "family_first_round_r2_heatmap.png")
    plot_all_other_bar(all_other_df, plots_dir / "all_other_first_round_r2_bar.png")
    plot_persistence_bar(persistence_df, plots_dir / "k1_persistence_bar.png")
    plot_corr_vs_r2(pairwise_df, plots_dir / "correlation_vs_prediction_power.png")

    summary_md = build_summary_markdown(pairwise_df, family_df, all_other_df, persistence_df, compare_df)
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Wrote MobLab statistical baseline analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
