from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_line(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path, dpi: int) -> str | None:
    data = df[[x, y]].dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data[x], data[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _save_fig(fig, out_path, dpi)


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path, dpi: int) -> str | None:
    data = df[[x, y]].dropna()
    if data.empty:
        return None
    width = max(8, 0.5 * len(data))
    fig, ax = plt.subplots(figsize=(width, 4))
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    return _save_fig(fig, out_path, dpi)


def _plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> str | None:
    if pivot.empty:
        return None
    values = pivot.values.astype(float)
    if values.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * pivot.shape[1]), max(4, 0.4 * pivot.shape[0])))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save_fig(fig, out_path, dpi)


def generate_all_plots(
    scored_df: pd.DataFrame,
    metrics_by_round: pd.DataFrame,
    metrics_by_game: pd.DataFrame,
    output_dir: Path,
    dpi: int = 160,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    contrib_err = pd.to_numeric(scored_df.get("contrib_error"), errors="coerce").dropna()
    if not contrib_err.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(contrib_err, bins=min(40, max(10, int(np.sqrt(len(contrib_err))))))
        ax.set_title("Contribution Error Histogram")
        ax.set_xlabel("Predicted - Actual")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        saved = _save_fig(fig, output_dir / "contribution_error_histogram.png", dpi)
        generated.append(saved)

    scatter_df = scored_df[["actual_contribution", "predicted_contribution"]].dropna()
    if not scatter_df.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(scatter_df["actual_contribution"], scatter_df["predicted_contribution"], alpha=0.6, s=16)
        mn = min(scatter_df["actual_contribution"].min(), scatter_df["predicted_contribution"].min())
        mx = max(scatter_df["actual_contribution"].max(), scatter_df["predicted_contribution"].max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        ax.set_title("Predicted vs Actual Contribution")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
        saved = _save_fig(fig, output_dir / "pred_vs_actual_contribution_scatter.png", dpi)
        generated.append(saved)

    line_specs = [
        ("contrib_mae", "Contribution MAE by Round", "MAE", "contribution_mae_by_round.png"),
        ("contrib_bias", "Contribution Bias by Round", "Bias (Predicted - Actual)", "contribution_bias_by_round.png"),
        ("action_exact_match", "Action Exact Match by Round", "Rate", "action_exact_match_by_round.png"),
        ("target_f1", "Target F1 by Round", "F1", "target_f1_by_round.png"),
        ("target_hit_any", "Target Hit-Any by Round", "Rate", "target_hit_any_by_round.png"),
    ]
    for metric_col, title, ylabel, file_name in line_specs:
        saved = _plot_line(
            metrics_by_round,
            x="roundIndex",
            y=metric_col,
            title=title,
            ylabel=ylabel,
            out_path=output_dir / file_name,
            dpi=dpi,
        )
        if saved:
            generated.append(saved)

    game_action = metrics_by_game[["gameId", "action_exact_match", "target_f1", "target_hit_any"]].dropna(how="all")
    if not game_action.empty:
        width = max(8, 0.7 * len(game_action))
        x = np.arange(len(game_action))
        bar_w = 0.25
        fig, ax = plt.subplots(figsize=(width, 4))
        ax.bar(x - bar_w, game_action["action_exact_match"], width=bar_w, label="Exact")
        ax.bar(x, game_action["target_f1"], width=bar_w, label="Target F1")
        ax.bar(x + bar_w, game_action["target_hit_any"], width=bar_w, label="Hit Any")
        ax.set_title("Action Metrics by Game")
        ax.set_ylabel("Rate")
        ax.set_xticks(x)
        ax.set_xticklabels(game_action["gameId"].astype(str), rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        saved = _save_fig(fig, output_dir / "action_metrics_by_game.png", dpi)
        generated.append(saved)

    saved = _plot_bar(
        metrics_by_game,
        x="gameId",
        y="contrib_mae",
        title="Contribution MAE by Game",
        ylabel="MAE",
        out_path=output_dir / "contribution_mae_by_game.png",
        dpi=dpi,
    )
    if saved:
        generated.append(saved)

    game_round_target = (
        scored_df.groupby(["gameId", "roundIndex"], dropna=False)["target_f1"].mean().reset_index().pivot(
            index="gameId", columns="roundIndex", values="target_f1"
        )
    )
    saved = _plot_heatmap(
        game_round_target,
        title="Game x Round Target F1",
        xlabel="Round",
        ylabel="Game",
        out_path=output_dir / "heatmap_game_round_target_f1.png",
        dpi=dpi,
    )
    if saved:
        generated.append(saved)

    game_round_mae = (
        scored_df.groupby(["gameId", "roundIndex"], dropna=False)["contrib_abs_error"].mean().reset_index().pivot(
            index="gameId", columns="roundIndex", values="contrib_abs_error"
        )
    )
    saved = _plot_heatmap(
        game_round_mae,
        title="Game x Round Contribution MAE",
        xlabel="Round",
        ylabel="Game",
        out_path=output_dir / "heatmap_game_round_contribution_mae.png",
        dpi=dpi,
    )
    if saved:
        generated.append(saved)

    return generated
