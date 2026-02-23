#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_output_Robin" / "learning_match_contribution"
PER_ROUND_PATH = BASE / "per_round_player_comparison.csv"
PER_PLAYER_PATH = BASE / "per_player_summary.csv"

PLOTS_DIR = BASE / "trajectory_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_ORDER = [
    "24_C",
    "24_T",
    "48_C",
    "48_T",
    "64_C",
    "64_T",
    "104_C",
    "104_T",
    "148_C",
    "148_T",
]


def classify_behavior(series: pd.Series, full_value: float = 20.0) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return "unknown"

    uniq = sorted(vals.unique())
    mean_v = float(vals.mean())
    std_v = float(vals.std(ddof=0))
    frac_zero = float((vals == 0).mean())
    frac_full = float((vals == full_value).mean())

    if (vals == full_value).all():
        return "always_full"
    if (vals == 0).all():
        return "always_zero"
    if set(uniq).issubset({0.0, full_value}) and len(uniq) > 1:
        return "binary_0_20"
    if mean_v >= 15 and frac_full >= 0.5:
        return "high_cooperator"
    if mean_v <= 5 and frac_zero >= 0.5:
        return "low_contributor"
    if std_v <= 2:
        return "stable_mid"
    return "variable_mid"


def make_game_panel_plot(game_df: pd.DataFrame, exp: str, out_png: Path) -> None:
    players = sorted(game_df["playerId"].dropna().unique().tolist())
    n = len(players)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, pid in enumerate(players):
        ax = axes_flat[i]
        p = game_df[game_df["playerId"] == pid].sort_values("round")
        ax.plot(p["round"], p["real_contribution"], marker="o", linewidth=1.8, label="Real")
        ax.plot(p["round"], p["sim_contribution"], marker="s", linewidth=1.8, label="Sim")
        mae = float(np.mean(np.abs(p["sim_contribution"] - p["real_contribution"])))
        ax.set_title(f"{pid[:8]}...  MAE={mae:.2f}", fontsize=9)
        ax.set_xlabel("Round")
        ax.set_ylabel("Contribution")
        ax.set_ylim(-0.5, 20.5)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{exp}: Real vs Sim Contribution Trajectories", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    if not PER_ROUND_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {PER_ROUND_PATH}")

    cmp_df = pd.read_csv(PER_ROUND_PATH)
    if PER_PLAYER_PATH.exists():
        player_summary = pd.read_csv(PER_PLAYER_PATH)
    else:
        player_summary = (
            cmp_df.groupby(["experiment", "real_gameId", "playerId"], as_index=False)
            .agg(
                n_rounds_compared=("round", "size"),
                mean_diff_sim_minus_real=("diff_sim_minus_real", "mean"),
                mae=("abs_diff", "mean"),
                rmse=("diff_sim_minus_real", lambda s: float(np.sqrt((s**2).mean()))),
            )
        )

    cmp_df["real_contribution"] = pd.to_numeric(cmp_df["real_contribution"], errors="coerce")
    cmp_df["sim_contribution"] = pd.to_numeric(cmp_df["sim_contribution"], errors="coerce")
    cmp_df["round"] = pd.to_numeric(cmp_df["round"], errors="coerce")
    cmp_df = cmp_df.dropna(subset=["experiment", "playerId", "round", "real_contribution", "sim_contribution"])

    # 1) Trajectory plots per game + one multi-page PDF
    pdf_path = PLOTS_DIR / "all_games_player_trajectories.pdf"
    with PdfPages(pdf_path) as pdf:
        for exp in EXPERIMENT_ORDER:
            g = cmp_df[cmp_df["experiment"] == exp].copy()
            if g.empty:
                continue
            out_png = PLOTS_DIR / f"{exp}_player_trajectories.png"
            make_game_panel_plot(g, exp, out_png)

            img = plt.imread(out_png)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(exp)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # 2) Mismatch patterns by player type
    real_type = (
        cmp_df.groupby(["experiment", "real_gameId", "playerId"])["real_contribution"]
        .apply(classify_behavior)
        .reset_index(name="real_type")
    )
    sim_type = (
        cmp_df.groupby(["experiment", "real_gameId", "playerId"])["sim_contribution"]
        .apply(classify_behavior)
        .reset_index(name="sim_type")
    )

    labels = real_type.merge(sim_type, on=["experiment", "real_gameId", "playerId"], how="inner")
    labels = labels.merge(
        player_summary[
            [
                "experiment",
                "real_gameId",
                "playerId",
                "n_rounds_compared",
                "mean_diff_sim_minus_real",
                "mae",
                "rmse",
                "pearson_r",
            ]
        ],
        on=["experiment", "real_gameId", "playerId"],
        how="left",
    )
    labels.to_csv(BASE / "player_type_labels.csv", index=False)

    by_real_type = (
        labels.groupby("real_type", as_index=False)
        .agg(
            n_players=("playerId", "size"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("mean_diff_sim_minus_real", "mean"),
            exact_match_rate=("mae", lambda s: float((s == 0).mean())),
            high_error_rate=("mae", lambda s: float((s > 5).mean())),
            mean_pearson_r=("pearson_r", "mean"),
        )
        .sort_values("mean_mae", ascending=False)
    )
    by_real_type.to_csv(BASE / "mismatch_by_real_player_type.csv", index=False)

    pair_error = (
        labels.groupby(["real_type", "sim_type"], as_index=False)
        .agg(
            n_players=("playerId", "size"),
            mean_mae=("mae", "mean"),
            mean_bias=("mean_diff_sim_minus_real", "mean"),
        )
        .sort_values(["n_players", "mean_mae"], ascending=[False, False])
    )
    pair_error.to_csv(BASE / "type_pair_error_summary.csv", index=False)

    confusion = pd.crosstab(labels["real_type"], labels["sim_type"])
    confusion.to_csv(BASE / "type_confusion_matrix_counts.csv")

    # Optional visuals for the type mismatch summary
    plt.figure(figsize=(8.5, 4.5))
    order = by_real_type["real_type"].tolist()
    sns.barplot(data=by_real_type, x="real_type", y="mean_mae", order=order, color="#4c78a8")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Real Behavior Type")
    plt.ylabel("Mean MAE")
    plt.title("Contribution Mismatch by Real Player Type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mae_by_real_type.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Type Confusion: Real Type (rows) vs Sim Type (cols)")
    plt.xlabel("Simulated Type")
    plt.ylabel("Real Type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "type_confusion_heatmap.png", dpi=180)
    plt.close()

    # Short human-readable report
    top_real = by_real_type.head(3)
    lines = []
    lines.append("# Learning-Match Contribution Analysis")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Per-game trajectory panels in `trajectory_plots/`")
    lines.append("- Multi-page PDF: `trajectory_plots/all_games_player_trajectories.pdf`")
    lines.append("- Type summary: `mismatch_by_real_player_type.csv`")
    lines.append("- Type confusion counts: `type_confusion_matrix_counts.csv`")
    lines.append("- Player type labels: `player_type_labels.csv`")
    lines.append("")
    lines.append("## Highest-Error Real Types (Mean MAE)")
    for _, row in top_real.iterrows():
        lines.append(
            f"- {row['real_type']}: mean_mae={row['mean_mae']:.3f}, "
            f"n_players={int(row['n_players'])}, mean_bias={row['mean_bias']:.3f}"
        )
    (BASE / "type_mismatch_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote plots to: {PLOTS_DIR}")
    print(f"Wrote: {BASE / 'player_type_labels.csv'}")
    print(f"Wrote: {BASE / 'mismatch_by_real_player_type.csv'}")
    print(f"Wrote: {BASE / 'type_pair_error_summary.csv'}")
    print(f"Wrote: {BASE / 'type_confusion_matrix_counts.csv'}")
    print(f"Wrote: {BASE / 'type_mismatch_report.md'}")


if __name__ == "__main__":
    main()
