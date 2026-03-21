from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("contribution_rate_mae", "Contribution Rate MAE", False),
    ("future_normalized_efficiency_abs_error", "Future Normalized Efficiency Error", False),
    ("punish_target_f1", "Punish Target F1", True),
    ("reward_target_f1", "Reward Target F1", True),
]

PREFERRED_BASELINE_ORDER = [
    "compact_observer_llm",
    "gpt_4_1_mini",
    "gpt_5_1",
    "persistence",
    "ewma",
    "within_game_ar",
]

BASELINE_LABELS = {
    "compact_observer_llm": "LLM",
    "gpt_4_1_mini": "GPT-4.1-mini",
    "gpt_5_1": "GPT-5.1",
    "persistence": "Persistence",
    "ewma": "EWMA",
    "within_game_ar": "Within-Game AR",
}

BASELINE_COLORS = {
    "compact_observer_llm": "#1f77b4",
    "gpt_4_1_mini": "#1f77b4",
    "gpt_5_1": "#17becf",
    "persistence": "#d62728",
    "ewma": "#2ca02c",
    "within_game_ar": "#ff7f0e",
}


def _summarize_for_plot(game_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (baseline, k), group in game_summary_df.groupby(["baseline", "k"], sort=True):
        row: dict[str, object] = {
            "baseline": baseline,
            "k": int(k),
            "num_games": int(group["game_id"].nunique()),
        }
        for metric, _, _ in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_se"] = float(values.std(ddof=1) / (len(values) ** 0.5)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "baseline"]).reset_index(drop=True)


def _baseline_order(plot_df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(plot_df["baseline"].astype(str).tolist()))
    ordered = [baseline for baseline in PREFERRED_BASELINE_ORDER if baseline in present]
    ordered.extend(baseline for baseline in present if baseline not in ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot compact observer LLM vs baseline comparison with standard-error bars."
    )
    parser.add_argument("--game-summary-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Trajectory Completion Comparison")
    parser.add_argument("--baselines", type=str, default="")
    args = parser.parse_args()

    game_summary_df = pd.read_csv(args.game_summary_csv)
    if args.baselines.strip():
        allowed = {value.strip() for value in args.baselines.split(",") if value.strip()}
        game_summary_df = game_summary_df[game_summary_df["baseline"].isin(allowed)].copy()
        if game_summary_df.empty:
            raise RuntimeError("No rows remain after filtering to the requested baselines.")
    plot_df = _summarize_for_plot(game_summary_df)
    baseline_order = _baseline_order(plot_df)

    if args.output_summary_csv is not None:
        args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(args.output_summary_csv, index=False)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, higher_is_better) in zip(axes, METRICS):
        for baseline in baseline_order:
            baseline_df = plot_df[plot_df["baseline"] == baseline].sort_values("k")
            if baseline_df.empty:
                continue
            ax.errorbar(
                baseline_df["k"],
                baseline_df[f"{metric}_mean"],
                yerr=baseline_df[f"{metric}_se"],
                label=BASELINE_LABELS.get(baseline, baseline),
                color=BASELINE_COLORS.get(baseline),
                marker="o",
                linewidth=2,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_xticks(sorted(plot_df["k"].unique()))
        ax.grid(True, alpha=0.25)
        if higher_is_better:
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(max(len(labels), 1), 4),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.01),
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.suptitle(args.title, fontsize=16, y=1.03)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print(f"Wrote plot to {args.output_png}")
    if args.output_summary_csv is not None:
        print(f"Wrote plot summary to {args.output_summary_csv}")


if __name__ == "__main__":
    main()
