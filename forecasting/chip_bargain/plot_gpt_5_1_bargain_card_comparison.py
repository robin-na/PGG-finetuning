from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    import plot_model_family_macro_panels as macro
    import plot_model_family_micro_panels as micro
except ModuleNotFoundError:  # pragma: no cover
    from forecasting.chip_bargain import plot_model_family_macro_panels as macro
    from forecasting.chip_bargain import plot_model_family_micro_panels as micro


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
RUNS = [
    "baseline_gpt_5_1_pgg_aligned_v3",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1",
]
RUN_LABELS = {
    "baseline_gpt_5_1_pgg_aligned_v3": "Baseline",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3": "Twin Unadjusted",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1": "Twin Bargain Card",
}
RUN_COLORS = ["#4C78A8", "#F4A3A3", "#E45756"]


def _mean_and_stderr(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] <= 1:
        return mean, float("nan")
    return mean, float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))


def _row_ylim(
    metric_name: str,
    *,
    summary_df: pd.DataFrame,
    noise_summary: dict[str, dict[str, float]],
    random_summary: dict[str, dict[str, float]],
) -> float:
    metric_summary = summary_df[summary_df["metric"] == metric_name]
    upper = 0.0
    for _, row in metric_summary.iterrows():
        stderr = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0
        upper = max(upper, float(row["mean_value"]) + stderr)
    upper = max(upper, float(noise_summary[metric_name]["bootstrap_p95"]))
    upper = max(upper, float(random_summary[metric_name]["bootstrap_p95"]))
    return upper * 1.15 if upper > 0 else 1.0


def _plot_panel(
    ax: plt.Axes,
    *,
    metric_name: str,
    metric_label: str,
    summary_df: pd.DataFrame,
    noise_summary: dict[str, float],
    random_summary: dict[str, float],
    ylim: float,
) -> tuple[list, list]:
    metric_summary = summary_df[summary_df["metric"] == metric_name].set_index("run_name")
    x = np.arange(len(RUNS) + 1)
    handles: list = []
    labels: list[str] = []
    for idx, run_name in enumerate(RUNS):
        row = metric_summary.loc[run_name]
        mean_value = float(row["mean_value"])
        stderr = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0
        bar = ax.bar(
            x[idx],
            mean_value,
            width=0.72,
            color=RUN_COLORS[idx],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.92,
            zorder=3,
        )
        if np.isfinite(stderr) and stderr > 0:
            ax.errorbar(
                x[idx],
                mean_value,
                yerr=stderr,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                zorder=4,
            )
        handles.append(bar[0])
        labels.append(RUN_LABELS[run_name])

    noise_mean = float(noise_summary["bootstrap_mean"])
    noise_low = max(noise_mean - float(noise_summary["bootstrap_p05"]), 0.0)
    noise_high = max(float(noise_summary["bootstrap_p95"]) - noise_mean, 0.0)
    noise_bar = ax.bar(
        x[len(RUNS)],
        noise_mean,
        width=0.72,
        color=macro.NOISE_CEILING_COLOR,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )
    ax.errorbar(
        x[len(RUNS)],
        noise_mean,
        yerr=np.array([[noise_low], [noise_high]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
        zorder=4,
    )
    handles.append(noise_bar[0])
    labels.append("Human Ceiling")

    random_mean = float(random_summary["bootstrap_mean"])
    if np.isfinite(random_mean):
        ax.axhline(
            random_mean,
            color=macro.RANDOM_BASELINE_COLOR,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )

    ax.set_ylim(0, ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[r] for r in RUNS] + ["Human Ceiling"], rotation=24, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylabel(f"{metric_label}\nWasserstein Distance", fontsize=10)
    return handles, labels


def _macro_summary(
    bootstrap_iters: int,
    rng_seed: int,
    random_baseline_iters: int,
    ceiling_method: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    run_tables = {run_name: macro._load_run_tables(run_name) for run_name in RUNS}
    summary_rows: list[dict[str, float | str]] = []
    for run_name, (generated_game_df, human_game_df) in run_tables.items():
        for metric_name, _ in macro.METRICS:
            scores_df = macro._treatment_scores_for_metric(
                metric_name,
                generated_game_df=generated_game_df,
                human_game_df=human_game_df,
            )
            mean_value, stderr = _mean_and_stderr(scores_df["score"])
            summary_rows.append(
                {
                    "metric": metric_name,
                    "run_name": run_name,
                    "mean_value": mean_value,
                    "stderr": stderr,
                    "n_groups": int(scores_df["treatment_name"].nunique()) if not scores_df.empty else 0,
                }
            )
    shared_generated_count_map = macro._shared_generated_count_map(run_tables)
    example_human_df = next(iter(run_tables.values()))[1]
    noise_summary = {
        metric_name: macro._noise_ceiling_summary(
            metric_name,
            human_game_df=example_human_df,
            shared_generated_count_map=shared_generated_count_map,
            bootstrap_iters=bootstrap_iters,
            rng_seed=rng_seed,
            ceiling_method=ceiling_method,
        )
        for metric_name, _ in macro.METRICS
    }
    request_manifest_csv = macro._request_manifest_path_for_run(RUNS[0])
    random_summary = {
        metric_name: macro._random_baseline_summary(
            metric_name,
            request_manifest_csv=request_manifest_csv,
            human_game_df=example_human_df,
            shared_generated_count_map=shared_generated_count_map,
            random_baseline_iters=random_baseline_iters,
            rng_seed=rng_seed,
        )
        for metric_name, _ in macro.METRICS
    }
    return pd.DataFrame(summary_rows), noise_summary, random_summary


def _micro_summary(
    bootstrap_iters: int,
    rng_seed: int,
    random_baseline_iters: int,
    ceiling_method: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    run_tables = {run_name: micro._load_run_tables(run_name) for run_name in RUNS}
    summary_rows: list[dict[str, float | str]] = []
    for run_name, tables in run_tables.items():
        _, _, generated_player_df, human_player_df, generated_round_df, human_round_df = tables
        for metric_name, _ in micro.METRICS:
            scores_df = micro._treatment_scores_for_metric(
                metric_name,
                generated_player_df=generated_player_df,
                human_player_df=human_player_df,
                generated_round_df=generated_round_df,
                human_round_df=human_round_df,
            )
            mean_value, stderr = _mean_and_stderr(scores_df["score"])
            summary_rows.append(
                {
                    "metric": metric_name,
                    "run_name": run_name,
                    "mean_value": mean_value,
                    "stderr": stderr,
                    "n_groups": int(scores_df["treatment_name"].nunique()) if not scores_df.empty else 0,
                }
            )
    shared_generated_count_map = micro._shared_generated_count_map(run_tables)
    example_tables = next(iter(run_tables.values()))
    noise_summary = {
        metric_name: micro._noise_ceiling_summary(
            metric_name,
            human_game_df=example_tables[1],
            human_player_df=example_tables[3],
            human_round_df=example_tables[5],
            shared_generated_count_map=shared_generated_count_map,
            bootstrap_iters=bootstrap_iters,
            rng_seed=rng_seed,
            ceiling_method=ceiling_method,
        )
        for metric_name, _ in micro.METRICS
    }
    request_manifest_csv = micro._request_manifest_path_for_run(RUNS[0])
    random_summary = {
        metric_name: micro._random_baseline_summary(
            metric_name,
            request_manifest_csv=request_manifest_csv,
            human_player_df=example_tables[3],
            human_round_df=example_tables[5],
            shared_generated_count_map=shared_generated_count_map,
            random_baseline_iters=random_baseline_iters,
            rng_seed=rng_seed,
        )
        for metric_name, _ in micro.METRICS
    }
    return pd.DataFrame(summary_rows), noise_summary, random_summary


def _render_figure(
    *,
    metrics: list[tuple[str, str]],
    summary_df: pd.DataFrame,
    noise_summary: dict[str, dict[str, float]],
    random_summary: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.6, 3.2 * len(metrics)), constrained_layout=False)
    axes = np.asarray(axes).reshape(len(metrics))
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.19, right=0.98, hspace=0.38)

    legend_handles = None
    legend_labels = None
    for row_idx, (metric_name, metric_label) in enumerate(metrics):
        ylim = _row_ylim(
            metric_name,
            summary_df=summary_df,
            noise_summary=noise_summary,
            random_summary=random_summary,
        )
        handles, labels = _plot_panel(
            axes[row_idx],
            metric_name=metric_name,
            metric_label=metric_label,
            summary_df=summary_df,
            noise_summary=noise_summary[metric_name],
            random_summary=random_summary[metric_name],
            ylim=ylim,
        )
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels

    if legend_handles and legend_labels:
        legend_handles = list(legend_handles) + [
            Line2D([0], [0], color=macro.RANDOM_BASELINE_COLOR, linestyle="--", linewidth=1.5)
        ]
        legend_labels = list(legend_labels) + ["Random Baseline"]
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False)
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot chip-bargain gpt-5.1 baseline vs old/new Twin-card comparisons.")
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--ceiling-method", choices=["bootstrap", "split_half"], default="split_half")
    parser.add_argument("--rng-seed", type=int, default=29)
    parser.add_argument("--random-baseline-iters", type=int, default=50)
    args = parser.parse_args()

    macro_summary_df, macro_noise, macro_random = _macro_summary(
        bootstrap_iters=args.bootstrap_iters,
        rng_seed=args.rng_seed,
        random_baseline_iters=args.random_baseline_iters,
        ceiling_method=args.ceiling_method,
    )
    micro_summary_df, micro_noise, micro_random = _micro_summary(
        bootstrap_iters=args.bootstrap_iters,
        rng_seed=args.rng_seed,
        random_baseline_iters=max(30, min(args.random_baseline_iters, 50)),
        ceiling_method=args.ceiling_method,
    )

    _render_figure(
        metrics=macro.METRICS,
        summary_df=macro_summary_df,
        noise_summary=macro_noise,
        random_summary=macro_random,
        output_path=RESULTS_ROOT / "gpt_5_1_bargain_card_macro_comparison.png",
        title="Chip Bargaining (gpt-5.1)",
    )
    _render_figure(
        metrics=micro.METRICS,
        summary_df=micro_summary_df,
        noise_summary=micro_noise,
        random_summary=micro_random,
        output_path=RESULTS_ROOT / "gpt_5_1_bargain_card_micro_comparison.png",
        title="Chip Bargaining Micro-Level (gpt-5.1)",
    )

    macro_summary_df.to_csv(RESULTS_ROOT / "gpt_5_1_bargain_card_macro_comparison_summary.csv", index=False)
    micro_summary_df.to_csv(RESULTS_ROOT / "gpt_5_1_bargain_card_micro_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
