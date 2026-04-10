from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    PRIMARY_METRIC_FAMILY,
    build_primary_distribution_summary,
    compute_treatment_metric_tables,
)
from common import (
    ROLE_A_CHECK,
    ROLE_A_TIME,
    ROLE_B_HIDDEN_CHECK,
    ROLE_B_HIDDEN_TIME,
    ROLE_B_OBSERVABLE_CHECK,
    ROLE_B_OBSERVABLE_TIME,
    _flatten_target_dict,
    build_human_records_df,
    write_csv,
    write_json,
)


RUN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
NOISE_CEILING_COLOR = "#2CA02C"
PRETTY_METRIC_LABELS = {
    "mean_send_field_distance": "Send",
    "mean_process_premium_distance": "Process Signal",
    "mean_action_premium_distance": "Act Signal",
    "return_pct": "Return %",
    "joint_pattern_distribution": "Joint Pattern",
    "process_premium_if_act": "Process Premium if Act",
    "process_premium_if_no_act": "Process Premium if No Act",
    "action_premium": "Act Premium",
    "action_premium_without_check": "Act Premium Without Check",
    "action_premium_after_check": "Act Premium After Check",
    "action_premium_fast": "Act Premium Fast",
    "action_premium_slow": "Act Premium Slow",
    "send_if_act": "Send if Act",
    "send_if_act_after_check": "Send if Act After Check",
    "send_if_act_fast": "Send if Act Fast",
    "send_if_act_slow": "Send if Act Slow",
    "send_if_act_without_check": "Send if Act Without Check",
    "send_if_no_act": "Send if No Act",
    "send_if_no_act_after_check": "Send if No Act After Check",
    "send_if_no_act_fast": "Send if No Act Fast",
    "send_if_no_act_slow": "Send if No Act Slow",
    "send_if_no_act_without_check": "Send if No Act Without Check",
}
HEADLINE_METRICS = [
    "mean_send_field_distance",
    "return_pct",
]


def _mean_stderr(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] <= 1:
        return mean, float("nan")
    stderr = float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))
    return mean, stderr


def _simulate_random_target(schema_type: str, rng: np.random.Generator) -> dict[str, Any]:
    if schema_type == ROLE_A_CHECK:
        return {
            "check": str(rng.choice(["YES", "NO"])),
            "act": str(rng.choice(["YES", "NO"])),
            "return_pct": int(rng.integers(0, 101)),
        }
    if schema_type == ROLE_A_TIME:
        return {
            "decision_time_bucket": str(rng.choice(["FAST", "SLOW"])),
            "act": str(rng.choice(["YES", "NO"])),
            "return_pct": int(rng.integers(0, 101)),
        }
    if schema_type == ROLE_B_OBSERVABLE_CHECK:
        return {
            "send_if_act_without_check": int(rng.integers(0, 11)),
            "send_if_act_after_check": int(rng.integers(0, 11)),
            "send_if_no_act_without_check": int(rng.integers(0, 11)),
            "send_if_no_act_after_check": int(rng.integers(0, 11)),
        }
    if schema_type == ROLE_B_HIDDEN_CHECK:
        return {
            "send_if_act": int(rng.integers(0, 11)),
            "send_if_no_act": int(rng.integers(0, 11)),
        }
    if schema_type == ROLE_B_OBSERVABLE_TIME:
        return {
            "send_if_act_fast": int(rng.integers(0, 11)),
            "send_if_no_act_fast": int(rng.integers(0, 11)),
            "send_if_act_slow": int(rng.integers(0, 11)),
            "send_if_no_act_slow": int(rng.integers(0, 11)),
        }
    if schema_type == ROLE_B_HIDDEN_TIME:
        return {
            "send_if_act": int(rng.integers(0, 11)),
            "send_if_no_act": int(rng.integers(0, 11)),
        }
    raise ValueError(f"Unsupported schema_type: {schema_type}")


def _simulate_random_distribution_summary(
    *,
    human_records: pd.DataFrame,
    iters: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    summary_parts: list[pd.DataFrame] = []
    base_columns = human_records.columns.tolist()

    for iter_idx in range(iters):
        generated_rows: list[dict[str, Any]] = []
        for row in human_records.to_dict(orient="records"):
            schema_type = str(row["schema_type"])
            target = _simulate_random_target(schema_type, rng)
            generated_rows.append(
                {
                    **{column: row.get(column) for column in base_columns if column not in {"data_source", *target.keys()}},
                    "data_source": "random_uniform",
                    **_flatten_target_dict(schema_type, target),
                }
            )
        generated_df = pd.DataFrame(generated_rows)
        _, _, overall_df = compute_treatment_metric_tables(
            generated_records=generated_df,
            human_records=human_records,
        )
        primary_df = build_primary_distribution_summary(overall_df).copy()
        primary_df["iter"] = iter_idx
        summary_parts.append(primary_df)

    combined = pd.concat(summary_parts, ignore_index=True)
    summary = (
        combined.groupby(["metric_family", "metric", "distance_kind", "n_treatments"], as_index=False)
        .agg(
            random_mean=("mean_value", "mean"),
            random_median=("mean_value", "median"),
            random_p05=("mean_value", lambda s: float(np.quantile(s, 0.05))),
            random_p95=("mean_value", lambda s: float(np.quantile(s, 0.95))),
        )
    )
    return summary


def _pretty_run_label(run_name: str) -> str:
    mapping = {
        "baseline_gpt_5_mini": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
        "baseline_gpt_5_1": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_1": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_1": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
    }
    return mapping.get(run_name, run_name.replace("_", " "))


def _plot_metric_panels(
    *,
    comparison_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    def distance_axis_label(metric: str) -> str:
        distance_kinds = comparison_df.loc[comparison_df["metric"] == metric, "distance_kind"].dropna().unique().tolist()
        if not distance_kinds:
            return "Distance"
        distance_kind = str(distance_kinds[0])
        if distance_kind in {"wasserstein_1d", "mean_wasserstein_1d"}:
            return "Wasserstein Distance"
        if distance_kind == "total_variation":
            return "Total Variation Distance"
        return "Distance"

    n_metrics = len(metrics)
    ncols = 3 if n_metrics <= 3 else 4
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.6 * nrows),
        constrained_layout=False,
    )
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.99, hspace=0.42, wspace=0.28)

    run_names = comparison_df["run_name"].drop_duplicates().tolist()
    series_order = [*run_names, "noise_ceiling"]
    x = np.arange(len(metrics), dtype=float)
    center_offset = (len(series_order) - 1) / 2.0
    bar_width = min(0.16, 0.84 / max(len(series_order), 1))

    for ax, metric in zip(axes_list, metrics):
        metric_df = comparison_df[comparison_df["metric"] == metric].copy()
        metric_df = metric_df.set_index("run_name").reindex(run_names).reset_index()
        noise_row = noise_df[noise_df["metric"] == metric]
        noise_mean = float(noise_row["bootstrap_mean"].iloc[0]) if not noise_row.empty else float("nan")
        noise_p05 = float(noise_row["bootstrap_p05"].iloc[0]) if not noise_row.empty else float("nan")
        noise_p95 = float(noise_row["bootstrap_p95"].iloc[0]) if not noise_row.empty else float("nan")

        upper_candidates: list[float] = [0.0]
        for index, run_name in enumerate(run_names):
            run_row = metric_df.loc[metric_df["run_name"] == run_name].iloc[0]
            run_value = float(run_row["mean_value"])
            run_stderr = float(run_row["stderr"]) if pd.notna(run_row["stderr"]) else float("nan")
            positions = x[metrics.index(metric)] + ((index - center_offset) * bar_width)
            bar = ax.bar(
                positions,
                run_value,
                width=bar_width,
                color=RUN_COLORS[index % len(RUN_COLORS)],
                edgecolor="white",
                linewidth=0.8,
                label=_pretty_run_label(run_name) if metric == metrics[0] else None,
                yerr=None if not math.isfinite(run_stderr) else np.array([[run_stderr], [run_stderr]]),
                capsize=3 if math.isfinite(run_stderr) else 0,
            )
            upper_candidates.extend([run_value, run_value + (run_stderr if math.isfinite(run_stderr) else 0.0)])
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2.0,
                run_value,
                f"{run_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if math.isfinite(noise_mean):
            noise_index = len(series_order) - 1
            noise_position = x[metrics.index(metric)] + ((noise_index - center_offset) * bar_width)
            yerr = np.array([[max(noise_mean - noise_p05, 0.0)], [max(noise_p95 - noise_mean, 0.0)]])
            noise_bar = ax.bar(
                noise_position,
                noise_mean,
                width=bar_width,
                color=NOISE_CEILING_COLOR,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
                label="Noise ceiling" if metric == metrics[0] else None,
                yerr=yerr,
                capsize=3,
            )
            upper_candidates.extend([noise_mean, noise_p95])
            ax.text(
                noise_bar[0].get_x() + noise_bar[0].get_width() / 2.0,
                noise_mean,
                f"{noise_mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        upper = max(upper_candidates)
        ax.set_ylim(0.0, upper * 1.18 + 1e-9)
        ax.set_title(PRETTY_METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylabel(distance_axis_label(metric))
        ax.set_xticks([x[metrics.index(metric)]])
        ax.set_xticklabels([PRETTY_METRIC_LABELS.get(metric, metric)], rotation=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)

    for ax in axes_list[n_metrics:]:
        ax.axis("off")

    handles = []
    labels = []
    if axes_list:
        for handle, label in zip(*axes_list[0].get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6), frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, fontsize=14, y=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot primary two-stage distribution-alignment metrics against the human noise ceiling."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names with completed vs-human treatment analysis.",
    )
    parser.add_argument("--random-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    default_output_name = "__vs__".join(args.runs) + "__plots"
    output_dir = args.output_dir or (args.forecasting_root / "results" / default_output_name)

    comparison_parts: list[pd.DataFrame] = []
    noise_parts: list[pd.DataFrame] = []
    for run_name in args.runs:
        summary_path = (
            args.forecasting_root / "results" / f"{run_name}__vs_human_treatments" / "primary_distribution_summary.csv"
        )
        run_df = pd.read_csv(summary_path)
        treatment_path = (
            args.forecasting_root / "results" / f"{run_name}__vs_human_treatments" / "treatment_distribution_distance.csv"
        )
        treatment_df = pd.read_csv(treatment_path)
        stderr_rows: list[dict[str, Any]] = []
        for (metric, distance_kind), group in treatment_df.groupby(["metric", "distance_kind"], sort=True):
            mean_score, stderr = _mean_stderr(group["score"])
            stderr_rows.append(
                {
                    "metric": metric,
                    "distance_kind": distance_kind,
                    "stderr": stderr,
                    "treatment_mean_check": mean_score,
                }
            )
        stderr_df = pd.DataFrame(stderr_rows)
        run_df = run_df.merge(
            stderr_df,
            on=["metric", "distance_kind"],
            how="left",
        )
        run_df["run_name"] = run_name
        comparison_parts.append(run_df)
        noise_path = (
            args.forecasting_root / "results" / f"{run_name}__noise_ceiling" / "primary_noise_ceiling_summary.csv"
        )
        noise_df = pd.read_csv(noise_path)
        noise_df["run_name"] = run_name
        noise_parts.append(noise_df)
    comparison_df = pd.concat(comparison_parts, ignore_index=True)
    noise_comparison_df = pd.concat(noise_parts, ignore_index=True)

    noise_df = (
        noise_comparison_df.groupby(["metric_family", "metric", "distance_kind"], as_index=False)
        .agg(
            n_treatments=("n_treatments", "first"),
            bootstrap_mean=("bootstrap_mean", "mean"),
            bootstrap_median=("bootstrap_median", "mean"),
            bootstrap_p05=("bootstrap_p05", "mean"),
            bootstrap_p95=("bootstrap_p95", "mean"),
        )
    )

    metrics = comparison_df["metric"].drop_duplicates().tolist()
    headline_metrics = [metric for metric in HEADLINE_METRICS if metric in metrics]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_primary_distribution_comparison.csv", comparison_df)
    write_csv(output_dir / "noise_ceiling_primary_distribution_summary.csv", noise_df)
    write_json(
        output_dir / "manifest.json",
        {
            "runs": args.runs,
            "output_dir": str(output_dir),
            "primary_metric_family": PRIMARY_METRIC_FAMILY,
        },
    )

    _plot_metric_panels(
        comparison_df=comparison_df,
        noise_df=noise_df,
        metrics=headline_metrics,
        output_path=output_dir / "headline_primary_distribution_comparison.png",
        title="Two-Stage Benchmark: Headline Distribution Metrics Vs Noise Ceiling",
    )
    _plot_metric_panels(
        comparison_df=comparison_df,
        noise_df=noise_df,
        metrics=metrics,
        output_path=output_dir / "all_primary_distribution_metrics.png",
        title="Two-Stage Benchmark: All Primary Distribution Metrics Vs Noise Ceiling",
    )


if __name__ == "__main__":
    main()
