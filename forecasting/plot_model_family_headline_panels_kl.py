from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "forecasting").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.kl_divergence_utils import bootstrap_summary, histogram_kl_divergence, mean_and_stderr


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
RUN_FAMILIES = {
    "gpt-5-mini": {
        "runs": [
            "baseline_gpt_5_mini",
            "demographic_only_row_resampled_seed_0_gpt_5_mini",
            "twin_sampled_seed_0_gpt_5_mini",
            "twin_sampled_unadjusted_seed_0_gpt_5_mini",
        ],
        "baseline_run": "baseline_gpt_5_mini",
    },
    "gpt-5.1": {
        "runs": [
            "baseline_gpt_5_1",
            "demographic_only_row_resampled_seed_0_gpt_5_1",
            "twin_sampled_seed_0_gpt_5_1",
            "twin_sampled_unadjusted_seed_0_gpt_5_1",
        ],
        "baseline_run": "baseline_gpt_5_1",
    },
}
RUN_LABELS = {
    "baseline_gpt_5_mini": "Baseline",
    "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
    "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
    "baseline_gpt_5_1": "Baseline",
    "demographic_only_row_resampled_seed_0_gpt_5_1": "Demographic Only",
    "twin_sampled_seed_0_gpt_5_1": "Twin-Sampled",
    "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
}
RUN_COLORS = ["#4C78A8", "#9ECAE1", "#D62728", "#F4A3A3"]
NOISE_CEILING_COLOR = "#8C8C8C"
METRICS = [
    ("mean_round_normalized_efficiency", "Mean Normalized Efficiency"),
    ("mean_total_contribution_rate", "Mean Contribution Rate"),
]
ALPHA = 1.0
BOOTSTRAP_ITERS = 300
RNG_SEED = 23


def _load_run_game_tables(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = RESULTS_ROOT / f"{run_name}__vs_human_treatments"
    generated_df = pd.read_csv(run_dir / "generated_game_summary.csv")
    human_df = pd.read_csv(run_dir / "human_game_summary.csv")
    return human_df, generated_df


def _build_bin_edges(all_values: list[pd.Series]) -> np.ndarray:
    concatenated = pd.concat([series.dropna().astype(float) for series in all_values if not series.dropna().empty], ignore_index=True)
    min_value = float(concatenated.min())
    max_value = float(concatenated.max())
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    edges = np.linspace(min_value, max_value, 21)
    edges[-1] += 1e-9
    return edges


def _metric_treatment_scores(
    generated_df: pd.DataFrame,
    human_df: pd.DataFrame,
    metric_name: str,
    *,
    bin_edges: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    common_treatments = sorted(
        set(generated_df["treatment_name"].dropna().astype(str))
        & set(human_df["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_df[generated_df["treatment_name"] == treatment_name]
        human_group = human_df[human_df["treatment_name"] == treatment_name]
        score = histogram_kl_divergence(
            human_group[metric_name].dropna().astype(float),
            generated_group[metric_name].dropna().astype(float),
            bin_edges=bin_edges,
            alpha=ALPHA,
        )
        if np.isfinite(score):
            rows.append({"treatment_name": treatment_name, "score": score})
    return pd.DataFrame(rows)


def _noise_ceiling_summary(
    *,
    human_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    metric_name: str,
    bin_edges: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(RNG_SEED)
    common_treatments = sorted(
        set(generated_df["treatment_name"].dropna().astype(str))
        & set(human_df["treatment_name"].dropna().astype(str))
    )
    bootstrap_scores: list[float] = []
    for _ in range(BOOTSTRAP_ITERS):
        treatment_scores: list[float] = []
        for treatment_name in common_treatments:
            human_group = human_df[human_df["treatment_name"] == treatment_name].copy().reset_index(drop=True)
            generated_group = generated_df[generated_df["treatment_name"] == treatment_name].copy().reset_index(drop=True)
            pseudo_generated = human_group.iloc[rng.integers(0, len(human_group), size=len(generated_group))].copy()
            pseudo_human = human_group.iloc[rng.integers(0, len(human_group), size=len(human_group))].copy()
            score = histogram_kl_divergence(
                pseudo_human[metric_name].dropna().astype(float),
                pseudo_generated[metric_name].dropna().astype(float),
                bin_edges=bin_edges,
                alpha=ALPHA,
            )
            if np.isfinite(score):
                treatment_scores.append(score)
        if treatment_scores:
            bootstrap_scores.append(float(np.mean(treatment_scores)))
    return bootstrap_summary(bootstrap_scores)


def _row_ylim(
    metric_name: str,
    family_scores: dict[str, pd.DataFrame],
    family_noise: dict[str, dict[str, float]],
) -> float:
    upper = 0.0
    for family_name, summary_df in family_scores.items():
        metric_summary = summary_df[summary_df["metric"] == metric_name]
        for _, row in metric_summary.iterrows():
            stderr = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0
            upper = max(upper, float(row["mean_value"]) + stderr)
        upper = max(upper, float(family_noise[family_name][metric_name]["bootstrap_p95"]))
    return upper * 1.15 if upper > 0 else 1.0


def _plot_panel(
    ax: plt.Axes,
    *,
    family_name: str,
    metric_name: str,
    metric_label: str,
    summary_df: pd.DataFrame,
    noise_summary: dict[str, float],
    ylim: float,
    show_ylabel: bool,
) -> tuple[list, list]:
    metric_summary = summary_df[summary_df["metric"] == metric_name].set_index("run_name")
    runs = RUN_FAMILIES[family_name]["runs"]
    x = np.arange(len(runs) + 1)
    handles = []
    labels = []
    for idx, run_name in enumerate(runs):
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
            yerr=[[stderr], [stderr]],
            capsize=4,
            alpha=0.92,
            label=RUN_LABELS[run_name],
        )
        handles.append(bar[0])
        labels.append(RUN_LABELS[run_name])
    noise_mean = float(noise_summary["bootstrap_mean"])
    noise_low = max(noise_mean - float(noise_summary["bootstrap_p05"]), 0.0)
    noise_high = max(float(noise_summary["bootstrap_p95"]) - noise_mean, 0.0)
    noise_bar = ax.bar(
        x[-1],
        noise_mean,
        width=0.72,
        color=NOISE_CEILING_COLOR,
        edgecolor="black",
        linewidth=0.4,
        yerr=[[noise_low], [noise_high]],
        capsize=4,
        alpha=0.92,
        label="Human Ceiling",
    )
    handles.append(noise_bar[0])
    labels.append("Human Ceiling")
    ax.set_ylim(0, ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[run_name] for run_name in runs] + ["Human Ceiling"], rotation=24, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel(f"{metric_label}\nKL Divergence", fontsize=10)
    return handles, labels


def main() -> None:
    family_run_tables: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    all_metric_series: dict[str, list[pd.Series]] = {metric_name: [] for metric_name, _ in METRICS}
    for family_name, spec in RUN_FAMILIES.items():
        run_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for run_name in spec["runs"]:
            human_df, generated_df = _load_run_game_tables(run_name)
            run_tables[run_name] = (human_df, generated_df)
            for metric_name, _ in METRICS:
                all_metric_series[metric_name].append(human_df[metric_name])
                all_metric_series[metric_name].append(generated_df[metric_name])
        family_run_tables[family_name] = run_tables
    metric_bins = {metric_name: _build_bin_edges(series_list) for metric_name, series_list in all_metric_series.items()}

    family_scores: dict[str, pd.DataFrame] = {}
    family_noise: dict[str, dict[str, dict[str, float]]] = {}
    for family_name, spec in RUN_FAMILIES.items():
        summary_rows: list[dict[str, float | str]] = []
        for run_name in spec["runs"]:
            human_df, generated_df = family_run_tables[family_name][run_name]
            for metric_name, _ in METRICS:
                scores_df = _metric_treatment_scores(
                    generated_df,
                    human_df,
                    metric_name,
                    bin_edges=metric_bins[metric_name],
                )
                mean_value, stderr = mean_and_stderr(scores_df["score"])
                summary_rows.append(
                    {
                        "metric": metric_name,
                        "run_name": run_name,
                        "mean_value": mean_value,
                        "stderr": stderr,
                        "n_groups": int(scores_df["treatment_name"].nunique()) if not scores_df.empty else 0,
                    }
                )
        family_scores[family_name] = pd.DataFrame(summary_rows)
        baseline_human, baseline_generated = family_run_tables[family_name][spec["baseline_run"]]
        family_noise[family_name] = {
            metric_name: _noise_ceiling_summary(
                human_df=baseline_human,
                generated_df=baseline_generated,
                metric_name=metric_name,
                bin_edges=metric_bins[metric_name],
            )
            for metric_name, _ in METRICS
        }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=False)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.11, right=0.99, hspace=0.34, wspace=0.12)

    legend_handles = None
    legend_labels = None
    family_names = list(RUN_FAMILIES.keys())
    for row_idx, (metric_name, metric_label) in enumerate(METRICS):
        ylim = _row_ylim(metric_name, family_scores, family_noise)
        for col_idx, family_name in enumerate(family_names):
            handles, labels = _plot_panel(
                axes[row_idx, col_idx],
                family_name=family_name,
                metric_name=metric_name,
                metric_label=metric_label,
                summary_df=family_scores[family_name],
                noise_summary=family_noise[family_name][metric_name],
                ylim=ylim,
                show_ylabel=(col_idx == 0),
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(family_name, fontsize=12)
            if legend_handles is None:
                legend_handles, legend_labels = handles, labels

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False)
    fig.suptitle("Public Goods Game", fontsize=14, y=0.985)

    output_path = RESULTS_ROOT / "headline_model_family_panels_kl.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows: list[dict[str, float | str]] = []
    for family_name, summary_df in family_scores.items():
        for row in summary_df.to_dict(orient="records"):
            summary_rows.append({"family_name": family_name, **row})
        for metric_name, noise in family_noise[family_name].items():
            summary_rows.append({"family_name": family_name, "metric": metric_name, "run_name": "human_ceiling", **noise})
    pd.DataFrame(summary_rows).to_csv(RESULTS_ROOT / "headline_model_family_panels_kl_summary.csv", index=False)


if __name__ == "__main__":
    main()
