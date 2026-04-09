from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analyze_vs_human_treatments import _wasserstein_distance_1d
from .compare_models_with_noise_ceiling import (
    _build_shared_count_table,
    _subset_generated_to_shared_count,
)
from .random_action_baseline import (
    build_uniform_random_rollout_tables,
    select_shared_game_skeletons,
    summarize_random_baseline_draws,
)


MACRO_METRICS = [
    ("mean_total_contribution_rate", "Mean contrib"),
    ("first_round_total_contribution_rate", "First-round contrib"),
    ("final_total_contribution_rate", "Final-round contrib"),
    ("mean_round_normalized_efficiency", "Mean eff"),
    ("first_round_normalized_efficiency", "First-round eff"),
    ("final_round_normalized_efficiency", "Final-round eff"),
]

SCORE_FAMILY_SPECS = [
    ("rmse_of_config_means", "RMSE Of Config Means"),
    ("mean_wasserstein_distance", "Mean Within-Config Wasserstein Distance"),
]

RUN_NAME_TO_LABEL = {
    "baseline_gpt_5_1": "gpt-5.1 baseline",
    "twin_sampled_seed_0_gpt_5_1": "gpt-5.1 twin",
    "twin_sampled_unadjusted_seed_0_gpt_5_1": "gpt-5.1 twin unadj",
    "demographic_only_row_resampled_seed_0_gpt_5_1": "gpt-5.1 demo-only",
    "baseline_gpt_5_mini": "gpt-5-mini baseline",
    "twin_sampled_seed_0_gpt_5_mini": "gpt-5-mini twin",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "gpt-5-mini twin unadj",
    "demographic_only_row_resampled_seed_0_gpt_5_mini": "gpt-5-mini demo-only",
}

MODEL_STYLE = {
    "gpt-5.1 baseline": {"color": "#1f77b4"},
    "gpt-5.1 twin": {"color": "#6baed6"},
    "gpt-5.1 twin unadj": {"color": "#9ecae1"},
    "gpt-5.1 demo-only": {"color": "#c6dbef"},
    "gpt-5-mini baseline": {"color": "#ff7f0e"},
    "gpt-5-mini twin": {"color": "#fdae6b"},
    "gpt-5-mini twin unadj": {"color": "#fdd0a2"},
    "gpt-5-mini demo-only": {"color": "#fee6ce"},
    "noise_ceiling": {"color": "#2ca02c"},
}
def _rmse_from_mse_components(mse_components: pd.Series) -> tuple[float, float]:
    clean = mse_components.dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(clean.mean()))
    if clean.shape[0] < 2 or rmse == 0:
        return rmse, float("nan")
    mse_se = float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))
    rmse_se = float(mse_se / (2.0 * rmse))
    return rmse, rmse_se


def _mean_with_stderr(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] < 2:
        return mean, float("nan")
    stderr = float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))
    return mean, stderr


def _per_config_rmse_components(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    generated_means = generated_game_df.groupby("treatment_name")[metric].mean()
    human_means = human_game_df.groupby("treatment_name")[metric].mean()
    paired = pd.DataFrame(
        {
            "generated_mean": generated_means,
            "human_mean": human_means,
        }
    ).dropna()
    paired = paired.reset_index()
    paired["mse_component"] = (paired["generated_mean"] - paired["human_mean"]) ** 2
    return paired[["treatment_name", "mse_component"]]


def _per_config_wasserstein_components(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    treatment_names = sorted(set(generated_game_df["treatment_name"]) & set(human_game_df["treatment_name"]))
    rows: list[dict[str, Any]] = []
    for treatment_name in treatment_names:
        generated_values = generated_game_df.loc[
            generated_game_df["treatment_name"] == treatment_name,
            metric,
        ]
        human_values = human_game_df.loc[
            human_game_df["treatment_name"] == treatment_name,
            metric,
        ]
        rows.append(
            {
                "treatment_name": treatment_name,
                "wd_component": _wasserstein_distance_1d(generated_values, human_values),
            }
        )
    return pd.DataFrame(rows)


def _build_model_score_table(
    generated_by_model: dict[str, pd.DataFrame],
    human_game_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, generated_game_df in generated_by_model.items():
        for metric, metric_label in MACRO_METRICS:
            rmse_components = _per_config_rmse_components(generated_game_df, human_game_df, metric)
            rmse_score, rmse_stderr = _rmse_from_mse_components(rmse_components["mse_component"])
            rows.append(
                {
                    "model_name": model_name,
                    "score_family": "rmse_of_config_means",
                    "metric": metric,
                    "metric_label": metric_label,
                    "score": rmse_score,
                    "stderr": rmse_stderr,
                    "num_treatments": int(rmse_components["treatment_name"].nunique()),
                }
            )

            wd_components = _per_config_wasserstein_components(generated_game_df, human_game_df, metric)
            wd_score, wd_stderr = _mean_with_stderr(wd_components["wd_component"])
            rows.append(
                {
                    "model_name": model_name,
                    "score_family": "mean_wasserstein_distance",
                    "metric": metric,
                    "metric_label": metric_label,
                    "score": wd_score,
                    "stderr": wd_stderr,
                    "num_treatments": int(wd_components["treatment_name"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _build_uniform_random_score_table(
    human_game_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
    *,
    random_iters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_game_df, selected_round_df, selected_actor_df = select_shared_game_skeletons(
        human_game_df,
        human_round_df,
        human_actor_df,
        shared_counts,
    )

    rng = np.random.default_rng(random_seed)
    draw_rows: list[dict[str, Any]] = []
    for draw_index in range(random_iters):
        random_game_df, _, _ = build_uniform_random_rollout_tables(
            selected_game_df,
            selected_round_df,
            selected_actor_df,
            rng=rng,
        )
        for metric, metric_label in MACRO_METRICS:
            rmse_components = _per_config_rmse_components(random_game_df, human_game_df, metric)
            rmse_score, _ = _rmse_from_mse_components(rmse_components["mse_component"])
            draw_rows.append(
                {
                    "score_family": "rmse_of_config_means",
                    "metric": metric,
                    "metric_label": metric_label,
                    "draw_index": draw_index,
                    "score": rmse_score,
                }
            )

            wd_components = _per_config_wasserstein_components(random_game_df, human_game_df, metric)
            wd_score, _ = _mean_with_stderr(wd_components["wd_component"])
            draw_rows.append(
                {
                    "score_family": "mean_wasserstein_distance",
                    "metric": metric,
                    "metric_label": metric_label,
                    "draw_index": draw_index,
                    "score": wd_score,
                }
            )

    draws_df = pd.DataFrame(draw_rows)
    summary_df = summarize_random_baseline_draws(
        draws_df,
        score_cols=["score_family", "metric", "metric_label"],
    )
    return draws_df, summary_df


def _bootstrap_noise_ceiling(
    human_game_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
    *,
    bootstrap_iters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    shared_generated_count_map = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    human_count_map = {
        str(row["treatment_name"]): int(row["human_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    human_values_by_treatment = {
        str(treatment_name): group.reset_index(drop=True)
        for treatment_name, group in human_game_df.groupby("treatment_name", sort=True)
    }

    bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_index in range(bootstrap_iters):
        for treatment_name, human_group in human_values_by_treatment.items():
            generated_count = shared_generated_count_map.get(treatment_name, 0)
            human_count = human_count_map.get(treatment_name, 0)
            if generated_count <= 0 or human_count <= 0:
                continue
            generated_indices = rng.integers(0, len(human_group), size=generated_count)
            human_indices = rng.integers(0, len(human_group), size=human_count)
            for metric, metric_label in MACRO_METRICS:
                generated_values = human_group.iloc[generated_indices][metric]
                human_values = human_group.iloc[human_indices][metric]
                generated_mean = float(generated_values.mean())
                human_mean = float(human_values.mean())
                bootstrap_rows.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "treatment_name": treatment_name,
                        "metric": metric,
                        "metric_label": metric_label,
                        "mse_component": (generated_mean - human_mean) ** 2,
                        "wd_component": _wasserstein_distance_1d(generated_values, human_values),
                    }
                )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    summary_rows: list[dict[str, Any]] = []
    for (metric, metric_label), group in bootstrap_df.groupby(
        ["metric", "metric_label"],
        sort=True,
    ):
        mse_by_treatment = (
            group.groupby("treatment_name", as_index=False)["mse_component"]
            .mean()
            .rename(columns={"mse_component": "component"})
        )
        rmse_score, rmse_stderr = _rmse_from_mse_components(mse_by_treatment["component"])
        summary_rows.append(
            {
                "score_family": "rmse_of_config_means",
                "metric": metric,
                "metric_label": metric_label,
                "noise_ceiling_mean": rmse_score,
                "noise_ceiling_stderr": rmse_stderr,
                "bootstrap_iters": bootstrap_iters,
                "num_treatments": int(mse_by_treatment["treatment_name"].nunique()),
            }
        )

        wd_by_treatment = (
            group.groupby("treatment_name", as_index=False)["wd_component"]
            .mean()
            .rename(columns={"wd_component": "component"})
        )
        wd_score, wd_stderr = _mean_with_stderr(wd_by_treatment["component"])
        summary_rows.append(
            {
                "score_family": "mean_wasserstein_distance",
                "metric": metric,
                "metric_label": metric_label,
                "noise_ceiling_mean": wd_score,
                "noise_ceiling_stderr": wd_stderr,
                "bootstrap_iters": bootstrap_iters,
                "num_treatments": int(wd_by_treatment["treatment_name"].nunique()),
            }
        )
    return bootstrap_df, pd.DataFrame(summary_rows)


def _format_markdown_table(table_df: pd.DataFrame) -> str:
    columns = list(table_df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in table_df.to_dict(orient="records"):
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def _plot_comparison_figure(
    plot_df: pd.DataFrame,
    output_path: Path,
) -> None:
    family_titles = {key: title for key, title in SCORE_FAMILY_SPECS}
    metric_labels = {key: label for key, label in MACRO_METRICS}
    model_order = [
        "gpt-5.1 baseline",
        "gpt-5.1 demo-only",
        "gpt-5.1 twin unadj",
        "gpt-5.1 twin",
        "gpt-5-mini baseline",
        "gpt-5-mini demo-only",
        "gpt-5-mini twin unadj",
        "gpt-5-mini twin",
        "noise_ceiling",
    ]
    bar_width = min(0.11, 0.84 / len(model_order))

    fig, axes = plt.subplots(2, 1, figsize=(18.5, 9.4), constrained_layout=False, sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.28)
    for ax, (score_family, title) in zip(axes, SCORE_FAMILY_SPECS, strict=True):
        family_df = plot_df[plot_df["score_family"] == score_family].copy()
        metric_order = [metric for metric, _ in MACRO_METRICS]
        x = np.arange(len(metric_order), dtype=float)
        center_offset = (len(model_order) - 1) / 2.0

        for index, model_name in enumerate(model_order):
            model_df = family_df[family_df["model_name"] == model_name].set_index("metric").reindex(metric_order)
            positions = x + ((index - center_offset) * bar_width)
            if model_name == "noise_ceiling":
                heights = model_df["noise_ceiling_mean"].to_numpy(dtype=float)
                yerr = model_df["noise_ceiling_stderr"].to_numpy(dtype=float)
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=MODEL_STYLE[model_name]["color"],
                    alpha=0.8,
                    label="Noise ceiling",
                    yerr=yerr,
                    capsize=3,
                )
            else:
                heights = model_df["score"].to_numpy(dtype=float)
                yerr = model_df["stderr"].to_numpy(dtype=float)
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=MODEL_STYLE[model_name]["color"],
                    alpha=0.9,
                    yerr=yerr,
                    capsize=3,
                    label=model_name if score_family == SCORE_FAMILY_SPECS[0][0] else None,
                )

        family_line_df = family_df.drop_duplicates("metric").set_index("metric").reindex(metric_order)
        cluster_half_width = ((len(model_order) - 1) * bar_width) / 2.0
        for metric_index, metric in enumerate(metric_order):
            line_y = family_line_df.loc[metric, "uniform_random_mean"]
            if pd.isna(line_y):
                continue
            ax.hlines(
                y=float(line_y),
                xmin=x[metric_index] - cluster_half_width - (0.18 * bar_width),
                xmax=x[metric_index] + cluster_half_width + (0.18 * bar_width),
                colors="#4d4d4d",
                linestyles=(0, (5, 3)),
                linewidth=1.8,
            )

        ax.set_title(title, pad=10)
        ax.set_ylabel("Lower is better")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(np.arange(len(MACRO_METRICS), dtype=float))
    axes[-1].set_xticklabels([metric_labels[metric] for metric, _ in MACRO_METRICS], rotation=20, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#4d4d4d",
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            label="Uniform random contrib baseline",
        )
    )
    labels.append("Uniform random contrib baseline")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=False,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot macro RMSE and Wasserstein alignment for forecasting runs."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=list(RUN_NAME_TO_LABEL.keys()),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--random-baseline-iters", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "macro_pointwise_alignment__llms"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = {
        run_name: args.forecasting_root / "results" / f"{run_name}__vs_human_treatments"
        for run_name in args.run_names
    }
    generated_by_model: dict[str, pd.DataFrame] = {}
    human_game_df: pd.DataFrame | None = None
    human_round_df: pd.DataFrame | None = None
    human_actor_df: pd.DataFrame | None = None
    for run_name, result_dir in result_dirs.items():
        if not result_dir.exists():
            raise FileNotFoundError(f"Missing result directory: {result_dir}")
        generated_by_model[RUN_NAME_TO_LABEL.get(run_name, run_name)] = pd.read_csv(
            result_dir / "generated_game_summary.csv"
        )
        if human_game_df is None:
            human_game_df = pd.read_csv(result_dir / "human_game_summary.csv")
            human_round_df = pd.read_csv(result_dir / "human_round_summary.csv")
            human_actor_df = pd.read_csv(result_dir / "human_actor_summary.csv")

    assert human_game_df is not None and human_round_df is not None and human_actor_df is not None
    shared_counts = _build_shared_count_table(human_game_df, generated_by_model)
    trimmed_generated_by_model = {
        model_name: _subset_generated_to_shared_count(generated_game_df, shared_counts)
        for model_name, generated_game_df in generated_by_model.items()
    }

    model_scores_df = _build_model_score_table(trimmed_generated_by_model, human_game_df)
    uniform_random_draws_df, uniform_random_df = _build_uniform_random_score_table(
        human_game_df,
        human_round_df,
        human_actor_df,
        shared_counts,
        random_iters=args.random_baseline_iters,
        random_seed=args.random_seed + 1000,
    )
    bootstrap_df, noise_ceiling_df = _bootstrap_noise_ceiling(
        human_game_df,
        shared_counts,
        bootstrap_iters=args.bootstrap_iters,
        random_seed=args.random_seed,
    )
    combined_df = model_scores_df.merge(
        noise_ceiling_df.drop(columns=["bootstrap_iters"], errors="ignore"),
        on=["score_family", "metric", "metric_label"],
        how="left",
    ).merge(
        uniform_random_df,
        on=["score_family", "metric", "metric_label"],
        how="left",
    )
    combined_df["gap_to_noise_ceiling"] = combined_df["score"] - combined_df["noise_ceiling_mean"]
    combined_df["ratio_to_noise_ceiling"] = combined_df["score"] / combined_df["noise_ceiling_mean"]

    noise_rows = noise_ceiling_df.copy()
    noise_rows["model_name"] = "noise_ceiling"
    noise_rows["score"] = noise_rows["noise_ceiling_mean"]
    noise_rows["stderr"] = noise_rows["noise_ceiling_stderr"]
    noise_rows["gap_to_noise_ceiling"] = 0.0
    noise_rows["ratio_to_noise_ceiling"] = 1.0
    plot_df = pd.concat(
        [combined_df, noise_rows.reindex(columns=combined_df.columns)],
        ignore_index=True,
    )

    combined_df.sort_values(["score_family", "metric", "model_name"]).to_csv(
        args.output_dir / "macro_pointwise_alignment_summary.csv",
        index=False,
    )
    bootstrap_df.sort_values(["metric", "treatment_name", "bootstrap_index"]).to_csv(
        args.output_dir / "macro_pointwise_alignment_bootstrap.csv",
        index=False,
    )
    uniform_random_draws_df.sort_values(["score_family", "metric", "draw_index"]).to_csv(
        args.output_dir / "macro_uniform_random_baseline_draws.csv",
        index=False,
    )
    uniform_random_df.sort_values(["score_family", "metric"]).to_csv(
        args.output_dir / "macro_uniform_random_baseline.csv",
        index=False,
    )

    table_df = combined_df[
        [
            "score_family",
            "metric_label",
            "model_name",
            "score",
            "stderr",
            "noise_ceiling_mean",
            "noise_ceiling_stderr",
            "uniform_random_mean",
            "uniform_random_stderr",
            "gap_to_noise_ceiling",
            "ratio_to_noise_ceiling",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "score_family": "score_family",
            "metric_label": "metric",
            "model_name": "run",
        }
    )
    numeric_cols = [col for col in table_df.columns if col not in {"score_family", "metric", "run"}]
    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda value: round(float(value), 4))
    table_df.to_csv(args.output_dir / "macro_pointwise_alignment_table.csv", index=False)
    (args.output_dir / "macro_pointwise_alignment_table.md").write_text(_format_markdown_table(table_df))

    _plot_comparison_figure(plot_df, args.output_dir / "macro_pointwise_alignment.png")


if __name__ == "__main__":
    main()
