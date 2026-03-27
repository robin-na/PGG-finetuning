from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze_vs_human_treatments import _abs_log_ratio, _safe_ratio, _wasserstein_distance_1d


SCALAR_METRICS = [
    ("mean_total_contribution_rate", "Mean contrib"),
    ("mean_round_normalized_efficiency", "Mean eff"),
    ("final_total_contribution_rate", "Final contrib"),
    ("final_round_normalized_efficiency", "Final eff"),
    ("total_contribution_rate_decay_slope", "Contrib slope"),
    ("round_normalized_efficiency_decay_slope", "Eff slope"),
    ("mean_within_round_contribution_rate_var", "Within-round var"),
]

SCORE_FAMILY_SPECS = [
    ("rmse_of_config_means", "RMSE Of Config Means"),
    ("mean_wasserstein_distance", "Mean Wasserstein Distance"),
    ("mean_abs_log_sd_ratio", "Mean |log SD ratio|"),
    ("mean_abs_log_iqr_ratio", "Mean |log IQR ratio|"),
]

RUN_NAME_TO_LABEL = {
    "baseline_gpt_5_1": "gpt-5.1 baseline",
    "twin_sampled_seed_0_gpt_5_1": "gpt-5.1 twin",
    "baseline_gpt_5_mini": "gpt-5-mini baseline",
    "twin_sampled_seed_0_gpt_5_mini": "gpt-5-mini twin",
}

MODEL_STYLE = {
    "gpt-5.1 baseline": {"color": "#1f77b4"},
    "gpt-5.1 twin": {"color": "#6baed6"},
    "gpt-5-mini baseline": {"color": "#ff7f0e"},
    "gpt-5-mini twin": {"color": "#fdae6b"},
    "noise_ceiling": {"color": "#2ca02c"},
}


def _iqr(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def _load_run_tables(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    generated_game_df = pd.read_csv(result_dir / "generated_game_summary.csv")
    human_game_df = pd.read_csv(result_dir / "human_game_summary.csv")
    return generated_game_df, human_game_df


def _build_shared_count_table(
    human_game_df: pd.DataFrame,
    generated_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    shared = (
        human_game_df.groupby("treatment_name")["game_id"]
        .nunique()
        .rename("human_count")
        .to_frame()
    )
    for model_name, generated_game_df in generated_by_model.items():
        shared[model_name] = generated_game_df.groupby("treatment_name")["custom_id"].nunique()
    shared = shared.fillna(0).astype(int)
    shared["shared_generated_count"] = shared[[*generated_by_model.keys()]].min(axis=1)
    shared["shared_count"] = shared[["human_count", "shared_generated_count"]].min(axis=1)
    return shared.reset_index()


def _subset_generated_to_shared_count(
    generated_game_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
) -> pd.DataFrame:
    count_map = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    frames: list[pd.DataFrame] = []
    for treatment_name, group in generated_game_df.groupby("treatment_name", sort=True):
        shared_count = count_map.get(str(treatment_name), 0)
        if shared_count <= 0:
            continue
        frames.append(group.sort_values("custom_id").head(shared_count).copy())
    if not frames:
        return generated_game_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def _score_rmse_of_config_means(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
) -> float:
    generated_means = generated_game_df.groupby("treatment_name")[metric].mean()
    human_means = human_game_df.groupby("treatment_name")[metric].mean()
    paired = pd.DataFrame(
        {
            "generated_mean": generated_means,
            "human_mean": human_means,
        }
    ).dropna()
    if paired.empty:
        return float("nan")
    return float(np.sqrt(((paired["generated_mean"] - paired["human_mean"]) ** 2).mean()))


def _score_mean_wasserstein_distance(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
) -> float:
    distances: list[float] = []
    treatment_names = sorted(set(generated_game_df["treatment_name"]) & set(human_game_df["treatment_name"]))
    for treatment_name in treatment_names:
        generated_values = generated_game_df.loc[
            generated_game_df["treatment_name"] == treatment_name,
            metric,
        ]
        human_values = human_game_df.loc[
            human_game_df["treatment_name"] == treatment_name,
            metric,
        ]
        distances.append(_wasserstein_distance_1d(generated_values, human_values))
    valid = [value for value in distances if not math.isnan(value)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _score_mean_abs_log_dispersion_ratio(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
    *,
    use_iqr: bool,
) -> float:
    distances: list[float] = []
    treatment_names = sorted(set(generated_game_df["treatment_name"]) & set(human_game_df["treatment_name"]))
    for treatment_name in treatment_names:
        generated_values = generated_game_df.loc[
            generated_game_df["treatment_name"] == treatment_name,
            metric,
        ].dropna().astype(float)
        human_values = human_game_df.loc[
            human_game_df["treatment_name"] == treatment_name,
            metric,
        ].dropna().astype(float)
        if use_iqr:
            generated_dispersion = _iqr(generated_values)
            human_dispersion = _iqr(human_values)
        else:
            generated_dispersion = float(generated_values.std(ddof=0)) if not generated_values.empty else float("nan")
            human_dispersion = float(human_values.std(ddof=0)) if not human_values.empty else float("nan")
        ratio = _safe_ratio(generated_dispersion, human_dispersion)
        distances.append(_abs_log_ratio(ratio))
    valid = [value for value in distances if not math.isnan(value)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _score_family(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
    metric: str,
    score_family: str,
) -> float:
    if score_family == "rmse_of_config_means":
        return _score_rmse_of_config_means(generated_game_df, human_game_df, metric)
    if score_family == "mean_wasserstein_distance":
        return _score_mean_wasserstein_distance(generated_game_df, human_game_df, metric)
    if score_family == "mean_abs_log_sd_ratio":
        return _score_mean_abs_log_dispersion_ratio(generated_game_df, human_game_df, metric, use_iqr=False)
    if score_family == "mean_abs_log_iqr_ratio":
        return _score_mean_abs_log_dispersion_ratio(generated_game_df, human_game_df, metric, use_iqr=True)
    raise ValueError(f"Unknown score family: {score_family}")


def _bootstrap_noise_ceiling(
    human_game_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
    *,
    bootstrap_iters: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    shared_count_map = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    human_count_map = {
        str(row["treatment_name"]): int(row["human_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    human_values_by_treatment = {
        treatment_name: group.reset_index(drop=True)
        for treatment_name, group in human_game_df.groupby("treatment_name", sort=True)
    }

    bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_index in range(bootstrap_iters):
        sampled_model_frames: list[pd.DataFrame] = []
        sampled_human_frames: list[pd.DataFrame] = []
        for treatment_name, human_group in human_values_by_treatment.items():
            sampled_model_count = shared_count_map.get(str(treatment_name), 0)
            sampled_human_count = human_count_map.get(str(treatment_name), 0)
            if sampled_model_count <= 0 or sampled_human_count <= 0:
                continue
            model_indices = rng.integers(0, len(human_group), size=sampled_model_count)
            human_indices = rng.integers(0, len(human_group), size=sampled_human_count)
            sampled_model_frames.append(human_group.iloc[model_indices].copy())
            sampled_human_frames.append(human_group.iloc[human_indices].copy())

        sampled_model_df = pd.concat(sampled_model_frames, ignore_index=True)
        sampled_human_df = pd.concat(sampled_human_frames, ignore_index=True)

        for score_family, _ in SCORE_FAMILY_SPECS:
            for metric, _ in SCALAR_METRICS:
                bootstrap_rows.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "score_family": score_family,
                        "metric": metric,
                        "score": _score_family(
                            sampled_model_df,
                            sampled_human_df,
                            metric,
                            score_family,
                        ),
                    }
                )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    summary_rows: list[dict[str, Any]] = []
    for (score_family, metric), group in bootstrap_df.groupby(["score_family", "metric"], sort=True):
        values = group["score"].dropna().astype(float)
        summary_rows.append(
            {
                "score_family": score_family,
                "metric": metric,
                "noise_ceiling_mean": float(values.mean()) if not values.empty else float("nan"),
                "noise_ceiling_p05": float(values.quantile(0.05)) if not values.empty else float("nan"),
                "noise_ceiling_p50": float(values.quantile(0.50)) if not values.empty else float("nan"),
                "noise_ceiling_p95": float(values.quantile(0.95)) if not values.empty else float("nan"),
                "bootstrap_iters": bootstrap_iters,
            }
        )
    return bootstrap_df, pd.DataFrame(summary_rows)


def _build_model_score_table(
    generated_by_model: dict[str, pd.DataFrame],
    human_game_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, generated_game_df in generated_by_model.items():
        for score_family, _ in SCORE_FAMILY_SPECS:
            for metric, _ in SCALAR_METRICS:
                rows.append(
                    {
                        "model_name": model_name,
                        "score_family": score_family,
                        "metric": metric,
                        "score": _score_family(
                            generated_game_df,
                            human_game_df,
                            metric,
                            score_family,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _plot_comparison_figure(
    combined_df: pd.DataFrame,
    output_path: Path,
) -> None:
    family_titles = {key: title for key, title in SCORE_FAMILY_SPECS}
    metric_labels = {key: label for key, label in SCALAR_METRICS}
    plot_model_names = [name for name in combined_df["model_name"].drop_duplicates().tolist() if name != "noise_ceiling"]
    preferred_order = [
        "gpt-5.1 baseline",
        "gpt-5.1 twin",
        "gpt-5-mini baseline",
        "gpt-5-mini twin",
    ]
    ordered_models = [name for name in preferred_order if name in plot_model_names]
    ordered_models.extend([name for name in plot_model_names if name not in ordered_models])
    model_order = [*ordered_models, "noise_ceiling"]
    bar_width = min(0.18, 0.82 / max(len(model_order), 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for ax, (score_family, title) in zip(axes.flat, SCORE_FAMILY_SPECS):
        plot_df = combined_df[combined_df["score_family"] == score_family].copy()
        metrics = [metric for metric, _ in SCALAR_METRICS]
        x = np.arange(len(metrics), dtype=float)
        center_offset = (len(model_order) - 1) / 2.0

        for index, model_name in enumerate(model_order):
            model_df = plot_df[plot_df["model_name"] == model_name].set_index("metric").reindex(metrics)
            positions = x + ((index - center_offset) * bar_width)
            if model_name == "noise_ceiling":
                heights = model_df["noise_ceiling_mean"].to_numpy(dtype=float)
                yerr_lower = heights - model_df["noise_ceiling_p05"].to_numpy(dtype=float)
                yerr_upper = model_df["noise_ceiling_p95"].to_numpy(dtype=float) - heights
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=MODEL_STYLE[model_name]["color"],
                    alpha=0.8,
                    label="Noise ceiling",
                    yerr=np.vstack([yerr_lower, yerr_upper]),
                    capsize=3,
                )
            else:
                heights = model_df["score"].to_numpy(dtype=float)
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=MODEL_STYLE[model_name]["color"],
                    alpha=0.9,
                    label=model_name if score_family == SCORE_FAMILY_SPECS[0][0] else None,
                )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[metric] for metric in metrics], rotation=20, ha="right")
        ax.set_ylabel("Lower is better")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 5),
        frameon=False,
    )
    fig.suptitle("Forecasting Models Vs Human Noise Ceiling", fontsize=15, y=1.06)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare forecasting models against a matched human noise ceiling."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=[
            "baseline_gpt_5_1",
            "twin_sampled_seed_0_gpt_5_1",
            "baseline_gpt_5_mini",
            "twin_sampled_seed_0_gpt_5_mini",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "model_comparison__noise_ceiling"

    result_dirs = {
        run_name: args.forecasting_root / "results" / f"{run_name}__vs_human_treatments"
        for run_name in args.run_names
    }
    generated_by_model: dict[str, pd.DataFrame] = {}
    human_tables: list[pd.DataFrame] = []
    for run_name, result_dir in result_dirs.items():
        generated_game_df, human_game_df = _load_run_tables(result_dir)
        generated_by_model[RUN_NAME_TO_LABEL.get(run_name, run_name)] = generated_game_df
        human_tables.append(human_game_df)

    human_game_df = human_tables[0].copy()
    shared_counts = _build_shared_count_table(human_game_df, generated_by_model)
    trimmed_generated_by_model = {
        model_name: _subset_generated_to_shared_count(generated_game_df, shared_counts)
        for model_name, generated_game_df in generated_by_model.items()
    }

    model_scores_df = _build_model_score_table(trimmed_generated_by_model, human_game_df)
    bootstrap_df, noise_ceiling_df = _bootstrap_noise_ceiling(
        human_game_df,
        shared_counts,
        bootstrap_iters=args.bootstrap_iters,
        random_seed=args.random_seed,
    )

    combined_df = model_scores_df.merge(
        noise_ceiling_df,
        on=["score_family", "metric"],
        how="left",
    )
    combined_df["gap_to_noise_ceiling"] = combined_df["score"] - combined_df["noise_ceiling_mean"]
    combined_df["ratio_to_noise_ceiling"] = combined_df["score"] / combined_df["noise_ceiling_mean"]
    noise_rows = noise_ceiling_df.copy()
    noise_rows["model_name"] = "noise_ceiling"
    noise_rows["score"] = noise_rows["noise_ceiling_mean"]
    noise_rows["gap_to_noise_ceiling"] = 0.0
    noise_rows["ratio_to_noise_ceiling"] = 1.0
    plot_df = pd.concat(
        [
            combined_df,
            noise_rows[combined_df.columns],
        ],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_counts.to_csv(args.output_dir / "shared_treatment_counts.csv", index=False)
    model_scores_df.sort_values(["score_family", "metric", "model_name"]).to_csv(
        args.output_dir / "model_scores.csv",
        index=False,
    )
    bootstrap_df.sort_values(["score_family", "metric", "bootstrap_index"]).to_csv(
        args.output_dir / "noise_ceiling_bootstrap.csv",
        index=False,
    )
    noise_ceiling_df.sort_values(["score_family", "metric"]).to_csv(
        args.output_dir / "noise_ceiling_summary.csv",
        index=False,
    )
    combined_df.sort_values(["score_family", "metric", "model_name"]).to_csv(
        args.output_dir / "model_vs_noise_ceiling_summary.csv",
        index=False,
    )
    _plot_comparison_figure(
        plot_df,
        args.output_dir / "model_vs_noise_ceiling.png",
    )

    manifest = {
        "run_names": args.run_names,
        "bootstrap_iters": args.bootstrap_iters,
        "random_seed": args.random_seed,
        "score_families": [score_family for score_family, _ in SCORE_FAMILY_SPECS],
        "metrics": [metric for metric, _ in SCALAR_METRICS],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(combined_df.sort_values(['score_family', 'metric', 'model_name']).to_string(index=False))


if __name__ == "__main__":
    main()
