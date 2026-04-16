from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import build_random_record_frames, wasserstein_distance_1d


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
BASE_RUN_FAMILIES = {
    "gpt-5-mini": {
        "runs": [
            "baseline_gpt_5_mini",
            "twin_sampled_unadjusted_seed_0_gpt_5_mini",
        ],
    },
    "gpt-5.1": {
        "runs": [
            "baseline_gpt_5_1",
            "twin_sampled_unadjusted_seed_0_gpt_5_1",
        ],
    },
}
BASE_RUN_LABELS = {
    "baseline_gpt_5_mini": "Baseline",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
    "baseline_gpt_5_1": "Baseline",
    "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
}
RUN_COLORS = ["#4C78A8", "#F4A3A3"]
NOISE_CEILING_COLOR = "#8C8C8C"
RANDOM_BASELINE_COLOR = "#555555"
METRICS = [
    ("final_surplus_ratio", "Final Surplus Ratio"),
    ("mean_trade_ratio", "Mean Trade Ratio"),
    ("mean_acceptance_rate", "Mean Acceptance Rate"),
]
DEFAULT_BOOTSTRAP_ITERS = 300
DEFAULT_RNG_SEED = 29
DEFAULT_RANDOM_BASELINE_ITERS = 50
DEFAULT_CEILING_METHOD = "split_half"


def _with_suffix(run_name: str, run_suffix: str) -> str:
    if not run_suffix:
        return run_name
    return f"{run_name}_{run_suffix}"


def _resolve_run_families(run_suffix: str) -> dict[str, dict[str, list[str]]]:
    return {
        family_name: {
            "runs": [_with_suffix(run_name, run_suffix) for run_name in spec["runs"]],
        }
        for family_name, spec in BASE_RUN_FAMILIES.items()
    }


def _resolve_run_label(run_name: str) -> str:
    for base_run_name, label in BASE_RUN_LABELS.items():
        if run_name == base_run_name or run_name.startswith(f"{base_run_name}_"):
            return label
    return run_name


def _mean_and_stderr(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] <= 1:
        return mean, float("nan")
    return mean, float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))


def _bootstrap_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {
            "bootstrap_mean": float("nan"),
            "bootstrap_p05": float("nan"),
            "bootstrap_p95": float("nan"),
        }
    return {
        "bootstrap_mean": float(arr.mean()),
        "bootstrap_p05": float(np.quantile(arr, 0.05)),
        "bootstrap_p95": float(np.quantile(arr, 0.95)),
    }


def _sample_disjoint_halves(frame: pd.DataFrame, sample_size: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_size = min(int(sample_size), int(len(frame) // 2))
    if sample_size <= 0:
        return pd.DataFrame(), pd.DataFrame()
    permutation = rng.permutation(len(frame))
    lhs = frame.iloc[permutation[:sample_size]].reset_index(drop=True)
    rhs = frame.iloc[permutation[sample_size : 2 * sample_size]].reset_index(drop=True)
    return lhs, rhs


def _load_run_tables(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = RESULTS_ROOT / f"{run_name}__vs_human_treatments"
    return (
        pd.read_csv(run_dir / "generated_game_records.csv"),
        pd.read_csv(run_dir / "human_game_records.csv"),
    )


def _available_families() -> dict[str, dict[str, list[str]]]:
    return _available_families_for_run_set(_resolve_run_families(""))


def _available_families_for_run_set(run_families: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    available: dict[str, dict[str, list[str]]] = {}
    for family_name, spec in run_families.items():
        run_dirs = [RESULTS_ROOT / f"{run_name}__vs_human_treatments" for run_name in spec["runs"]]
        if all(run_dir.exists() for run_dir in run_dirs):
            available[family_name] = spec
    if not available:
        raise FileNotFoundError("No complete chip-bargain model-family results are available for plotting.")
    return available


def _treatment_scores_for_metric(metric_name: str, *, generated_game_df: pd.DataFrame, human_game_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    common_treatments = sorted(
        set(generated_game_df["treatment_name"].dropna().astype(str))
        & set(human_game_df["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_values = generated_game_df.loc[
            generated_game_df["treatment_name"] == treatment_name,
            metric_name,
        ]
        human_values = human_game_df.loc[
            human_game_df["treatment_name"] == treatment_name,
            metric_name,
        ]
        score = wasserstein_distance_1d(generated_values, human_values)
        if np.isfinite(score):
            rows.append({"treatment_name": treatment_name, "score": score})
    return pd.DataFrame(rows)


def _shared_generated_count_map(run_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, int]:
    per_run_maps: list[dict[str, int]] = []
    for generated_game_df, _ in run_tables.values():
        per_run_maps.append(
            {
                str(treatment_name): int(len(group))
                for treatment_name, group in generated_game_df.groupby("treatment_name", sort=True)
            }
        )
    common_treatments = sorted(set.intersection(*(set(m.keys()) for m in per_run_maps))) if per_run_maps else []
    return {
        treatment_name: min(run_map[treatment_name] for run_map in per_run_maps)
        for treatment_name in common_treatments
    }


def _noise_ceiling_summary(
    metric_name: str,
    *,
    human_game_df: pd.DataFrame,
    shared_generated_count_map: dict[str, int],
    bootstrap_iters: int,
    rng_seed: int,
    ceiling_method: str,
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    bootstrap_scores: list[float] = []
    human_values_by_treatment = {
        str(treatment_name): group.reset_index(drop=True)
        for treatment_name, group in human_game_df.groupby("treatment_name", sort=True)
    }
    human_count_map = {treatment_name: int(len(group)) for treatment_name, group in human_values_by_treatment.items()}

    for _ in range(bootstrap_iters):
        treatment_scores: list[float] = []
        for treatment_name, generated_count in shared_generated_count_map.items():
            human_group = human_values_by_treatment[treatment_name]
            if ceiling_method == "split_half":
                pseudo_generated, pseudo_human = _sample_disjoint_halves(human_group, generated_count, rng)
                if pseudo_generated.empty or pseudo_human.empty:
                    continue
            else:
                human_count = human_count_map[treatment_name]
                pseudo_generated = human_group.iloc[rng.integers(0, len(human_group), size=generated_count)]
                pseudo_human = human_group.iloc[rng.integers(0, len(human_group), size=human_count)]
            score = wasserstein_distance_1d(pseudo_generated[metric_name], pseudo_human[metric_name])
            if np.isfinite(score):
                treatment_scores.append(score)
        if treatment_scores:
            bootstrap_scores.append(float(np.mean(treatment_scores)))
    return _bootstrap_summary(bootstrap_scores)


def _request_manifest_path_for_run(run_name: str) -> Path:
    return FORECASTING_ROOT / "metadata" / run_name / "request_manifest.csv"


def _random_baseline_summary(
    metric_name: str,
    *,
    request_manifest_csv: Path,
    human_game_df: pd.DataFrame,
    shared_generated_count_map: dict[str, int],
    random_baseline_iters: int,
    rng_seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    random_scores: list[float] = []
    for _ in range(random_baseline_iters):
        random_games, _, _ = build_random_record_frames(
            request_manifest_csv=request_manifest_csv,
            sample_count_map=shared_generated_count_map,
            rng=rng,
        )
        scores_df = _treatment_scores_for_metric(
            metric_name,
            generated_game_df=random_games,
            human_game_df=human_game_df,
        )
        if not scores_df.empty:
            random_scores.append(float(scores_df["score"].mean()))
    return _bootstrap_summary(random_scores)


def _row_ylim(
    metric_name: str,
    family_scores: dict[str, pd.DataFrame],
    family_noise: dict[str, dict[str, float]],
    family_random: dict[str, dict[str, float]],
) -> float:
    upper = 0.0
    for family_name, summary_df in family_scores.items():
        metric_summary = summary_df[summary_df["metric"] == metric_name]
        for _, row in metric_summary.iterrows():
            stderr = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0
            upper = max(upper, float(row["mean_value"]) + stderr)
        upper = max(upper, float(family_noise[family_name][metric_name]["bootstrap_p95"]))
        upper = max(upper, float(family_random[family_name][metric_name]["bootstrap_p95"]))
    return upper * 1.15 if upper > 0 else 1.0


def _plot_panel(
    ax: plt.Axes,
    *,
    family_name: str,
    metric_name: str,
    metric_label: str,
    summary_df: pd.DataFrame,
    noise_summary: dict[str, float],
    random_summary: dict[str, float],
    ylim: float,
    show_ylabel: bool,
    run_families: dict[str, dict[str, list[str]]],
) -> tuple[list, list]:
    metric_summary = summary_df[summary_df["metric"] == metric_name].set_index("run_name")
    runs = run_families[family_name]["runs"]
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
            label=_resolve_run_label(run_name),
        )
        handles.append(bar[0])
        labels.append(_resolve_run_label(run_name))

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
    random_mean = float(random_summary["bootstrap_mean"])
    if np.isfinite(random_mean):
        ax.axhline(
            random_mean,
            color=RANDOM_BASELINE_COLOR,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_ylim(0, ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([_resolve_run_label(run_name) for run_name in runs] + ["Human Ceiling"], rotation=24, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel(f"{metric_label}\nWasserstein Distance", fontsize=10)
    return handles, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot chip-bargain macro-level model-family comparison panels.")
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--output-stem", type=str, default="headline_model_family_macro_panels")
    parser.add_argument("--title", type=str, default="Chip Bargaining")
    parser.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    parser.add_argument("--ceiling-method", choices=["bootstrap", "split_half"], default=DEFAULT_CEILING_METHOD)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--random-baseline-iters", type=int, default=DEFAULT_RANDOM_BASELINE_ITERS)
    args = parser.parse_args()

    run_families = _resolve_run_families(args.run_suffix)
    available_families = _available_families_for_run_set(run_families)
    family_tables: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    family_scores: dict[str, pd.DataFrame] = {}
    family_noise: dict[str, dict[str, dict[str, float]]] = {}
    family_random: dict[str, dict[str, dict[str, float]]] = {}

    for family_name, spec in available_families.items():
        run_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        summary_rows: list[dict[str, float | str]] = []
        for run_name in spec["runs"]:
            tables = _load_run_tables(run_name)
            run_tables[run_name] = tables
            generated_game_df, human_game_df = tables
            for metric_name, _ in METRICS:
                scores_df = _treatment_scores_for_metric(
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
        family_tables[family_name] = run_tables
        family_scores[family_name] = pd.DataFrame(summary_rows)
        shared_generated_count_map = _shared_generated_count_map(run_tables)
        example_human_df = next(iter(run_tables.values()))[1]
        family_noise[family_name] = {
            metric_name: _noise_ceiling_summary(
                metric_name,
                human_game_df=example_human_df,
                shared_generated_count_map=shared_generated_count_map,
                bootstrap_iters=args.bootstrap_iters,
                rng_seed=args.rng_seed,
                ceiling_method=args.ceiling_method,
            )
            for metric_name, _ in METRICS
        }
        request_manifest_csv = _request_manifest_path_for_run(spec["runs"][0])
        family_random[family_name] = {
            metric_name: _random_baseline_summary(
                metric_name,
                request_manifest_csv=request_manifest_csv,
                human_game_df=example_human_df,
                shared_generated_count_map=shared_generated_count_map,
                random_baseline_iters=args.random_baseline_iters,
                rng_seed=args.rng_seed,
            )
            for metric_name, _ in METRICS
        }

    family_names = list(available_families.keys())
    ncols = len(family_names)
    fig_width = 6.0 if ncols == 1 else 11.5
    fig, axes = plt.subplots(len(METRICS), ncols, figsize=(fig_width, 3.2 * len(METRICS)), constrained_layout=False)
    if ncols == 1:
        axes = np.asarray(axes).reshape(len(METRICS), 1)
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.18, right=0.98, hspace=0.36)
    else:
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.11, right=0.99, hspace=0.36, wspace=0.12)

    legend_handles = None
    legend_labels = None
    for row_idx, (metric_name, metric_label) in enumerate(METRICS):
        ylim = _row_ylim(metric_name, family_scores, family_noise, family_random)
        for col_idx, family_name in enumerate(family_names):
            handles, labels = _plot_panel(
                axes[row_idx, col_idx],
                family_name=family_name,
                metric_name=metric_name,
                metric_label=metric_label,
                summary_df=family_scores[family_name],
                noise_summary=family_noise[family_name][metric_name],
                random_summary=family_random[family_name][metric_name],
                ylim=ylim,
                show_ylabel=(col_idx == 0),
                run_families=run_families,
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(family_name, fontsize=12)
            if legend_handles is None:
                legend_handles, legend_labels = handles, labels

    if legend_handles and legend_labels:
        legend_handles = list(legend_handles) + [
            Line2D([0], [0], color=RANDOM_BASELINE_COLOR, linestyle="--", linewidth=1.5)
        ]
        legend_labels = list(legend_labels) + ["Random Baseline"]
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
    fig.suptitle(args.title, fontsize=14, y=0.985)

    output_path = RESULTS_ROOT / f"{args.output_stem}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_parts: list[pd.DataFrame] = []
    for family_name, summary_df in family_scores.items():
        family_df = summary_df.copy()
        family_df["family_name"] = family_name
        summary_parts.append(family_df)
        random_rows = pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "run_name": "random_baseline",
                    "mean_value": family_random[family_name][metric_name]["bootstrap_mean"],
                    "stderr": float("nan"),
                    "n_groups": int(len(_shared_generated_count_map(family_tables[family_name]))),
                    "family_name": family_name,
                }
                for metric_name, _ in METRICS
            ]
        )
        summary_parts.append(random_rows)
    pd.concat(summary_parts, ignore_index=True).to_csv(
        RESULTS_ROOT / f"{args.output_stem}_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
