from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    ("player_mean_payoff", "Player Mean Payoff"),
    ("round_efficiency", "Round Efficiency"),
]
BOOTSTRAP_ITERS = 300
RNG_SEED = 29


def _wasserstein_distance_1d(x_values: pd.Series, y_values: pd.Series) -> float:
    x = np.sort(pd.Series(x_values).dropna().astype(float).to_numpy())
    y = np.sort(pd.Series(y_values).dropna().astype(float).to_numpy())
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size == 1 and y.size == 1:
        return float(abs(x[0] - y[0]))
    support = np.sort(np.concatenate([x, y]))
    deltas = np.diff(support)
    if deltas.size == 0:
        return 0.0
    x_cdf = np.searchsorted(x, support[:-1], side="right") / float(x.size)
    y_cdf = np.searchsorted(y, support[:-1], side="right") / float(y.size)
    return float(np.sum(np.abs(x_cdf - y_cdf) * deltas))


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


def _load_run_tables(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = RESULTS_ROOT / f"{run_name}__vs_human_treatments"
    generated_actor_df = pd.read_csv(run_dir / "generated_actor_summary.csv")
    human_actor_df = pd.read_csv(run_dir / "human_actor_summary.csv")
    generated_round_df = pd.read_csv(run_dir / "generated_round_summary.csv")
    human_round_df = pd.read_csv(run_dir / "human_round_summary.csv")
    return generated_actor_df, human_actor_df, generated_round_df, human_round_df


def _player_mean_payoff_frame(actor_df: pd.DataFrame) -> pd.DataFrame:
    entity_col = "entity_id" if "entity_id" in actor_df.columns else "game_id"
    grouped = (
        actor_df.groupby(["treatment_name", entity_col, "player_id"], as_index=False)
        .agg(player_mean_payoff=("round_payoff", "mean"))
    )
    return grouped


def _treatment_scores_for_metric(
    metric_name: str,
    *,
    generated_actor_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    generated_round_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if metric_name == "player_mean_payoff":
        generated_metric_df = _player_mean_payoff_frame(generated_actor_df)
        human_metric_df = _player_mean_payoff_frame(human_actor_df)
        common_treatments = sorted(
            set(generated_metric_df["treatment_name"].dropna().astype(str))
            & set(human_metric_df["treatment_name"].dropna().astype(str))
        )
        for treatment_name in common_treatments:
            generated_values = generated_metric_df.loc[
                generated_metric_df["treatment_name"] == treatment_name,
                "player_mean_payoff",
            ]
            human_values = human_metric_df.loc[
                human_metric_df["treatment_name"] == treatment_name,
                "player_mean_payoff",
            ]
            score = _wasserstein_distance_1d(generated_values, human_values)
            if np.isfinite(score):
                rows.append({"treatment_name": treatment_name, "score": score})
    elif metric_name == "round_efficiency":
        common_treatments = sorted(
            set(generated_round_df["treatment_name"].dropna().astype(str))
            & set(human_round_df["treatment_name"].dropna().astype(str))
        )
        for treatment_name in common_treatments:
            generated_values = generated_round_df.loc[
                generated_round_df["treatment_name"] == treatment_name,
                "round_normalized_efficiency",
            ]
            human_values = human_round_df.loc[
                human_round_df["treatment_name"] == treatment_name,
                "round_normalized_efficiency",
            ]
            score = _wasserstein_distance_1d(generated_values, human_values)
            if np.isfinite(score):
                rows.append({"treatment_name": treatment_name, "score": score})
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")
    return pd.DataFrame(rows)


def _noise_ceiling_summary(
    metric_name: str,
    *,
    generated_actor_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    generated_round_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
) -> dict[str, float]:
    rng = np.random.default_rng(RNG_SEED)
    bootstrap_scores: list[float] = []

    if metric_name == "player_mean_payoff":
        generated_metric_df = _player_mean_payoff_frame(generated_actor_df)
        human_metric_df = _player_mean_payoff_frame(human_actor_df)
        common_treatments = sorted(
            set(generated_metric_df["treatment_name"].dropna().astype(str))
            & set(human_metric_df["treatment_name"].dropna().astype(str))
        )
        for _ in range(BOOTSTRAP_ITERS):
            treatment_scores: list[float] = []
            for treatment_name in common_treatments:
                generated_values = generated_metric_df.loc[
                    generated_metric_df["treatment_name"] == treatment_name,
                    "player_mean_payoff",
                ].reset_index(drop=True)
                human_values = human_metric_df.loc[
                    human_metric_df["treatment_name"] == treatment_name,
                    "player_mean_payoff",
                ].reset_index(drop=True)
                pseudo_generated = human_values.iloc[rng.integers(0, len(human_values), size=len(generated_values))]
                pseudo_human = human_values.iloc[rng.integers(0, len(human_values), size=len(human_values))]
                score = _wasserstein_distance_1d(pseudo_generated, pseudo_human)
                if np.isfinite(score):
                    treatment_scores.append(score)
            if treatment_scores:
                bootstrap_scores.append(float(np.mean(treatment_scores)))
    elif metric_name == "round_efficiency":
        common_treatments = sorted(
            set(generated_round_df["treatment_name"].dropna().astype(str))
            & set(human_round_df["treatment_name"].dropna().astype(str))
        )
        for _ in range(BOOTSTRAP_ITERS):
            treatment_scores: list[float] = []
            for treatment_name in common_treatments:
                generated_values = generated_round_df.loc[
                    generated_round_df["treatment_name"] == treatment_name,
                    "round_normalized_efficiency",
                ].reset_index(drop=True)
                human_values = human_round_df.loc[
                    human_round_df["treatment_name"] == treatment_name,
                    "round_normalized_efficiency",
                ].reset_index(drop=True)
                pseudo_generated = human_values.iloc[rng.integers(0, len(human_values), size=len(generated_values))]
                pseudo_human = human_values.iloc[rng.integers(0, len(human_values), size=len(human_values))]
                score = _wasserstein_distance_1d(pseudo_generated, pseudo_human)
                if np.isfinite(score):
                    treatment_scores.append(score)
            if treatment_scores:
                bootstrap_scores.append(float(np.mean(treatment_scores)))
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return _bootstrap_summary(bootstrap_scores)


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
        ax.set_ylabel(f"{metric_label}\nWasserstein Distance", fontsize=10)
    return handles, labels


def main() -> None:
    family_tables: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
    family_scores: dict[str, pd.DataFrame] = {}
    family_noise: dict[str, dict[str, dict[str, float]]] = {}

    for family_name, spec in RUN_FAMILIES.items():
        run_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        summary_rows: list[dict[str, float | str]] = []
        for run_name in spec["runs"]:
            tables = _load_run_tables(run_name)
            run_tables[run_name] = tables
            generated_actor_df, human_actor_df, generated_round_df, human_round_df = tables
            for metric_name, _ in METRICS:
                scores_df = _treatment_scores_for_metric(
                    metric_name,
                    generated_actor_df=generated_actor_df,
                    human_actor_df=human_actor_df,
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
        family_tables[family_name] = run_tables
        family_scores[family_name] = pd.DataFrame(summary_rows)
        baseline_tables = run_tables[spec["baseline_run"]]
        family_noise[family_name] = {
            metric_name: _noise_ceiling_summary(
                metric_name,
                generated_actor_df=baseline_tables[0],
                human_actor_df=baseline_tables[1],
                generated_round_df=baseline_tables[2],
                human_round_df=baseline_tables[3],
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

    fig.suptitle("Public Goods Game Micro-Level", fontsize=14, y=0.985)
    output_path = RESULTS_ROOT / "headline_model_family_micro_panels.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows: list[dict[str, float | str]] = []
    for family_name, summary_df in family_scores.items():
        for row in summary_df.to_dict(orient="records"):
            summary_rows.append({"family_name": family_name, **row})
        for metric_name, noise in family_noise[family_name].items():
            summary_rows.append({"family_name": family_name, "metric": metric_name, "run_name": "human_ceiling", **noise})
    pd.DataFrame(summary_rows).to_csv(RESULTS_ROOT / "headline_model_family_micro_panels_summary.csv", index=False)


if __name__ == "__main__":
    main()
