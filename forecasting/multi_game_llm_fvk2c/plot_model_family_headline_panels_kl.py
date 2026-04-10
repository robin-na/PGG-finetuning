from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "forecasting").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.kl_divergence_utils import (
    bootstrap_summary,
    categorical_kl_divergence,
    histogram_kl_divergence,
    mean_and_stderr,
)
from forecasting.multi_game_llm_fvk2c.analysis_utils import _augment_scenario_indices
from forecasting.multi_game_llm_fvk2c.common import (
    DELEGATION_FIELDS,
    build_generated_scenarios_df,
    build_generated_sessions_df,
    build_human_scenarios_df,
    build_human_sessions_df,
)


FORECASTING_ROOT = Path(__file__).resolve().parent
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
    ("mean_delegation", "Mean Delegation"),
    ("trust_proxy", "Trust Proxy"),
]
ALPHA = 1.0
BOOTSTRAP_ITERS = 300
RNG_SEED = 19
PSI_BIN_EDGES = np.linspace(0.0, 1.0, 21)
PSI_BIN_EDGES[-1] += 1e-9


def _load_run_data(run_name: str) -> dict[str, pd.DataFrame]:
    metadata_dir = FORECASTING_ROOT / "metadata" / run_name
    request_manifest = metadata_dir / "request_manifest.csv"
    return {
        "human_sessions": build_human_sessions_df(
            gold_targets_jsonl=metadata_dir / "gold_targets.jsonl",
            request_manifest_csv=request_manifest,
        ),
        "generated_sessions": build_generated_sessions_df(
            parsed_output_jsonl=metadata_dir / "parsed_output.jsonl",
            request_manifest_csv=request_manifest,
        ),
        "human_scenarios": build_human_scenarios_df(
            gold_targets_jsonl=metadata_dir / "gold_targets.jsonl",
            request_manifest_csv=request_manifest,
        ),
        "generated_scenarios": build_generated_scenarios_df(
            parsed_output_jsonl=metadata_dir / "parsed_output.jsonl",
            request_manifest_csv=request_manifest,
        ),
    }


def _mean_delegation_scores(
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    common_treatments = sorted(
        set(generated_sessions["treatment_name"].dropna().astype(str))
        & set(human_sessions["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_sessions[generated_sessions["treatment_name"] == treatment_name].copy()
        human_group = human_sessions[human_sessions["treatment_name"] == treatment_name].copy()
        scores: list[float] = []
        for field_name in DELEGATION_FIELDS:
            score = categorical_kl_divergence(
                human_group[field_name].dropna().astype(int),
                generated_group[field_name].dropna().astype(int),
                support=[0, 1],
                alpha=ALPHA,
            )
            if np.isfinite(score):
                scores.append(score)
        if scores:
            rows.append({"cell_name": treatment_name, "score": float(np.mean(scores))})
    return pd.DataFrame(rows)


def _trust_proxy_scores(
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    generated_aug = _augment_scenario_indices(generated_scenarios)
    human_aug = _augment_scenario_indices(human_scenarios)
    rows: list[dict[str, float | str]] = []
    common_cells = sorted(
        set(generated_aug["paper_role_cell_name"].dropna().astype(str))
        & set(human_aug["paper_role_cell_name"].dropna().astype(str))
    )
    for cell_name in common_cells:
        generated_group = generated_aug[generated_aug["paper_role_cell_name"] == cell_name].copy()
        human_group = human_aug[human_aug["paper_role_cell_name"] == cell_name].copy()
        score = histogram_kl_divergence(
            human_group["PSI"].dropna().astype(float).clip(0.0, 1.0),
            generated_group["PSI"].dropna().astype(float).clip(0.0, 1.0),
            bin_edges=PSI_BIN_EDGES,
            alpha=ALPHA,
        )
        if np.isfinite(score):
            rows.append({"cell_name": cell_name, "score": score})
    return pd.DataFrame(rows)


def _metric_scores(run_data: dict[str, pd.DataFrame], metric_name: str) -> pd.DataFrame:
    if metric_name == "mean_delegation":
        return _mean_delegation_scores(run_data["generated_sessions"], run_data["human_sessions"])
    if metric_name == "trust_proxy":
        return _trust_proxy_scores(run_data["generated_scenarios"], run_data["human_scenarios"])
    raise ValueError(f"Unsupported metric: {metric_name}")


def _noise_ceiling_summary(
    *,
    baseline_data: dict[str, pd.DataFrame],
    metric_name: str,
) -> dict[str, float]:
    rng = np.random.default_rng(RNG_SEED)
    bootstrap_scores: list[float] = []
    if metric_name == "mean_delegation":
        human_df = baseline_data["human_sessions"].copy()
        generated_df = baseline_data["generated_sessions"].copy()
        common_treatments = sorted(
            set(generated_df["treatment_name"].dropna().astype(str))
            & set(human_df["treatment_name"].dropna().astype(str))
        )
        for _ in range(BOOTSTRAP_ITERS):
            scores: list[float] = []
            for treatment_name in common_treatments:
                human_group = human_df[human_df["treatment_name"] == treatment_name].copy().reset_index(drop=True)
                generated_group = generated_df[generated_df["treatment_name"] == treatment_name].copy().reset_index(drop=True)
                pseudo_generated = human_group.iloc[rng.integers(0, len(human_group), size=len(generated_group))].copy()
                pseudo_human = human_group.iloc[rng.integers(0, len(human_group), size=len(human_group))].copy()
                scores_df = _mean_delegation_scores(pseudo_generated, pseudo_human)
                if not scores_df.empty:
                    scores.extend(scores_df["score"].astype(float).tolist())
            if scores:
                bootstrap_scores.append(float(np.mean(scores)))
    elif metric_name == "trust_proxy":
        human_df = _augment_scenario_indices(baseline_data["human_scenarios"].copy())
        generated_df = _augment_scenario_indices(baseline_data["generated_scenarios"].copy())
        common_cells = sorted(
            set(generated_df["paper_role_cell_name"].dropna().astype(str))
            & set(human_df["paper_role_cell_name"].dropna().astype(str))
        )
        for _ in range(BOOTSTRAP_ITERS):
            scores: list[float] = []
            for cell_name in common_cells:
                human_group = human_df[human_df["paper_role_cell_name"] == cell_name].copy().reset_index(drop=True)
                generated_group = generated_df[generated_df["paper_role_cell_name"] == cell_name].copy().reset_index(drop=True)
                pseudo_generated = human_group.iloc[rng.integers(0, len(human_group), size=len(generated_group))].copy()
                pseudo_human = human_group.iloc[rng.integers(0, len(human_group), size=len(human_group))].copy()
                score = histogram_kl_divergence(
                    pseudo_human["PSI"].dropna().astype(float).clip(0.0, 1.0),
                    pseudo_generated["PSI"].dropna().astype(float).clip(0.0, 1.0),
                    bin_edges=PSI_BIN_EDGES,
                    alpha=ALPHA,
                )
                if np.isfinite(score):
                    scores.append(score)
            if scores:
                bootstrap_scores.append(float(np.mean(scores)))
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")
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
    family_run_data: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    family_scores: dict[str, pd.DataFrame] = {}
    family_noise: dict[str, dict[str, dict[str, float]]] = {}
    for family_name, spec in RUN_FAMILIES.items():
        run_data_map: dict[str, dict[str, pd.DataFrame]] = {}
        summary_rows: list[dict[str, float | str]] = []
        for run_name in spec["runs"]:
            run_data = _load_run_data(run_name)
            run_data_map[run_name] = run_data
            for metric_name, _ in METRICS:
                scores_df = _metric_scores(run_data, metric_name)
                mean_value, stderr = mean_and_stderr(scores_df["score"])
                summary_rows.append(
                    {
                        "metric": metric_name,
                        "run_name": run_name,
                        "mean_value": mean_value,
                        "stderr": stderr,
                        "n_groups": int(scores_df["cell_name"].nunique()) if not scores_df.empty else 0,
                    }
                )
        family_run_data[family_name] = run_data_map
        family_scores[family_name] = pd.DataFrame(summary_rows)
        baseline_data = run_data_map[spec["baseline_run"]]
        family_noise[family_name] = {
            metric_name: _noise_ceiling_summary(baseline_data=baseline_data, metric_name=metric_name)
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
    fig.suptitle("Multi-Game LLM Delegation", fontsize=14, y=0.985)

    output_path = FORECASTING_ROOT / "results" / "headline_model_family_panels_kl.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows: list[dict[str, float | str]] = []
    for family_name, summary_df in family_scores.items():
        for row in summary_df.to_dict(orient="records"):
            summary_rows.append({"family_name": family_name, **row})
        for metric_name, noise in family_noise[family_name].items():
            summary_rows.append({"family_name": family_name, "metric": metric_name, "run_name": "human_ceiling", **noise})
    pd.DataFrame(summary_rows).to_csv(
        FORECASTING_ROOT / "results" / "headline_model_family_panels_kl_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
