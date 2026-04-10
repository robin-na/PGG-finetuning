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
    integer_kl_divergence,
    mean_and_stderr,
)
from forecasting.two_stage_trust_punishment_y2hgu.common import (
    ROLE_A_CHECK,
    ROLE_A_TIME,
    ROLE_B_HIDDEN_CHECK,
    ROLE_B_HIDDEN_TIME,
    ROLE_B_OBSERVABLE_CHECK,
    ROLE_B_OBSERVABLE_TIME,
    build_generated_records_df,
    build_human_records_df,
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
    ("mean_send_field_distance", "Send"),
    ("return_pct", "Return %"),
]
OBSERVABLE_SEND_SCHEMAS = {ROLE_B_OBSERVABLE_CHECK, ROLE_B_OBSERVABLE_TIME}
HIDDEN_SEND_SCHEMAS = {ROLE_B_HIDDEN_CHECK, ROLE_B_HIDDEN_TIME}
RETURN_SCHEMAS = {ROLE_A_CHECK, ROLE_A_TIME}
ALPHA = 1.0
BOOTSTRAP_ITERS = 300
RNG_SEED = 17


def _load_run_records(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_dir = FORECASTING_ROOT / "metadata" / run_name
    human_df = build_human_records_df(
        gold_targets_jsonl=metadata_dir / "gold_targets.jsonl",
        request_manifest_csv=metadata_dir / "request_manifest.csv",
    )
    generated_df = build_generated_records_df(
        parsed_output_jsonl=metadata_dir / "parsed_output.jsonl",
        request_manifest_csv=metadata_dir / "request_manifest.csv",
    )
    return human_df, generated_df


def _send_fields_for_schema(schema_type: str) -> list[str]:
    if schema_type == ROLE_B_OBSERVABLE_CHECK:
        return [
            "send_if_act_without_check",
            "send_if_act_after_check",
            "send_if_no_act_without_check",
            "send_if_no_act_after_check",
        ]
    if schema_type == ROLE_B_OBSERVABLE_TIME:
        return [
            "send_if_act_fast",
            "send_if_no_act_fast",
            "send_if_act_slow",
            "send_if_no_act_slow",
        ]
    if schema_type in HIDDEN_SEND_SCHEMAS:
        return ["send_if_act", "send_if_no_act"]
    return []


def _metric_treatment_scores(
    generated_df: pd.DataFrame,
    human_df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    common_treatments = sorted(
        set(generated_df["treatment_name"].dropna().astype(str))
        & set(human_df["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_df[generated_df["treatment_name"] == treatment_name].copy()
        human_group = human_df[human_df["treatment_name"] == treatment_name].copy()
        if generated_group.empty or human_group.empty:
            continue
        schema_type = str(human_group["schema_type"].iloc[0])
        if metric_name == "mean_send_field_distance":
            field_names = _send_fields_for_schema(schema_type)
            if not field_names:
                continue
            scores: list[float] = []
            for field_name in field_names:
                score = integer_kl_divergence(
                    human_group[field_name],
                    generated_group[field_name],
                    min_value=0,
                    max_value=10,
                    alpha=ALPHA,
                )
                if np.isfinite(score):
                    scores.append(score)
            if not scores:
                continue
            rows.append({"treatment_name": treatment_name, "score": float(np.mean(scores))})
        elif metric_name == "return_pct":
            if schema_type not in RETURN_SCHEMAS:
                continue
            score = integer_kl_divergence(
                human_group["return_pct"],
                generated_group["return_pct"],
                min_value=0,
                max_value=100,
                alpha=ALPHA,
            )
            if np.isfinite(score):
                rows.append({"treatment_name": treatment_name, "score": score})
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    return pd.DataFrame(rows)


def _noise_ceiling_summary(
    *,
    human_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    metric_name: str,
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
            if human_group.empty or generated_group.empty:
                continue
            pseudo_generated = human_group.iloc[rng.integers(0, len(human_group), size=len(generated_group))].copy()
            pseudo_human = human_group.iloc[rng.integers(0, len(human_group), size=len(human_group))].copy()
            scores_df = _metric_treatment_scores(pseudo_generated, pseudo_human, metric_name)
            if not scores_df.empty:
                treatment_scores.extend(scores_df["score"].astype(float).tolist())
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
    family_run_records: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    family_scores: dict[str, pd.DataFrame] = {}
    family_noise: dict[str, dict[str, dict[str, float]]] = {}

    for family_name, spec in RUN_FAMILIES.items():
        run_records: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        summary_rows: list[dict[str, float | str]] = []
        for run_name in spec["runs"]:
            human_df, generated_df = _load_run_records(run_name)
            run_records[run_name] = (human_df, generated_df)
            for metric_name, _ in METRICS:
                scores_df = _metric_treatment_scores(generated_df, human_df, metric_name)
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
        family_run_records[family_name] = run_records
        family_scores[family_name] = pd.DataFrame(summary_rows)
        baseline_human, baseline_generated = run_records[spec["baseline_run"]]
        family_noise[family_name] = {
            metric_name: _noise_ceiling_summary(
                human_df=baseline_human,
                generated_df=baseline_generated,
                metric_name=metric_name,
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
    fig.suptitle("Two-Stage Trust / Punishment / Helping", fontsize=14, y=0.985)

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
