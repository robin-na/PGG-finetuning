from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import build_primary_distribution_summary, compute_metric_tables
from common import (
    CHOICE_SUPPORT,
    DELEGATION_FIELDS,
    NUMERIC_SCENARIO_FIELDS,
    SCENARIO_FIELDS,
    build_generated_scenarios_df,
    build_generated_sessions_df,
    build_human_scenarios_df,
    build_human_sessions_df,
    total_variation_distance,
    wasserstein_distance_1d,
    write_csv,
    write_json,
)


RUN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
NOISE_CEILING_COLOR = "#2CA02C"
PRETTY_METRIC_LABELS = {
    "fairness_wd": "UG Offer + Responder Threshold",
    "trust_wd": "TG Sender Send",
    "trustworthiness_wd": "TG Receiver Return",
    "cooperation_wd": "PD + SH Choice",
    "coordination_wd": "Earth Choice (Paper Coding)",
    "delegation_index_wd": "Delegation Index",
    "fairness_offer_wd": "UG Proposer Offer",
    "inequality_tolerance_wd": "UG Responder Minimum Threshold",
    "trust_send_wd": "TG Sender Send",
    "trustworthiness_return_wd": "TG Receiver Return",
    "cooperation_pd_wd": "PD Choice",
    "cooperation_sh_wd": "SH Choice",
    "coordination_earth_wd": "Earth Choice",
    "payoff_index_wd": "Payoff Index",
    "prosociality_index_wd": "Prosociality Index",
    "kindness_index_wd": "Kindness Index",
    "intentions_index_wd": "Intentions Index",
    "predictability_index_wd": "Predictability Index",
    "equality_index_wd": "Equality Index",
    "mean_delegation_wd": "Mean Delegation",
    "mean_role_state_tv": "Mean Role State",
    "mean_numeric_direct_value_wd": "Mean Numeric Direct Value",
    "C_delegated": "Coordination Delegation",
    "PD_delegated": "PD Delegation",
    "SH_delegated": "SH Delegation",
    "TGReceiver_delegated": "TG Receiver Delegation",
    "TGSender_delegated": "TG Sender Delegation",
    "UGProposer_delegated": "UG Proposer Delegation",
    "UGResponder_delegated": "UG Responder Delegation",
    "C_decision_state": "Coordination State",
    "PD_decision_state": "PD State",
    "SH_decision_state": "SH State",
    "TGReceiver_decision_state": "TG Receiver State",
    "TGSender_decision_state": "TG Sender State",
    "UGProposer_decision_state": "UG Proposer State",
    "UGResponder_decision_state": "UG Responder State",
    "UGProposer_decision_direct": "UG Proposer Direct",
    "UGResponder_decision_direct": "UG Responder Direct",
    "TGReceiver_decision_direct": "TG Receiver Direct",
}
HEADLINE_METRICS = [
    "fairness_offer_wd",
    "inequality_tolerance_wd",
    "trust_send_wd",
    "trustworthiness_return_wd",
    "cooperation_pd_wd",
    "cooperation_sh_wd",
    "coordination_earth_wd",
    "delegation_index_wd",
]


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


def _load_scenario_manifest_json(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    return payload if isinstance(payload, list) else []


def _sample_random_choice(field_name: str, rng: np.random.Generator) -> Any:
    if field_name in NUMERIC_SCENARIO_FIELDS:
        low, high = NUMERIC_SCENARIO_FIELDS[field_name]
        return int(rng.integers(low, high + 1))
    return str(rng.choice(CHOICE_SUPPORT[field_name]))


def _simulate_random_session_rows(
    *,
    request_manifest_csv: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    manifest = pd.read_csv(request_manifest_csv)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        treatment = str(row["Treatment"])
        scenario_manifest = _load_scenario_manifest_json(str(row["scenario_manifest_json"]))

        if treatment == "TransparentRandom":
            delegation_payload = {field_name: None for field_name in DELEGATION_FIELDS}
        else:
            delegation_payload = {
                field_name: int(rng.integers(0, 2))
                for field_name in DELEGATION_FIELDS
            }

        scenario_outputs: list[dict[str, Any]] = []
        for scenario_item in scenario_manifest:
            scenario_name = str(scenario_item["scenario"])
            case_name = str(scenario_item["case"])
            scenario_obj: dict[str, Any] = {
                "scenario": scenario_name,
                "case": case_name,
            }
            for field_name in SCENARIO_FIELDS:
                if scenario_name == "NoAISupport":
                    scenario_obj[field_name] = _sample_random_choice(field_name, rng)
                    continue
                if treatment == "TransparentRandom":
                    scenario_obj[field_name] = _sample_random_choice(field_name, rng)
                    continue
                delegated = delegation_payload[
                    {
                        "UGProposer_decision": "UGProposer_delegated",
                        "UGResponder_decision": "UGResponder_delegated",
                        "TGSender_decision": "TGSender_delegated",
                        "TGReceiver_decision": "TGReceiver_delegated",
                        "PD_decision": "PD_delegated",
                        "SH_decision": "SH_delegated",
                        "C_decision": "C_delegated",
                    }[field_name]
                ]
                scenario_obj[field_name] = None if delegated == 1 else _sample_random_choice(field_name, rng)
            scenario_outputs.append(scenario_obj)

        rows.append(
            {
                "custom_id": str(row["custom_id"]),
                "parse_success": True,
                "parsed_target": {
                    **delegation_payload,
                    "scenario_outputs": scenario_outputs,
                },
            }
        )
    return rows


def _build_random_primary_distribution_summary(
    *,
    request_manifest_csv: Path,
    gold_targets_jsonl: Path,
    iters: int,
    seed: int,
) -> pd.DataFrame:
    human_sessions = build_human_sessions_df(
        gold_targets_jsonl=gold_targets_jsonl,
        request_manifest_csv=request_manifest_csv,
    )
    human_scenarios = build_human_scenarios_df(
        gold_targets_jsonl=gold_targets_jsonl,
        request_manifest_csv=request_manifest_csv,
    )

    iter_frames: list[pd.DataFrame] = []
    tmp_path = Path("/tmp/multi_game_random_parsed_output.jsonl")
    for iter_idx in range(iters):
        rows = _simulate_random_session_rows(
            request_manifest_csv=request_manifest_csv,
            seed=seed + iter_idx,
        )
        tmp_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )
        generated_sessions = build_generated_sessions_df(
            parsed_output_jsonl=tmp_path,
            request_manifest_csv=request_manifest_csv,
        )
        generated_scenarios = build_generated_scenarios_df(
            parsed_output_jsonl=tmp_path,
            request_manifest_csv=request_manifest_csv,
        )
        _, _, overall_df = compute_metric_tables(
            generated_sessions=generated_sessions,
            human_sessions=human_sessions,
            generated_scenarios=generated_scenarios,
            human_scenarios=human_scenarios,
        )
        summary_df = build_primary_distribution_summary(overall_df).copy()
        summary_df["iter"] = iter_idx
        iter_frames.append(summary_df)

    combined = pd.concat(iter_frames, ignore_index=True)
    summary = (
        combined.groupby(["metric_family", "metric", "distance_kind"], as_index=False)
        .agg(
            n_groups=("n_groups", "first"),
            random_mean=("mean_value", "mean"),
            random_median=("mean_value", "median"),
            random_p05=("mean_value", lambda s: float(np.quantile(s, 0.05))),
            random_p95=("mean_value", lambda s: float(np.quantile(s, 0.95))),
        )
    )
    return summary


def _compute_distribution_rows(
    *,
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    common_treatments = sorted(
        set(generated_sessions["treatment_name"].dropna().astype(str))
        & set(human_sessions["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_sessions[generated_sessions["treatment_name"] == treatment_name].copy()
        human_group = human_sessions[human_sessions["treatment_name"] == treatment_name].copy()
        field_scores: list[float] = []
        for field_name in DELEGATION_FIELDS:
            generated_values = generated_group[field_name].dropna().astype(float)
            human_values = human_group[field_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            score = wasserstein_distance_1d(generated_values, human_values)
            field_scores.append(score)
            rows.append(
                {
                    "metric_family": "delegation_distribution_distance",
                    "metric_scope": "distribution",
                    "metric": field_name,
                    "distance_kind": "wasserstein_1d",
                    "n_groups": 4,
                    "mean_value": score,
                }
            )
        if field_scores:
            rows.append(
                {
                    "metric_family": "delegation_distribution_distance",
                    "metric_scope": "distribution",
                    "metric": "mean_delegation_wd",
                    "distance_kind": "mean_wasserstein_1d",
                    "n_groups": 4,
                    "mean_value": float(np.mean(field_scores)),
                }
            )

    common_cells = sorted(
        set(generated_scenarios["cell_name"].dropna().astype(str))
        & set(human_scenarios["cell_name"].dropna().astype(str))
    )
    for cell_name in common_cells:
        generated_group = generated_scenarios[generated_scenarios["cell_name"] == cell_name].copy()
        human_group = human_scenarios[human_scenarios["cell_name"] == cell_name].copy()
        state_scores: list[float] = []
        numeric_scores: list[float] = []

        for field_name in SCENARIO_FIELDS:
            state_score = total_variation_distance(
                generated_group[f"{field_name}_state"],
                human_group[f"{field_name}_state"],
                support=_state_support(field_name),
            )
            state_scores.append(state_score)
            rows.append(
                {
                    "metric_family": "scenario_state_distribution_distance",
                    "metric_scope": "distribution",
                    "metric": f"{field_name}_state",
                    "distance_kind": "total_variation",
                    "n_groups": 6,
                    "mean_value": state_score,
                }
            )

        if state_scores:
            rows.append(
                {
                    "metric_family": "scenario_state_distribution_distance",
                    "metric_scope": "distribution",
                    "metric": "mean_role_state_tv",
                    "distance_kind": "total_variation",
                    "n_groups": 6,
                    "mean_value": float(np.mean(state_scores)),
                }
            )

        for field_name in NUMERIC_SCENARIO_FIELDS:
            generated_values = generated_group[field_name].dropna().astype(float)
            human_values = human_group[field_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            score = wasserstein_distance_1d(generated_values, human_values)
            numeric_scores.append(score)
            rows.append(
                {
                    "metric_family": "scenario_direct_value_distance",
                    "metric_scope": "distribution",
                    "metric": f"{field_name}_direct",
                    "distance_kind": "wasserstein_1d",
                    "n_groups": 6,
                    "mean_value": score,
                }
            )

        if numeric_scores:
            rows.append(
                {
                    "metric_family": "scenario_direct_value_distance",
                    "metric_scope": "distribution",
                    "metric": "mean_numeric_direct_value_wd",
                    "distance_kind": "mean_wasserstein_1d",
                    "n_groups": 6,
                    "mean_value": float(np.mean(numeric_scores)),
                }
            )

    summary = (
        pd.DataFrame(rows)
        .groupby(["metric_family", "metric_scope", "metric", "distance_kind", "n_groups"], as_index=False)
        .agg(
            mean_value=("mean_value", "mean"),
            median_value=("mean_value", "median"),
        )
    )
    summary["stderr"] = np.nan
    return summary.to_dict(orient="records")


def _state_support(field_name: str) -> list[str]:
    if field_name in {"UGProposer_decision", "UGResponder_decision"}:
        return [str(value) for value in range(0, 11)] + ["NULL"]
    if field_name == "TGReceiver_decision":
        return [str(value) for value in range(0, 7)] + ["NULL"]
    if field_name == "TGSender_decision":
        return ["YES", "NO", "NULL"]
    if field_name == "PD_decision":
        return ["A", "B", "NULL"]
    if field_name == "SH_decision":
        return ["X", "Y", "NULL"]
    if field_name == "C_decision":
        return ["Mercury", "Venus", "Earth", "Mars", "Saturn", "NULL"]
    raise ValueError(f"Unsupported field_name: {field_name}")


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
        description="Plot primary multi-game distribution-alignment metrics against the human noise ceiling."
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
            n_groups=("n_groups", "first"),
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
            "headline_metrics": HEADLINE_METRICS,
        },
    )

    _plot_metric_panels(
        comparison_df=comparison_df,
        noise_df=noise_df,
        metrics=headline_metrics,
        output_path=output_dir / "headline_primary_distribution_comparison.png",
        title="Multi-Game Benchmark: Headline Distribution Metrics Vs Noise Ceiling",
    )
    _plot_metric_panels(
        comparison_df=comparison_df,
        noise_df=noise_df,
        metrics=metrics,
        output_path=output_dir / "all_primary_distribution_metrics.png",
        title="Multi-Game Benchmark: All Primary Distribution Metrics Vs Noise Ceiling",
    )


if __name__ == "__main__":
    main()
