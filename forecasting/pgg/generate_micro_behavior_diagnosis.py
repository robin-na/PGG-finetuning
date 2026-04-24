from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
METADATA_ROOT = FORECASTING_ROOT / "metadata"
OUTPUT_DIR = RESULTS_ROOT / "diagnosis"

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

SERIES_ORDER = [
    "Human",
    "Baseline",
    "Demographic Only",
    "Twin-Sampled",
    "Twin Unadjusted",
]

SERIES_COLORS = {
    "Human": "#222222",
    "Baseline": "#4C78A8",
    "Demographic Only": "#9ECAE1",
    "Twin-Sampled": "#D62728",
    "Twin Unadjusted": "#F4A3A3",
}

ACTION_SPECS = [
    {
        "key": "contribution",
        "title": "Contribution",
        "source_col": "contribution_rate",
        "transform": lambda s: (pd.to_numeric(s, errors="coerce") * 20.0).round().astype("Int64"),
        "filter_col": None,
        "x_label": "Coins Contributed",
        "description": "All actor-rounds. Contribution is shown in coins from 0 to 20.",
    },
    {
        "key": "punishment",
        "title": "Punishment",
        "source_col": "punish_target_count",
        "transform": lambda s: pd.to_numeric(s, errors="coerce").round().astype("Int64"),
        "filter_col": "punishment_enabled",
        "x_label": "Punishment Targets Chosen",
        "description": "Actor-rounds from punishment-enabled treatments only.",
    },
    {
        "key": "reward",
        "title": "Reward",
        "source_col": "reward_target_count",
        "transform": lambda s: pd.to_numeric(s, errors="coerce").round().astype("Int64"),
        "filter_col": "reward_enabled",
        "x_label": "Reward Targets Chosen",
        "description": "Actor-rounds from reward-enabled treatments only.",
    },
]


def _tvd(dist_a: dict[int, float], dist_b: dict[int, float]) -> float:
    support = sorted(set(dist_a) | set(dist_b))
    return 0.5 * float(sum(abs(dist_a.get(value, 0.0) - dist_b.get(value, 0.0)) for value in support))


def _distribution(values: pd.Series) -> dict[int, float]:
    clean = pd.Series(values).dropna().astype(int)
    if clean.empty:
        return {}
    counts = clean.value_counts(normalize=True).sort_index()
    return {int(index): float(value) for index, value in counts.items()}


def _load_treatment_flags(baseline_run: str) -> pd.DataFrame:
    manifest_path = METADATA_ROOT / baseline_run / "request_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    cols = ["treatment_name", "punishment_enabled", "reward_enabled"]
    return manifest_df[cols].drop_duplicates("treatment_name").copy()


def _load_actor_table(run_name: str, *, generated: bool) -> pd.DataFrame:
    suffix = "generated_actor_summary.csv" if generated else "human_actor_summary.csv"
    path = RESULTS_ROOT / f"{run_name}__vs_human_treatments" / suffix
    return pd.read_csv(path)


def _prepare_actor_table(
    actor_df: pd.DataFrame,
    *,
    flags_df: pd.DataFrame,
    action_spec: dict[str, Any],
) -> pd.Series:
    merged = actor_df.merge(flags_df, on="treatment_name", how="left")
    filter_col = action_spec["filter_col"]
    if filter_col is not None:
        merged = merged[merged[filter_col].astype(bool)].copy()
    transformed = action_spec["transform"](merged[action_spec["source_col"]])
    return pd.Series(transformed, copy=False)


def _panel_text_lines(
    *,
    family_runs: list[str],
    distributions: dict[str, dict[int, float]],
) -> str:
    human_dist = distributions["Human"]
    lines: list[str] = []
    for run_name in family_runs:
        label = RUN_LABELS[run_name]
        value = _tvd(distributions[label], human_dist)
        lines.append(f"{label}: TVD {value:.3f}")
    return "\n".join(lines)


def _family_slug(family_name: str) -> str:
    return family_name.replace("-", "_").replace(".", "_")


def _plot_family_figure(
    *,
    family_name: str,
    family_runs: list[str],
    flags_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    generated_actor_by_run: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(ACTION_SPECS),
        figsize=(18, 4.8),
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    summary_rows: list[dict[str, Any]] = []

    for col_index, action_spec in enumerate(ACTION_SPECS):
        ax = axes[col_index]

        distributions: dict[str, dict[int, float]] = {}
        counts: dict[str, int] = {}

        human_values = _prepare_actor_table(
            human_actor_df,
            flags_df=flags_df,
            action_spec=action_spec,
        )
        distributions["Human"] = _distribution(human_values)
        counts["Human"] = int(human_values.dropna().shape[0])

        for run_name in family_runs:
            label = RUN_LABELS[run_name]
            values = _prepare_actor_table(
                generated_actor_by_run[run_name],
                flags_df=flags_df,
                action_spec=action_spec,
            )
            distributions[label] = _distribution(values)
            counts[label] = int(values.dropna().shape[0])

        support = sorted({value for dist in distributions.values() for value in dist})
        if not support:
            support = [0]
        x = np.arange(len(support))
        width = 0.15
        offsets = (np.arange(len(SERIES_ORDER)) - (len(SERIES_ORDER) - 1) / 2.0) * width

        for offset, series_name in zip(offsets, SERIES_ORDER):
            values = [distributions.get(series_name, {}).get(value, 0.0) for value in support]
            ax.bar(
                x + offset,
                values,
                width=width * 0.95,
                color=SERIES_COLORS[series_name],
                alpha=0.7 if series_name != "Human" else 0.9,
                edgecolor="none",
                label=series_name,
            )

        if action_spec["key"] == "contribution":
            tick_values = support
            tick_idx = x[::2] if len(x) > 12 else x
            tick_labels = [str(value) for value in tick_values[::2]] if len(x) > 12 else [str(value) for value in tick_values]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=45)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(value) for value in support], rotation=0)

        ax.set_title(action_spec["title"])
        ax.set_xlabel(action_spec["x_label"])
        if col_index == 0:
            ax.set_ylabel("Mass")
        ax.set_ylim(0, 1)

        ax.text(
            0.98,
            0.98,
            _panel_text_lines(family_runs=family_runs, distributions=distributions),
            ha="right",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )

        summary_rows.append(
            {
                "model_family": family_name,
                "action_key": action_spec["key"],
                "action_title": action_spec["title"],
                "description": action_spec["description"],
                "support_json": json.dumps(support),
                "human_n": counts["Human"],
                "human_distribution_json": json.dumps(distributions["Human"], sort_keys=True),
                "baseline_n": counts["Baseline"],
                "baseline_distribution_json": json.dumps(distributions["Baseline"], sort_keys=True),
                "baseline_tvd": _tvd(distributions["Baseline"], distributions["Human"]),
                "demographic_only_n": counts["Demographic Only"],
                "demographic_only_distribution_json": json.dumps(
                    distributions["Demographic Only"],
                    sort_keys=True,
                ),
                "demographic_only_tvd": _tvd(distributions["Demographic Only"], distributions["Human"]),
                "twin_sampled_n": counts["Twin-Sampled"],
                "twin_sampled_distribution_json": json.dumps(
                    distributions["Twin-Sampled"],
                    sort_keys=True,
                ),
                "twin_sampled_tvd": _tvd(distributions["Twin-Sampled"], distributions["Human"]),
                "twin_unadjusted_n": counts["Twin Unadjusted"],
                "twin_unadjusted_distribution_json": json.dumps(
                    distributions["Twin Unadjusted"],
                    sort_keys=True,
                ),
                "twin_unadjusted_tvd": _tvd(distributions["Twin Unadjusted"], distributions["Human"]),
            }
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(f"PGG Micro Behavioral Distribution Diagnosis: {family_name}", y=1.08, fontsize=16)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    figure_path = output_dir / f"pgg_micro_behavior_distribution_diagnosis_{_family_slug(family_name)}.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def build_diagnosis(*, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for family_name, family_spec in RUN_FAMILIES.items():
        family_runs = list(family_spec["runs"])
        flags_df = _load_treatment_flags(family_spec["baseline_run"])
        human_actor_df = _load_actor_table(family_spec["baseline_run"], generated=False)
        generated_actor_by_run = {
            run_name: _load_actor_table(run_name, generated=True)
            for run_name in family_runs
        }
        summary_rows.extend(
            _plot_family_figure(
                family_name=family_name,
                family_runs=family_runs,
                flags_df=flags_df,
                human_actor_df=human_actor_df,
                generated_actor_by_run=generated_actor_by_run,
                output_dir=output_dir,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "behavior_distribution_summary.csv", index=False)

    readme_lines = [
        "# PGG Diagnosis",
        "",
        "- Figure (gpt-5-mini): [pgg_micro_behavior_distribution_diagnosis_gpt_5_mini.png](./pgg_micro_behavior_distribution_diagnosis_gpt_5_mini.png)",
        "- Figure (gpt-5.1): [pgg_micro_behavior_distribution_diagnosis_gpt_5_1.png](./pgg_micro_behavior_distribution_diagnosis_gpt_5_1.png)",
        "- Summary: [behavior_distribution_summary.csv](./behavior_distribution_summary.csv)",
        "",
        "Notes:",
        "- Contribution uses all actor-rounds and is shown in coins from 0 to 20.",
        "- Punishment uses actor-rounds from punishment-enabled treatments only.",
        "- Reward uses actor-rounds from reward-enabled treatments only.",
        "- Each panel compares raw empirical mass for Human, Baseline, Demographic Only, Twin-Sampled, and Twin Unadjusted.",
        "- The text box in each panel reports TVD against the human distribution for the four generated variants.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PGG micro behavioral distribution diagnosis plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for diagnosis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_diagnosis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
