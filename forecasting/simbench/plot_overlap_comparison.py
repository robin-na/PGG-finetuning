from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

MPLCONFIGDIR = str((Path(__file__).resolve().parent / ".mplconfig").resolve())
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)

import matplotlib.pyplot as plt


def _load_row_eval(results_root: Path, run_name: str) -> pd.DataFrame:
    path = results_root / f"{run_name}__gold_eval" / "row_level_evaluation.csv"
    return pd.read_csv(path)


def _normalized_entropy(predicted_entropy: float, option_count: int) -> float:
    if pd.isna(predicted_entropy) or pd.isna(option_count):
        return float("nan")
    option_count = int(option_count)
    if option_count <= 1:
        return float("nan")
    max_entropy = math.log(option_count)
    if max_entropy <= 0:
        return float("nan")
    return float(predicted_entropy) / max_entropy


def _serialize_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _standard_error(series: pd.Series) -> float:
    clean = series.dropna().astype(float)
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _build_overlap_summary(
    *,
    results_root: Path,
    baseline_run: str,
    twin_run: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _load_row_eval(results_root, baseline_run)
    twin = _load_row_eval(results_root, twin_run)

    merged = baseline.merge(
        twin,
        on=["simbench_row_id", "dataset_name"],
        suffixes=("_baseline", "_twin"),
    )
    overlap = merged[
        merged["evaluated_baseline"].astype(bool) & merged["evaluated_twin"].astype(bool)
    ].copy()

    overlap["normalized_predicted_entropy_baseline"] = overlap.apply(
        lambda row: _normalized_entropy(row["predicted_entropy_baseline"], row["option_count_baseline"]),
        axis=1,
    )
    overlap["normalized_predicted_entropy_twin"] = overlap.apply(
        lambda row: _normalized_entropy(row["predicted_entropy_twin"], row["option_count_twin"]),
        axis=1,
    )

    row_cols = [
        "simbench_row_id",
        "dataset_name",
        "group_size_baseline",
        "tvd_baseline",
        "tvd_twin",
        "jsd_baseline",
        "jsd_twin",
        "predicted_entropy_baseline",
        "predicted_entropy_twin",
        "normalized_predicted_entropy_baseline",
        "normalized_predicted_entropy_twin",
        "modal_match_baseline",
        "modal_match_twin",
    ]
    overlap[row_cols].rename(columns={"group_size_baseline": "group_size"}).to_csv(
        output_dir / "row_overlap.csv", index=False
    )

    dataset_rows: list[dict] = []
    for dataset_name, group in overlap.groupby("dataset_name", sort=True):
        dataset_rows.append(
            {
                "dataset_name": dataset_name,
                "n_rows": int(len(group)),
                "mean_tvd_baseline": float(group["tvd_baseline"].mean()),
                "mean_tvd_twin": float(group["tvd_twin"].mean()),
                "se_tvd_baseline": _standard_error(group["tvd_baseline"]),
                "se_tvd_twin": _standard_error(group["tvd_twin"]),
                "weighted_mean_tvd_baseline": float(
                    (group["tvd_baseline"] * group["group_size_baseline"]).sum()
                    / group["group_size_baseline"].sum()
                ),
                "weighted_mean_tvd_twin": float(
                    (group["tvd_twin"] * group["group_size_baseline"]).sum()
                    / group["group_size_baseline"].sum()
                ),
                "mean_normalized_predicted_entropy_baseline": float(
                    group["normalized_predicted_entropy_baseline"].mean()
                ),
                "mean_normalized_predicted_entropy_twin": float(
                    group["normalized_predicted_entropy_twin"].mean()
                ),
                "mean_predicted_entropy_baseline": float(group["predicted_entropy_baseline"].mean()),
                "mean_predicted_entropy_twin": float(group["predicted_entropy_twin"].mean()),
                "mean_jsd_baseline": float(group["jsd_baseline"].mean()),
                "mean_jsd_twin": float(group["jsd_twin"].mean()),
                "modal_match_baseline": float(group["modal_match_baseline"].mean()),
                "modal_match_twin": float(group["modal_match_twin"].mean()),
            }
        )

    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(output_dir / "dataset_overlap_summary.csv", index=False)

    overall = {
        "base_run": baseline_run,
        "twin_run": twin_run,
        "n_overlap_rows": int(len(overlap)),
        "mean_tvd_baseline": float(overlap["tvd_baseline"].mean()),
        "mean_tvd_twin": float(overlap["tvd_twin"].mean()),
        "weighted_mean_tvd_baseline": float(
            (overlap["tvd_baseline"] * overlap["group_size_baseline"]).sum()
            / overlap["group_size_baseline"].sum()
        ),
        "weighted_mean_tvd_twin": float(
            (overlap["tvd_twin"] * overlap["group_size_baseline"]).sum()
            / overlap["group_size_baseline"].sum()
        ),
        "mean_normalized_predicted_entropy_baseline": float(
            overlap["normalized_predicted_entropy_baseline"].mean()
        ),
        "mean_normalized_predicted_entropy_twin": float(
            overlap["normalized_predicted_entropy_twin"].mean()
        ),
        "mean_predicted_entropy_baseline": float(overlap["predicted_entropy_baseline"].mean()),
        "mean_predicted_entropy_twin": float(overlap["predicted_entropy_twin"].mean()),
        "mean_jsd_baseline": float(overlap["jsd_baseline"].mean()),
        "mean_jsd_twin": float(overlap["jsd_twin"].mean()),
        "modal_match_baseline": float(overlap["modal_match_baseline"].mean()),
        "modal_match_twin": float(overlap["modal_match_twin"].mean()),
    }
    _serialize_json(output_dir / "overall_overlap_summary.json", overall)

    return overlap, dataset_df


def _render_grouped_bar_plot(
    *,
    dataset_df: pd.DataFrame,
    baseline_column: str,
    twin_column: str,
    y_label: str,
    output_png: Path,
    output_pdf: Path,
    baseline_error_column: str | None = None,
    twin_error_column: str | None = None,
) -> None:
    plot_df = dataset_df.sort_values("dataset_name").reset_index(drop=True)
    labels = plot_df["dataset_name"].tolist()
    baseline_values = plot_df[baseline_column].tolist()
    twin_values = plot_df[twin_column].tolist()
    baseline_errors = (
        plot_df[baseline_error_column].fillna(0.0).tolist()
        if baseline_error_column and baseline_error_column in plot_df.columns
        else None
    )
    twin_errors = (
        plot_df[twin_error_column].fillna(0.0).tolist()
        if twin_error_column and twin_error_column in plot_df.columns
        else None
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = range(len(labels))
    width = 0.36
    error_kw = {
        "elinewidth": 1.3,
        "capsize": 4,
        "capthick": 1.3,
        "ecolor": "#374151",
    }
    ax.bar(
        [i - width / 2 for i in x],
        baseline_values,
        width=width,
        label="Baseline",
        color="#3b82f6",
        yerr=baseline_errors,
        error_kw=error_kw if baseline_errors is not None else None,
    )
    ax.bar(
        [i + width / 2 for i in x],
        twin_values,
        width=width,
        label="Twin",
        color="#ef4444",
        yerr=twin_errors,
        error_kw=error_kw if twin_errors is not None else None,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    low_candidates = baseline_values + twin_values
    high_candidates = baseline_values + twin_values
    if baseline_errors is not None:
        low_candidates.extend([v - e for v, e in zip(baseline_values, baseline_errors)])
        high_candidates.extend([v + e for v, e in zip(baseline_values, baseline_errors)])
    if twin_errors is not None:
        low_candidates.extend([v - e for v, e in zip(twin_values, twin_errors)])
        high_candidates.extend([v + e for v, e in zip(twin_values, twin_errors)])
    lower = min(low_candidates)
    upper = max(high_candidates)
    span = upper - lower
    pad = 0.08 * span if span > 0 else 0.05
    ax.set_ylim(lower - pad, upper + pad)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build overlap comparison summaries and plots for SimBench runs.")
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--baseline-run",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--twin-run",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.forecasting_root / "results"
    output_dir = args.output_dir or (
        results_root / f"{args.baseline_run}__vs__{args.twin_run}__overlap"
    )

    _, dataset_df = _build_overlap_summary(
        results_root=results_root,
        baseline_run=args.baseline_run,
        twin_run=args.twin_run,
        output_dir=output_dir,
    )

    _render_grouped_bar_plot(
        dataset_df=dataset_df,
        baseline_column="mean_tvd_baseline",
        twin_column="mean_tvd_twin",
        y_label="Mean TVD (lower is better)",
        output_png=output_dir / "mean_tvd_by_task_overlap.png",
        output_pdf=output_dir / "mean_tvd_by_task_overlap.pdf",
        baseline_error_column="se_tvd_baseline",
        twin_error_column="se_tvd_twin",
    )
    _render_grouped_bar_plot(
        dataset_df=dataset_df,
        baseline_column="mean_normalized_predicted_entropy_baseline",
        twin_column="mean_normalized_predicted_entropy_twin",
        y_label="Mean Normalized Entropy (higher = more spread)",
        output_png=output_dir / "mean_normalized_entropy_by_task_overlap.png",
        output_pdf=output_dir / "mean_normalized_entropy_by_task_overlap.pdf",
    )


if __name__ == "__main__":
    main()
