from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "simbenchpop__persona_summary_ablation_compare__gpt_5_nano__us_only"


RUNS = [
    {
        "label": "baseline",
        "kind": "baseline",
        "run_name": "simbenchpop__baseline_group_batched_explained__gpt_5_nano__us_only",
    },
    {
        "label": "full_card",
        "kind": "full_card",
        "run_name": "simbenchpop__twin_persona_summary_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "background_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_background_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "direct_social_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_direct_social_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "self_report_social_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_self_report_social_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "non_social_econ_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_non_social_econ_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "cognitive_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_cognitive_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
    {
        "label": "misc_heuristics_pricing_text_only",
        "kind": "ablation",
        "run_name": "simbenchpop__twin_persona_summary_misc_heuristics_pricing_text_only_batched_seed_0__n64__gpt_5_nano__us_only",
    },
]


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    value = pd.to_numeric(frame[value_col], errors="coerce")
    weight = pd.to_numeric(frame[weight_col], errors="coerce")
    mask = value.notna() & weight.notna() & weight.gt(0)
    if not mask.any():
        return float("nan")
    return float((value[mask] * weight[mask]).sum() / weight[mask].sum())


def _summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    evaluated = frame[frame["evaluated"].fillna(False).astype(bool)].copy()
    return {
        "row_count": int(len(evaluated)),
        "mean_tvd": _safe_mean(evaluated["tvd"]),
        "weighted_mean_tvd_by_group_size": _weighted_mean(evaluated, "tvd", "group_size"),
        "mean_simbench_score": _safe_mean(evaluated["simbench_score"]),
        "weighted_mean_simbench_score_by_group_size": _weighted_mean(
            evaluated, "simbench_score", "group_size"
        ),
        "mean_jsd": _safe_mean(evaluated["jsd"]),
        "modal_match_rate": _safe_mean(evaluated["modal_match"]),
    }


def _load_run_frame(run_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{run_name}__gold_eval" / "row_level_evaluation.csv"
    frame = pd.read_csv(path)
    frame["simbench_row_id"] = frame["simbench_row_id"].astype(str)
    frame["run_name"] = run_name
    return frame


def _comparison_rows(
    run_frames: dict[str, pd.DataFrame], reference_label: str, include_labels: list[str], scope_name: str
) -> pd.DataFrame:
    ref = run_frames[reference_label][["simbench_row_id", "tvd", "simbench_score", "jsd", "modal_match"]].rename(
        columns={
            "tvd": "reference_tvd",
            "simbench_score": "reference_simbench_score",
            "jsd": "reference_jsd",
            "modal_match": "reference_modal_match",
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for label in include_labels:
        if label == reference_label:
            continue
        merged = run_frames[label][
            ["simbench_row_id", "tvd", "simbench_score", "jsd", "modal_match"]
        ].merge(ref, on="simbench_row_id", how="inner")
        rows.append(
            {
                "scope": scope_name,
                "reference": reference_label,
                "label": label,
                "row_count": int(len(merged)),
                "delta_mean_tvd": float((merged["tvd"] - merged["reference_tvd"]).mean()),
                "delta_mean_simbench_score": float(
                    (merged["simbench_score"] - merged["reference_simbench_score"]).mean()
                ),
                "pvalue_paired_t_simbench_score": float(
                    stats.ttest_rel(merged["simbench_score"], merged["reference_simbench_score"]).pvalue
                )
                if len(merged) > 1
                else float("nan"),
                "delta_mean_jsd": float((merged["jsd"] - merged["reference_jsd"]).mean()),
                "delta_modal_match_rate": float((merged["modal_match"] - merged["reference_modal_match"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_frames: dict[str, pd.DataFrame] = {}
    for spec in RUNS:
        frame = _load_run_frame(spec["run_name"])
        frame["label"] = spec["label"]
        frame["kind"] = spec["kind"]
        run_frames[spec["label"]] = frame

    common_row_ids: set[str] | None = None
    for frame in run_frames.values():
        evaluated_ids = set(frame.loc[frame["evaluated"].fillna(False).astype(bool), "simbench_row_id"].astype(str))
        common_row_ids = evaluated_ids if common_row_ids is None else (common_row_ids & evaluated_ids)
    assert common_row_ids is not None

    overall_rows: list[dict[str, float | int | str]] = []
    common_rows: list[dict[str, float | int | str]] = []
    common_no_choices_rows: list[dict[str, float | int | str]] = []
    task_rows: list[dict[str, float | int | str]] = []

    for spec in RUNS:
        label = spec["label"]
        frame = run_frames[label].copy()
        overall_rows.append({"label": label, "kind": spec["kind"], **_summarize(frame)})

        common = frame[frame["simbench_row_id"].isin(common_row_ids)].copy()
        common_rows.append({"label": label, "kind": spec["kind"], **_summarize(common)})

        common_no_choices = common[common["dataset_name"] != "Choices13k"].copy()
        common_no_choices_rows.append({"label": label, "kind": spec["kind"], **_summarize(common_no_choices)})

        for dataset_name, group in common.groupby("dataset_name", dropna=False):
            task_rows.append(
                {
                    "label": label,
                    "kind": spec["kind"],
                    "dataset_name": str(dataset_name),
                    **_summarize(group),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    common_df = pd.DataFrame(common_rows)
    common_no_choices_df = pd.DataFrame(common_no_choices_rows)
    task_df = pd.DataFrame(task_rows)

    overall_df.to_csv(OUTPUT_DIR / "overall_summary_all_available_rows.csv", index=False)
    common_df.to_csv(OUTPUT_DIR / "overall_summary_common_rows.csv", index=False)
    common_no_choices_df.to_csv(OUTPUT_DIR / "overall_summary_common_rows_no_choices13k.csv", index=False)
    task_df.to_csv(OUTPUT_DIR / "dataset_summary_common_rows.csv", index=False)
    task_df.loc[task_df.groupby("dataset_name")["mean_simbench_score"].idxmax()].sort_values(
        ["label", "dataset_name"]
    ).to_csv(OUTPUT_DIR / "best_label_by_dataset_common_rows.csv", index=False)

    include_labels = [spec["label"] for spec in RUNS]
    common_run_frames = {
        label: frame[frame["simbench_row_id"].isin(common_row_ids)].copy() for label, frame in run_frames.items()
    }
    common_no_choices_run_frames = {
        label: frame[frame["dataset_name"] != "Choices13k"].copy() for label, frame in common_run_frames.items()
    }

    pairwise_baseline = pd.concat(
        [
            _comparison_rows(common_run_frames, "baseline", include_labels, "common_rows"),
            _comparison_rows(common_no_choices_run_frames, "baseline", include_labels, "common_rows_no_choices13k"),
        ],
        ignore_index=True,
    )
    pairwise_full = pd.concat(
        [
            _comparison_rows(common_run_frames, "full_card", include_labels, "common_rows"),
            _comparison_rows(common_no_choices_run_frames, "full_card", include_labels, "common_rows_no_choices13k"),
        ],
        ignore_index=True,
    )
    pairwise_baseline.to_csv(OUTPUT_DIR / "pairwise_vs_baseline.csv", index=False)
    pairwise_full.to_csv(OUTPUT_DIR / "pairwise_vs_full_card.csv", index=False)

    best_by_metric = []
    for metric, ascending in [("mean_simbench_score", False), ("mean_tvd", True)]:
        sorted_df = common_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        best = sorted_df.iloc[0]
        best_by_metric.append(
            {
                "scope": "common_rows",
                "metric": metric,
                "best_label": best["label"],
                "best_value": float(best[metric]),
            }
        )
        sorted_no_choices = common_no_choices_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        best_no_choices = sorted_no_choices.iloc[0]
        best_by_metric.append(
            {
                "scope": "common_rows_no_choices13k",
                "metric": metric,
                "best_label": best_no_choices["label"],
                "best_value": float(best_no_choices[metric]),
            }
        )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "common_row_count": int(len(common_row_ids)),
                "best_by_metric": best_by_metric,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
