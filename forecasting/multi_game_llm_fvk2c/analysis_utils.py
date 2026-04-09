from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import (
    CHOICE_SUPPORT,
    DELEGATION_FIELDS,
    NUMERIC_SCENARIO_FIELDS,
    SCENARIO_FIELDS,
    STATE_SUPPORT,
    total_variation_distance,
    wasserstein_distance_1d,
)


PRIMARY_METRIC_FAMILIES = {
    "delegation_distribution_distance",
    "scenario_state_distribution_distance",
    "scenario_direct_value_distance",
}

_METRIC_FAMILY_SORT_ORDER = {
    "delegation_distribution_distance": 0,
    "scenario_state_distribution_distance": 1,
    "scenario_direct_value_distance": 2,
    "mean_alignment_abs_error": 3,
}

_DISTANCE_KIND_SORT_ORDER = {
    "mean_wasserstein_1d": 0,
    "wasserstein_1d": 1,
    "total_variation": 2,
    "": 3,
}

_METRIC_SORT_ORDER = {
    "mean_delegation_wd": 0,
    "mean_role_state_tv": 1,
    "mean_numeric_direct_value_wd": 2,
    "UGProposer_decision_direct": 3,
    "UGResponder_decision_direct": 4,
    "TGReceiver_decision_direct": 5,
}


def _mean_stderr(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] <= 1:
        return mean, float("nan")
    return mean, float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))


def _build_delegation_mean_rows(
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_treatments = sorted(
        set(generated_sessions["treatment_name"].dropna().astype(str))
        & set(human_sessions["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_sessions[generated_sessions["treatment_name"] == treatment_name].copy()
        human_group = human_sessions[human_sessions["treatment_name"] == treatment_name].copy()
        for field_name in DELEGATION_FIELDS:
            generated_values = generated_group[field_name].dropna().astype(float)
            human_values = human_group[field_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            generated_value = float(generated_values.mean())
            human_value = float(human_values.mean())
            rows.append(
                {
                    "metric_scope": "session",
                    "treatment_name": treatment_name,
                    "cell_name": None,
                    "scenario": None,
                    "case": None,
                    "metric": field_name,
                    "generated_value": generated_value,
                    "human_value": human_value,
                    "abs_error": abs(generated_value - human_value),
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )
    return rows


def _build_scenario_mean_rows(
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_cells = sorted(
        set(generated_scenarios["cell_name"].dropna().astype(str))
        & set(human_scenarios["cell_name"].dropna().astype(str))
    )
    for cell_name in common_cells:
        generated_group = generated_scenarios[generated_scenarios["cell_name"] == cell_name].copy()
        human_group = human_scenarios[human_scenarios["cell_name"] == cell_name].copy()
        treatment_name = str(human_group["treatment_name"].iloc[0])
        scenario = str(human_group["scenario"].iloc[0])
        case = str(human_group["case"].iloc[0])

        for field_name in SCENARIO_FIELDS:
            nonnull_col = f"{field_name}_nonnull"
            generated_value = float(generated_group[nonnull_col].mean())
            human_value = float(human_group[nonnull_col].mean())
            rows.append(
                {
                    "metric_scope": "scenario",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": f"{field_name}_nonnull_rate",
                    "generated_value": generated_value,
                    "human_value": human_value,
                    "abs_error": abs(generated_value - human_value),
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )

        for field_name in NUMERIC_SCENARIO_FIELDS:
            generated_values = generated_group[field_name].dropna().astype(float)
            human_values = human_group[field_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            generated_value = float(generated_values.mean())
            human_value = float(human_values.mean())
            rows.append(
                {
                    "metric_scope": "scenario",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": f"{field_name}_direct_mean",
                    "generated_value": generated_value,
                    "human_value": human_value,
                    "abs_error": abs(generated_value - human_value),
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )
    return rows


def _build_delegation_distance_rows(
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
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
                    "treatment_name": treatment_name,
                    "cell_name": None,
                    "scenario": None,
                    "case": None,
                    "metric": field_name,
                    "distance_kind": "wasserstein_1d",
                    "score": score,
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )
        if field_scores:
            rows.append(
                {
                    "metric_family": "delegation_distribution_distance",
                    "treatment_name": treatment_name,
                    "cell_name": None,
                    "scenario": None,
                    "case": None,
                    "metric": "mean_delegation_wd",
                    "distance_kind": "mean_wasserstein_1d",
                    "score": float(np.mean(field_scores)),
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )
    return rows


def _build_scenario_distance_rows(
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_cells = sorted(
        set(generated_scenarios["cell_name"].dropna().astype(str))
        & set(human_scenarios["cell_name"].dropna().astype(str))
    )
    for cell_name in common_cells:
        generated_group = generated_scenarios[generated_scenarios["cell_name"] == cell_name].copy()
        human_group = human_scenarios[human_scenarios["cell_name"] == cell_name].copy()
        treatment_name = str(human_group["treatment_name"].iloc[0])
        scenario = str(human_group["scenario"].iloc[0])
        case = str(human_group["case"].iloc[0])

        state_scores: list[float] = []
        numeric_direct_scores: list[float] = []
        for field_name in SCENARIO_FIELDS:
            state_score = total_variation_distance(
                generated_group[f"{field_name}_state"],
                human_group[f"{field_name}_state"],
                STATE_SUPPORT[field_name],
            )
            state_scores.append(state_score)
            rows.append(
                {
                    "metric_family": "scenario_state_distribution_distance",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": f"{field_name}_state",
                    "distance_kind": "total_variation",
                    "score": state_score,
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )

        if state_scores:
            rows.append(
                {
                    "metric_family": "scenario_state_distribution_distance",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": "mean_role_state_tv",
                    "distance_kind": "total_variation",
                    "score": float(np.mean(state_scores)),
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )

        for field_name in NUMERIC_SCENARIO_FIELDS:
            generated_values = generated_group[field_name].dropna().astype(float)
            human_values = human_group[field_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            score = wasserstein_distance_1d(generated_values, human_values)
            numeric_direct_scores.append(score)
            rows.append(
                {
                    "metric_family": "scenario_direct_value_distance",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": f"{field_name}_direct",
                    "distance_kind": "wasserstein_1d",
                    "score": score,
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )

        if numeric_direct_scores:
            rows.append(
                {
                    "metric_family": "scenario_direct_value_distance",
                    "treatment_name": treatment_name,
                    "cell_name": cell_name,
                    "scenario": scenario,
                    "case": case,
                    "metric": "mean_numeric_direct_value_wd",
                    "distance_kind": "mean_wasserstein_1d",
                    "score": float(np.mean(numeric_direct_scores)),
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )
    return rows


def compute_metric_tables(
    *,
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_rows = [
        *_build_delegation_mean_rows(generated_sessions, human_sessions),
        *_build_scenario_mean_rows(generated_scenarios, human_scenarios),
    ]
    dist_rows = [
        *_build_delegation_distance_rows(generated_sessions, human_sessions),
        *_build_scenario_distance_rows(generated_scenarios, human_scenarios),
    ]

    mean_df = pd.DataFrame(mean_rows)
    dist_df = pd.DataFrame(dist_rows)

    overall_rows: list[dict[str, Any]] = []
    if not mean_df.empty:
        summary = (
            mean_df.groupby(["metric_scope", "metric"], as_index=False)
            .agg(
                n_groups=("treatment_name", "nunique"),
                mean_value=("abs_error", "mean"),
                median_value=("abs_error", "median"),
                stderr=("abs_error", lambda s: _mean_stderr(s)[1]),
            )
        )
        for row in summary.to_dict(orient="records"):
            overall_rows.append(
                {
                    "metric_family": "mean_alignment_abs_error",
                    "metric_scope": row["metric_scope"],
                    "metric": row["metric"],
                    "distance_kind": "",
                    "n_groups": int(row["n_groups"]),
                    "mean_value": float(row["mean_value"]),
                    "median_value": float(row["median_value"]),
                    "stderr": float(row["stderr"]) if pd.notna(row["stderr"]) else float("nan"),
                }
            )

    if not dist_df.empty:
        summary = (
            dist_df.groupby(["metric_family", "metric", "distance_kind"], as_index=False)
            .agg(
                n_groups=("treatment_name", "nunique"),
                mean_value=("score", "mean"),
                median_value=("score", "median"),
                stderr=("score", lambda s: _mean_stderr(s)[1]),
            )
        )
        for row in summary.to_dict(orient="records"):
            overall_rows.append(
                {
                    "metric_family": row["metric_family"],
                    "metric_scope": "distribution",
                    "metric": row["metric"],
                    "distance_kind": row["distance_kind"],
                    "n_groups": int(row["n_groups"]),
                    "mean_value": float(row["mean_value"]),
                    "median_value": float(row["median_value"]),
                    "stderr": float(row["stderr"]) if pd.notna(row["stderr"]) else float("nan"),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    return mean_df, dist_df, overall_df


def sort_overall_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.copy()
    sorted_frame["_metric_family_order"] = (
        sorted_frame["metric_family"].map(_METRIC_FAMILY_SORT_ORDER).fillna(99).astype(int)
    )
    sorted_frame["_distance_kind_order"] = (
        sorted_frame["distance_kind"].fillna("").map(_DISTANCE_KIND_SORT_ORDER).fillna(99).astype(int)
    )
    sorted_frame["_metric_order"] = (
        sorted_frame["metric"].map(_METRIC_SORT_ORDER).fillna(99).astype(int)
    )
    sorted_frame = sorted_frame.sort_values(
        ["_metric_family_order", "_distance_kind_order", "_metric_order", "metric"],
        kind="stable",
    )
    return sorted_frame.drop(
        columns=["_metric_family_order", "_distance_kind_order", "_metric_order"]
    ).reset_index(drop=True)


def build_primary_distribution_summary(overall_metric_summary: pd.DataFrame) -> pd.DataFrame:
    if overall_metric_summary.empty:
        return overall_metric_summary.copy()
    primary = overall_metric_summary[
        overall_metric_summary["metric_family"].isin(PRIMARY_METRIC_FAMILIES)
    ].copy()
    return sort_overall_metric_summary(primary)
