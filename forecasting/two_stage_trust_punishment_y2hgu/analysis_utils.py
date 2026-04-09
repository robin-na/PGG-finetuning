from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from common import (
    ROLE_A_CHECK,
    ROLE_A_TIME,
    ROLE_B_HIDDEN_CHECK,
    ROLE_B_HIDDEN_TIME,
    ROLE_B_OBSERVABLE_CHECK,
    ROLE_B_OBSERVABLE_TIME,
    total_variation_distance,
    wasserstein_distance_1d,
)


ROLE_A_CHECK_SUPPORT = ["checked_act", "checked_no_act", "unchecked_act", "unchecked_no_act"]
ROLE_A_TIME_SUPPORT = ["fast_act", "fast_no_act", "slow_act", "slow_no_act"]
PRIMARY_METRIC_FAMILY = "distribution_distance"

_METRIC_FAMILY_SORT_ORDER = {
    PRIMARY_METRIC_FAMILY: 0,
    "mean_alignment_abs_error": 1,
}
_DISTANCE_KIND_SORT_ORDER = {
    "mean_wasserstein_1d": 0,
    "wasserstein_1d": 1,
    "total_variation": 2,
    "": 3,
}
_METRIC_SORT_ORDER = {
    "mean_send_field_distance": 0,
    "return_pct": 1,
    "joint_pattern_distribution": 2,
}


def _mean_metrics_for_schema(frame: pd.DataFrame, schema_type: str) -> dict[str, float]:
    if schema_type == ROLE_A_CHECK:
        return {
            "check_rate": float(frame["process_binary"].mean()),
            "act_rate": float(frame["act_binary"].mean()),
            "mean_return_pct": float(frame["return_pct"].mean()),
            "pattern_checked_act": float((frame["pattern_label"] == "checked_act").mean()),
            "pattern_checked_no_act": float((frame["pattern_label"] == "checked_no_act").mean()),
            "pattern_unchecked_act": float((frame["pattern_label"] == "unchecked_act").mean()),
            "pattern_unchecked_no_act": float((frame["pattern_label"] == "unchecked_no_act").mean()),
        }
    if schema_type == ROLE_A_TIME:
        return {
            "fast_rate": float(frame["process_binary"].mean()),
            "act_rate": float(frame["act_binary"].mean()),
            "mean_return_pct": float(frame["return_pct"].mean()),
            "pattern_fast_act": float((frame["pattern_label"] == "fast_act").mean()),
            "pattern_fast_no_act": float((frame["pattern_label"] == "fast_no_act").mean()),
            "pattern_slow_act": float((frame["pattern_label"] == "slow_act").mean()),
            "pattern_slow_no_act": float((frame["pattern_label"] == "slow_no_act").mean()),
        }
    if schema_type == ROLE_B_OBSERVABLE_CHECK:
        return {
            "mean_send_if_act_without_check": float(frame["send_if_act_without_check"].mean()),
            "mean_send_if_act_after_check": float(frame["send_if_act_after_check"].mean()),
            "mean_send_if_no_act_without_check": float(frame["send_if_no_act_without_check"].mean()),
            "mean_send_if_no_act_after_check": float(frame["send_if_no_act_after_check"].mean()),
            "mean_action_premium_without_check": float(frame["action_premium_without_check"].mean()),
            "mean_action_premium_after_check": float(frame["action_premium_after_check"].mean()),
            "mean_process_premium_if_act": float(frame["process_premium_if_act"].mean()),
            "mean_process_premium_if_no_act": float(frame["process_premium_if_no_act"].mean()),
        }
    if schema_type == ROLE_B_HIDDEN_CHECK:
        return {
            "mean_send_if_act": float(frame["send_if_act"].mean()),
            "mean_send_if_no_act": float(frame["send_if_no_act"].mean()),
            "mean_action_premium": float(frame["action_premium"].mean()),
        }
    if schema_type == ROLE_B_OBSERVABLE_TIME:
        return {
            "mean_send_if_act_fast": float(frame["send_if_act_fast"].mean()),
            "mean_send_if_no_act_fast": float(frame["send_if_no_act_fast"].mean()),
            "mean_send_if_act_slow": float(frame["send_if_act_slow"].mean()),
            "mean_send_if_no_act_slow": float(frame["send_if_no_act_slow"].mean()),
            "mean_action_premium_fast": float(frame["action_premium_fast"].mean()),
            "mean_action_premium_slow": float(frame["action_premium_slow"].mean()),
            "mean_fast_premium_if_act": float(frame["process_premium_if_act"].mean()),
            "mean_fast_premium_if_no_act": float(frame["process_premium_if_no_act"].mean()),
        }
    if schema_type == ROLE_B_HIDDEN_TIME:
        return {
            "mean_send_if_act": float(frame["send_if_act"].mean()),
            "mean_send_if_no_act": float(frame["send_if_no_act"].mean()),
            "mean_action_premium": float(frame["action_premium"].mean()),
        }
    raise ValueError(f"Unsupported schema_type: {schema_type}")


def _distribution_metrics_for_schema(
    generated: pd.DataFrame,
    human: pd.DataFrame,
    schema_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if schema_type == ROLE_A_CHECK:
        rows.append(
            {
                "metric": "return_pct",
                "distance_kind": "wasserstein_1d",
                "score": wasserstein_distance_1d(generated["return_pct"], human["return_pct"]),
            }
        )
        rows.append(
            {
                "metric": "joint_pattern_distribution",
                "distance_kind": "total_variation",
                "score": total_variation_distance(
                    generated["pattern_label"],
                    human["pattern_label"],
                    ROLE_A_CHECK_SUPPORT,
                ),
            }
        )
        return rows
    if schema_type == ROLE_A_TIME:
        rows.append(
            {
                "metric": "return_pct",
                "distance_kind": "wasserstein_1d",
                "score": wasserstein_distance_1d(generated["return_pct"], human["return_pct"]),
            }
        )
        rows.append(
            {
                "metric": "joint_pattern_distribution",
                "distance_kind": "total_variation",
                "score": total_variation_distance(
                    generated["pattern_label"],
                    human["pattern_label"],
                    ROLE_A_TIME_SUPPORT,
                ),
            }
        )
        return rows

    if schema_type == ROLE_B_OBSERVABLE_CHECK:
        fields = [
            "send_if_act_without_check",
            "send_if_act_after_check",
            "send_if_no_act_without_check",
            "send_if_no_act_after_check",
        ]
    elif schema_type == ROLE_B_OBSERVABLE_TIME:
        fields = [
            "send_if_act_fast",
            "send_if_no_act_fast",
            "send_if_act_slow",
            "send_if_no_act_slow",
        ]
    elif schema_type in {ROLE_B_HIDDEN_CHECK, ROLE_B_HIDDEN_TIME}:
        fields = ["send_if_act", "send_if_no_act"]
    else:
        raise ValueError(f"Unsupported schema_type: {schema_type}")

    field_scores: list[float] = []
    for field_name in fields:
        score = wasserstein_distance_1d(generated[field_name], human[field_name])
        field_scores.append(score)
        rows.append(
            {
                "metric": field_name,
                "distance_kind": "wasserstein_1d",
                "score": score,
            }
        )
    rows.append(
        {
            "metric": "mean_send_field_distance",
            "distance_kind": "mean_wasserstein_1d",
            "score": float(np.mean(field_scores)),
        }
    )
    return rows


def compute_treatment_metric_tables(
    *,
    generated_records: pd.DataFrame,
    human_records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_treatments = sorted(
        set(generated_records["treatment_name"].dropna().astype(str))
        & set(human_records["treatment_name"].dropna().astype(str))
    )

    mean_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []

    for treatment_name in common_treatments:
        generated_group = generated_records[generated_records["treatment_name"] == treatment_name].copy()
        human_group = human_records[human_records["treatment_name"] == treatment_name].copy()
        if generated_group.empty or human_group.empty:
            continue
        schema_type = str(human_group["schema_type"].iloc[0])
        role = str(human_group["role"].iloc[0])
        visibility = str(human_group["visibility"].iloc[0])
        experiment_code = str(human_group["experiment_code"].iloc[0])

        generated_means = _mean_metrics_for_schema(generated_group, schema_type)
        human_means = _mean_metrics_for_schema(human_group, schema_type)
        for metric, human_value in human_means.items():
            generated_value = generated_means[metric]
            mean_rows.append(
                {
                    "treatment_name": treatment_name,
                    "experiment_code": experiment_code,
                    "role": role,
                    "visibility": visibility,
                    "schema_type": schema_type,
                    "metric": metric,
                    "generated_value": generated_value,
                    "human_value": human_value,
                    "abs_error": abs(float(generated_value) - float(human_value)),
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                }
            )

        for row in _distribution_metrics_for_schema(generated_group, human_group, schema_type):
            dist_rows.append(
                {
                    "treatment_name": treatment_name,
                    "experiment_code": experiment_code,
                    "role": role,
                    "visibility": visibility,
                    "schema_type": schema_type,
                    "generated_n": int(len(generated_group)),
                    "human_n": int(len(human_group)),
                    **row,
                }
            )

    mean_df = pd.DataFrame(mean_rows)
    dist_df = pd.DataFrame(dist_rows)

    overall_rows: list[dict[str, Any]] = []
    if not mean_df.empty:
        summary = (
            mean_df.groupby("metric", as_index=False)
            .agg(
                n_treatments=("treatment_name", "nunique"),
                mean_value=("abs_error", "mean"),
                median_value=("abs_error", "median"),
            )
        )
        for row in summary.to_dict(orient="records"):
            overall_rows.append(
                {
                    "metric_family": "mean_alignment_abs_error",
                    "metric": row["metric"],
                    "distance_kind": "",
                    "n_treatments": int(row["n_treatments"]),
                    "mean_value": float(row["mean_value"]),
                    "median_value": float(row["median_value"]),
                }
            )
    if not dist_df.empty:
        summary = (
            dist_df.groupby(["metric", "distance_kind"], as_index=False)
            .agg(
                n_treatments=("treatment_name", "nunique"),
                mean_value=("score", "mean"),
                median_value=("score", "median"),
            )
        )
        for row in summary.to_dict(orient="records"):
            overall_rows.append(
                {
                    "metric_family": "distribution_distance",
                    "metric": row["metric"],
                    "distance_kind": row["distance_kind"],
                    "n_treatments": int(row["n_treatments"]),
                    "mean_value": float(row["mean_value"]),
                    "median_value": float(row["median_value"]),
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
        overall_metric_summary["metric_family"] == PRIMARY_METRIC_FAMILY
    ].copy()
    return sort_overall_metric_summary(primary)
