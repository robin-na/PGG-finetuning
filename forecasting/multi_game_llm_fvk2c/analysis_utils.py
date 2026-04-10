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
    "behavior_distribution_distance",
    "paper_index_distribution_distance",
    "delegation_distribution_distance",
    "scenario_state_distribution_distance",
    "scenario_direct_value_distance",
}

_METRIC_FAMILY_SORT_ORDER = {
    "behavior_distribution_distance": 0,
    "paper_index_distribution_distance": 1,
    "delegation_distribution_distance": 2,
    "scenario_state_distribution_distance": 3,
    "scenario_direct_value_distance": 4,
    "mean_alignment_abs_error": 5,
}

_DISTANCE_KIND_SORT_ORDER = {
    "mean_wasserstein_1d": 0,
    "wasserstein_1d": 1,
    "total_variation": 2,
    "": 3,
}

_METRIC_SORT_ORDER = {
    "fairness_wd": 0,
    "trust_wd": 1,
    "trustworthiness_wd": 2,
    "cooperation_wd": 3,
    "coordination_wd": 4,
    "delegation_index_wd": 5,
    "fairness_offer_wd": 6,
    "inequality_tolerance_wd": 7,
    "trust_send_wd": 8,
    "trustworthiness_return_wd": 9,
    "cooperation_pd_wd": 10,
    "cooperation_sh_wd": 11,
    "coordination_earth_wd": 12,
    "payoff_index_wd": 13,
    "prosociality_index_wd": 14,
    "kindness_index_wd": 15,
    "intentions_index_wd": 16,
    "predictability_index_wd": 17,
    "equality_index_wd": 18,
    "mean_delegation_wd": 19,
    "mean_role_state_tv": 20,
    "mean_numeric_direct_value_wd": 21,
    "UGProposer_decision_direct": 22,
    "UGResponder_decision_direct": 23,
    "TGReceiver_decision_direct": 24,
}

_PAPER_SCENARIO_INDEX_METRICS = {
    "payoff_index_wd": "WI",
    "prosociality_index_wd": "PSI",
    "kindness_index_wd": "KI",
    "intentions_index_wd": "II",
    "predictability_index_wd": "PM",
    "equality_index_wd": "FM",
}

_BEHAVIOR_METRIC_COMPONENTS = {
    "fairness_wd": [
        ("fairness_offer_wd", "UG_S"),
        ("inequality_tolerance_wd", "UG_R"),
    ],
    "trust_wd": [
        ("trust_send_wd", "TG_S"),
    ],
    "trustworthiness_wd": [
        ("trustworthiness_return_wd", "TG_T"),
    ],
    "cooperation_wd": [
        ("cooperation_pd_wd", "PD_bin"),
        ("cooperation_sh_wd", "SH_bin"),
    ],
    "coordination_wd": [
        ("coordination_earth_wd", "Earth"),
    ],
}


def _role_from_row(row: pd.Series) -> str:
    if str(row["scenario"]) == "AISupport":
        return "support"
    if str(row["case"]) == "AgainstAI":
        return "AI"
    if str(row["case"]) == "AgainstHuman":
        return "human"
    return "unknown"


def _binary_choice(series: pd.Series, *, positive: str) -> pd.Series:
    return series.map(
        lambda value: np.nan if pd.isna(value) else 1.0 if str(value) == positive else 0.0
    )


def _augment_scenario_indices(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["role"] = data.apply(_role_from_row, axis=1)

    data["UG_S"] = pd.to_numeric(data["UGProposer_decision"], errors="coerce") / 5.0
    data["UG_R"] = pd.to_numeric(data["UGResponder_decision"], errors="coerce") / 5.0
    data["rUG_R"] = 1.0 - data["UG_R"]
    data["TG_S"] = _binary_choice(data["TGSender_decision"], positive="YES")
    data["TG_T"] = pd.to_numeric(data["TGReceiver_decision"], errors="coerce") / 6.0
    data["PD_bin"] = _binary_choice(data["PD_decision"], positive="A")
    data["SH_bin"] = _binary_choice(data["SH_decision"], positive="X")
    data["Earth"] = data["C_decision"].map(
        lambda value: np.nan if pd.isna(value) else 1.0 if str(value) == "Earth" else 0.0
    )

    data["WI"] = data[["UG_S", "UG_R", "TG_S", "TG_T", "PD_bin", "SH_bin", "Earth"]].mean(
        axis=1,
        skipna=True,
    )
    data["PSI"] = data[["UG_S", "rUG_R", "TG_S", "TG_T", "PD_bin", "SH_bin"]].mean(
        axis=1,
        skipna=True,
    )
    data["KI"] = data[["TG_S", "PD_bin", "SH_bin"]].mean(axis=1, skipna=True)
    data["II"] = data[["UG_R", "TG_T"]].mean(axis=1, skipna=True)
    data["PM"] = data["Earth"]
    data["FM"] = data["UG_S"]
    data["paper_role_cell_name"] = data["treatment_name"].astype(str) + "__" + data["role"].astype(str)
    return data


def _augment_session_indices(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    delegation_cols = [
        "UGProposer_delegated",
        "UGResponder_delegated",
        "TGSender_delegated",
        "TGReceiver_delegated",
        "PD_delegated",
        "SH_delegated",
        "C_delegated",
    ]
    for field_name in delegation_cols:
        data[field_name] = pd.to_numeric(data[field_name], errors="coerce")
    data["DI"] = data[delegation_cols].mean(axis=1, skipna=True)
    return data


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


def _build_paper_index_mean_rows(
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    generated_sessions_aug = _augment_session_indices(generated_sessions)
    human_sessions_aug = _augment_session_indices(human_sessions)
    common_treatments = sorted(
        set(generated_sessions_aug["treatment_name"].dropna().astype(str))
        & set(human_sessions_aug["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_sessions_aug[generated_sessions_aug["treatment_name"] == treatment_name]
        human_group = human_sessions_aug[human_sessions_aug["treatment_name"] == treatment_name]
        generated_values = generated_group["DI"].dropna().astype(float)
        human_values = human_group["DI"].dropna().astype(float)
        if generated_values.empty or human_values.empty:
            continue
        rows.append(
            {
                "metric_scope": "paper_index_session",
                "treatment_name": treatment_name,
                "cell_name": None,
                "scenario": None,
                "case": None,
                "metric": "delegation_index_mean",
                "generated_value": float(generated_values.mean()),
                "human_value": float(human_values.mean()),
                "abs_error": abs(float(generated_values.mean()) - float(human_values.mean())),
                "generated_n": int(len(generated_values)),
                "human_n": int(len(human_values)),
            }
        )

    generated_scenarios_aug = _augment_scenario_indices(generated_scenarios)
    human_scenarios_aug = _augment_scenario_indices(human_scenarios)
    common_role_cells = sorted(
        set(generated_scenarios_aug["paper_role_cell_name"].dropna().astype(str))
        & set(human_scenarios_aug["paper_role_cell_name"].dropna().astype(str))
    )
    for role_cell in common_role_cells:
        generated_group = generated_scenarios_aug[
            generated_scenarios_aug["paper_role_cell_name"] == role_cell
        ].copy()
        human_group = human_scenarios_aug[
            human_scenarios_aug["paper_role_cell_name"] == role_cell
        ].copy()
        treatment_name = str(human_group["treatment_name"].iloc[0])
        role = str(human_group["role"].iloc[0])
        for metric_name, column_name in _PAPER_SCENARIO_INDEX_METRICS.items():
            generated_values = generated_group[column_name].dropna().astype(float)
            human_values = human_group[column_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            mean_label = metric_name.replace("_wd", "_mean")
            rows.append(
                {
                    "metric_scope": "paper_index_scenario",
                    "treatment_name": treatment_name,
                    "cell_name": role_cell,
                    "scenario": role,
                    "case": None,
                    "metric": mean_label,
                    "generated_value": float(generated_values.mean()),
                    "human_value": float(human_values.mean()),
                    "abs_error": abs(float(generated_values.mean()) - float(human_values.mean())),
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )
    return rows


def _build_paper_index_distance_rows(
    generated_sessions: pd.DataFrame,
    human_sessions: pd.DataFrame,
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    generated_sessions_aug = _augment_session_indices(generated_sessions)
    human_sessions_aug = _augment_session_indices(human_sessions)
    common_treatments = sorted(
        set(generated_sessions_aug["treatment_name"].dropna().astype(str))
        & set(human_sessions_aug["treatment_name"].dropna().astype(str))
    )
    for treatment_name in common_treatments:
        generated_group = generated_sessions_aug[generated_sessions_aug["treatment_name"] == treatment_name]
        human_group = human_sessions_aug[human_sessions_aug["treatment_name"] == treatment_name]
        generated_values = generated_group["DI"].dropna().astype(float)
        human_values = human_group["DI"].dropna().astype(float)
        if generated_values.empty or human_values.empty:
            continue
        rows.append(
            {
                "metric_family": "paper_index_distribution_distance",
                "treatment_name": treatment_name,
                "cell_name": None,
                "scenario": None,
                "case": None,
                "metric": "delegation_index_wd",
                "distance_kind": "wasserstein_1d",
                "score": wasserstein_distance_1d(generated_values, human_values),
                "generated_n": int(len(generated_values)),
                "human_n": int(len(human_values)),
            }
        )

    generated_scenarios_aug = _augment_scenario_indices(generated_scenarios)
    human_scenarios_aug = _augment_scenario_indices(human_scenarios)
    common_role_cells = sorted(
        set(generated_scenarios_aug["paper_role_cell_name"].dropna().astype(str))
        & set(human_scenarios_aug["paper_role_cell_name"].dropna().astype(str))
    )
    for role_cell in common_role_cells:
        generated_group = generated_scenarios_aug[
            generated_scenarios_aug["paper_role_cell_name"] == role_cell
        ].copy()
        human_group = human_scenarios_aug[
            human_scenarios_aug["paper_role_cell_name"] == role_cell
        ].copy()
        treatment_name = str(human_group["treatment_name"].iloc[0])
        role = str(human_group["role"].iloc[0])
        for metric_name, column_name in _PAPER_SCENARIO_INDEX_METRICS.items():
            generated_values = generated_group[column_name].dropna().astype(float)
            human_values = human_group[column_name].dropna().astype(float)
            if generated_values.empty or human_values.empty:
                continue
            rows.append(
                {
                    "metric_family": "paper_index_distribution_distance",
                    "treatment_name": treatment_name,
                    "cell_name": role_cell,
                    "scenario": role,
                    "case": None,
                    "metric": metric_name,
                    "distance_kind": "wasserstein_1d",
                    "score": wasserstein_distance_1d(generated_values, human_values),
                    "generated_n": int(len(generated_values)),
                    "human_n": int(len(human_values)),
                }
            )
    return rows


def _build_behavior_distance_rows(
    generated_scenarios: pd.DataFrame,
    human_scenarios: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generated_scenarios_aug = _augment_scenario_indices(generated_scenarios)
    human_scenarios_aug = _augment_scenario_indices(human_scenarios)

    generated_no_support = generated_scenarios_aug[
        generated_scenarios_aug["scenario"].astype(str) == "NoAISupport"
    ].copy()
    human_no_support = human_scenarios_aug[
        human_scenarios_aug["scenario"].astype(str) == "NoAISupport"
    ].copy()

    common_cells = sorted(
        set(generated_no_support["cell_name"].dropna().astype(str))
        & set(human_no_support["cell_name"].dropna().astype(str))
    )
    for cell_name in common_cells:
        generated_group = generated_no_support[generated_no_support["cell_name"] == cell_name].copy()
        human_group = human_no_support[human_no_support["cell_name"] == cell_name].copy()
        treatment_name = str(human_group["treatment_name"].iloc[0])
        scenario = str(human_group["scenario"].iloc[0])
        case = str(human_group["case"].iloc[0])

        for metric_name, components in _BEHAVIOR_METRIC_COMPONENTS.items():
            component_scores: list[float] = []
            for component_metric, column_name in components:
                generated_values = generated_group[column_name].dropna().astype(float)
                human_values = human_group[column_name].dropna().astype(float)
                if generated_values.empty or human_values.empty:
                    continue
                score = wasserstein_distance_1d(generated_values, human_values)
                component_scores.append(score)
                rows.append(
                    {
                        "metric_family": "behavior_distribution_distance",
                        "treatment_name": treatment_name,
                        "cell_name": cell_name,
                        "scenario": scenario,
                        "case": case,
                        "metric": component_metric,
                        "distance_kind": "wasserstein_1d",
                        "score": score,
                        "generated_n": int(len(generated_values)),
                        "human_n": int(len(human_values)),
                    }
                )
            if component_scores:
                rows.append(
                    {
                        "metric_family": "behavior_distribution_distance",
                        "treatment_name": treatment_name,
                        "cell_name": cell_name,
                        "scenario": scenario,
                        "case": case,
                        "metric": metric_name,
                        "distance_kind": "mean_wasserstein_1d",
                        "score": float(np.mean(component_scores)),
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
        *_build_paper_index_mean_rows(
            generated_sessions,
            human_sessions,
            generated_scenarios,
            human_scenarios,
        ),
    ]
    dist_rows = [
        *_build_behavior_distance_rows(generated_scenarios, human_scenarios),
        *_build_paper_index_distance_rows(
            generated_sessions,
            human_sessions,
            generated_scenarios,
            human_scenarios,
        ),
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
