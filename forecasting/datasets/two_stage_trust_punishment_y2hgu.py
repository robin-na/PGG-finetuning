from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    DatasetBundle,
    age_to_bracket,
    clean_numeric_string,
    first_present_value,
    simple_demographic_summary,
    yes_no,
)


def _two_stage_gender_to_display(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    mapping = {1: "Male", 2: "Female", 3: "Other", 4: "Other"}
    return mapping.get(code)


def _two_stage_gender_to_match(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    mapping = {1: "male", 2: "female"}
    return mapping.get(code)


def _make_record(
    *,
    record_id: str,
    unit_id: str,
    treatment_name: str,
    experiment_code: str,
    role: str,
    visibility: str,
    schema_type: str,
    action_context: str,
    deliberation_dimension: str,
    gold_target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "unit_id": unit_id,
        "treatment_name": treatment_name,
        "experiment_code": experiment_code,
        "role": role,
        "visibility": visibility,
        "schema_type": schema_type,
        "action_context": action_context,
        "deliberation_dimension": deliberation_dimension,
        "gold_target_json": json.dumps(gold_target),
    }


def build_bundle(repo_root: Path) -> DatasetBundle:
    data_dir = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "two_stage_trust_punishment_y2hgu"
        / "Data Files"
    )
    config_map = {
        "helpcostcheckE1.csv": ("E1", "help", "cost"),
        "puncostcheckE2.csv": ("E2", "punish", "cost"),
        "helpimpactcheckE4.csv": ("E4", "help", "impact"),
        "punimpactcheckE5.csv": ("E5", "punish", "impact"),
    }

    record_rows: list[dict[str, Any]] = []
    demo_rows: list[dict[str, Any]] = []
    demo_counter = 0

    for filename, (experiment_code, action_context, deliberation_dimension) in config_map.items():
        df = pd.read_csv(data_dir / filename, sep=";")
        for _, row in df.iterrows():
            demo_counter += 1
            age = clean_numeric_string(row.get("age"))
            sex = _two_stage_gender_to_display(row.get("gender"))
            summary, markdown = simple_demographic_summary(age=age, sex_or_gender=sex)
            demo_rows.append(
                {
                    "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                    "summary": summary,
                    "markdown": markdown,
                    "matching_age_bracket": age_to_bracket(age),
                    "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                    "matching_education": None,
                }
            )

            condition = int(row["Condition"])
            pid = str(row["PID"])
            if condition in {3, 4}:
                visibility = "observable" if condition == 3 else "hidden"
                check = int(row["checkObs"] if condition == 3 else row["checkHid"])
                action_col_checked = "calcHelp" if action_context == "help" else "calcPun"
                action_col_unchecked = "uncalcHelp" if action_context == "help" else "uncalcPun"
                acted = int(row[action_col_checked] if check == 1 else row[action_col_unchecked])
                target = {"check": yes_no(check), "act": yes_no(acted), "return_pct": int(row["return"])}
                record_rows.append(
                    _make_record(
                        record_id=f"{experiment_code}__A__{pid}",
                        unit_id=f"{experiment_code}__A__{pid}",
                        treatment_name=f"{experiment_code}_ROLE_A_{visibility.upper()}",
                        experiment_code=experiment_code,
                        role="A",
                        visibility=visibility,
                        schema_type="role_a_check",
                        action_context=action_context,
                        deliberation_dimension=deliberation_dimension,
                        gold_target=target,
                    )
                )
            elif condition in {5, 6}:
                visibility = "observable" if condition == 5 else "hidden"
                if visibility == "observable":
                    if action_context == "help":
                        target = {
                            "send_if_act_without_check": int(row["helpUncalc"]),
                            "send_if_act_after_check": int(row["helpCalc"]),
                            "send_if_no_act_without_check": int(row["noUncalc"]),
                            "send_if_no_act_after_check": int(row["noCalc"]),
                        }
                    else:
                        target = {
                            "send_if_act_without_check": int(row["punUncalc"]),
                            "send_if_act_after_check": int(row["punCalc"]),
                            "send_if_no_act_without_check": int(row["noUncalc"]),
                            "send_if_no_act_after_check": int(row["noCalc"]),
                        }
                    schema_type = "role_b_observable_check"
                else:
                    if action_context == "help":
                        target = {
                            "send_if_act": int(first_present_value(row, "helpedHidB", "helpHid")),
                            "send_if_no_act": int(first_present_value(row, "noHelpdHidB", "noHelpHid")),
                        }
                    else:
                        target = {
                            "send_if_act": int(row["punHid"]),
                            "send_if_no_act": int(row["noPunHid"]),
                        }
                    schema_type = "role_b_hidden_check"
                record_rows.append(
                    _make_record(
                        record_id=f"{experiment_code}__B__{pid}",
                        unit_id=f"{experiment_code}__B__{pid}",
                        treatment_name=f"{experiment_code}_ROLE_B_{visibility.upper()}",
                        experiment_code=experiment_code,
                        role="B",
                        visibility=visibility,
                        schema_type=schema_type,
                        action_context=action_context,
                        deliberation_dimension=deliberation_dimension,
                        gold_target=target,
                    )
                )

    e3a = pd.read_csv(data_dir / "puntimeE3a.csv", sep=";")
    median_decision_time = float(e3a["decisionT"].median())
    for _, row in e3a.iterrows():
        demo_counter += 1
        age = clean_numeric_string(row.get("age"))
        sex = _two_stage_gender_to_display(row.get("gender"))
        summary, markdown = simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": age_to_bracket(age),
                "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                "matching_education": None,
            }
        )
        visibility = "observable" if int(row["Condition"]) == 1 else "hidden"
        target = {
            "decision_time_bucket": "FAST" if float(row["decisionT"]) <= median_decision_time else "SLOW",
            "act": yes_no(row["punishing"]),
            "return_pct": int(row["return"]),
        }
        pid = str(row["ID"])
        record_rows.append(
            _make_record(
                record_id=f"E3A__A__{pid}",
                unit_id=f"E3A__A__{pid}",
                treatment_name=f"E3A_ROLE_A_{visibility.upper()}",
                experiment_code="E3A",
                role="A",
                visibility=visibility,
                schema_type="role_a_time",
                action_context="punish",
                deliberation_dimension="decision_time",
                gold_target=target,
            )
        )

    e3b = pd.read_csv(data_dir / "puntimeE3b.csv", sep=";")
    for _, row in e3b.iterrows():
        demo_counter += 1
        age = clean_numeric_string(row.get("age"))
        sex = _two_stage_gender_to_display(row.get("gender"))
        summary, markdown = simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": age_to_bracket(age),
                "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                "matching_education": None,
            }
        )
        visibility = "observable" if int(row["Condition"]) == 1 else "hidden"
        if visibility == "observable":
            target = {
                "send_if_act_fast": int(row["punFast"]),
                "send_if_no_act_fast": int(row["noFast"]),
                "send_if_act_slow": int(row["punSlow"]),
                "send_if_no_act_slow": int(row["noSlow"]),
            }
            schema_type = "role_b_observable_time"
        else:
            target = {
                "send_if_act": int(row["punHid"]),
                "send_if_no_act": int(row["noPunHid"]),
            }
            schema_type = "role_b_hidden_time"
        pid = str(row["ID"])
        record_rows.append(
            _make_record(
                record_id=f"E3B__B__{pid}",
                unit_id=f"E3B__B__{pid}",
                treatment_name=f"E3B_ROLE_B_{visibility.upper()}",
                experiment_code="E3B",
                role="B",
                visibility=visibility,
                schema_type=schema_type,
                action_context="punish",
                deliberation_dimension="decision_time",
                gold_target=target,
            )
        )

    records = pd.DataFrame(record_rows).sort_values(["treatment_name", "record_id"]).reset_index(drop=True)
    units_df = records[["unit_id"]].drop_duplicates("unit_id").sort_values("unit_id").reset_index(drop=True)
    demographic_source = pd.DataFrame(demo_rows)
    return DatasetBundle(
        dataset_key="two_stage_trust_punishment_y2hgu",
        display_name="Two-Stage Trust / Punishment / Helping",
        records=records,
        units=units_df,
        demographic_source=demographic_source,
        twin_matching_fields=["matching_age_bracket", "matching_sex"],
    )

