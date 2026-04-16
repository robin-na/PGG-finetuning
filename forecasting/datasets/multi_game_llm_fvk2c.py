from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import DatasetBundle, age_to_bracket, clean_numeric_string, clean_text, simple_demographic_summary


def _multi_game_education_to_harmonized(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "Primary school": "high school",
        "Secondary school up to age of 16": "high school",
        "Higher or secondary or further education (A-levels, BTEC, etc.)": "high school",
        "College or university": "college/postsecondary",
        "Post-graduate degree": "postgraduate",
    }
    return mapping.get(text)


def _multi_game_gender_to_display(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "man": "Male",
        "woman": "Female",
        "non-binary": "Non-binary",
        "prefer to self-describe:": "Self-described",
    }
    return mapping.get(text.lower(), text)


def _multi_game_gender_to_match(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {"man": "male", "woman": "female"}
    return mapping.get(text.lower())


def _recode_numeric_like(value: Any) -> Any:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _decision_value(raw_value: Any, game: str) -> Any:
    cleaned = _recode_numeric_like(raw_value)
    if cleaned is None:
        return None
    if game in {"UGProposer", "UGResponder", "TGReceiver"}:
        return int(float(cleaned))
    if game == "TGSender":
        return "YES" if cleaned == "1" or cleaned.lower() == "yes" else "NO"
    if game == "PD":
        return "A" if cleaned == "1" or cleaned.upper() == "A" else "B"
    if game == "SH":
        return "X" if cleaned == "1" or cleaned.upper() == "X" else "Y"
    if game == "C":
        lowered = cleaned.lower()
        mapping = {
            "mercury": "Mercury",
            "venus": "Venus",
            "earth": "Earth",
            "mars": "Mars",
            "saturn": "Saturn",
            "-2": "Mercury",
            "-1": "Venus",
            "0": "Earth",
            "1": "Mars",
            "2": "Saturn",
        }
        return mapping.get(lowered, cleaned)
    return cleaned


def _scenario_descriptor(scenario_code: str) -> tuple[str, str, int]:
    if scenario_code == "11":
        return "AISupport", "AgainstHuman", 1
    if scenario_code == "21":
        return "NoAISupport", "AgainstHuman", 2
    if scenario_code == "22":
        return "NoAISupport", "AgainstAI", 3
    if scenario_code == "2":
        return "NoAISupport", "Opaque", 2
    raise ValueError(f"Unsupported multi-game scenario code: {scenario_code}")


def build_bundle(repo_root: Path) -> DatasetBundle:
    raw_path = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "multi_game_llm_fvk2c"
        / "Package"
        / "data"
        / "MainDataRawClean.csv"
    )
    df = pd.read_csv(raw_path)
    treatment_map = {
        ("TransparentRandom", 1): "TRP",
        ("TransparentRandom", 0): "TRU",
        ("TransparentDelegation", 1): "TDP",
        ("TransparentDelegation", 0): "TDU",
        ("OpaqueDelegation", 1): "ODP",
        ("OpaqueDelegation", 0): "ODU",
    }
    df["TreatmentCode"] = [
        treatment_map[(treatment, int(personalized))]
        for treatment, personalized in zip(df["Treatment"], df["PersonalizedTreatment"])
    ]

    games = ["UGProposer", "UGResponder", "TGSender", "TGReceiver", "PD", "SH", "C"]
    for game in games:
        for scenario_code in ["11", "21", "22"]:
            cols = [col for col in df.columns if f"{game}_{scenario_code}" in col and "JUS" not in col]
            df[f"{game}_{scenario_code}"] = df[cols].apply(
                lambda row: "".join(str(value) for value in row if pd.notna(value)),
                axis=1,
            )
        cols = [col for col in df.columns if col.startswith(f"OD_{game}_2") and "JUS" not in col]
        df[f"{game}_2"] = df[cols].apply(
            lambda row: "".join(str(value) for value in row if pd.notna(value)),
            axis=1,
        )
        delegation_name = "TGSender_Delegation" if game == "TGSender" else f"{game}_Delegation"
        df[delegation_name] = np.where(
            df["Treatment"] != "TransparentRandom",
            (df[f"{game}_11"] == "").astype(int),
            np.nan,
        )

    age = df[["AgeTR", "AgeTDOD"]].bfill(axis=1).iloc[:, 0]
    gender = df[["GenderTR", "GenderTDOD"]].bfill(axis=1).iloc[:, 0]
    education = df[["EducationTR", "EducationTDOD"]].bfill(axis=1).iloc[:, 0]
    subject_demo_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(
        pd.DataFrame({"SubjectID": df["SubjectID"], "age": age, "gender": gender, "education": education})
        .drop_duplicates("SubjectID")
        .itertuples(index=False),
        start=1,
    ):
        age_value = clean_numeric_string(row.age)
        sex = _multi_game_gender_to_display(row.gender)
        edu = clean_text(row.education)
        summary, markdown = simple_demographic_summary(age=age_value, sex_or_gender=sex, education=edu)
        subject_demo_rows.append(
            {
                "source_row_id": f"multi_game_demo_source_{idx:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": age_to_bracket(age_value),
                "matching_sex": _multi_game_gender_to_match(row.gender),
                "matching_education": _multi_game_education_to_harmonized(row.education),
                "unit_id": str(row.SubjectID),
            }
        )
    demographic_source = pd.DataFrame(subject_demo_rows)
    units_df = demographic_source[["unit_id"]].drop_duplicates("unit_id").sort_values("unit_id").reset_index(drop=True)

    record_rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        subject_id = str(getattr(row, "SubjectID"))
        treatment = str(getattr(row, "Treatment"))
        treatment_code = str(getattr(row, "TreatmentCode"))
        personalized = int(getattr(row, "PersonalizedTreatment"))
        delegation_target = {
            "UGProposer_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "UGProposer_Delegation")),
            "UGResponder_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "UGResponder_Delegation")),
            "TGSender_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "TGSender_Delegation")),
            "TGReceiver_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "TGReceiver_Delegation")),
            "PD_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "PD_Delegation")),
            "SH_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "SH_Delegation")),
            "C_delegated": None if treatment == "TransparentRandom" else int(getattr(row, "C_Delegation")),
        }
        scenario_codes = ["11", "21", "22"] if treatment in {"TransparentRandom", "TransparentDelegation"} else ["11", "2"]
        scenario_outputs: list[dict[str, Any]] = []
        scenario_manifest: list[dict[str, Any]] = []
        for scenario_code in scenario_codes:
            scenario, case, scenario_order = _scenario_descriptor(scenario_code)
            scenario_outputs.append(
                {
                    "scenario": scenario,
                    "case": case,
                    "UGProposer_decision": _decision_value(getattr(row, f"UGProposer_{scenario_code}"), "UGProposer"),
                    "UGResponder_decision": _decision_value(getattr(row, f"UGResponder_{scenario_code}"), "UGResponder"),
                    "TGSender_decision": _decision_value(getattr(row, f"TGSender_{scenario_code}"), "TGSender"),
                    "TGReceiver_decision": _decision_value(getattr(row, f"TGReceiver_{scenario_code}"), "TGReceiver"),
                    "PD_decision": _decision_value(getattr(row, f"PD_{scenario_code}"), "PD"),
                    "SH_decision": _decision_value(getattr(row, f"SH_{scenario_code}"), "SH"),
                    "C_decision": _decision_value(getattr(row, f"C_{scenario_code}"), "C"),
                }
            )
            scenario_manifest.append(
                {
                    "scenario_code": scenario_code,
                    "scenario": scenario,
                    "case": case,
                    "order": scenario_order,
                }
            )
        scenario_outputs.sort(
            key=lambda item: next(
                order["order"]
                for order in scenario_manifest
                if order["scenario"] == item["scenario"] and order["case"] == item["case"]
            )
        )
        scenario_manifest.sort(key=lambda item: item["order"])

        target = {**delegation_target, "scenario_outputs": scenario_outputs}
        record_rows.append(
            {
                "record_id": subject_id,
                "unit_id": subject_id,
                "treatment_name": treatment_code,
                "TreatmentCode": treatment_code,
                "Treatment": treatment,
                "PersonalizedTreatment": personalized,
                "num_scenarios": len(scenario_outputs),
                "scenario_manifest_json": json.dumps(scenario_manifest),
                "gold_target_json": json.dumps(target),
            }
        )

    records = pd.DataFrame(record_rows).sort_values(["treatment_name", "record_id"]).reset_index(drop=True)
    return DatasetBundle(
        dataset_key="multi_game_llm_fvk2c",
        display_name="Multi-Game Battery with LLM Delegation",
        records=records,
        units=units_df,
        demographic_source=demographic_source.drop(columns=["unit_id"]),
        twin_matching_fields=["matching_age_bracket", "matching_sex", "matching_education"],
    )

