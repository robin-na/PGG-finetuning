from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    DatasetBundle,
    age_to_bracket,
    canonical_sex,
    clean_numeric_string,
    clean_text,
    format_bullet_markdown,
)


def _minority_education_to_harmonized(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "No formal qualifications": "high school",
        "Secondary education (e.g. GED/GCSE)": "high school",
        "High school diploma/A-levels": "high school",
        "Technical/community college": "college/postsecondary",
        "Undergraduate degree (BA/BSc/other)": "college/postsecondary",
        "Graduate degree (MA/MSc/MPhil/other)": "postgraduate",
        "Doctorate degree (PhD/other)": "postgraduate",
    }
    return mapping.get(text)


def _minority_sex_to_display(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {"male": "Male", "female": "Female"}
    return mapping.get(text.lower(), text)


def _minority_demographic_summary(row: dict[str, Any]) -> tuple[str, str]:
    parts: list[str] = []
    age = row.get("age")
    sex = row.get("sex_or_gender")
    education = row.get("education")
    ethnicity = row.get("ethnicity")
    employment = row.get("employment_status")
    student = row.get("student_status")
    nationality = row.get("nationality")
    residence = row.get("country_of_residence")
    birth = row.get("country_of_birth")

    if age is not None and sex is not None:
        parts.append(f"{age}-year-old, {sex.lower()}")
    elif age is not None:
        parts.append(f"{age}-year-old")
    elif sex is not None:
        parts.append(sex.lower())
    if ethnicity is not None:
        parts.append(ethnicity)
    if education is not None:
        parts.append(education)
    if student is not None and student != "No":
        parts.append("student")
    if employment is not None:
        parts.append(employment.lower())
    if nationality is not None and residence is not None:
        if birth is not None and birth != residence:
            parts.append(f"{nationality} citizen living in {residence}, born in {birth}")
        else:
            parts.append(f"{nationality} citizen living in {residence}")
    elif residence is not None:
        parts.append(f"living in {residence}")
    elif birth is not None:
        parts.append(f"born in {birth}")

    summary = ", ".join(parts).strip()
    if summary:
        summary = summary[0].upper() + summary[1:] + "."
    markdown = format_bullet_markdown(
        [
            ("Age", age),
            ("Sex/gender", sex),
            ("Education", education),
            ("Ethnicity", ethnicity),
            ("Student status", student),
            ("Employment", employment),
            ("Country of birth", birth),
            ("Country of residence", residence),
            ("Nationality", nationality),
        ]
    )
    return summary, markdown


def build_bundle(repo_root: Path) -> DatasetBundle:
    main_path = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "minority_game_bret_njzas"
        / "experiment_data"
        / "all_apps_wide-2022-08-31.csv"
    )
    prolific_path = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "minority_game_bret_njzas"
        / "experiment_data"
        / "prolific_export_62fcafdbdaec84519e0c272b.csv"
    )
    decision_cols = [f"bonus_game.{i}.player.decision" for i in range(1, 12)]
    main = pd.read_csv(
        main_path,
        usecols=[
            "participant.code",
            "participant.label",
            "participant.finished",
            "participant.in_deception",
            "bret.1.player.boxes_collected",
            *decision_cols,
        ],
    )
    main = main[main["participant.finished"] == 1].copy()
    prolific = pd.read_csv(prolific_path)
    merged = main.merge(
        prolific,
        left_on="participant.label",
        right_on="Participant id",
        how="left",
    )

    demo_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(merged.to_dict(orient="records"), start=1):
        age = clean_numeric_string(row.get("Age"))
        sex = _minority_sex_to_display(row.get("Sex"))
        education = clean_text(row.get("Highest education level completed"))
        ethnicity = clean_text(row.get("Ethnicity simplified")) or clean_text(row.get("Ethnicity_simplified"))
        country_of_birth = clean_text(row.get("Country of birth")) or clean_text(row.get("Country_of_birth"))
        country_of_residence = clean_text(row.get("Country of residence")) or clean_text(row.get("Country_of_residence"))
        nationality = clean_text(row.get("Nationality"))
        student_status = clean_text(row.get("Student status")) or clean_text(row.get("Student_status"))
        employment = clean_text(row.get("Employment status")) or clean_text(row.get("Employment_status"))
        summary, markdown = _minority_demographic_summary(
            {
                "age": age,
                "sex_or_gender": sex,
                "education": education,
                "ethnicity": ethnicity,
                "student_status": student_status,
                "employment_status": employment,
                "country_of_birth": country_of_birth,
                "country_of_residence": country_of_residence,
                "nationality": nationality,
            }
        )
        demo_rows.append(
            {
                "source_row_id": f"minority_demo_source_{idx:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": age_to_bracket(age),
                "matching_sex": canonical_sex(sex),
                "matching_education": _minority_education_to_harmonized(education),
            }
        )
    demographic_source = pd.DataFrame(demo_rows)

    record_rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        participant_code = str(row["participant.code"])
        units.append({"unit_id": participant_code})
        target = {
            "bonus_game_choices": [str(row[col]) for col in decision_cols],
            "bret_boxes": int(round(float(row["bret.1.player.boxes_collected"]))),
        }
        record_rows.append(
            {
                "record_id": participant_code,
                "unit_id": participant_code,
                "treatment_name": f"DECEPTION_{int(row['participant.in_deception'])}",
                "deception_condition": int(row["participant.in_deception"]),
                "participant_label": str(row["participant.label"]),
                "gold_target_json": json.dumps(target),
            }
        )

    records = pd.DataFrame(record_rows).sort_values(["treatment_name", "record_id"]).reset_index(drop=True)
    units_df = pd.DataFrame(units).drop_duplicates("unit_id").sort_values("unit_id").reset_index(drop=True)
    return DatasetBundle(
        dataset_key="minority_game_bret_njzas",
        display_name="Minority Game + BRET",
        records=records,
        units=units_df,
        demographic_source=demographic_source,
        twin_matching_fields=["matching_age_bracket", "matching_sex", "matching_education"],
    )

