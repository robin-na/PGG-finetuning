from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import DatasetBundle, age_to_bracket, clean_numeric_string, simple_demographic_summary


def _longitudinal_gender_to_display(value: object) -> str | None:
    text = str(value).strip() if pd.notna(value) else None
    if not text:
        return None
    mapping = {"male": "Male", "female": "Female", "non-binary": "Non-binary"}
    return mapping.get(text.lower(), text)


def _longitudinal_gender_to_match(value: object) -> str | None:
    text = str(value).strip() if pd.notna(value) else None
    if not text:
        return None
    mapping = {"male": "male", "female": "female"}
    return mapping.get(text.lower())


def build_bundle(repo_root: Path) -> DatasetBundle:
    data_dir = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "longitudinal_trust_game_ht863"
        / "Data"
    )
    raw_files = sorted(
        [p for p in data_dir.glob("Repeated_trust_game+-+day+*.csv")],
        key=lambda p: int(p.name.split("day+")[1].split("_")[0]),
    )
    day_frames: list[pd.DataFrame] = []
    day10_demo: pd.DataFrame | None = None
    for fp in raw_files:
        day = int(fp.name.split("day+")[1].split("_")[0])
        df = pd.read_csv(fp, skiprows=[1, 2])
        if day == 5:
            df.loc[df["Q52"] == "613c9c83c9cd63d09d4ed30 ", "Q52"] = "613c9c83c9cd63d09d4ed300"
            df.loc[df["IPAddress"] == "51.9.95.189", "Q52"] = "615c49ef513583533427c961"
        df = df[df["Q52"].notna()].copy()
        rating_cols = [f"{i}_Q38" for i in range(1, 17)]
        for col in rating_cols:
            df[col] = (
                df[col]
                .replace({"Not at all": "1", "Extremely": "9"})
                .apply(pd.to_numeric, errors="coerce")
            )
        keep_cols = ["Q52", *rating_cols]
        if day == 10:
            keep_cols.extend(["Q49", "Q50"])
        out = df[keep_cols].copy()
        out["day"] = day
        day_frames.append(out)
        if day == 10:
            day10_demo = out[["Q52", "Q49", "Q50"]].drop_duplicates("Q52").copy()

    full = pd.concat(day_frames, ignore_index=True)
    day_counts = full.groupby("Q52")["day"].nunique()
    complete_pids = sorted(day_counts[day_counts == 10].index.tolist())
    full = full[full["Q52"].isin(complete_pids)].copy()
    if day10_demo is None:
        raise ValueError("Longitudinal trust: missing day-10 demographic table.")
    day10_demo = day10_demo[day10_demo["Q52"].isin(complete_pids)].copy()

    demo_rows: list[dict[str, object]] = []
    for idx, row in enumerate(day10_demo.itertuples(index=False), start=1):
        age = clean_numeric_string(row.Q50)
        sex = _longitudinal_gender_to_display(row.Q49)
        summary, markdown = simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"longitudinal_demo_source_{idx:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": age_to_bracket(age),
                "matching_sex": _longitudinal_gender_to_match(row.Q49),
                "matching_education": None,
            }
        )
    demographic_source = pd.DataFrame(demo_rows)

    record_rows: list[dict[str, object]] = []
    units = [{"unit_id": pid} for pid in complete_pids]
    for pid in complete_pids:
        person = full[full["Q52"] == pid].copy().sort_values("day")
        days_payload: list[dict[str, object]] = []
        flat_ratings: list[int] = []
        for day in range(1, 11):
            day_row = person[person["day"] == day]
            if day_row.empty:
                raise ValueError(f"Longitudinal trust: missing day {day} for participant {pid}")
            ratings = [int(day_row.iloc[0][f"{i}_Q38"]) for i in range(1, 17)]
            days_payload.append({"day": day, "ratings": ratings})
            flat_ratings.extend(ratings)
        target = {"days": days_payload}
        record_rows.append(
            {
                "record_id": pid,
                "unit_id": pid,
                "treatment_name": "LONGITUDINAL_TRUST_PANEL",
                "gold_target_json": json.dumps(target),
                "num_days": 10,
                "num_trials_per_day": 16,
                "num_ratings": len(flat_ratings),
            }
        )
    records = pd.DataFrame(record_rows).sort_values("record_id").reset_index(drop=True)
    units_df = pd.DataFrame(units).drop_duplicates("unit_id").sort_values("unit_id").reset_index(drop=True)
    return DatasetBundle(
        dataset_key="longitudinal_trust_game_ht863",
        display_name="Longitudinal Trust Game",
        records=records,
        units=units_df,
        demographic_source=demographic_source,
        twin_matching_fields=["matching_age_bracket", "matching_sex"],
    )

