from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DEMOGRAPHIC_FIELDS = ["data.age", "data.gender", "data.education"]
THIS_DIR = Path(__file__).resolve().parent

GENDER_CODE_MAP = {
    "unknown": 0,
    "man": 1,
    "woman": 2,
    "non_binary": 3,
}

EDUCATION_CODE_MAP = {
    "unknown": 0,
    "high_school": 1,
    "bachelor": 2,
    "master": 3,
    "other": 4,
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_gender(value) -> str:
    """
    Normalize raw/open-ended gender input into:
    {'man', 'woman', 'non_binary', 'unknown'}.
    """
    raw = _clean_text(value)
    if not raw:
        return "unknown"

    if re.fullmatch(r"\d+", raw):
        return "unknown"

    if raw in {
        "none",
        "prefer not to say",
        "prefer not",
        "na",
        "n/a",
        "unspecified",
        "questioning",
        "?",
        "idk",
        "i don't know",
        "decline",
        "declined",
        "i don't have a gender, i have a sex.",
        "zx",
        "25",
        "cabbage",
    }:
        return "unknown"

    normalized = re.sub(r"[\W_]+", " ", raw)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = set(normalized.split())

    non_binary_phrases = {
        "non binary",
        "nonbinary",
        "non binary afab",
        "non binary amab",
        "non binary man",
        "genderqueer",
        "gender queer",
        "genderfluid",
        "gender fluid",
        "agender",
        "genderless",
        "nb",
        "enby",
        "queer",
        "gq",
    }

    if any(phrase in normalized for phrase in non_binary_phrases):
        return "non_binary"
    if "non" in tokens and "binary" in tokens:
        return "non_binary"

    male_aliases = {
        "m",
        "male",
        "man",
        "masculine",
        "mle",
        "msle",
        "malee",
        "malel",
        "males",
        "malde",
        "malw",
        "make",
        "mail",
        "mann",
        "transman",
    }
    female_aliases = {
        "f",
        "female",
        "woman",
        "women",
        "girl",
        "fem",
        "feminine",
        "femail",
        "femae",
        "femaile",
        "femal",
        "famel",
        "feame",
        "femake",
        "fenale",
        "females",
        "femals",
        "fena",
        "femsle",
        "femmine",
        "feminin",
        "womsn",
        "femme",
    }

    male_hit = bool(tokens & male_aliases) or any(stem in normalized for stem in ("male", "man", "mascul"))
    female_hit = bool(tokens & female_aliases) or any(stem in normalized for stem in ("female", "woman", "girl", "fem"))
    trans_hit = "trans" in tokens or "transgender" in tokens or "transgender" in normalized

    if trans_hit:
        if "trans male" in normalized or "transman" in normalized or ("trans" in tokens and male_hit):
            return "man"
        if "trans female" in normalized or "trans woman" in normalized or ("trans" in tokens and female_hit):
            return "woman"

    if male_hit and not female_hit:
        return "man"
    if female_hit and not male_hit:
        return "woman"
    return "unknown"


def _normalize_education(value) -> str:
    """
    Normalize education into:
    {'high_school', 'bachelor', 'master', 'other', 'unknown'}.
    """
    raw = _clean_text(value).replace("_", "-")
    if not raw:
        return "unknown"
    if raw in {"high-school", "high school"}:
        return "high_school"
    if raw == "bachelor":
        return "bachelor"
    if raw == "master":
        return "master"
    if raw == "other":
        return "other"
    return "other"


def _select_one_row_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per (gameId, playerId).
    If duplicates exist, keep the most complete demographics row.
    """
    required = {"gameId", "playerId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    for col in DEMOGRAPHIC_FIELDS:
        if col not in work.columns:
            work[col] = pd.NA

    work["_created_at"] = pd.to_datetime(work.get("createdAt"), errors="coerce", utc=True)
    work["_filled_demographics"] = work[DEMOGRAPHIC_FIELDS].notna().sum(axis=1)

    work = work.sort_values(
        by=["gameId", "playerId", "_filled_demographics", "_created_at"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    work = work.drop_duplicates(subset=["gameId", "playerId"], keep="first")
    return work.reset_index(drop=True)


def build_demographic_numeric_table(df_demographic: pd.DataFrame) -> pd.DataFrame:
    """
    Convert demographic inputs to numeric model-ready features.

    Output schema:
    - keys: gameId, playerId
    - numeric: age, age_missing, gender_code, education_code
    - one-hot indicators for gender and education categories
    """
    dedup = _select_one_row_per_player(df_demographic)
    out = dedup[["gameId", "playerId"]].copy()

    age = pd.to_numeric(dedup.get("data.age"), errors="coerce")
    age = age.where((age > 0) & (age <= 100))
    out["age"] = age.astype("float32")
    out["age_missing"] = out["age"].isna().astype("int8")

    gender_cat = dedup.get("data.gender", pd.Series(index=dedup.index, dtype="object")).map(_normalize_gender)
    out["gender_code"] = gender_cat.map(GENDER_CODE_MAP).astype("int8")
    for category in ("man", "woman", "non_binary", "unknown"):
        out[f"gender_{category}"] = (gender_cat == category).astype("int8")

    education_cat = dedup.get("data.education", pd.Series(index=dedup.index, dtype="object")).map(_normalize_education)
    out["education_code"] = education_cat.map(EDUCATION_CODE_MAP).astype("int8")
    for category in ("high_school", "bachelor", "master", "other", "unknown"):
        out[f"education_{category}"] = (education_cat == category).astype("int8")

    return out


def generate_and_save_tables(
    learn_input: Path = Path("data/raw_data/learning_wave/player-inputs.csv"),
    val_input: Path = Path("data/raw_data/validation_wave/player-inputs.csv"),
    learn_output: Path = THIS_DIR / "demographics_numeric_learn.csv",
    val_output: Path = THIS_DIR / "demographics_numeric_val.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learn_df = pd.read_csv(learn_input)
    val_df = pd.read_csv(val_input)

    learn_table = build_demographic_numeric_table(learn_df)
    val_table = build_demographic_numeric_table(val_df)

    learn_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    learn_table.to_csv(learn_output, index=False)
    val_table.to_csv(val_output, index=False)
    return learn_table, val_table


if __name__ == "__main__":
    learn_table, val_table = generate_and_save_tables()
    print(f"Saved learn table: {len(learn_table)} rows")
    print(f"Saved val table:   {len(val_table)} rows")
