#!/usr/bin/env python3
"""Extract demographics from Twin-2k-500 HuggingFace dataset and map to PGG-compatible format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from config import (
    AGE_MAP,
    D_FEATURE_COLUMNS,
    EDUCATION_MAP,
    EDUCATION_MAP_DEFAULT,
    HF_CONFIG_WAVE_SPLIT,
    HF_DATASET_NAME,
    OUTPUT_ROOT,
    SEX_MAP,
    SEX_MAP_DEFAULT,
)


def extract_demographics_from_persona_json(persona_json_str: str) -> Dict[str, Any]:
    """Extract QID12 (sex), QID13 (age), QID14 (education) from wave1_3_persona_json."""
    blocks = json.loads(persona_json_str)
    demo = {}
    for block in blocks:
        if block.get("BlockName") != "Demographics":
            continue
        for q in block.get("Questions", []):
            qid = q.get("QuestionID")
            answers = q.get("Answers", {})
            if qid in ("QID12", "QID13", "QID14"):
                demo[qid] = {
                    "position": answers.get("SelectedByPosition"),
                    "text": answers.get("SelectedText"),
                }
        break
    return demo


def map_to_pgg_features(demo: Dict[str, Any]) -> Dict[str, Any]:
    """Map Twin-2k-500 demographic answers to PGG-compatible feature vector."""
    row: Dict[str, Any] = {}

    # Age
    age_pos = demo.get("QID13", {}).get("position")
    if age_pos and age_pos in AGE_MAP:
        row["age"] = AGE_MAP[age_pos]
        row["age_missing"] = 0
    else:
        row["age"] = float("nan")
        row["age_missing"] = 1

    # Sex → gender one-hot
    sex_pos = demo.get("QID12", {}).get("position")
    gender = SEX_MAP.get(sex_pos, SEX_MAP_DEFAULT)
    row.update(gender)

    # Education → education one-hot
    edu_pos = demo.get("QID14", {}).get("position")
    edu = EDUCATION_MAP.get(edu_pos, EDUCATION_MAP_DEFAULT)
    row.update(edu)

    # Also store readable labels for prompt construction
    row["sex_label"] = demo.get("QID12", {}).get("text", "Unknown")
    row["age_label"] = demo.get("QID13", {}).get("text", "Unknown")
    row["education_label"] = demo.get("QID14", {}).get("text", "Unknown")

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Twin-2k-500 demographics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "twin2k_demographics_pgg_compatible.csv",
    )
    args = parser.parse_args()

    print("Loading Twin-2K-500 wave_split from HuggingFace...")
    ds = load_dataset(HF_DATASET_NAME, HF_CONFIG_WAVE_SPLIT)["data"]
    print(f"Loaded {len(ds)} participants.")

    rows: List[Dict[str, Any]] = []
    for i, example in enumerate(ds):
        pid = example["pid"]
        demo = extract_demographics_from_persona_json(example["wave1_3_persona_json"])
        features = map_to_pgg_features(demo)
        features["pid"] = pid
        rows.append(features)

    df = pd.DataFrame(rows)
    # Reorder: pid first, then D features, then labels
    label_cols = ["sex_label", "age_label", "education_label"]
    col_order = ["pid"] + D_FEATURE_COLUMNS + label_cols
    df = df[col_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Summary stats
    print(f"\nAge distribution:\n{df['age'].value_counts().sort_index()}")
    print(f"\nGender distribution:")
    for g in ["gender_man", "gender_woman", "gender_non_binary", "gender_unknown"]:
        print(f"  {g}: {df[g].sum()}")
    print(f"\nEducation distribution:")
    for e in ["education_high_school", "education_bachelor", "education_master", "education_other", "education_unknown"]:
        print(f"  {e}: {df[e].sum()}")


if __name__ == "__main__":
    main()
