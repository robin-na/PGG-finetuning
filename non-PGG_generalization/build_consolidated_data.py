#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "non-PGG_generalization" / "data" / "PGG"


def nonempty_jsonl_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_demographics(learn_path: Path, val_path: Path, out_path: Path) -> int:
    learn_df = pd.read_csv(learn_path)
    val_df = pd.read_csv(val_path)

    learn_df.insert(0, "wave", "learning_wave")
    val_df.insert(0, "wave", "validation_wave")

    combined = pd.concat([learn_df, val_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["wave", "gameId", "playerId"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return int(len(combined))


def build_oracle_union(
    learn_path: Path, val_path: Path, out_path: Path
) -> Tuple[int, int, int]:
    rows: List[Dict] = []
    for wave, src in (
        ("learning_wave", learn_path),
        ("validation_wave", val_path),
    ):
        for row in nonempty_jsonl_rows(src):
            row["_wave"] = wave
            rows.append(row)

    seen = set()
    unique_rows: List[Dict] = []
    for row in rows:
        key = (str(row.get("experiment")), str(row.get("participant")))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in unique_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    return len(rows), len(unique_rows), len(rows) - len(unique_rows)


def main() -> int:
    demo_learn = REPO_ROOT / "demographics" / "demographics_numeric_learn.csv"
    demo_val = REPO_ROOT / "demographics" / "demographics_numeric_val.csv"
    oracle_learn = REPO_ROOT / "Persona" / "archetype_oracle_gpt51_learn.jsonl"
    oracle_val = REPO_ROOT / "Persona" / "archetype_oracle_gpt51_val.jsonl"

    out_demo = OUT_DIR / "demographics_numeric_learn_val_consolidated.csv"
    out_oracle = OUT_DIR / "archetype_oracle_gpt51_learn_val_union_finished.jsonl"
    out_manifest = OUT_DIR / "manifest.json"

    demo_rows = build_demographics(demo_learn, demo_val, out_demo)
    oracle_total, oracle_unique, oracle_dropped = build_oracle_union(
        oracle_learn, oracle_val, out_oracle
    )

    manifest = {
        "inputs": {
            "demographics_learn": str(demo_learn.relative_to(REPO_ROOT)),
            "demographics_val": str(demo_val.relative_to(REPO_ROOT)),
            "oracle_learn": str(oracle_learn.relative_to(REPO_ROOT)),
            "oracle_val": str(oracle_val.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "demographics_consolidated_csv": str(out_demo.relative_to(REPO_ROOT)),
            "oracle_consolidated_jsonl": str(out_oracle.relative_to(REPO_ROOT)),
        },
        "row_counts": {
            "demographics_learning_wave": int(pd.read_csv(demo_learn).shape[0]),
            "demographics_validation_wave": int(pd.read_csv(demo_val).shape[0]),
            "demographics_consolidated": demo_rows,
            "oracle_learning_wave": count_nonempty_lines(oracle_learn),
            "oracle_validation_wave": count_nonempty_lines(oracle_val),
            "oracle_consolidated_total": oracle_total,
            "oracle_consolidated_unique": oracle_unique,
            "oracle_duplicates_dropped": oracle_dropped,
        },
        "notes": {
            "oracle_extra_field": "_wave",
            "oracle_dedupe_key": ["experiment", "participant"],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {out_demo.relative_to(REPO_ROOT)}")
    print(f"Wrote: {out_oracle.relative_to(REPO_ROOT)}")
    print(f"Wrote: {out_manifest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
