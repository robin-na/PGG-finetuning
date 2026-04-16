#!/usr/bin/env python3
"""Build a first-pass Twin-2K-500 task inventory and PGG-transfer grounding files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download


REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
CONFIG = "wave_split"
QUESTION_CATALOG_FILE = "question_catalog_and_human_response_csv/question_catalog.json"
OUT_DIR = Path("non-PGG_generalization/twin_profiles")
LOCAL_TWIN_ROOT = Path("non-PGG_generalization/data/Twin-2k-500")
LOCAL_QUESTION_CATALOG = LOCAL_TWIN_ROOT / "snapshot" / QUESTION_CATALOG_FILE
LOCAL_WAVE_SPLIT = LOCAL_TWIN_ROOT / "wave_split_dataset"


CORE_GROUPS = {
    "trust": {"QID117", "QID118", "QID119", "QID120", "QID121", "QID122", "QID271", "QID272"},
    "ultimatum": {"QID224", "QID225", "QID226", "QID227", "QID228", "QID229", "QID230"},
    "dictator": {"QID231", "QID275"},
    "mental_accounting": {"QID149", "QID150", "QID151", "QID152"},
    "time_preference": {"QID84", "QID243", "QID244", "QID245", "QID246", "QID247", "QID248"},
    "risk_preference_gain": {"QID249", "QID250", "QID251", "QID252"},
    "risk_preference_loss": {"QID276", "QID277", "QID278", "QID279"},
}


def load_question_catalog() -> List[Dict]:
    try:
        catalog_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=QUESTION_CATALOG_FILE,
            repo_type="dataset",
        )
    except Exception:
        if not LOCAL_QUESTION_CATALOG.exists():
            raise
        catalog_path = str(LOCAL_QUESTION_CATALOG)
    with Path(catalog_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_wave4_target_qids() -> Tuple[List[str], List[str]]:
    try:
        ds = load_dataset(REPO_ID, CONFIG, split="data[:1]")
        row = ds[0]
    except Exception:
        if not LOCAL_WAVE_SPLIT.exists():
            raise
        row = load_from_disk(str(LOCAL_WAVE_SPLIT))["data"][0]
    prior_blocks = json.loads(row["wave4_Q_wave1_3_A"])
    wave4_blocks = json.loads(row["wave4_Q_wave4_A"])

    prior_qids: List[str] = []
    wave4_qids: List[str] = []

    for block in prior_blocks:
        for q in block.get("Questions", []):
            qid = q.get("QuestionID")
            if qid:
                prior_qids.append(qid)

    for block in wave4_blocks:
        for q in block.get("Questions", []):
            qid = q.get("QuestionID")
            if qid:
                wave4_qids.append(qid)

    return prior_qids, wave4_qids


def classify_family(question_id: str, block_name: str) -> str:
    normalized = " ".join((block_name or "").split()).lower()
    if "economic preferences" in normalized:
        for family, qids in CORE_GROUPS.items():
            if question_id in qids:
                return family

    if "demographics" in normalized:
        return "demographics"
    if "personality" in normalized:
        return "personality"
    if "cognitive tests" in normalized:
        return "cognitive_tests"
    if "pricing" in normalized:
        return "pricing"
    if "heuristics" in normalized or "anchoring" in normalized or "allais" in normalized:
        return "heuristics_biases"
    if "economic preferences" in normalized:
        return "economic_preferences_other"
    return "other"


def candidate_role(family: str, is_wave4_target: bool) -> str:
    if family in {"trust", "ultimatum", "dictator"}:
        return "core_pgg_transfer_target"
    if family in {"mental_accounting", "time_preference", "risk_preference_gain", "risk_preference_loss"}:
        return "secondary_econ_target"
    if is_wave4_target:
        return "digital_twin_sim_target"
    if family in {"demographics", "personality", "cognitive_tests", "heuristics_biases", "pricing"}:
        return "profile_input_candidate"
    return "other"


def short_text(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def write_inventory(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_wave4_target_full_text(
    catalog: List[Dict],
    wave4_qids: List[str],
    out_path: Path,
) -> None:
    qid_set = set(wave4_qids)
    rows: List[Dict[str, str]] = []
    for q in catalog:
        qid = q.get("QuestionID", "")
        if qid not in qid_set:
            continue
        rows.append(
            {
                "question_id": qid,
                "block_name": q.get("BlockName", ""),
                "question_type": q.get("QuestionType", ""),
                "question_text_full": " ".join((q.get("QuestionText", "") or "").split()),
                "options_json": json.dumps(q.get("Options", []), ensure_ascii=False),
                "rows_json": json.dumps(q.get("Rows", []), ensure_ascii=False),
                "csv_columns_json": json.dumps(q.get("csv_columns", []), ensure_ascii=False),
            }
        )
    rows.sort(key=lambda r: (r["block_name"], r["question_id"]))
    write_inventory(rows, out_path)


def build_summary_md(inventory_rows: List[Dict], out_path: Path) -> None:
    lines = [
        "# Twin Profiles",
        "",
        "## Digital-Twin-Simulation Setup",
        "",
        "- Same participant appears across waves 1 through 4.",
        "- `wave1_3_persona_text` is the prompt source: it is a text rendering of that participant's answers from waves 1-3.",
        "- `wave4_Q_wave4_A` is the held-out target answered by that same participant in wave 4.",
        "- `wave4_Q_wave1_3_A` is the matched earlier-wave answer block for the same target questions and same participant, giving a human consistency comparator.",
        "- The standard wave-4 benchmark targets 64 question IDs and is dominated by heuristics/biases and pricing blocks, not trust/ultimatum/dictator.",
        "",
        "## Recommended PGG-Transfer Benchmark",
        "",
        "- Primary targets: trust, ultimatum, dictator.",
        "- Secondary targets: mental accounting, time preference, risk preference.",
        "- Use Twin demographics, personality, cognitive tests, and non-target behavioral tasks as profile inputs.",
        "- Exclude target-family items from the Twin profile during prediction.",
        "",
        "## First-Pass Leakage Exclusions",
        "",
        "- Trust target: exclude `QID117-122` and trust thought-text `QID271-272`.",
        "- Ultimatum target: exclude `QID224-230`.",
        "- Dictator target: exclude `QID231` and dictator thought-text `QID275`.",
        "- Secondary target families: exclude same-family items when those families are targets.",
        "",
        "## Why Not Reuse Their Benchmark As-Is",
        "",
        "- Their benchmark is well-suited for Twin-internal persona completion.",
        "- It is not the cleanest primary benchmark for PGG transfer because the main wave-4 target set is mostly outside the canonical social-preference game space.",
        "",
        "## Files",
        "",
        f"- `twin_question_inventory.csv`",
        f"- `wave4_target_inventory.csv`",
        f"- `wave4_target_inventory_full_text.csv`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_question_catalog()
    prior_qids, wave4_qids = get_wave4_target_qids()
    wave4_qid_set = set(wave4_qids)
    prior_qid_set = set(prior_qids)

    inventory_rows: List[Dict] = []
    for q in catalog:
        qid = q.get("QuestionID", "")
        block_name = q.get("BlockName", "")
        family = classify_family(qid, block_name)
        inventory_rows.append(
            {
                "question_id": qid,
                "block_name": block_name,
                "question_type": q.get("QuestionType", ""),
                "n_csv_columns": len(q.get("csv_columns", []) or []),
                "is_wave4_target": qid in wave4_qid_set,
                "is_wave1_3_match_for_wave4_target": qid in prior_qid_set,
                "family": family,
                "candidate_role": candidate_role(family, qid in wave4_qid_set),
                "exclusion_group": family if family in CORE_GROUPS else "",
                "question_text_short": short_text(q.get("QuestionText", "")),
            }
        )

    inventory_rows.sort(key=lambda r: (r["block_name"], r["question_id"]))
    write_inventory(inventory_rows, OUT_DIR / "twin_question_inventory.csv")
    write_inventory(
        [r for r in inventory_rows if r["is_wave4_target"]],
        OUT_DIR / "wave4_target_inventory.csv",
    )
    write_wave4_target_full_text(
        catalog,
        wave4_qids,
        OUT_DIR / "wave4_target_inventory_full_text.csv",
    )
    build_summary_md(inventory_rows, OUT_DIR / "TWIN_TASK_GROUNDING.md")

    print(f"Wrote {OUT_DIR / 'twin_question_inventory.csv'}")
    print(f"Wrote {OUT_DIR / 'wave4_target_inventory.csv'}")
    print(f"Wrote {OUT_DIR / 'wave4_target_inventory_full_text.csv'}")
    print(f"Wrote {OUT_DIR / 'TWIN_TASK_GROUNDING.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
