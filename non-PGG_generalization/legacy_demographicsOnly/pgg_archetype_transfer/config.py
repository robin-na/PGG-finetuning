"""Constants, paths, and demographic mappings for PGG archetype transfer pipeline."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # PGG-finetuning/
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
NON_PGG_ROOT = PROJECT_ROOT / "non-PGG_generalization"
TRANSFER_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = TRANSFER_ROOT / "output"

# PGG consolidated data
PGG_DATA_DIR = NON_PGG_ROOT / "data" / "PGG"
PGG_DEMOGRAPHICS_CSV = PGG_DATA_DIR / "demographics_numeric_learn_val_consolidated.csv"
PGG_ARCHETYPE_JSONL = PGG_DATA_DIR / "archetype_oracle_gpt51_learn_val_union_finished.jsonl"

# Twin-2k-500 legacy local paths
TWIN_DIR = NON_PGG_ROOT / "twin-20k-500"
QUESTION_CATALOG_JSON = (
    TWIN_DIR / "question_catalog_and_human_response_csv" / "question_catalog.json"
)
WAVE4_MAPPING_JSON = (
    TWIN_DIR / "LLM_simulation_results" / "wave4_formatted_to_catalog_mapping.json"
)

# HuggingFace dataset
HF_DATASET_NAME = "LLM-Digital-Twin/Twin-2K-500"
HF_CONFIG_WAVE_SPLIT = "wave_split"

# Retrieval common utilities (for reuse)
RETRIEVAL_COMMON_DIR = PROJECT_ROOT / "Persona" / "archetype_retrieval"

# ── Demographic feature columns (PGG-compatible) ─────────────────────────────
D_FEATURE_COLUMNS = [
    "age",
    "age_missing",
    "gender_man",
    "gender_woman",
    "gender_non_binary",
    "gender_unknown",
    "education_high_school",
    "education_bachelor",
    "education_master",
    "education_other",
    "education_unknown",
]

# ── Twin-2k-500 → PGG demographic mappings ───────────────────────────────────

# QID12: sex (1-indexed position in options list)
SEX_MAP = {
    1: {"gender_man": 1, "gender_woman": 0, "gender_non_binary": 0, "gender_unknown": 0},
    2: {"gender_man": 0, "gender_woman": 1, "gender_non_binary": 0, "gender_unknown": 0},
}
SEX_MAP_DEFAULT = {"gender_man": 0, "gender_woman": 0, "gender_non_binary": 0, "gender_unknown": 1}

# QID13: age bracket → midpoint
AGE_MAP = {
    1: 24,   # 18-29
    2: 40,   # 30-49
    3: 57,   # 50-64
    4: 72,   # 65+
}

# QID14: education level → PGG one-hot
EDUCATION_MAP = {
    1: {"education_high_school": 0, "education_bachelor": 0, "education_master": 0, "education_other": 1, "education_unknown": 0},  # Less than HS
    2: {"education_high_school": 1, "education_bachelor": 0, "education_master": 0, "education_other": 0, "education_unknown": 0},  # HS grad
    3: {"education_high_school": 0, "education_bachelor": 0, "education_master": 0, "education_other": 1, "education_unknown": 0},  # Some college
    4: {"education_high_school": 0, "education_bachelor": 0, "education_master": 0, "education_other": 1, "education_unknown": 0},  # Associate's
    5: {"education_high_school": 0, "education_bachelor": 1, "education_master": 0, "education_other": 0, "education_unknown": 0},  # College grad
    6: {"education_high_school": 0, "education_bachelor": 0, "education_master": 1, "education_other": 0, "education_unknown": 0},  # Postgraduate
}
EDUCATION_MAP_DEFAULT = {"education_high_school": 0, "education_bachelor": 0, "education_master": 0, "education_other": 0, "education_unknown": 1}

# ── Economic game questions (target questions for pilot) ──────────────────────
# BlockName == "Economic preferences" in question_catalog.json
ECONOMIC_GAME_BLOCK = "Economic preferences"

# ── Game-specific QID sets (for --games filter) ───────────────────────────────
GAME_QIDS = {
    "trust":     ["QID117", "QID118", "QID119", "QID120", "QID121", "QID122"],
    "ultimatum": ["QID224", "QID225", "QID226", "QID227", "QID228", "QID229", "QID230"],
    "dictator":  ["QID231"],
}

# ── Model settings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
PREDICTION_MODEL = "gpt-4o"
SUMMARIZATION_MODEL = "gpt-4o-mini"
TOP_K = 10
PILOT_N_PARTICIPANTS = 200
RIDGE_ALPHA = 10.0
RANDOM_STATE = 42
