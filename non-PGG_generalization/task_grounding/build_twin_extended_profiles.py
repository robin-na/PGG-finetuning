#!/usr/bin/env python3
"""Build deterministic Twin extended profiles from wave 1-3 responses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from jsonschema import Draft202012Validator


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

DEFAULT_RESPONSES_CSV = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "wave1_3_response.csv"
)
DEFAULT_CATALOG_JSON = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "question_catalog.json"
)
DEFAULT_MAPPING_CSV = THIS_DIR / "twin_extended_profile_mapping.csv"
DEFAULT_SCHEMA_JSON = THIS_DIR / "twin_extended_profile_schema.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output" / "twin_extended_profiles"


BDI_QIDS = [
    "Personality::QID126",
    "Personality::QID128",
    "Personality::QID129",
    "Personality::QID130",
    "Personality::QID131",
    "Personality::QID132",
    "Personality::QID133",
    "Personality::QID134",
    "Personality::QID136",
    "Personality::QID137",
    "Personality::QID138",
    "Personality::QID139",
    "Personality::QID140",
    "Personality::QID141",
    "Personality::QID142",
    "Personality::QID143",
    "Personality::QID144",
    "Personality::QID145",
    "Personality::QID146",
    "Personality::QID147",
]

MENTAL_ACCOUNTING_KEYS = {
    "Economic preferences::QID149": 1,
    "Economic preferences::QID150": 1,
    "Economic preferences::QID151": 1,
    "Economic preferences::QID152": 2,
}

WTP_WTA_AMOUNTS = [
    10,
    100,
    1000,
    10000,
    50000,
    100000,
    250000,
    500000,
    1000000,
    5000000,
]

FINANCIAL_LITERAL_KEYS = {
    "Cognitive tests::QID36": 1,
    "Cognitive tests::QID37": 3,
    "Cognitive tests::QID38": 2,
    "Cognitive tests::QID39": 1,
    "Cognitive tests::QID40": 2,
    "Cognitive tests::QID41": 2,
    "Cognitive tests::QID42": 1,
}

NUMERACY_FREE_TEXT_KEYS = {
    "Cognitive tests::QID44": 500.0,
    "Cognitive tests::QID45": 10.0,
    "Cognitive tests::QID46": 20.0,
    "Cognitive tests::QID47": 0.1,
    "Cognitive tests::QID48": 100.0,
    "Cognitive tests::QID49": 5.0,
    "Cognitive tests::QID50": 0.05,
    "Cognitive tests::QID51": 47.0,
    "Cognitive tests::QID53": 0.0,
    "Cognitive tests::QID54": 2.0,
    "Cognitive tests::QID55": 8.0,
}

TEXT_ANSWER_KEYS = {
    "Cognitive tests::QID52": "emily",
}

LOGIC_KEYS = {
    "Cognitive tests::QID217": 1,
    "Cognitive tests::QID218": 1,
    "Cognitive tests::QID219": 1,
    "Cognitive tests::QID220": 1,
    "Cognitive tests::QID268": 3,
    "Cognitive tests::QID269": 3,
    "Cognitive tests::QID270": 2,
    "Cognitive tests::QID271": 1,
    "Cognitive tests::QID272": 1,
    "Cognitive tests::QID273": 2,
    "Cognitive tests::QID274": 2,
    "Cognitive tests::QID275": 1,
    "Cognitive tests::QID276": 2,
    "Cognitive tests::QID277": 2,
    "Cognitive tests::QID278": 1,
    "Cognitive tests::QID279": 1,
}

AFRICA_TRUE_COUNTRIES = 54.0
AFRICA_HIGH_ANCHOR = 65.0
AFRICA_LOW_ANCHOR = 12.0
REDWOOD_TRUE_FEET = 380.0
REDWOOD_HIGH_ANCHOR = 1000.0
REDWOOD_LOW_ANCHOR = 85.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-csv", type=Path, default=DEFAULT_RESPONSES_CSV)
    parser.add_argument("--catalog-json", type=Path, default=DEFAULT_CATALOG_JSON)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pid", type=int, default=None)
    return parser.parse_args()


def normalize_block_name(value: str) -> str:
    return " ".join((value or "").split())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def text_excerpt(value: Any, limit: int = 160) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def safe_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def score_to_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score < 15:
        return "very_low"
    if score < 35:
        return "low"
    if score < 45:
        return "mixed"
    if score < 65:
        return "medium"
    if score < 85:
        return "high"
    return "very_high"


def dimension_payload(
    score: Optional[float],
    rationale: str,
    evidence_refs: Sequence[str],
    confidence: str,
    scope_note: Optional[str] = None,
) -> Dict[str, Any]:
    if score is None:
        payload: Dict[str, Any] = {
            "label": "unknown",
            "score_0_to_100": 50,
            "confidence": "low",
            "rationale": rationale,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
        }
    else:
        payload = {
            "label": score_to_label(score),
            "score_0_to_100": int(round(clamp(score, 0, 100))),
            "confidence": confidence,
            "rationale": rationale,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
        }
    if scope_note is not None:
        payload["scope_note"] = scope_note
    return payload


def feature_payload(name: str, raw_value: Any, score: Optional[float], refs: Sequence[str]) -> Dict[str, Any]:
    value: Dict[str, Any] = {"raw": raw_value}
    if score is not None:
        value["score_0_to_100"] = int(round(clamp(score, 0, 100)))
        value["label"] = score_to_label(score)
    return {
        "name": name,
        "value": value,
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def confidence_from_sources(n_sources: int, n_signals: int) -> str:
    if n_sources >= 2 and n_signals >= 3:
        return "high"
    if n_signals >= 2:
        return "medium"
    return "low"


class QuestionMeta:
    def __init__(self, raw: Dict[str, Any]):
        self.question_id = raw["QuestionID"]
        self.block_name = normalize_block_name(raw["BlockName"])
        self.question_ref = f"{self.block_name}::{self.question_id}"
        self.question_type = raw["QuestionType"]
        self.question_text = normalize_text(raw.get("QuestionText", ""))
        self.rows = list(raw.get("Rows", []) or [])
        self.options = list(raw.get("Options", []) or [])
        self.csv_columns = list(raw.get("csv_columns", []) or [])


def load_catalog(path: Path) -> Dict[str, QuestionMeta]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, QuestionMeta] = {}
    for raw in rows:
        meta = QuestionMeta(raw)
        out[meta.question_ref] = meta
    return out


def load_mapping(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def infer_scale_max(df: pd.DataFrame, meta: QuestionMeta) -> Optional[int]:
    if meta.question_type == "MC" and meta.options:
        return len(meta.options)
    numeric_vals: List[float] = []
    for col in meta.csv_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        series = series[series.map(math.isfinite)]
        if not series.empty:
            numeric_vals.append(float(series.max()))
    if not numeric_vals:
        return None
    return int(max(numeric_vals))


def build_scale_max_map(df: pd.DataFrame, catalog: Dict[str, QuestionMeta]) -> Dict[str, Optional[int]]:
    return {question_ref: infer_scale_max(df, meta) for question_ref, meta in catalog.items()}


def response_present(row: pd.Series, meta: QuestionMeta) -> bool:
    for col in meta.csv_columns:
        if col not in row.index:
            continue
        if not is_missing(row[col]):
            return True
    return False


def multi_select_indices(row: pd.Series, meta: QuestionMeta) -> List[int]:
    selected: List[int] = []
    for idx, col in enumerate(meta.csv_columns, start=1):
        if col not in row.index or is_missing(row[col]):
            continue
        value = safe_float(row[col])
        if value is not None and int(round(value)) == 1:
            selected.append(idx)
    return selected


def mc_choice_index(row: pd.Series, meta: QuestionMeta) -> Optional[int]:
    if not meta.csv_columns:
        return None
    if len(meta.csv_columns) == 1:
        value = row.get(meta.csv_columns[0])
        numeric = safe_float(value)
        return int(round(numeric)) if numeric is not None else None
    selected = multi_select_indices(row, meta)
    if not selected:
        return None
    return max(selected)


def option_label(meta: QuestionMeta, index: Optional[int]) -> Optional[str]:
    if index is None or not meta.options:
        return None
    if 1 <= index <= len(meta.options):
        return normalize_text(meta.options[index - 1])
    return None


def selected_option_labels(row: pd.Series, meta: QuestionMeta) -> List[str]:
    labels: List[str] = []
    for idx in multi_select_indices(row, meta):
        label = option_label(meta, idx)
        if label:
            labels.append(label)
    return labels


def matrix_value(row: pd.Series, meta: QuestionMeta, item_index: int) -> Optional[float]:
    if item_index < 1 or item_index > len(meta.csv_columns):
        return None
    col = meta.csv_columns[item_index - 1]
    return safe_float(row.get(col))


def normalized_scale_value(value: Optional[float], scale_max: Optional[int], reverse: bool = False) -> Optional[float]:
    if value is None or scale_max is None or scale_max <= 1:
        return None
    score = (value - 1.0) / float(scale_max - 1)
    score = clamp(score, 0.0, 1.0)
    if reverse:
        score = 1.0 - score
    return score


def mean_matrix_items(
    row: pd.Series,
    meta: QuestionMeta,
    scale_max: Optional[int],
    positive_items: Sequence[int],
    reverse_items: Sequence[int] = (),
) -> Optional[float]:
    values: List[float] = []
    for idx in positive_items:
        value = normalized_scale_value(matrix_value(row, meta, idx), scale_max, reverse=False)
        if value is not None:
            values.append(value)
    for idx in reverse_items:
        value = normalized_scale_value(matrix_value(row, meta, idx), scale_max, reverse=True)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def mc_scaled_choice(row: pd.Series, meta: QuestionMeta, scale_max: Optional[int], reverse: bool = False) -> Optional[float]:
    return normalized_scale_value(float(mc_choice_index(row, meta)) if mc_choice_index(row, meta) is not None else None, scale_max, reverse=reverse)


def one_hot_scaled_choice(row: pd.Series, meta: QuestionMeta) -> Optional[float]:
    choice = mc_choice_index(row, meta)
    if choice is None or not meta.options:
        return None
    return normalized_scale_value(float(choice), len(meta.options), reverse=False)


def numeric_column_value(row: pd.Series, column: str) -> Optional[float]:
    return safe_float(row.get(column))


def text_column_value(row: pd.Series, column: str) -> Optional[str]:
    value = row.get(column)
    if is_missing(value):
        return None
    return normalize_text(value)


def compute_bdi_average(row: pd.Series, catalog: Dict[str, QuestionMeta]) -> Optional[float]:
    severities: List[float] = []
    for question_ref in BDI_QIDS:
        meta = catalog[question_ref]
        choice = mc_choice_index(row, meta)
        if choice is None:
            continue
        severity = (choice - 1) / 3.0
        severities.append(severity)
    if not severities:
        return None
    return float(sum(severities) / len(severities))


def compute_anxiety_average(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID125"]
    return mean_matrix_items(row, meta, scale_max_map[meta.question_ref], list(range(1, len(meta.csv_columns) + 1)))


def compute_big_five(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Dict[str, Optional[float]]:
    meta = catalog["Personality::QID25"]
    scale_max = scale_max_map[meta.question_ref]
    return {
        "extraversion": mean_matrix_items(row, meta, scale_max, [1, 11, 16, 26, 36], [6, 21, 31]),
        "agreeableness": mean_matrix_items(row, meta, scale_max, [7, 17, 22, 32, 42], [2, 12, 27, 37]),
        "conscientiousness": mean_matrix_items(row, meta, scale_max, [3, 13, 28, 33, 38], [8, 18, 23, 43]),
        "neuroticism": mean_matrix_items(row, meta, scale_max, [4, 14, 19, 29, 39], [9, 24, 34]),
        "openness": mean_matrix_items(row, meta, scale_max, [5, 10, 15, 20, 25, 30, 40, 44], [35, 41]),
    }


def compute_need_for_cognition(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID26"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(
        row,
        meta,
        scale_max,
        [1, 2, 6, 10, 11, 13, 14, 15, 18],
        [3, 4, 5, 7, 8, 9, 12, 16, 17],
    )


def compute_values_scores(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Dict[str, Optional[float]]:
    meta = catalog["Personality::QID29"]
    scale_max = scale_max_map[meta.question_ref]
    prosocial = mean_matrix_items(row, meta, scale_max, [5, 7, 9, 11, 12, 14, 16, 17, 21])
    competitive = mean_matrix_items(row, meta, scale_max, [1, 4, 8, 10, 13, 18, 22, 24])
    return {
        "prosocial_values": prosocial,
        "competitive_values": competitive,
    }


def compute_empathy(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID232"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, [2, 3, 5, 9, 10, 11, 12, 14, 15, 16, 17], [1, 6, 7, 8, 13, 18, 19, 20])


def compute_cooperation_competition(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Dict[str, Optional[float]]:
    meta = catalog["Personality::QID233"]
    scale_max = scale_max_map[meta.question_ref]
    return {
        "cooperation_orientation": mean_matrix_items(row, meta, scale_max, [9, 10, 11, 12, 13, 14, 15, 16]),
        "competition_orientation": mean_matrix_items(row, meta, scale_max, [5, 6, 7, 8], [9, 10, 11, 12]),
        "self_reliance": mean_matrix_items(row, meta, scale_max, [1, 2, 3, 4]),
    }


def compute_social_sensitivity(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID236"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13], [4, 6])


def compute_uncertainty_aversion(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID238"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, list(range(1, len(meta.csv_columns) + 1)))


def compute_self_concept_clarity(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID237"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, [6, 11], [1, 2, 3, 4, 5, 7, 8, 9, 10, 12])


def compute_maximizing(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID239"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, list(range(1, len(meta.csv_columns) + 1)))


def compute_minimalism(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID35"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, list(range(1, len(meta.csv_columns) + 1)))


def compute_orderliness(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID30"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, [1, 2, 3, 4], [5, 6, 7, 8])


def compute_consumer_uniqueness(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Optional[float]:
    meta = catalog["Personality::QID234"]
    scale_max = scale_max_map[meta.question_ref]
    return mean_matrix_items(row, meta, scale_max, list(range(1, len(meta.csv_columns) + 1)))


def compute_spending_measures(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Dict[str, Optional[float]]:
    q31 = mc_scaled_choice(row, catalog["Personality::QID31"], scale_max_map["Personality::QID31"])
    q32 = mc_scaled_choice(row, catalog["Personality::QID32"], scale_max_map["Personality::QID32"])
    q33 = mc_scaled_choice(row, catalog["Personality::QID33"], scale_max_map["Personality::QID33"])
    q34 = mc_scaled_choice(row, catalog["Personality::QID34"], scale_max_map["Personality::QID34"])
    spendthrift = mean([q31, q32, 1.0 - q33 if q33 is not None else None, 1.0 - q34 if q34 is not None else None])
    tightwad = mean([1.0 - q31 if q31 is not None else None, q33, q34])
    spending_restraint = mean([1.0 - q32 if q32 is not None else None, q33, 1.0 - q31 if q31 is not None else None])
    return {
        "spendthrift": spendthrift,
        "tightwad": tightwad,
        "spending_restraint": spending_restraint,
    }


def compute_revenge_tendency(row: pd.Series, catalog: Dict[str, QuestionMeta]) -> Optional[float]:
    meta = catalog["Personality::QID27"]
    # QID27 appears to use True/False coded as 1/2 in the current snapshot.
    items = []
    for idx in [2, 4, 6, 8, 11, 12]:
        choice = matrix_value(row, meta, idx)
        if choice is not None:
            items.append(1.0 if int(round(choice)) == 1 else 0.0)
    forgiving = matrix_value(row, catalog["Personality::QID25"], 17)
    forgiving_norm = normalized_scale_value(forgiving, 5, reverse=False)
    if forgiving_norm is not None:
        items.append(1.0 - forgiving_norm)
    if not items:
        return None
    return float(sum(items) / len(items))


def trust_send_amount(row: pd.Series) -> Optional[float]:
    value = safe_float(row.get("QID117"))
    if value is None:
        return None
    return float(6 - int(round(value)))


def trust_return_shares(row: pd.Series) -> List[float]:
    shares: List[float] = []
    totals = {
        "QID118": 15.0,
        "QID119": 12.0,
        "QID120": 9.0,
        "QID121": 6.0,
        "QID122": 3.0,
    }
    for col, total in totals.items():
        value = safe_float(row.get(col))
        if value is None:
            continue
        returned = total - (int(round(value)) - 1)
        shares.append(returned / total if total else 0.0)
    return shares


def ultimatum_offer_to_other(row: pd.Series) -> Optional[float]:
    value = safe_float(row.get("QID224"))
    if value is None:
        return None
    return float(6 - int(round(value)))


def ultimatum_min_acceptable_to_self(row: pd.Series) -> Optional[float]:
    accepted_amounts: List[float] = []
    mapping = {
        "QID226": 4.0,
        "QID227": 3.0,
        "QID228": 2.0,
        "QID229": 1.0,
        "QID230": 0.0,
    }
    for col, amount_to_self in mapping.items():
        value = safe_float(row.get(col))
        if value is None:
            continue
        if int(round(value)) == 1:
            accepted_amounts.append(amount_to_self)
    if not accepted_amounts:
        return None
    return min(accepted_amounts)


def ultimatum_rejection_rate(row: pd.Series) -> Optional[float]:
    seen = 0
    rejected = 0
    for col in ["QID226", "QID227", "QID228", "QID229", "QID230"]:
        value = safe_float(row.get(col))
        if value is None:
            continue
        seen += 1
        if int(round(value)) == 2:
            rejected += 1
    if seen == 0:
        return None
    return rejected / float(seen)


def dictator_offer_to_other(row: pd.Series) -> Optional[float]:
    value = safe_float(row.get("QID231"))
    if value is None:
        return None
    return float(6 - int(round(value)))


def mental_accounting_score(row: pd.Series) -> Optional[float]:
    scores: List[float] = []
    for question_ref, keyed_answer in MENTAL_ACCOUNTING_KEYS.items():
        col = question_ref.split("::", 1)[1]
        value = safe_float(row.get(col))
        if value is None:
            continue
        scores.append(1.0 if int(round(value)) == keyed_answer else 0.0)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def later_choice_rate(row: pd.Series) -> Optional[float]:
    cols = [
        *[f"QID84_{i}" for i in range(4, 13)],
        *[f"QID244_{i}" for i in range(4, 14)],
        *[f"QID245_{i}" for i in range(4, 15)],
        *[f"QID246_{i}" for i in range(4, 13)],
        *[f"QID247_{i}" for i in range(4, 14)],
        *[f"QID248_{i}" for i in range(4, 15)],
    ]
    choices: List[float] = []
    for col in cols:
        value = safe_float(row.get(col))
        if value is None:
            continue
        # 1 = later/larger left option, 2 = sooner/smaller right option.
        choices.append(1.0 if int(round(value)) == 1 else 0.0)
    if not choices:
        return None
    return float(sum(choices) / len(choices))


def lottery_choice_rate(row: pd.Series, prefixes: Sequence[str]) -> Optional[float]:
    choices: List[float] = []
    for prefix in prefixes:
        for i in range(1, 15):
            col = f"{prefix}_{i}"
            value = safe_float(row.get(col))
            if value is None:
                continue
            # 1 = lottery left option, 2 = sure alternative right option.
            choices.append(1.0 if int(round(value)) == 1 else 0.0)
    if not choices:
        return None
    return float(sum(choices) / len(choices))


def numeric_answer_is_correct(value: Optional[float], target: float) -> Optional[bool]:
    if value is None:
        return None
    return math.isclose(float(value), float(target), rel_tol=1e-4, abs_tol=1e-4)


def compute_financial_literacy_accuracy(row: pd.Series) -> Optional[float]:
    scores: List[float] = []
    for question_ref, correct in FINANCIAL_LITERAL_KEYS.items():
        col = question_ref.split("::", 1)[1]
        value = safe_float(row.get(col))
        if value is None:
            continue
        scores.append(1.0 if int(round(value)) == correct else 0.0)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def compute_numeracy_accuracy(row: pd.Series) -> Optional[float]:
    scores: List[float] = []
    for question_ref, correct in NUMERACY_FREE_TEXT_KEYS.items():
        col = question_ref.split("::", 1)[1]
        if not col.endswith("_TEXT"):
            col = f"{col}_TEXT"
        value = safe_float(row.get(col))
        result = numeric_answer_is_correct(value, correct)
        if result is None:
            continue
        scores.append(1.0 if result else 0.0)
    for question_ref, correct_text in TEXT_ANSWER_KEYS.items():
        col = question_ref.split("::", 1)[1]
        if not col.endswith("_TEXT"):
            col = f"{col}_TEXT"
        value = text_column_value(row, col)
        if value is None:
            continue
        scores.append(1.0 if value.lower() == correct_text else 0.0)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def compute_crt_accuracy(row: pd.Series) -> Optional[float]:
    keys = {
        "QID49_TEXT": 5.0,
        "QID50_TEXT": 0.05,
        "QID51_TEXT": 47.0,
        "QID52_TEXT": "emily",
        "QID53_TEXT": 0.0,
        "QID54_TEXT": 2.0,
        "QID55_TEXT": 8.0,
    }
    scores: List[float] = []
    for col, target in keys.items():
        if isinstance(target, str):
            value = text_column_value(row, col)
            if value is None:
                continue
            scores.append(1.0 if value.lower() == target else 0.0)
        else:
            value = safe_float(row.get(col))
            result = numeric_answer_is_correct(value, target)
            if result is None:
                continue
            scores.append(1.0 if result else 0.0)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def compute_logic_accuracy(row: pd.Series) -> Optional[float]:
    scores: List[float] = []
    for question_ref, correct in LOGIC_KEYS.items():
        col = question_ref.split("::", 1)[1]
        value = safe_float(row.get(col))
        if value is None:
            continue
        scores.append(1.0 if int(round(value)) == correct else 0.0)

    q221_values = []
    for idx in range(1, 5):
        col = f"QID221_{idx}"
        value = safe_float(row.get(col))
        q221_values.append(1 if value is not None and int(round(value)) == 1 else 0)
    if any(q221_values):
        scores.append(1.0 if q221_values == [1, 0, 0, 1] else 0.0)

    if not scores:
        return None
    return float(sum(scores) / len(scores))


def mapped_amount(choice_index: Optional[float]) -> Optional[float]:
    if choice_index is None:
        return None
    idx = int(round(choice_index))
    if 1 <= idx <= len(WTP_WTA_AMOUNTS):
        return float(WTP_WTA_AMOUNTS[idx - 1])
    return None


def anchor_pull_high(estimate: Optional[float], truth: float, anchor: float) -> Optional[float]:
    if estimate is None:
        return None
    if anchor <= truth:
        return None
    return clamp((estimate - truth) / (anchor - truth), 0.0, 1.0)


def anchor_pull_low(estimate: Optional[float], truth: float, anchor: float) -> Optional[float]:
    if estimate is None:
        return None
    if anchor >= truth:
        return None
    return clamp((truth - estimate) / (truth - anchor), 0.0, 1.0)


def compute_anchor_susceptibility(row: pd.Series) -> Optional[float]:
    pulls: List[float] = []
    africa_high = safe_float(row.get("QID166_TEXT"))
    africa_low = safe_float(row.get("QID164_TEXT"))
    redwood_high = safe_float(row.get("QID170_TEXT"))
    redwood_low = safe_float(row.get("QID168_TEXT"))
    if africa_high is not None:
        pulls.append(anchor_pull_high(africa_high, AFRICA_TRUE_COUNTRIES, AFRICA_HIGH_ANCHOR))
    if africa_low is not None:
        pulls.append(anchor_pull_low(africa_low, AFRICA_TRUE_COUNTRIES, AFRICA_LOW_ANCHOR))
    if redwood_high is not None:
        pulls.append(anchor_pull_high(redwood_high, REDWOOD_TRUE_FEET, REDWOOD_HIGH_ANCHOR))
    if redwood_low is not None:
        pulls.append(anchor_pull_low(redwood_low, REDWOOD_TRUE_FEET, REDWOOD_LOW_ANCHOR))
    pulls = [pull for pull in pulls if pull is not None]
    if not pulls:
        return None
    return float(sum(pulls) / len(pulls))


def compute_framing_susceptibility(row: pd.Series, mental_accounting: Optional[float]) -> Optional[float]:
    calc = safe_float(row.get("QID183"))
    jacket = safe_float(row.get("QID184"))
    relative_gap = None
    if calc is not None and jacket is not None:
        calc_yes = 1.0 if int(round(calc)) == 1 else 0.0
        jacket_yes = 1.0 if int(round(jacket)) == 1 else 0.0
        if calc_yes == 1.0 and jacket_yes == 0.0:
            relative_gap = 1.0
        elif calc_yes == jacket_yes:
            relative_gap = 0.0
        else:
            relative_gap = 0.5

    wta_rank = safe_float(row.get("QID190"))
    wtp_rank = safe_float(row.get("QID189"))
    gap = None
    if wta_rank is not None and wtp_rank is not None:
        gap = clamp((wta_rank - wtp_rank + 9.0) / 18.0, 0.0, 1.0)

    parts = [value for value in [relative_gap, gap, mental_accounting] if value is not None]
    if not parts:
        return None
    return float(sum(parts) / len(parts))


def compute_purchase_yes_rate(row: pd.Series) -> Optional[float]:
    choices: List[float] = []
    for idx in range(1, 41):
        col = f"QID9_{idx}"
        value = safe_float(row.get(col))
        if value is None:
            continue
        choices.append(1.0 if int(round(value)) == 1 else 0.0)
    if not choices:
        return None
    return float(sum(choices) / len(choices))


def demographic_label(row: pd.Series, meta: QuestionMeta) -> Optional[str]:
    choice = mc_choice_index(row, meta)
    label = option_label(meta, choice)
    return label


def education_harmonized(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    if label in {"Less than high school", "High school graduate", "Some college, no degree"}:
        return "high school"
    if label in {"Associate's degree", "College graduate/some postgrad"}:
        return "college/postsecondary"
    if label == "Postgraduate":
        return "postgraduate"
    return None


def row_item_excerpt(meta: QuestionMeta, item_index: int, value: Optional[float], scale_max: Optional[int]) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    prompt = meta.rows[item_index - 1] if 0 < item_index <= len(meta.rows) else meta.question_text
    response = f"rating {int(round(value))}"
    if scale_max is not None:
        response += f"/{scale_max}"
    return {
        "question_ref": meta.question_ref,
        "prompt_excerpt": text_excerpt(prompt, 160),
        "response_excerpt": response,
    }


def selected_option_excerpt(meta: QuestionMeta, labels: Sequence[str]) -> Optional[Dict[str, str]]:
    if not labels:
        return None
    return {
        "question_ref": meta.question_ref,
        "prompt_excerpt": text_excerpt(meta.question_text, 160),
        "response_excerpt": text_excerpt(" | ".join(labels), 160),
    }


def single_option_excerpt(meta: QuestionMeta, choice: Optional[int]) -> Optional[Dict[str, str]]:
    label = option_label(meta, choice)
    if label is None:
        return None
    return {
        "question_ref": meta.question_ref,
        "prompt_excerpt": text_excerpt(meta.question_text, 160),
        "response_excerpt": text_excerpt(label, 160),
    }


def text_excerpt_item(meta: QuestionMeta, text: Optional[str], prompt_override: Optional[str] = None) -> Optional[Dict[str, str]]:
    if text is None:
        return None
    return {
        "question_ref": meta.question_ref,
        "prompt_excerpt": text_excerpt(prompt_override or meta.question_text, 160),
        "response_excerpt": text_excerpt(text, 160),
    }


def add_if_present(items: List[Dict[str, str]], item: Optional[Dict[str, str]]) -> None:
    if item is not None:
        items.append(item)


def build_background_context(row: pd.Series, catalog: Dict[str, QuestionMeta], answered_refs: Sequence[str]) -> Dict[str, Any]:
    question_refs = [ref for ref in answered_refs if ref.startswith("Demographics::")]
    features: List[Dict[str, Any]] = []
    for qid, name in [
        ("QID11", "region"),
        ("QID12", "sex_assigned_at_birth"),
        ("QID13", "age_bracket"),
        ("QID14", "education_completed_raw"),
        ("QID15", "race_or_origin"),
        ("QID16", "citizenship"),
        ("QID17", "relationship_status"),
        ("QID18", "religion"),
        ("QID19", "religious_service_attendance"),
        ("QID20", "party_identification"),
        ("QID21", "income_bracket"),
        ("QID22", "political_views"),
        ("QID23", "household_size"),
        ("QID24", "employment_status"),
    ]:
        meta = catalog[f"Demographics::{qid}"]
        label = demographic_label(row, meta)
        if label is not None:
            features.append(feature_payload(name, label, None, [meta.question_ref]))
    education_label = demographic_label(row, catalog["Demographics::QID14"])
    harmonized = education_harmonized(education_label)
    if harmonized is not None:
        features.append(feature_payload("education_completed_harmonized", harmonized, None, ["Demographics::QID14"]))
    return {
        "question_refs": question_refs,
        "harmonized_features": features,
        "context_notes": [
            "Demographics are retained as participant context and should not be over-interpreted as deep trait evidence.",
            "Twin records sex assigned at birth rather than gender identity.",
        ],
    }


def build_question_refs_by_subsection(
    answered_refs: Sequence[str],
    mapping_rows: Sequence[Dict[str, str]],
) -> Dict[str, List[str]]:
    by_subsection: Dict[str, List[str]] = {}
    mapping_by_ref = {row["question_ref"]: row for row in mapping_rows}
    for ref in answered_refs:
        mapping = mapping_by_ref.get(ref)
        if not mapping:
            continue
        subsection = mapping["profile_subsection"]
        by_subsection.setdefault(subsection, []).append(ref)
    for subsection in by_subsection:
        by_subsection[subsection] = sorted(by_subsection[subsection])
    return by_subsection


def block_with_default_features(question_refs: Sequence[str], features: List[Dict[str, Any]], diagnostics: List[Dict[str, str]]) -> Dict[str, Any]:
    all_features = [
        feature_payload("answered_question_count", len(question_refs), None, list(question_refs))
    ]
    all_features.extend(features)
    return {
        "question_refs": list(question_refs),
        "summary_features": all_features,
        "diagnostic_items": diagnostics,
    }


def compute_feature_bank(row: pd.Series, catalog: Dict[str, QuestionMeta], scale_max_map: Dict[str, Optional[int]]) -> Dict[str, Any]:
    big_five = compute_big_five(row, catalog, scale_max_map)
    values_scores = compute_values_scores(row, catalog, scale_max_map)
    coop_comp = compute_cooperation_competition(row, catalog, scale_max_map)
    spending = compute_spending_measures(row, catalog, scale_max_map)

    trust_send = trust_send_amount(row)
    trust_returns = trust_return_shares(row)
    trust_return_mean = mean(trust_returns)
    ultimatum_min_accept = ultimatum_min_acceptable_to_self(row)

    mental_accounting = mental_accounting_score(row)
    patience = later_choice_rate(row)
    risk_gains = lottery_choice_rate(row, ["QID250", "QID251", "QID252"])
    risk_losses = lottery_choice_rate(row, ["QID276", "QID277", "QID278", "QID279"])

    financial_lit = compute_financial_literacy_accuracy(row)
    numeracy = compute_numeracy_accuracy(row)
    crt = compute_crt_accuracy(row)
    logic = compute_logic_accuracy(row)

    anchor = compute_anchor_susceptibility(row)
    framing = compute_framing_susceptibility(row, mental_accounting)
    purchase_yes_rate = compute_purchase_yes_rate(row)

    q190_amount = mapped_amount(safe_float(row.get("QID190")))
    q189_amount = mapped_amount(safe_float(row.get("QID189")))
    q191_amount = mapped_amount(safe_float(row.get("QID191")))

    return {
        **big_five,
        **values_scores,
        **coop_comp,
        **spending,
        "need_for_cognition": compute_need_for_cognition(row, catalog, scale_max_map),
        "empathy": compute_empathy(row, catalog, scale_max_map),
        "social_sensitivity": compute_social_sensitivity(row, catalog, scale_max_map),
        "uncertainty_aversion": compute_uncertainty_aversion(row, catalog, scale_max_map),
        "depressive_affect": compute_bdi_average(row, catalog),
        "anxiety_symptoms": compute_anxiety_average(row, catalog, scale_max_map),
        "self_concept_clarity": compute_self_concept_clarity(row, catalog, scale_max_map),
        "maximizing_tendency": compute_maximizing(row, catalog, scale_max_map),
        "minimalism": compute_minimalism(row, catalog, scale_max_map),
        "orderliness": compute_orderliness(row, catalog, scale_max_map),
        "consumer_uniqueness": compute_consumer_uniqueness(row, catalog, scale_max_map),
        "revenge_tendency": compute_revenge_tendency(row, catalog),
        "trust_send_amount": trust_send,
        "trust_send_share": trust_send / 5.0 if trust_send is not None else None,
        "trust_return_share_mean": trust_return_mean,
        "trust_return_share_min": min(trust_returns) if trust_returns else None,
        "trust_return_share_max": max(trust_returns) if trust_returns else None,
        "ultimatum_offer_to_other": ultimatum_offer_to_other(row),
        "ultimatum_offer_share": ultimatum_offer_to_other(row) / 5.0 if ultimatum_offer_to_other(row) is not None else None,
        "ultimatum_min_acceptable_to_self": ultimatum_min_accept,
        "ultimatum_rejection_rate": ultimatum_rejection_rate(row),
        "dictator_offer_to_other": dictator_offer_to_other(row),
        "dictator_offer_share": dictator_offer_to_other(row) / 5.0 if dictator_offer_to_other(row) is not None else None,
        "mental_accounting": mental_accounting,
        "patience_later_choice_rate": patience,
        "risk_tolerance_gains": risk_gains,
        "risk_tolerance_losses": risk_losses,
        "financial_literacy_accuracy": financial_lit,
        "numeracy_accuracy": numeracy,
        "cognitive_reflection_accuracy": crt,
        "logical_reasoning_accuracy": logic,
        "cognitive_self_estimate": safe_float(row.get("QID123_TEXT")),
        "cognitive_population_estimate": safe_float(row.get("QID124_TEXT")),
        "cognitive_self_minus_population": (
            safe_float(row.get("QID123_TEXT")) - safe_float(row.get("QID124_TEXT"))
            if safe_float(row.get("QID123_TEXT")) is not None and safe_float(row.get("QID124_TEXT")) is not None
            else None
        ),
        "anchor_susceptibility": anchor,
        "framing_susceptibility": framing,
        "search_willingness": mean(
            [
                1.0 if safe_float(row.get("QID183")) == 1 else 0.0 if safe_float(row.get("QID183")) is not None else None,
                1.0 if safe_float(row.get("QID184")) == 1 else 0.0 if safe_float(row.get("QID184")) is not None else None,
            ]
        ),
        "absolute_relative_gap": (
            (1.0 if safe_float(row.get("QID183")) == 1 else 0.0)
            - (1.0 if safe_float(row.get("QID184")) == 1 else 0.0)
            if safe_float(row.get("QID183")) is not None and safe_float(row.get("QID184")) is not None
            else None
        ),
        "allais_certainty_pattern": (
            1.0
            if safe_float(row.get("QID192")) == 1 and safe_float(row.get("QID193")) == 2
            else 0.0
            if safe_float(row.get("QID192")) is not None and safe_float(row.get("QID193")) is not None
            else None
        ),
        "ratio_bias_correct_small_tray": (
            1.0 if safe_float(row.get("QID196")) == 1 else 0.0 if safe_float(row.get("QID196")) is not None else None
        ),
        "vaccine_acceptance": mc_scaled_choice(row, catalog["Non-experimental heuristics and biases::QID291"], scale_max_map["Non-experimental heuristics and biases::QID291"]),
        "wta_amount": q190_amount,
        "wtp_certainty_amount": q189_amount,
        "wtp_risk_reduction_amount": q191_amount,
        "purchase_yes_rate": purchase_yes_rate,
        "public_support_estimate_mean": mean([safe_float(row.get(f"QID290_{suffix}")) for suffix in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]]),
        "benefit_perception_mean": mean([safe_float(row.get(f"QID288_{i}")) for i in range(1, 5)]),
        "risk_perception_mean": mean([safe_float(row.get(f"QID289_{i}")) for i in range(1, 5)]),
        "aspire_text": text_column_value(row, "QID268_TEXT"),
        "ought_text": text_column_value(row, "QID269_TEXT"),
        "actual_text": text_column_value(row, "QID270_TEXT"),
        "forward_flow_words": [text_column_value(row, f"QID10_{i}") for i in range(1, 21) if text_column_value(row, f"QID10_{i}")],
    }


def make_personality_block(
    row: pd.Series,
    catalog: Dict[str, QuestionMeta],
    scale_max_map: Dict[str, Optional[int]],
    question_refs: Sequence[str],
    features: Dict[str, Any],
) -> Dict[str, Any]:
    payloads = [
        feature_payload("big_five_extraversion", round((features["extraversion"] or 0) * 100, 2) if features["extraversion"] is not None else None, features["extraversion"] * 100 if features["extraversion"] is not None else None, ["Personality::QID25"]),
        feature_payload("big_five_agreeableness", round((features["agreeableness"] or 0) * 100, 2) if features["agreeableness"] is not None else None, features["agreeableness"] * 100 if features["agreeableness"] is not None else None, ["Personality::QID25"]),
        feature_payload("big_five_conscientiousness", round((features["conscientiousness"] or 0) * 100, 2) if features["conscientiousness"] is not None else None, features["conscientiousness"] * 100 if features["conscientiousness"] is not None else None, ["Personality::QID25"]),
        feature_payload("big_five_neuroticism", round((features["neuroticism"] or 0) * 100, 2) if features["neuroticism"] is not None else None, features["neuroticism"] * 100 if features["neuroticism"] is not None else None, ["Personality::QID25"]),
        feature_payload("big_five_openness", round((features["openness"] or 0) * 100, 2) if features["openness"] is not None else None, features["openness"] * 100 if features["openness"] is not None else None, ["Personality::QID25"]),
        feature_payload("need_for_cognition", round((features["need_for_cognition"] or 0) * 100, 2) if features["need_for_cognition"] is not None else None, features["need_for_cognition"] * 100 if features["need_for_cognition"] is not None else None, ["Personality::QID26"]),
        feature_payload("prosocial_values", round((features["prosocial_values"] or 0) * 100, 2) if features["prosocial_values"] is not None else None, features["prosocial_values"] * 100 if features["prosocial_values"] is not None else None, ["Personality::QID29"]),
        feature_payload("competitive_values", round((features["competitive_values"] or 0) * 100, 2) if features["competitive_values"] is not None else None, features["competitive_values"] * 100 if features["competitive_values"] is not None else None, ["Personality::QID29"]),
        feature_payload("empathy", round((features["empathy"] or 0) * 100, 2) if features["empathy"] is not None else None, features["empathy"] * 100 if features["empathy"] is not None else None, ["Personality::QID232"]),
        feature_payload("cooperation_orientation", round((features["cooperation_orientation"] or 0) * 100, 2) if features["cooperation_orientation"] is not None else None, features["cooperation_orientation"] * 100 if features["cooperation_orientation"] is not None else None, ["Personality::QID233"]),
        feature_payload("competition_orientation", round((features["competition_orientation"] or 0) * 100, 2) if features["competition_orientation"] is not None else None, features["competition_orientation"] * 100 if features["competition_orientation"] is not None else None, ["Personality::QID233"]),
        feature_payload("social_sensitivity", round((features["social_sensitivity"] or 0) * 100, 2) if features["social_sensitivity"] is not None else None, features["social_sensitivity"] * 100 if features["social_sensitivity"] is not None else None, ["Personality::QID236"]),
        feature_payload("uncertainty_aversion", round((features["uncertainty_aversion"] or 0) * 100, 2) if features["uncertainty_aversion"] is not None else None, features["uncertainty_aversion"] * 100 if features["uncertainty_aversion"] is not None else None, ["Personality::QID238"]),
        feature_payload("depressive_affect", round((features["depressive_affect"] or 0) * 100, 2) if features["depressive_affect"] is not None else None, features["depressive_affect"] * 100 if features["depressive_affect"] is not None else None, BDI_QIDS),
        feature_payload("anxiety_symptoms", round((features["anxiety_symptoms"] or 0) * 100, 2) if features["anxiety_symptoms"] is not None else None, features["anxiety_symptoms"] * 100 if features["anxiety_symptoms"] is not None else None, ["Personality::QID125"]),
        feature_payload("self_concept_clarity", round((features["self_concept_clarity"] or 0) * 100, 2) if features["self_concept_clarity"] is not None else None, features["self_concept_clarity"] * 100 if features["self_concept_clarity"] is not None else None, ["Personality::QID237"]),
        feature_payload("maximizing_tendency", round((features["maximizing_tendency"] or 0) * 100, 2) if features["maximizing_tendency"] is not None else None, features["maximizing_tendency"] * 100 if features["maximizing_tendency"] is not None else None, ["Personality::QID239"]),
        feature_payload("minimalism", round((features["minimalism"] or 0) * 100, 2) if features["minimalism"] is not None else None, features["minimalism"] * 100 if features["minimalism"] is not None else None, ["Personality::QID35"]),
        feature_payload("spendthrift_tendency", round((features["spendthrift"] or 0) * 100, 2) if features["spendthrift"] is not None else None, features["spendthrift"] * 100 if features["spendthrift"] is not None else None, ["Personality::QID31", "Personality::QID32", "Personality::QID33", "Personality::QID34"]),
        feature_payload("tightwad_tendency", round((features["tightwad"] or 0) * 100, 2) if features["tightwad"] is not None else None, features["tightwad"] * 100 if features["tightwad"] is not None else None, ["Personality::QID31", "Personality::QID32", "Personality::QID33", "Personality::QID34"]),
        feature_payload("orderliness", round((features["orderliness"] or 0) * 100, 2) if features["orderliness"] is not None else None, features["orderliness"] * 100 if features["orderliness"] is not None else None, ["Personality::QID30"]),
    ]
    payloads = [payload for payload in payloads if payload["value"]["raw"] is not None]

    diagnostics: List[Dict[str, str]] = []
    q25 = catalog["Personality::QID25"]
    for idx in [7, 22, 32, 42, 12, 37]:
        add_if_present(diagnostics, row_item_excerpt(q25, idx, matrix_value(row, q25, idx), scale_max_map[q25.question_ref]))
    q233 = catalog["Personality::QID233"]
    for idx in [5, 6, 7, 10, 11, 12]:
        add_if_present(diagnostics, row_item_excerpt(q233, idx, matrix_value(row, q233, idx), scale_max_map[q233.question_ref]))
    q238 = catalog["Personality::QID238"]
    for idx in [1, 7, 8, 15]:
        add_if_present(diagnostics, row_item_excerpt(q238, idx, matrix_value(row, q238, idx), scale_max_map[q238.question_ref]))
    bdi = catalog["Personality::QID126"]
    add_if_present(diagnostics, selected_option_excerpt(bdi, selected_option_labels(row, bdi)))
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_social_game_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = []
    if features["trust_send_amount"] is not None:
        payloads.append(feature_payload("trust_send_amount", float(features["trust_send_amount"]), features["trust_send_share"] * 100 if features["trust_send_share"] is not None else None, ["Economic preferences::QID117"]))
    if features["trust_return_share_mean"] is not None:
        payloads.append(feature_payload("trust_return_share_mean", round(features["trust_return_share_mean"], 4), features["trust_return_share_mean"] * 100, ["Economic preferences::QID118", "Economic preferences::QID119", "Economic preferences::QID120", "Economic preferences::QID121", "Economic preferences::QID122"]))
    if features["ultimatum_offer_to_other"] is not None:
        payloads.append(feature_payload("ultimatum_offer_to_other", float(features["ultimatum_offer_to_other"]), features["ultimatum_offer_share"] * 100 if features["ultimatum_offer_share"] is not None else None, ["Economic preferences::QID224"]))
    if features["ultimatum_min_acceptable_to_self"] is not None:
        payloads.append(feature_payload("ultimatum_min_acceptable_to_self", float(features["ultimatum_min_acceptable_to_self"]), (features["ultimatum_min_acceptable_to_self"] / 4.0) * 100, ["Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230"]))
    if features["ultimatum_rejection_rate"] is not None:
        payloads.append(feature_payload("ultimatum_rejection_rate", round(features["ultimatum_rejection_rate"], 4), features["ultimatum_rejection_rate"] * 100, ["Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230"]))
    if features["dictator_offer_to_other"] is not None:
        payloads.append(feature_payload("dictator_offer_to_other", float(features["dictator_offer_to_other"]), features["dictator_offer_share"] * 100 if features["dictator_offer_share"] is not None else None, ["Economic preferences::QID231"]))

    diagnostics: List[Dict[str, str]] = []
    for ref in ["Economic preferences::QID117", "Economic preferences::QID224", "Economic preferences::QID231"]:
        add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    for ref in ["Economic preferences::QID118", "Economic preferences::QID119", "Economic preferences::QID120", "Economic preferences::QID121", "Economic preferences::QID122"]:
        add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    for ref in ["Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230"]:
        add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    for ref, prompt in [
        ("Economic preferences::QID271", "Trust-game thoughts"),
        ("Economic preferences::QID272", "Trust reciprocity thoughts"),
        ("Economic preferences::QID275", "Dictator-game thoughts"),
    ]:
        meta = catalog[ref]
        texts = []
        for col in meta.csv_columns:
            value = text_column_value(row, col)
            if value:
                texts.append(value)
        if texts:
            diagnostics.append(
                {
                    "question_ref": ref,
                    "prompt_excerpt": prompt,
                    "response_excerpt": text_excerpt(" | ".join(texts), 180),
                }
            )
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_non_social_econ_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = []
    if features["mental_accounting"] is not None:
        payloads.append(feature_payload("mental_accounting_endorsement_rate", round(features["mental_accounting"], 4), features["mental_accounting"] * 100, list(MENTAL_ACCOUNTING_KEYS.keys())))
    if features["patience_later_choice_rate"] is not None:
        payloads.append(feature_payload("later_choice_rate", round(features["patience_later_choice_rate"], 4), features["patience_later_choice_rate"] * 100, ["Economic preferences::QID84", "Economic preferences::QID244", "Economic preferences::QID245", "Economic preferences::QID246", "Economic preferences::QID247", "Economic preferences::QID248"]))
    if features["risk_tolerance_gains"] is not None:
        payloads.append(feature_payload("lottery_choice_rate_gains", round(features["risk_tolerance_gains"], 4), features["risk_tolerance_gains"] * 100, ["Economic preferences::QID250", "Economic preferences::QID251", "Economic preferences::QID252"]))
    if features["risk_tolerance_losses"] is not None:
        payloads.append(feature_payload("lottery_choice_rate_losses", round(features["risk_tolerance_losses"], 4), features["risk_tolerance_losses"] * 100, ["Economic preferences::QID276", "Economic preferences::QID277", "Economic preferences::QID278", "Economic preferences::QID279"]))
    diagnostics: List[Dict[str, str]] = []
    for ref in ["Economic preferences::QID149", "Economic preferences::QID150", "Economic preferences::QID151", "Economic preferences::QID152"]:
        add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    add_if_present(diagnostics, row_item_excerpt(catalog["Economic preferences::QID84"], 1, matrix_value(row, catalog["Economic preferences::QID84"], 1), 2))
    add_if_present(diagnostics, row_item_excerpt(catalog["Economic preferences::QID84"], len(catalog["Economic preferences::QID84"].csv_columns), matrix_value(row, catalog["Economic preferences::QID84"], len(catalog["Economic preferences::QID84"].csv_columns)), 2))
    add_if_present(diagnostics, row_item_excerpt(catalog["Economic preferences::QID250"], 1, matrix_value(row, catalog["Economic preferences::QID250"], 1), 2))
    add_if_present(diagnostics, row_item_excerpt(catalog["Economic preferences::QID276"], len(catalog["Economic preferences::QID276"].csv_columns), matrix_value(row, catalog["Economic preferences::QID276"], len(catalog["Economic preferences::QID276"].csv_columns)), 2))
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_cognitive_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = []
    if features["financial_literacy_accuracy"] is not None:
        payloads.append(feature_payload("financial_literacy_accuracy", round(features["financial_literacy_accuracy"], 4), features["financial_literacy_accuracy"] * 100, list(FINANCIAL_LITERAL_KEYS.keys())))
    if features["numeracy_accuracy"] is not None:
        payloads.append(feature_payload("numeracy_accuracy", round(features["numeracy_accuracy"], 4), features["numeracy_accuracy"] * 100, list(NUMERACY_FREE_TEXT_KEYS.keys()) + list(TEXT_ANSWER_KEYS.keys())))
    if features["cognitive_reflection_accuracy"] is not None:
        payloads.append(feature_payload("cognitive_reflection_accuracy", round(features["cognitive_reflection_accuracy"], 4), features["cognitive_reflection_accuracy"] * 100, ["Cognitive tests::QID49", "Cognitive tests::QID50", "Cognitive tests::QID51", "Cognitive tests::QID52", "Cognitive tests::QID53", "Cognitive tests::QID54", "Cognitive tests::QID55"]))
    if features["logical_reasoning_accuracy"] is not None:
        payloads.append(feature_payload("logical_reasoning_accuracy", round(features["logical_reasoning_accuracy"], 4), features["logical_reasoning_accuracy"] * 100, list(LOGIC_KEYS.keys()) + ["Cognitive tests::QID221"]))
    if features["cognitive_self_estimate"] is not None:
        payloads.append(feature_payload("self_estimated_cognitive_score", features["cognitive_self_estimate"], None, ["Cognitive tests::QID123"]))
    if features["cognitive_population_estimate"] is not None:
        payloads.append(feature_payload("estimated_population_cognitive_score", features["cognitive_population_estimate"], None, ["Cognitive tests::QID124"]))
    if features["cognitive_self_minus_population"] is not None:
        payloads.append(feature_payload("self_minus_population_estimate", features["cognitive_self_minus_population"], None, ["Cognitive tests::QID123", "Cognitive tests::QID124"]))
    diagnostics: List[Dict[str, str]] = []
    for qref, col in [
        ("Cognitive tests::QID50", "QID50_TEXT"),
        ("Cognitive tests::QID53", "QID53_TEXT"),
        ("Cognitive tests::QID54", "QID54_TEXT"),
        ("Cognitive tests::QID55", "QID55_TEXT"),
    ]:
        add_if_present(diagnostics, text_excerpt_item(catalog[qref], text_column_value(row, col)))
    for qref in ["Cognitive tests::QID217", "Cognitive tests::QID274", "Cognitive tests::QID268", "Cognitive tests::QID278"]:
        add_if_present(diagnostics, single_option_excerpt(catalog[qref], mc_choice_index(row, catalog[qref])))
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_heuristics_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = []
    if features["search_willingness"] is not None:
        payloads.append(feature_payload("search_willingness", round(features["search_willingness"], 4), features["search_willingness"] * 100, ["Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184"]))
    if features["absolute_relative_gap"] is not None:
        payloads.append(feature_payload("absolute_relative_gap", round(features["absolute_relative_gap"], 4), None, ["Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184"]))
    if features["allais_certainty_pattern"] is not None:
        payloads.append(feature_payload("allais_certainty_pattern", int(features["allais_certainty_pattern"]), features["allais_certainty_pattern"] * 100, ["Allais Form 1::QID192", "Allais Form 2::QID193"]))
    if features["ratio_bias_correct_small_tray"] is not None:
        payloads.append(feature_payload("ratio_bias_small_tray_choice", int(features["ratio_bias_correct_small_tray"]), features["ratio_bias_correct_small_tray"] * 100, ["Non-experimental heuristics and biases::QID196"]))
    if features["anchor_susceptibility"] is not None:
        payloads.append(feature_payload("anchor_pull_average", round(features["anchor_susceptibility"], 4), features["anchor_susceptibility"] * 100, ["Anchoring - African countries high::QID166", "Anchoring - African countries low::QID164", "Anchoring - redwood high::QID170", "Anchoring - redwood low::QID168"]))
    if features["framing_susceptibility"] is not None:
        payloads.append(feature_payload("framing_reference_sensitivity", round(features["framing_susceptibility"], 4), features["framing_susceptibility"] * 100, ["Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184", "WTA/WTP Thaler problem - WTP certainty::QID189", "WTA/WTP Thaler problem - WTA certainty::QID190"]))
    if features["vaccine_acceptance"] is not None:
        payloads.append(feature_payload("vaccine_acceptance", round(features["vaccine_acceptance"], 4), features["vaccine_acceptance"] * 100, ["Non-experimental heuristics and biases::QID291"]))
    if features["public_support_estimate_mean"] is not None:
        payloads.append(feature_payload("public_support_estimate_mean", round(features["public_support_estimate_mean"], 2), None, ["Non-experimental heuristics and biases::QID290"]))
    diagnostics: List[Dict[str, str]] = []
    for ref in [
        "Absolute vs. relative - calculator::QID183",
        "Absolute vs. relative - jacket::QID184",
        "Allais Form 1::QID192",
        "Allais Form 2::QID193",
        "Non-experimental heuristics and biases::QID196",
        "Non-experimental heuristics and biases::QID291",
    ]:
        add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    for ref, col in [
        ("Anchoring - African countries high::QID166", "QID166_TEXT"),
        ("Anchoring - African countries low::QID164", "QID164_TEXT"),
        ("Anchoring - redwood high::QID170", "QID170_TEXT"),
        ("Anchoring - redwood low::QID168", "QID168_TEXT"),
    ]:
        value = text_column_value(row, col)
        add_if_present(diagnostics, text_excerpt_item(catalog[ref], value))
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_pricing_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = []
    if features["purchase_yes_rate"] is not None:
        payloads.append(feature_payload("purchase_yes_rate", round(features["purchase_yes_rate"], 4), features["purchase_yes_rate"] * 100, [f"Product Preferences - Pricing::QID9_{i}" for i in range(1, 41)]))
    diagnostics: List[Dict[str, str]] = []
    for idx in [1, 2, 9, 20, 33]:
        ref = f"Product Preferences - Pricing::QID9_{idx}"
        if ref in catalog:
            add_if_present(diagnostics, single_option_excerpt(catalog[ref], mc_choice_index(row, catalog[ref])))
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def build_open_text_block(row: pd.Series, catalog: Dict[str, QuestionMeta], question_refs: Sequence[str], features: Dict[str, Any]) -> Dict[str, Any]:
    payloads = [
        feature_payload("aspire_text_present", bool(features["aspire_text"]), None, ["Personality::QID268"]),
        feature_payload("ought_text_present", bool(features["ought_text"]), None, ["Personality::QID269"]),
        feature_payload("actual_text_present", bool(features["actual_text"]), None, ["Personality::QID270"]),
        feature_payload("forward_flow_word_count", len(features["forward_flow_words"]), None, ["Forward Flow::QID10"]),
    ]
    diagnostics: List[Dict[str, str]] = []
    add_if_present(diagnostics, text_excerpt_item(catalog["Personality::QID268"], features["aspire_text"], "Aspired self"))
    add_if_present(diagnostics, text_excerpt_item(catalog["Personality::QID269"], features["ought_text"], "Ought self"))
    add_if_present(diagnostics, text_excerpt_item(catalog["Personality::QID270"], features["actual_text"], "Actual self"))
    if features["forward_flow_words"]:
        diagnostics.append(
            {
                "question_ref": "Forward Flow::QID10",
                "prompt_excerpt": "Forward-flow free association",
                "response_excerpt": text_excerpt(" -> ".join(features["forward_flow_words"][:10]), 180),
            }
        )
    return block_with_default_features(question_refs, payloads, diagnostics[:10])


def weighted_mean(parts: Sequence[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in parts:
        if value is None:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{round(value * 100, 1)}"


def build_derived_dimensions(features: Dict[str, Any]) -> Dict[str, Any]:
    trustingness = weighted_mean(
        [
            (features["trust_send_share"], 0.45),
            (features["agreeableness"], 0.20),
            (features["prosocial_values"], 0.15),
            (features["cooperation_orientation"], 0.10),
            (1.0 - features["uncertainty_aversion"] if features["uncertainty_aversion"] is not None else None, 0.10),
        ]
    )
    reciprocity = weighted_mean(
        [
            (features["trust_return_share_mean"], 0.45),
            (features["cooperation_orientation"], 0.20),
            (1.0 - features["dictator_offer_share"] if features["dictator_offer_share"] is not None else None, 0.05),
            (features["ultimatum_rejection_rate"], 0.10),
            (features["prosocial_values"], 0.20),
        ]
    )
    fairness_enforcement = weighted_mean(
        [
            ((features["ultimatum_min_acceptable_to_self"] / 4.0) if features["ultimatum_min_acceptable_to_self"] is not None else None, 0.55),
            (features["revenge_tendency"], 0.20),
            (1.0 - features["agreeableness"] if features["agreeableness"] is not None else None, 0.10),
            (features["competitive_values"], 0.15),
        ]
    )
    altruistic_sharing = weighted_mean(
        [
            (features["dictator_offer_share"], 0.45),
            (features["trust_send_share"], 0.20),
            (features["prosocial_values"], 0.20),
            (features["agreeableness"], 0.15),
        ]
    )
    exploitation_caution = weighted_mean(
        [
            (1.0 - features["trust_send_share"] if features["trust_send_share"] is not None else None, 0.35),
            ((features["ultimatum_min_acceptable_to_self"] / 4.0) if features["ultimatum_min_acceptable_to_self"] is not None else None, 0.20),
            (features["uncertainty_aversion"], 0.20),
            (features["self_reliance"], 0.15),
            (features["revenge_tendency"], 0.10),
        ]
    )

    patience = features["patience_later_choice_rate"]
    risk_gains = features["risk_tolerance_gains"]
    risk_losses = features["risk_tolerance_losses"]
    mental_accounting = features["mental_accounting"]
    cognitive_reflection = features["cognitive_reflection_accuracy"]
    numeracy = weighted_mean(
        [
            (features["financial_literacy_accuracy"], 0.45),
            (features["numeracy_accuracy"], 0.55),
        ]
    )
    logical_reasoning = features["logical_reasoning_accuracy"]
    anchor_sus = features["anchor_susceptibility"]
    framing_sus = features["framing_susceptibility"]

    price_sensitivity = weighted_mean(
        [
            (1.0 - features["purchase_yes_rate"] if features["purchase_yes_rate"] is not None else None, 0.30),
            (features["search_willingness"], 0.25),
            (features["tightwad"], 0.30),
            (1.0 - features["spendthrift"] if features["spendthrift"] is not None else None, 0.15),
        ]
    )
    willingness_to_search = weighted_mean(
        [
            (features["search_willingness"], 0.60),
            (features["maximizing_tendency"], 0.25),
            (features["consumer_uniqueness"], 0.15),
        ]
    )
    reference_dependence = weighted_mean(
        [
            (features["framing_susceptibility"], 0.40),
            (features["mental_accounting"], 0.35),
            (
                clamp(((math.log10(features["wta_amount"]) - math.log10(features["wtp_certainty_amount"])) + 6.0) / 12.0, 0.0, 1.0)
                if features["wta_amount"] is not None and features["wtp_certainty_amount"] is not None
                else None,
                0.25,
            ),
        ]
    )
    purchase_inhibition = weighted_mean(
        [
            (features["tightwad"], 0.45),
            (1.0 - features["purchase_yes_rate"] if features["purchase_yes_rate"] is not None else None, 0.20),
            (features["spending_restraint"], 0.35),
        ]
    )

    empathy = weighted_mean(
        [
            (features["empathy"], 0.70),
            (features["social_sensitivity"], 0.30),
        ]
    )
    cooperation_orientation = weighted_mean(
        [
            (features["cooperation_orientation"], 0.35),
            (features["agreeableness"], 0.25),
            (features["prosocial_values"], 0.25),
            (features["trust_send_share"], 0.15),
        ]
    )
    competition_orientation = weighted_mean(
        [
            (features["competition_orientation"], 0.50),
            (features["competitive_values"], 0.35),
            (1.0 - features["agreeableness"] if features["agreeableness"] is not None else None, 0.15),
        ]
    )
    uncertainty_aversion = features["uncertainty_aversion"]
    depressive_affect = weighted_mean(
        [
            (features["depressive_affect"], 0.75),
            (features["anxiety_symptoms"], 0.25),
        ]
    )
    spending_self_control = weighted_mean(
        [
            (features["spending_restraint"], 0.40),
            (features["orderliness"], 0.25),
            (features["conscientiousness"], 0.25),
            (features["minimalism"], 0.10),
        ]
    )

    return {
        "social_preferences": {
            "trustingness": dimension_payload(
                trustingness * 100 if trustingness is not None else None,
                f"Built from trust-game sending ({features['trust_send_amount']} of $5), agreeableness/prosocial self-report, and cooperation/value items.",
                ["Economic preferences::QID117", "Personality::QID25", "Personality::QID29", "Personality::QID233"],
                confidence_from_sources(2, sum(v is not None for v in [features["trust_send_share"], features["agreeableness"], features["prosocial_values"], features["cooperation_orientation"]])),
            ),
            "reciprocity": dimension_payload(
                reciprocity * 100 if reciprocity is not None else None,
                f"Built from trust-game return patterns (mean return share {fmt_pct(features['trust_return_share_mean'])}%), cooperation orientation, and social-game responses.",
                ["Economic preferences::QID118", "Economic preferences::QID119", "Economic preferences::QID120", "Economic preferences::QID121", "Economic preferences::QID122", "Personality::QID233"],
                confidence_from_sources(2, sum(v is not None for v in [features["trust_return_share_mean"], features["cooperation_orientation"], features["ultimatum_rejection_rate"]])),
            ),
            "fairness_enforcement": dimension_payload(
                fairness_enforcement * 100 if fairness_enforcement is not None else None,
                f"Built mainly from ultimatum acceptance thresholds (minimum accepted to self: {features['ultimatum_min_acceptable_to_self']}) and revenge/forgiveness related self-report items.",
                ["Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230", "Personality::QID27", "Personality::QID25"],
                confidence_from_sources(2, sum(v is not None for v in [features["ultimatum_min_acceptable_to_self"], features["revenge_tendency"], features["competitive_values"]])),
            ),
            "altruistic_sharing": dimension_payload(
                altruistic_sharing * 100 if altruistic_sharing is not None else None,
                f"Built from dictator giving ({features['dictator_offer_to_other']} of $5), trust-game sending, and prosocial/helpfulness value endorsements.",
                ["Economic preferences::QID231", "Economic preferences::QID117", "Personality::QID25", "Personality::QID29"],
                confidence_from_sources(2, sum(v is not None for v in [features["dictator_offer_share"], features["trust_send_share"], features["prosocial_values"]])),
            ),
            "exploitation_caution": dimension_payload(
                exploitation_caution * 100 if exploitation_caution is not None else None,
                f"Built from inverse trust-game sending, ultimatum rejection threshold, uncertainty aversion, and self-reliance cues.",
                ["Economic preferences::QID117", "Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230", "Personality::QID238", "Personality::QID233"],
                confidence_from_sources(2, sum(v is not None for v in [features["trust_send_share"], features["ultimatum_min_acceptable_to_self"], features["uncertainty_aversion"]])),
            ),
        },
        "decision_style": {
            "patience": dimension_payload(
                patience * 100 if patience is not None else None,
                f"Built from delayed-versus-sooner choices across the time-preference matrices; later/larger choice rate was {fmt_pct(patience)}%.",
                ["Economic preferences::QID84", "Economic preferences::QID244", "Economic preferences::QID245", "Economic preferences::QID246", "Economic preferences::QID247", "Economic preferences::QID248"],
                confidence_from_sources(1, 1 if patience is not None else 0),
            ),
            "risk_tolerance_gains": dimension_payload(
                risk_gains * 100 if risk_gains is not None else None,
                f"Built from gain-domain lottery matrices; lottery choice rate was {fmt_pct(risk_gains)}%.",
                ["Economic preferences::QID250", "Economic preferences::QID251", "Economic preferences::QID252"],
                confidence_from_sources(1, 1 if risk_gains is not None else 0),
            ),
            "risk_tolerance_losses": dimension_payload(
                risk_losses * 100 if risk_losses is not None else None,
                f"Built from loss-domain lottery matrices; lottery choice rate was {fmt_pct(risk_losses)}%.",
                ["Economic preferences::QID276", "Economic preferences::QID277", "Economic preferences::QID278", "Economic preferences::QID279"],
                confidence_from_sources(1, 1 if risk_losses is not None else 0),
            ),
            "mental_accounting_reliance": dimension_payload(
                mental_accounting * 100 if mental_accounting is not None else None,
                f"Built from the four mental-accounting scenarios; mental-accounting-consistent response rate was {fmt_pct(mental_accounting)}%.",
                list(MENTAL_ACCOUNTING_KEYS.keys()),
                confidence_from_sources(1, 1 if mental_accounting is not None else 0),
            ),
            "cognitive_reflection": dimension_payload(
                cognitive_reflection * 100 if cognitive_reflection is not None else None,
                f"Built from classic reflection / trick questions; accuracy was {fmt_pct(cognitive_reflection)}%.",
                ["Cognitive tests::QID49", "Cognitive tests::QID50", "Cognitive tests::QID51", "Cognitive tests::QID52", "Cognitive tests::QID53", "Cognitive tests::QID54", "Cognitive tests::QID55"],
                confidence_from_sources(1, 1 if cognitive_reflection is not None else 0),
            ),
            "numeracy": dimension_payload(
                numeracy * 100 if numeracy is not None else None,
                f"Built from financial-literacy and numeracy items; combined accuracy was {fmt_pct(numeracy)}%.",
                list(FINANCIAL_LITERAL_KEYS.keys()) + list(NUMERACY_FREE_TEXT_KEYS.keys()),
                confidence_from_sources(2, sum(v is not None for v in [features["financial_literacy_accuracy"], features["numeracy_accuracy"]])),
            ),
            "logical_reasoning": dimension_payload(
                logical_reasoning * 100 if logical_reasoning is not None else None,
                f"Built from conditional and syllogistic reasoning items; accuracy was {fmt_pct(logical_reasoning)}%.",
                list(LOGIC_KEYS.keys()) + ["Cognitive tests::QID221"],
                confidence_from_sources(1, 1 if logical_reasoning is not None else 0),
            ),
            "anchor_susceptibility": dimension_payload(
                anchor_sus * 100 if anchor_sus is not None else None,
                "Built from how far one-shot anchoring estimates were pulled toward the presented anchors. This is inherently noisy because each participant sees only one anchor version per item.",
                ["Anchoring - African countries high::QID166", "Anchoring - African countries low::QID164", "Anchoring - redwood high::QID170", "Anchoring - redwood low::QID168"],
                "low" if anchor_sus is not None else "low",
            ),
            "framing_susceptibility": dimension_payload(
                framing_sus * 100 if framing_sus is not None else None,
                "Built from reference- and framing-sensitive tasks such as calculator/jacket search asymmetry, WTA/WTP gap, and mental-accounting scenarios.",
                ["Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184", "WTA/WTP Thaler problem - WTP certainty::QID189", "WTA/WTP Thaler problem - WTA certainty::QID190", *MENTAL_ACCOUNTING_KEYS.keys()],
                "low" if framing_sus is not None else "low",
            ),
        },
        "consumer_style": {
            "price_sensitivity": dimension_payload(
                price_sensitivity * 100 if price_sensitivity is not None else None,
                "Built from pricing-block purchase rate, sale-search willingness, and tightwad/spendthrift self-report.",
                ["Product Preferences - Pricing::QID9_1", "Product Preferences - Pricing::QID9_40", "Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184", "Personality::QID31", "Personality::QID32", "Personality::QID33", "Personality::QID34"],
                confidence_from_sources(2, sum(v is not None for v in [features["purchase_yes_rate"], features["search_willingness"], features["tightwad"], features["spendthrift"]])),
            ),
            "willingness_to_search": dimension_payload(
                willingness_to_search * 100 if willingness_to_search is not None else None,
                "Built from sale-search items, maximizing tendencies, and product/brand differentiation motives.",
                ["Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184", "Personality::QID239", "Personality::QID234"],
                confidence_from_sources(2, sum(v is not None for v in [features["search_willingness"], features["maximizing_tendency"], features["consumer_uniqueness"]])),
            ),
            "reference_dependence": dimension_payload(
                reference_dependence * 100 if reference_dependence is not None else None,
                "Built from WTA/WTP asymmetry, calculator/jacket relative-savings asymmetry, and mental-accounting choices.",
                ["WTA/WTP Thaler problem - WTP certainty::QID189", "WTA/WTP Thaler problem - WTA certainty::QID190", "Absolute vs. relative - calculator::QID183", "Absolute vs. relative - jacket::QID184", *MENTAL_ACCOUNTING_KEYS.keys()],
                confidence_from_sources(2, sum(v is not None for v in [features["framing_susceptibility"], features["mental_accounting"], features["wta_amount"], features["wtp_certainty_amount"]])),
            ),
            "purchase_inhibition": dimension_payload(
                purchase_inhibition * 100 if purchase_inhibition is not None else None,
                "Built from tightwad/spendthrift self-report, restraint-related spending items, and the pricing purchase rate.",
                ["Personality::QID31", "Personality::QID32", "Personality::QID33", "Personality::QID34", "Product Preferences - Pricing::QID9_1", "Product Preferences - Pricing::QID9_40"],
                confidence_from_sources(2, sum(v is not None for v in [features["tightwad"], features["spending_restraint"], features["purchase_yes_rate"]])),
            ),
        },
        "self_regulation_and_affect": {
            "empathy": dimension_payload(
                empathy * 100 if empathy is not None else None,
                "Built from empathy contagion / perspective-taking items and social sensitivity items.",
                ["Personality::QID232", "Personality::QID236"],
                confidence_from_sources(2, sum(v is not None for v in [features["empathy"], features["social_sensitivity"]])),
            ),
            "cooperation_orientation": dimension_payload(
                cooperation_orientation * 100 if cooperation_orientation is not None else None,
                "Built from cooperation/collectivism items, agreeableness, prosocial values, and sharing behavior.",
                ["Personality::QID233", "Personality::QID25", "Personality::QID29", "Economic preferences::QID117", "Economic preferences::QID231"],
                confidence_from_sources(2, sum(v is not None for v in [features["cooperation_orientation"], features["agreeableness"], features["prosocial_values"]])),
            ),
            "competition_orientation": dimension_payload(
                competition_orientation * 100 if competition_orientation is not None else None,
                "Built from competition items, competitive/status values, and lower agreeableness where present.",
                ["Personality::QID233", "Personality::QID29", "Personality::QID25"],
                confidence_from_sources(2, sum(v is not None for v in [features["competition_orientation"], features["competitive_values"]])),
            ),
            "uncertainty_aversion": dimension_payload(
                uncertainty_aversion * 100 if uncertainty_aversion is not None else None,
                "Built from need-for-closure / uncertainty-avoidance items.",
                ["Personality::QID238"],
                confidence_from_sources(1, 1 if uncertainty_aversion is not None else 0),
            ),
            "depressive_affect": dimension_payload(
                depressive_affect * 100 if depressive_affect is not None else None,
                "Built mainly from BDI-style symptom items, with anxiety symptoms as a secondary signal.",
                [*BDI_QIDS, "Personality::QID125"],
                confidence_from_sources(2, sum(v is not None for v in [features["depressive_affect"], features["anxiety_symptoms"]])),
            ),
            "spending_self_control": dimension_payload(
                spending_self_control * 100 if spending_self_control is not None else None,
                "Built from spending-restraint items, orderliness/conscientiousness, and minimalist ownership tendencies.",
                ["Personality::QID31", "Personality::QID32", "Personality::QID33", "Personality::QID34", "Personality::QID30", "Personality::QID25", "Personality::QID35"],
                confidence_from_sources(2, sum(v is not None for v in [features["spending_restraint"], features["orderliness"], features["conscientiousness"]])),
            ),
        },
    }


def build_pgg_cues(derived_dimensions: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    social = derived_dimensions["social_preferences"]
    decision = derived_dimensions["decision_style"]
    affect = derived_dimensions["self_regulation_and_affect"]
    cooperation_orientation = weighted_mean(
        [
            (social["altruistic_sharing"]["score_0_to_100"] / 100.0 if social["altruistic_sharing"]["label"] != "unknown" else None, 0.45),
            (affect["cooperation_orientation"]["score_0_to_100"] / 100.0 if affect["cooperation_orientation"]["label"] != "unknown" else None, 0.55),
        ]
    )
    conditional_cooperation = weighted_mean(
        [
            (social["reciprocity"]["score_0_to_100"] / 100.0 if social["reciprocity"]["label"] != "unknown" else None, 0.65),
            (social["fairness_enforcement"]["score_0_to_100"] / 100.0 if social["fairness_enforcement"]["label"] != "unknown" else None, 0.35),
        ]
    )
    norm_enforcement = weighted_mean(
        [
            (social["fairness_enforcement"]["score_0_to_100"] / 100.0 if social["fairness_enforcement"]["label"] != "unknown" else None, 0.75),
            (features["revenge_tendency"], 0.25),
        ]
    )
    generosity_without_return = weighted_mean(
        [
            (social["altruistic_sharing"]["score_0_to_100"] / 100.0 if social["altruistic_sharing"]["label"] != "unknown" else None, 0.70),
            (social["trustingness"]["score_0_to_100"] / 100.0 if social["trustingness"]["label"] != "unknown" else None, 0.30),
        ]
    )
    exploitation_caution = social["exploitation_caution"]["score_0_to_100"] / 100.0 if social["exploitation_caution"]["label"] != "unknown" else None
    communication_coordination = weighted_mean(
        [
            (features["extraversion"], 0.30),
            (features["social_sensitivity"], 0.40),
            (features["empathy"], 0.30),
        ]
    )
    behavioral_stability = weighted_mean(
        [
            (features["conscientiousness"], 0.40),
            (features["self_concept_clarity"], 0.35),
            (1.0 - features["neuroticism"] if features["neuroticism"] is not None else None, 0.25),
        ]
    )

    return {
        "cooperation_orientation": dimension_payload(
            cooperation_orientation * 100 if cooperation_orientation is not None else None,
            "Transfer cue built from observed sharing behavior plus cooperation-oriented self-report.",
            ["Economic preferences::QID117", "Economic preferences::QID231", "Personality::QID233", "Personality::QID25", "Personality::QID29"],
            "medium",
            scope_note="Cue only. This is not a direct prediction of contribution levels in a specific PGG configuration.",
        ),
        "conditional_cooperation": dimension_payload(
            conditional_cooperation * 100 if conditional_cooperation is not None else None,
            "Transfer cue built from reciprocity and fairness-threshold signals in the Twin tasks.",
            ["Economic preferences::QID118", "Economic preferences::QID119", "Economic preferences::QID120", "Economic preferences::QID121", "Economic preferences::QID122", "Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230"],
            "medium",
            scope_note="Cue only. It indicates whether behavior looks more contingent versus fixed, not exactly how the participant would react round by round in a PGG.",
        ),
        "norm_enforcement": dimension_payload(
            norm_enforcement * 100 if norm_enforcement is not None else None,
            "Transfer cue built from ultimatum rejection thresholds and revenge/forgiveness self-report.",
            ["Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230", "Personality::QID27", "Personality::QID25"],
            "medium",
            scope_note="Cue only. It captures willingness to resist or sanction perceived unfairness, not specific punishment behavior in a public-goods game.",
        ),
        "generosity_without_return": dimension_payload(
            generosity_without_return * 100 if generosity_without_return is not None else None,
            "Transfer cue built from dictator giving, trust-game sending, and prosocial value orientation.",
            ["Economic preferences::QID231", "Economic preferences::QID117", "Personality::QID25", "Personality::QID29"],
            "medium",
            scope_note="Cue only. It indicates baseline willingness to give when return is uncertain or absent.",
        ),
        "exploitation_caution": dimension_payload(
            exploitation_caution * 100 if exploitation_caution is not None else None,
            "Transfer cue copied from the exploitation-caution derived dimension.",
            ["Economic preferences::QID117", "Economic preferences::QID226", "Economic preferences::QID227", "Economic preferences::QID228", "Economic preferences::QID229", "Economic preferences::QID230", "Personality::QID238"],
            "medium",
            scope_note="Cue only. It captures guardedness toward possible exploitation, not a specific contribution floor.",
        ),
        "communication_coordination": dimension_payload(
            communication_coordination * 100 if communication_coordination is not None else None,
            "Transfer cue inferred from extraversion, empathy, and social-sensitivity self-report; no real-time strategic communication task is observed in Twin.",
            ["Personality::QID25", "Personality::QID232", "Personality::QID236"],
            "low",
            scope_note="Cue only. This is especially uncertain because Twin does not directly observe repeated group communication.",
        ),
        "behavioral_stability": dimension_payload(
            behavioral_stability * 100 if behavioral_stability is not None else None,
            "Transfer cue built from conscientiousness, self-concept clarity, and lower neurotic volatility.",
            ["Personality::QID25", "Personality::QID237", "Personality::QID30"],
            "medium",
            scope_note="Cue only. It indicates whether behavior is likely to be rule-like and stable rather than volatile across contexts.",
        ),
    }


def summarize_style(dimension_payloads: Dict[str, Any], positive_keys: Sequence[str], caution_keys: Sequence[str], title: str) -> Dict[str, Any]:
    scored = [
        (key, value)
        for key, value in dimension_payloads.items()
        if value["label"] != "unknown"
    ]
    scored.sort(key=lambda item: abs(item[1]["score_0_to_100"] - 50), reverse=True)
    highlights: List[str] = []
    cautions: List[str] = []
    for key, value in scored:
        if value["score_0_to_100"] >= 65 and key in positive_keys and len(highlights) < 4:
            highlights.append(f"{key.replace('_', ' ')}: {value['label']}")
        if value["score_0_to_100"] <= 35 and key in caution_keys and len(cautions) < 3:
            cautions.append(f"{key.replace('_', ' ')}: {value['label']}")
    for key, value in scored:
        if len(highlights) >= 2:
            break
        candidate = f"{key.replace('_', ' ')}: {value['label']}"
        if candidate not in highlights:
            highlights.append(candidate)
    if not highlights:
        highlights = [
            "overall profile: mixed",
            "no single dimension dominates strongly",
        ]
    elif len(highlights) == 1:
        highlights.append("overall profile: mixed")
    if not cautions:
        cautions.append("Several cues are moderate rather than extreme, so the style is not sharply one-sided.")
    top_bits = ", ".join(highlights[:2])
    summary = f"{title} is anchored by {top_bits}."
    return {
        "summary": summary,
        "highlights": highlights[:4],
        "cautions": cautions[:3],
    }


def build_behavioral_signature(derived: Dict[str, Any], cues: Dict[str, Any]) -> List[str]:
    signature: List[str] = []
    social = derived["social_preferences"]
    decision = derived["decision_style"]
    affect = derived["self_regulation_and_affect"]

    if social["altruistic_sharing"]["score_0_to_100"] >= 65:
        signature.append("Shows a relatively strong sharing/prosocial orientation in direct allocation tasks.")
    elif social["altruistic_sharing"]["score_0_to_100"] <= 35:
        signature.append("Protects own resources more than a strongly altruistic/sharing type.")

    if social["fairness_enforcement"]["score_0_to_100"] >= 65:
        signature.append("Sets meaningful fairness thresholds rather than accepting any social split.")
    elif social["fairness_enforcement"]["score_0_to_100"] <= 35:
        signature.append("Looks more tolerant of unequal outcomes than a strong norm-enforcement type.")

    if decision["patience"]["label"] != "unknown":
        if decision["patience"]["score_0_to_100"] >= 65:
            signature.append("Consistently prefers larger later rewards over smaller sooner ones.")
        elif decision["patience"]["score_0_to_100"] <= 35:
            signature.append("Frequently prefers sooner rewards, indicating a shorter time horizon.")

    if affect["uncertainty_aversion"]["score_0_to_100"] >= 65:
        signature.append("Prefers structure, closure, and predictability over open-ended uncertainty.")

    if cues["communication_coordination"]["score_0_to_100"] >= 65:
        signature.append("Has self-report cues consistent with reading others and adapting socially.")

    if affect["depressive_affect"]["score_0_to_100"] >= 65:
        signature.append("Reports elevated affective distress relative to a low-symptom baseline.")

    fallback_dimensions = [
        ("trustingness", social["trustingness"]),
        ("reciprocity", social["reciprocity"]),
        ("cooperation orientation", affect["cooperation_orientation"]),
        ("competition orientation", affect["competition_orientation"]),
        ("cognitive reflection", decision["cognitive_reflection"]),
        ("logical reasoning", decision["logical_reasoning"]),
        ("behavioral stability", cues["behavioral_stability"]),
    ]
    fallback_dimensions.sort(key=lambda item: abs(item[1]["score_0_to_100"] - 50), reverse=True)
    for name, payload in fallback_dimensions:
        if len(signature) >= 3:
            break
        if payload["label"] == "unknown":
            continue
        sentence = f"{name.capitalize()} looks {payload['label'].replace('_', ' ')} rather than neutral."
        if sentence not in signature:
            signature.append(sentence)

    while len(signature) < 3:
        signature.append("Profile is broadly mixed rather than dominated by one or two extreme tendencies.")
    return signature[:6]


def build_uncertainties(answered_refs: Sequence[str], features: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [
        {
            "topic": "Transfer scope",
            "note": "This profile stops at transfer-relevant cues. It does not directly predict how the participant would behave in a specific public-goods-game configuration.",
            "evidence_refs": [],
        },
        {
            "topic": "Anchor susceptibility",
            "note": "Anchoring is estimated from single exposed conditions rather than within-person low/high anchor comparisons, so the anchor-susceptibility signal is low-confidence.",
            "evidence_refs": [
                "Anchoring - African countries high::QID166",
                "Anchoring - African countries low::QID164",
                "Anchoring - redwood high::QID170",
                "Anchoring - redwood low::QID168",
            ],
        },
        {
            "topic": "Communication",
            "note": "Communication/coordination cues come from self-report and social-sensitivity items, not from direct repeated-group communication behavior.",
            "evidence_refs": ["Personality::QID25", "Personality::QID232", "Personality::QID236"],
        },
        {
            "topic": "Pricing interpretation",
            "note": "The pricing block uses isolated yes/no purchase judgments without an explicit budget constraint, so consumer-style inferences should be read as coarse tendencies.",
            "evidence_refs": [ref for ref in answered_refs if ref.startswith("Product Preferences - Pricing::")][:5],
        },
    ]
    if features["actual_text"]:
        items.append(
            {
                "topic": "Open text versus structured items",
                "note": "Open-text self-description can add qualitative detail, but it may reflect self-presentation or current salience rather than stable average behavior.",
                "evidence_refs": ["Personality::QID268", "Personality::QID269", "Personality::QID270"],
            }
        )
    return items[:6]


def answered_question_refs(row: pd.Series, catalog: Dict[str, QuestionMeta], mapping_rows: Sequence[Dict[str, str]]) -> List[str]:
    refs: List[str] = []
    valid_refs = {mapping["question_ref"] for mapping in mapping_rows if mapping["profile_section"] != "exclude"}
    for ref in valid_refs:
        meta = catalog.get(ref)
        if meta is None:
            continue
        if response_present(row, meta):
            refs.append(ref)
    return sorted(refs)


def build_profile(
    row: pd.Series,
    catalog: Dict[str, QuestionMeta],
    mapping_rows: Sequence[Dict[str, str]],
    scale_max_map: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    answered_refs = answered_question_refs(row, catalog, mapping_rows)
    refs_by_subsection = build_question_refs_by_subsection(answered_refs, mapping_rows)
    feature_bank = compute_feature_bank(row, catalog, scale_max_map)
    derived = build_derived_dimensions(feature_bank)
    cues = build_pgg_cues(derived, feature_bank)

    observed = {
        "personality_and_self_report": make_personality_block(
            row, catalog, scale_max_map, refs_by_subsection.get("personality_and_self_report", []), feature_bank
        ),
        "social_game_behavior": build_social_game_block(
            row, catalog, refs_by_subsection.get("social_game_behavior", []), feature_bank
        ),
        "economic_preferences_non_social": build_non_social_econ_block(
            row, catalog, refs_by_subsection.get("economic_preferences_non_social", []), feature_bank
        ),
        "cognitive_performance": build_cognitive_block(
            row, catalog, refs_by_subsection.get("cognitive_performance", []), feature_bank
        ),
        "heuristics_and_biases": build_heuristics_block(
            row, catalog, refs_by_subsection.get("heuristics_and_biases", []), feature_bank
        ),
        "pricing_and_consumer_choice": build_pricing_block(
            row, catalog, refs_by_subsection.get("pricing_and_consumer_choice", []), feature_bank
        ),
        "open_text_responses": build_open_text_block(
            row, catalog, refs_by_subsection.get("open_text_responses", []), feature_bank
        ),
    }

    social_dimensions = {
        **derived["social_preferences"],
        "cooperation_orientation": derived["self_regulation_and_affect"]["cooperation_orientation"],
        "competition_orientation": derived["self_regulation_and_affect"]["competition_orientation"],
        "empathy": derived["self_regulation_and_affect"]["empathy"],
    }
    decision_dimensions = {
        **derived["decision_style"],
        **derived["consumer_style"],
    }

    excluded_scaffolding_refs = sorted(
        row_mapping["question_ref"] for row_mapping in mapping_rows if row_mapping["profile_section"] == "exclude"
    )

    profile = {
        "profile_spec_version": "twin_extended_profile_v1",
        "participant": {
            "pid": str(int(row["pid"])),
            "source_dataset": "Twin-2k",
            "source_waves": [1, 2, 3],
        },
        "coverage": {
            "included_question_refs": answered_refs,
            "excluded_scaffolding_refs": excluded_scaffolding_refs,
            "notes": [
                "Question refs are keyed as BlockName::QuestionID because bare QuestionID is not globally unique in the Twin catalog.",
                "The deterministic profile keeps observed Twin evidence and transfer-relevant cues separate from downstream game-specific forecasting.",
            ],
        },
        "background_context": build_background_context(row, catalog, answered_refs),
        "observed_in_twin": observed,
        "derived_dimensions": derived,
        "behavioral_signature": build_behavioral_signature(derived, cues),
        "social_style": summarize_style(
            social_dimensions,
            positive_keys=("trustingness", "reciprocity", "altruistic_sharing", "cooperation_orientation", "empathy"),
            caution_keys=("competition_orientation", "exploitation_caution"),
            title="Social style",
        ),
        "decision_style": summarize_style(
            decision_dimensions,
            positive_keys=("patience", "cognitive_reflection", "numeracy", "logical_reasoning", "price_sensitivity", "willingness_to_search"),
            caution_keys=("anchor_susceptibility", "framing_susceptibility", "purchase_inhibition"),
            title="Decision style",
        ),
        "pgg_relevant_cues": cues,
        "uncertainties": build_uncertainties(answered_refs, feature_bank),
    }
    return profile


def validate_profile(profile: Dict[str, Any], validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(profile), key=lambda err: list(err.absolute_path))
    if not errors:
        return
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    raise ValueError(f"Profile failed schema validation at {path}: {first.message}")


def main() -> int:
    args = parse_args()
    schema = json.loads(args.schema_json.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    catalog = load_catalog(args.catalog_json)
    mapping_rows = load_mapping(args.mapping_csv)
    df = pd.read_csv(args.responses_csv)

    if args.pid is not None:
        df = df[df["pid"] == args.pid]
    if args.limit is not None:
        df = df.head(args.limit)

    scale_max_map = build_scale_max_map(df if len(df) > 0 else pd.read_csv(args.responses_csv), catalog)

    profiles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        profile = build_profile(row, catalog, mapping_rows, scale_max_map)
        validate_profile(profile, validator)
        profiles.append(profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "twin_extended_profiles.jsonl"
    preview_json = args.output_dir / "preview_twin_extended_profiles.json"

    with output_jsonl.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")
    preview_json.write_text(json.dumps(profiles[:3], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote jsonl:   {output_jsonl}")
    print(f"Wrote preview: {preview_json}")
    print(f"Profiles:      {len(profiles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
