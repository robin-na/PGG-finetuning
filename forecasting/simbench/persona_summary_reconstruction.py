from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

RELEASED_SUMMARY_CACHE = SCRIPT_DIR / "cache" / "twin_persona_summary_cache.jsonl"
TWIN_EXTENDED_PROFILES = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "twin_profiles"
    / "output"
    / "twin_extended_profiles"
    / "twin_extended_profiles.jsonl"
)
WAVE1_SCORES = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "raw_data"
    / "wave 1 scores.csv"
)
WAVE2_SCORES = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "raw_data"
    / "wave 2 scores.csv"
)
WAVE3_SCORES = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "raw_data"
    / "wave 3 scores.csv"
)
WAVE123_RESPONSE_LABELS = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "wave1_3_response_label.csv"
)

VARIANT_RELEASED_STYLE_FULL = "twin_persona_summary_released_style_full"
VARIANT_BACKGROUND_ONLY = "twin_persona_summary_background_only"
VARIANT_DIRECT_SOCIAL_ONLY = "twin_persona_summary_direct_social_only"
VARIANT_SELF_REPORT_SOCIAL_ONLY = "twin_persona_summary_self_report_social_only"
VARIANT_NON_SOCIAL_ECON_ONLY = "twin_persona_summary_non_social_econ_only"
VARIANT_COGNITIVE_ONLY = "twin_persona_summary_cognitive_only"
VARIANT_MISC_HEURISTICS_PRICING_TEXT_ONLY = "twin_persona_summary_misc_heuristics_pricing_text_only"

CATEGORY_BACKGROUND = "background"
CATEGORY_DIRECT_SOCIAL = "direct_social"
CATEGORY_SELF_REPORT_SOCIAL = "self_report_social"
CATEGORY_NON_SOCIAL_ECON = "non_social_econ"
CATEGORY_COGNITIVE = "cognitive"
CATEGORY_MISC_HEURISTICS_PRICING_TEXT = "misc_heuristics_pricing_text"

VARIANT_TO_CATEGORIES = {
    VARIANT_BACKGROUND_ONLY: {CATEGORY_BACKGROUND},
    VARIANT_DIRECT_SOCIAL_ONLY: {CATEGORY_DIRECT_SOCIAL},
    VARIANT_SELF_REPORT_SOCIAL_ONLY: {CATEGORY_SELF_REPORT_SOCIAL},
    VARIANT_NON_SOCIAL_ECON_ONLY: {CATEGORY_NON_SOCIAL_ECON},
    VARIANT_COGNITIVE_ONLY: {CATEGORY_COGNITIVE},
    VARIANT_MISC_HEURISTICS_PRICING_TEXT_ONLY: {CATEGORY_MISC_HEURISTICS_PRICING_TEXT},
}

ALL_VARIANTS = [
    VARIANT_RELEASED_STYLE_FULL,
    VARIANT_BACKGROUND_ONLY,
    VARIANT_DIRECT_SOCIAL_ONLY,
    VARIANT_SELF_REPORT_SOCIAL_ONLY,
    VARIANT_NON_SOCIAL_ECON_ONLY,
    VARIANT_COGNITIVE_ONLY,
    VARIANT_MISC_HEURISTICS_PRICING_TEXT_ONLY,
]


@dataclass(frozen=True)
class MetricSpec:
    display_name: str
    source_kind: str
    source_group: str
    source_name: str


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    category: str
    title: str
    recoverability: str
    notes: str
    released_style: bool = True


INTRO_STUB = "The following is a description of a person."

DEMOGRAPHIC_LABELS = [
    ("region", "Geographic region"),
    ("sex_assigned_at_birth", "Gender"),
    ("age_bracket", "Age"),
    ("education_completed_raw", "Education level"),
    ("race_or_origin", "Race"),
    ("citizenship", "Citizen of the US"),
    ("relationship_status", "Marital status"),
    ("religion", "Religion"),
    ("religious_service_attendance", "Religious attendance"),
    ("party_identification", "Political affiliation"),
    ("income_bracket", "Income"),
    ("political_views", "Political views"),
    ("household_size", "Household size"),
    ("employment_status", "Employment status"),
]

RELEASED_METRIC_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.\-]+)\s*=\s*(?P<value>-?\d+(?:\.\d+)?)\s*\((?P<pct>\d+)(?:st|nd|rd|th) percentile\)"
)

SECTION_SPECS: list[SectionSpec] = [
    SectionSpec(
        section_id="demographics",
        category=CATEGORY_BACKGROUND,
        title="The person's demographics are the following...",
        recoverability="recoverable",
        notes="Demographics are available directly from harmonized Twin background fields.",
    ),
    SectionSpec(
        section_id="big5",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's Big 5 scores are the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="need_for_cognition",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's need for cognition score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="agency_communion",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's agentic / communal value scores are the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="minimalism",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's minimalism score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="basic_empathy",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's basic empathy scale score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="green",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's G.R.E.E.N. score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="crt",
        category=CATEGORY_COGNITIVE,
        title="The person's CRT score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="fluid_crystallized",
        category=CATEGORY_COGNITIVE,
        title="The person's fluid and crystallized intelligence scores are the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="syllogism",
        category=CATEGORY_COGNITIVE,
        title="The person's syllogism score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="total_intelligence",
        category=CATEGORY_COGNITIVE,
        title="The person's total intelligence scores, overconfidence score, and overplacement score are the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="ultimatum",
        category=CATEGORY_DIRECT_SOCIAL,
        title="The person's ultimatum game scores are the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="mental_accounting",
        category=CATEGORY_NON_SOCIAL_ECON,
        title="The person's mental accounting score is the following:",
        recoverability="recoverable",
        notes="Wave 1 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="social_desirability",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's social desirability score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="secondary_conscientiousness",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's secondary conscientiousness score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="beck_anxiety",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's Beck anxiety score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="individualism_collectivism",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's individualism vs collectivism scores are the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="financial_literacy",
        category=CATEGORY_COGNITIVE,
        title="The person's financial literacy score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="numeracy",
        category=CATEGORY_COGNITIVE,
        title="The person's numeracy score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="deductive_certainty",
        category=CATEGORY_COGNITIVE,
        title="The person's modus ponens deductive certainty score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="forward_flow",
        category=CATEGORY_COGNITIVE,
        title="The person's forward flow score is the following:",
        recoverability="probably_recoverable",
        notes="Score is direct from wave 2; chain text is recoverable from ordered QID10_1..20 responses.",
    ),
    SectionSpec(
        section_id="discount_presentbias",
        category=CATEGORY_NON_SOCIAL_ECON,
        title="The person's discount rate and present bias are the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="risk_aversion",
        category=CATEGORY_NON_SOCIAL_ECON,
        title="The person's risk aversion score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="loss_aversion",
        category=CATEGORY_NON_SOCIAL_ECON,
        title="The person's loss aversion score is the following:",
        recoverability="recoverable",
        notes="Wave 2 score table matches released values directly when available.",
    ),
    SectionSpec(
        section_id="trust_game",
        category=CATEGORY_DIRECT_SOCIAL,
        title="The person's trust game scores are the following:",
        recoverability="probably_recoverable",
        notes="Scores are direct from wave 2; thought-list text is recoverable from ordered QID271_* and QID272_* responses.",
    ),
    SectionSpec(
        section_id="regulatory_focus",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's regulatory focus scale score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="tightwad_spendthrift",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's tightwad-spendthrift score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="beck_depression",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's Beck depression score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="need_for_uniqueness",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's need for uniqueness score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="self_monitoring",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's self-monitoring score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="self_concept_clarity",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's self-concept clarity score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="need_for_closure",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's need for closure score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="maximization",
        category=CATEGORY_SELF_REPORT_SOCIAL,
        title="The person's maximization scale score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="wason",
        category=CATEGORY_COGNITIVE,
        title="The person's Wason selection score is the following:",
        recoverability="recoverable",
        notes="Wave 3 score table matches released values directly.",
    ),
    SectionSpec(
        section_id="dictator",
        category=CATEGORY_DIRECT_SOCIAL,
        title="The person's dictator game score is the following:",
        recoverability="probably_recoverable",
        notes="Score is direct from wave 3; thought-list text is recoverable from ordered QID275_* responses.",
    ),
    SectionSpec(
        section_id="heuristics_biases",
        category=CATEGORY_MISC_HEURISTICS_PRICING_TEXT,
        title="The person's heuristics and biases summary scores are the following:",
        recoverability="recoverable",
        notes="Not present in released summary, but recoverable from structured Twin heuristics features with percentile ranking over the Twin pool.",
        released_style=False,
    ),
    SectionSpec(
        section_id="pricing_consumer",
        category=CATEGORY_MISC_HEURISTICS_PRICING_TEXT,
        title="The person's pricing and consumer choice summary score is the following:",
        recoverability="recoverable",
        notes="Not present in released summary, but recoverable from structured Twin pricing features with percentile ranking over the Twin pool.",
        released_style=False,
    ),
    SectionSpec(
        section_id="qualitative_self",
        category=CATEGORY_MISC_HEURISTICS_PRICING_TEXT,
        title="The person also answered three purely qualitative questions about their concept of self.",
        recoverability="recoverable",
        notes="Recoverable directly from QID268_TEXT, QID269_TEXT, and QID270_TEXT.",
    ),
]

SECTION_SPEC_BY_ID = {spec.section_id: spec for spec in SECTION_SPECS}

SCORE_SECTION_METRICS: dict[str, list[MetricSpec]] = {
    "big5": [
        MetricSpec("score_extraversion", "score", "wave1", "score_extraversion"),
        MetricSpec("score_agreeableness", "score", "wave1", "score_agreeableness"),
        MetricSpec("wave1_score_conscientiousness", "score", "wave1", "score_conscientiousness"),
        MetricSpec("score_openness", "score", "wave1", "score_openness"),
        MetricSpec("score_neuroticism", "score", "wave1", "score_neuroticism"),
    ],
    "need_for_cognition": [
        MetricSpec("score_needforcognition", "score", "wave1", "score_needforcognition"),
    ],
    "agency_communion": [
        MetricSpec("score_agency", "score", "wave1", "score_agency"),
        MetricSpec("score_communion", "score", "wave1", "score_communion"),
    ],
    "minimalism": [
        MetricSpec("score_minimalism", "score", "wave1", "score_minimalism"),
    ],
    "basic_empathy": [
        MetricSpec("score_BES", "score", "wave1", "score_BES"),
    ],
    "green": [
        MetricSpec("score_GREEN", "score", "wave1", "score_GREEN"),
    ],
    "crt": [
        MetricSpec("crt2_score", "score", "wave1", "crt2_score"),
    ],
    "fluid_crystallized": [
        MetricSpec("score_fluid", "score", "wave1", "score_fluid"),
        MetricSpec("score_crystallized", "score", "wave1", "score_crystallized"),
    ],
    "syllogism": [
        MetricSpec("score_syllogism_merged", "score", "wave1", "score_syllogism_merged"),
    ],
    "total_intelligence": [
        MetricSpec("score_actual_total", "score", "wave1", "score_actual_total"),
        MetricSpec("score_overconfidence", "score", "wave1", "score_overconfidence"),
        MetricSpec("score_overplacement", "score", "wave1", "score_overplacement"),
    ],
    "ultimatum": [
        MetricSpec("score_ultimatum_sender", "score", "wave1", "score_ultimatum_sender"),
        MetricSpec("score_ultimatum_accepted", "score", "wave1", "score_ultimatum_accepted"),
    ],
    "mental_accounting": [
        MetricSpec("score_mentalaccounting", "score", "wave1", "score_mentalaccounting"),
    ],
    "social_desirability": [
        MetricSpec("score_socialdesirability", "score", "wave2", "score_socialdesirability"),
    ],
    "secondary_conscientiousness": [
        MetricSpec("wave2_score_conscientiousness", "score", "wave2", "score_conscientiousness"),
    ],
    "beck_anxiety": [
        MetricSpec("score_anxiety", "score", "wave2", "score_anxiety"),
    ],
    "individualism_collectivism": [
        MetricSpec("score_HI", "score", "wave2", "score_HI"),
        MetricSpec("score_HC", "score", "wave2", "score_HC"),
        MetricSpec("score_VI", "score", "wave2", "score_VI"),
        MetricSpec("score_VC", "score", "wave2", "score_VC"),
    ],
    "financial_literacy": [
        MetricSpec("score_finliteracy", "score", "wave2", "score_finliteracy"),
    ],
    "numeracy": [
        MetricSpec("score_numeracy", "score", "wave2", "score_numeracy"),
    ],
    "deductive_certainty": [
        MetricSpec("score_deductive_certainty", "score", "wave2", "score_deductive_certainty"),
    ],
    "forward_flow": [
        MetricSpec("score_forwardflow", "score", "wave2", "score_forwardflow"),
    ],
    "discount_presentbias": [
        MetricSpec("score_discount", "score", "wave2", "score_discount"),
        MetricSpec("score_presentbias", "score", "wave2", "score_presentbias"),
    ],
    "risk_aversion": [
        MetricSpec("score_riskaversion", "score", "wave2", "score_riskaversion"),
    ],
    "loss_aversion": [
        MetricSpec("score_lossaversion", "score", "wave2", "score_lossaversion"),
    ],
    "trust_game": [
        MetricSpec("score_trustgame_sender", "score", "wave2", "score_trustgame_sender"),
        MetricSpec("score_trustgame_receiver", "score", "wave2", "score_trustgame_receiver"),
    ],
    "regulatory_focus": [
        MetricSpec("score_RFS", "score", "wave3", "score_RFS"),
    ],
    "tightwad_spendthrift": [
        MetricSpec("score_ST-TW", "score", "wave3", "score_ST-TW"),
    ],
    "beck_depression": [
        MetricSpec("score_depression", "score", "wave3", "score_depression"),
    ],
    "need_for_uniqueness": [
        MetricSpec("score_CNFU-S", "score", "wave3", "score_CNFU-S"),
    ],
    "self_monitoring": [
        MetricSpec("score_selfmonitor", "score", "wave3", "score_selfmonitor"),
    ],
    "self_concept_clarity": [
        MetricSpec("score_SCC", "score", "wave3", "score_SCC"),
    ],
    "need_for_closure": [
        MetricSpec("score_needforclosure", "score", "wave3", "score_needforclosure"),
    ],
    "maximization": [
        MetricSpec("score_maximization", "score", "wave3", "score_maximization"),
    ],
    "wason": [
        MetricSpec("score_wason", "score", "wave3", "score_wason"),
    ],
    "dictator": [
        MetricSpec("score_dictator_sender", "score", "wave3", "score_dictator_sender"),
    ],
    "heuristics_biases": [
        MetricSpec("search_willingness", "feature", "heuristics_and_biases", "search_willingness"),
        MetricSpec(
            "ratio_bias_small_tray_choice",
            "feature",
            "heuristics_and_biases",
            "ratio_bias_small_tray_choice",
        ),
        MetricSpec("anchor_pull_average", "feature", "heuristics_and_biases", "anchor_pull_average"),
        MetricSpec(
            "framing_reference_sensitivity",
            "feature",
            "heuristics_and_biases",
            "framing_reference_sensitivity",
        ),
        MetricSpec("vaccine_acceptance", "feature", "heuristics_and_biases", "vaccine_acceptance"),
        MetricSpec(
            "public_support_estimate_mean",
            "feature",
            "heuristics_and_biases",
            "public_support_estimate_mean",
        ),
    ],
    "pricing_consumer": [
        MetricSpec("purchase_yes_rate", "feature", "pricing_and_consumer_choice", "purchase_yes_rate"),
    ],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))
    rounded = round(value, 3)
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _ordinal(percentile: int) -> str:
    if 10 <= percentile % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")
    return f"{percentile}{suffix}"


def _compute_rank_max_percentiles(values_by_pid: dict[str, float | None]) -> dict[str, int]:
    present = [(pid, value) for pid, value in values_by_pid.items() if value is not None]
    if not present:
        return {}
    sorted_present = sorted(present, key=lambda item: item[1])
    counts_by_value: dict[float, int] = {}
    for _, value in sorted_present:
        counts_by_value[value] = counts_by_value.get(value, 0) + 1
    cumulative = 0
    pct_by_value: dict[float, int] = {}
    total = len(sorted_present)
    for value in sorted(counts_by_value):
        cumulative += counts_by_value[value]
        pct_by_value[value] = int(round(100.0 * cumulative / total))
    return {pid: pct_by_value[value] for pid, value in present}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PersonaSummaryReconstructor:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        released_summary_cache: Path = RELEASED_SUMMARY_CACHE,
        extended_profiles_path: Path = TWIN_EXTENDED_PROFILES,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.released_summary_cache = Path(released_summary_cache)
        self.extended_profiles_path = Path(extended_profiles_path)

        self.released_summary_map = self._load_released_summary_map()
        self.extended_profiles = self._load_extended_profiles()
        self.response_rows, self.response_fieldnames = self._load_response_rows()
        self.score_tables = self._load_score_tables()
        self.score_raw_strings = self._load_score_raw_strings()
        self.score_percentiles = self._load_score_percentiles()
        self.feature_values = self._load_feature_values()
        self.feature_percentiles = self._load_feature_percentiles()

        self.available_pids = sorted(
            set(self.extended_profiles)
            & set(self.response_rows)
            & set(self.score_tables["wave1"])
            & set(self.score_tables["wave2"])
            & set(self.score_tables["wave3"])
        )

    def _load_released_summary_map(self) -> dict[str, str]:
        rows = _read_jsonl(self.released_summary_cache)
        return {
            str(row.get("pid")): str(row.get("persona_summary") or "")
            for row in rows
            if row.get("pid") is not None and row.get("persona_summary")
        }

    def _load_extended_profiles(self) -> dict[str, dict[str, Any]]:
        rows = _read_jsonl(self.extended_profiles_path)
        by_pid: dict[str, dict[str, Any]] = {}
        for row in rows:
            pid = str(((row.get("participant") or {}).get("pid")) or "").strip()
            if pid:
                by_pid[pid] = row
        return by_pid

    def _load_response_rows(self) -> tuple[dict[str, dict[str, str]], list[str]]:
        rows, fieldnames = _read_csv_dicts(WAVE123_RESPONSE_LABELS)
        by_pid = {str(row["pid"]).strip(): row for row in rows if str(row.get("pid") or "").strip()}
        return by_pid, fieldnames

    def _load_score_tables(self) -> dict[str, dict[str, dict[str, float | None]]]:
        paths = {
            "wave1": WAVE1_SCORES,
            "wave2": WAVE2_SCORES,
            "wave3": WAVE3_SCORES,
        }
        tables: dict[str, dict[str, dict[str, float | None]]] = {}
        for table_name, path in paths.items():
            rows, _ = _read_csv_dicts(path)
            tables[table_name] = {}
            for row in rows:
                pid = str(row.get("TWIN_ID") or "").strip()
                if not pid:
                    continue
                tables[table_name][pid] = {key: _parse_float(value) for key, value in row.items() if key != "TWIN_ID"}
        return tables

    def _load_score_raw_strings(self) -> dict[str, dict[str, dict[str, str]]]:
        paths = {
            "wave1": WAVE1_SCORES,
            "wave2": WAVE2_SCORES,
            "wave3": WAVE3_SCORES,
        }
        tables: dict[str, dict[str, dict[str, str]]] = {}
        for table_name, path in paths.items():
            rows, _ = _read_csv_dicts(path)
            tables[table_name] = {}
            for row in rows:
                pid = str(row.get("TWIN_ID") or "").strip()
                if not pid:
                    continue
                tables[table_name][pid] = {
                    key: str(value or "").strip() for key, value in row.items() if key != "TWIN_ID"
                }
        return tables

    def _load_score_percentiles(self) -> dict[str, dict[str, dict[str, int]]]:
        percentiles: dict[str, dict[str, dict[str, int]]] = {}
        for table_name, rows in self.score_tables.items():
            percentiles[table_name] = {}
            if not rows:
                continue
            columns = sorted({column for row in rows.values() for column in row})
            for column in columns:
                values_by_pid = {pid: row.get(column) for pid, row in rows.items()}
                percentiles[table_name][column] = _compute_rank_max_percentiles(values_by_pid)
        return percentiles

    def _load_feature_values(self) -> dict[str, dict[str, dict[str, float | None]]]:
        values: dict[str, dict[str, dict[str, float | None]]] = {
            "heuristics_and_biases": {},
            "pricing_and_consumer_choice": {},
        }
        for pid, profile in self.extended_profiles.items():
            observed = profile.get("observed_in_twin") or {}
            for block_name in values:
                feature_map: dict[str, float | None] = {}
                block = observed.get(block_name) or {}
                for item in block.get("summary_features", []):
                    name = str(item.get("name") or "").strip()
                    if not name or name == "answered_question_count":
                        continue
                    raw_value = (item.get("value") or {}).get("raw")
                    if isinstance(raw_value, bool):
                        feature_map[name] = 1.0 if raw_value else 0.0
                    else:
                        feature_map[name] = _parse_float(raw_value)
                values[block_name][pid] = feature_map
        return values

    def _load_feature_percentiles(self) -> dict[str, dict[str, dict[str, int]]]:
        percentiles: dict[str, dict[str, dict[str, int]]] = {}
        for block_name, rows in self.feature_values.items():
            percentiles[block_name] = {}
            columns = sorted({column for row in rows.values() for column in row})
            for column in columns:
                values_by_pid = {pid: row.get(column) for pid, row in rows.items()}
                percentiles[block_name][column] = _compute_rank_max_percentiles(values_by_pid)
        return percentiles

    def _background_value(self, pid: str, feature_name: str) -> str | None:
        profile = self.extended_profiles[pid]
        block = profile.get("background_context") or {}
        for item in block.get("harmonized_features", []):
            if str(item.get("name") or "") == feature_name:
                raw_value = (item.get("value") or {}).get("raw")
                if raw_value is None:
                    return None
                return str(raw_value)
        return None

    def _metric_value(self, pid: str, metric: MetricSpec) -> float | None:
        if metric.source_kind == "score":
            return self.score_tables[metric.source_group].get(pid, {}).get(metric.source_name)
        return self.feature_values[metric.source_group].get(pid, {}).get(metric.source_name)

    def _metric_percentile(self, pid: str, metric: MetricSpec) -> int | None:
        if metric.source_kind == "score":
            return self.score_percentiles[metric.source_group].get(metric.source_name, {}).get(pid)
        return self.feature_percentiles[metric.source_group].get(metric.source_name, {}).get(pid)

    def _metric_rendered_value(self, pid: str, metric: MetricSpec) -> str | None:
        value = self._metric_value(pid, metric)
        if value is None:
            return None
        if metric.source_kind == "score" and metric.source_name == "score_discount":
            raw_text = self.score_raw_strings.get(metric.source_group, {}).get(pid, {}).get(metric.source_name, "")
            if raw_text:
                if "." not in raw_text:
                    return raw_text
                whole, frac = raw_text.split(".", 1)
                trimmed = frac[:3].rstrip("0")
                return f"{whole}.{trimmed}" if trimmed else whole
        return _format_number(value)

    def _ordered_response_values(self, pid: str, prefix: str) -> list[str]:
        row = self.response_rows[pid]
        values: list[str] = []
        for column in self.response_fieldnames:
            if not column.startswith(prefix):
                continue
            text = str(row.get(column) or "").strip()
            if not text or text.lower() in {"none", "nan", "na", "null"}:
                continue
            values.append(text)
        return values

    def _render_demographics(self, pid: str) -> str | None:
        lines = [SECTION_SPEC_BY_ID["demographics"].title]
        any_value = False
        for feature_name, label in DEMOGRAPHIC_LABELS:
            value = self._background_value(pid, feature_name)
            if value is None or not str(value).strip():
                continue
            any_value = True
            lines.append(f"{label}: {value}")
        if not any_value:
            return None
        return "\n".join(lines)

    def _render_score_section(self, pid: str, section_id: str, explanation: str) -> str | None:
        metrics = SCORE_SECTION_METRICS[section_id]
        rendered_lines = [SECTION_SPEC_BY_ID[section_id].title]
        for metric in metrics:
            percentile = self._metric_percentile(pid, metric)
            rendered_value = self._metric_rendered_value(pid, metric)
            if rendered_value is None or percentile is None:
                return None
            rendered_lines.append(
                f"{metric.display_name} = {rendered_value} ({_ordinal(percentile)} percentile)"
            )
        rendered_lines.append(explanation)
        return "\n".join(rendered_lines)

    def _render_forward_flow(self, pid: str) -> str | None:
        score_metric = SCORE_SECTION_METRICS["forward_flow"][0]
        percentile = self._metric_percentile(pid, score_metric)
        rendered_value = self._metric_rendered_value(pid, score_metric)
        chain = self._ordered_response_values(pid, "QID10_")
        if rendered_value is None or percentile is None or not chain:
            return None
        return "\n".join(
            [
                SECTION_SPEC_BY_ID["forward_flow"].title,
                f"{score_metric.display_name} = {rendered_value} ({_ordinal(percentile)} percentile)",
                (
                    "A higher score indicates the person was able to generate more distant words in a sequence "
                    '(e.g. "candle -> bee -> sugar" instead of "candle -> fire -> flame"). '
                    "The actual word chain that the subject generated was the following: "
                    + " -> ".join(chain)
                ),
            ]
        )

    def _render_trust_game(self, pid: str) -> str | None:
        metrics = SCORE_SECTION_METRICS["trust_game"]
        sender_pct = self._metric_percentile(pid, metrics[0])
        receiver_pct = self._metric_percentile(pid, metrics[1])
        sender_text_value = self._metric_rendered_value(pid, metrics[0])
        receiver_text_value = self._metric_rendered_value(pid, metrics[1])
        sender_thoughts = self._ordered_response_values(pid, "QID271_")
        receiver_thoughts = self._ordered_response_values(pid, "QID272_")
        if None in {sender_text_value, sender_pct, receiver_text_value, receiver_pct}:
            return None
        sender_text = "; ".join(sender_thoughts)
        receiver_text = "; ".join(receiver_thoughts)
        return "\n".join(
            [
                SECTION_SPEC_BY_ID["trust_game"].title,
                f"{metrics[0].display_name} = {sender_text_value} ({_ordinal(sender_pct)} percentile)",
                f"{metrics[1].display_name} = {receiver_text_value} ({_ordinal(receiver_pct)} percentile)",
                (
                    "The sender score is the percent sent in the trust game (where one player sends money to another, "
                    "the money is multiplied, and the second player can return some to the first player). "
                    f'The person was asked to list up to 6 thoughts that crossed their mind while playing as the sender and they answered: "{sender_text}". '
                    f'The person was then asked to list up to 6 thoughts that crossed their mind while playing as the receiver and they answered: "{receiver_text}"'
                ),
            ]
        )

    def _render_dictator(self, pid: str) -> str | None:
        metric = SCORE_SECTION_METRICS["dictator"][0]
        percentile = self._metric_percentile(pid, metric)
        rendered_value = self._metric_rendered_value(pid, metric)
        thoughts = self._ordered_response_values(pid, "QID275_")
        if rendered_value is None or percentile is None:
            return None
        thought_text = "; ".join(thoughts)
        return "\n".join(
            [
                SECTION_SPEC_BY_ID["dictator"].title,
                f"{metric.display_name} = {rendered_value} ({_ordinal(percentile)} percentile)",
                (
                    "This is the percent split sent when the person played the dictator game as the sender. "
                    f'The person was asked to list up to 6 thoughts that crossed their mind when playing this game and they answered: "{thought_text}"'
                ),
            ]
        )

    def _render_qualitative_self(self, pid: str) -> str | None:
        row = self.response_rows[pid]
        aspire = str(row.get("QID268_TEXT") or "").strip()
        ought = str(row.get("QID269_TEXT") or "").strip()
        actual = str(row.get("QID270_TEXT") or "").strip()
        if not aspire or not ought or not actual:
            return None
        return "\n".join(
            [
                SECTION_SPEC_BY_ID["qualitative_self"].title,
                (
                    'The person was asked, "Please describe the type of person you aspire to be. That is, write about the '
                    'traits and behaviors you would like ideally to possess, your ultimate goals for yourself. Please write at least 3 sentences." '
                    f'They answered: "{aspire}"'
                ),
                (
                    'The person was then asked the question "Please describe the type of person you ought to be. That is, write about '
                    'the traits and behaviors attributes that you should or ought to possess, based on your responsibilities and what other people expect from you. '
                    f'Please write at least 3 sentences." They answered: "{ought}"'
                ),
                (
                    'The person was finally asked the question "Please describe the type of person you actually are. That is, write about the traits and behaviors you actually possess. '
                    f'Please write at least 3 sentences." They answered: "{actual}"'
                ),
            ]
        )

    def _render_heuristics_biases(self, pid: str) -> str | None:
        metrics = SCORE_SECTION_METRICS["heuristics_biases"]
        lines = [SECTION_SPEC_BY_ID["heuristics_biases"].title]
        at_least_one = False
        for metric in metrics:
            percentile = self._metric_percentile(pid, metric)
            rendered_value = self._metric_rendered_value(pid, metric)
            if rendered_value is None or percentile is None:
                continue
            at_least_one = True
            lines.append(f"{metric.display_name} = {rendered_value} ({_ordinal(percentile)} percentile)")
        if not at_least_one:
            return None
        lines.append(
            "These scores summarize the person's behavior on classic heuristics-and-biases tasks. "
            "Higher search_willingness indicates a stronger tendency to seek additional information before deciding. "
            "Higher ratio_bias_small_tray_choice indicates a greater tendency to choose the smaller tray in the ratio-bias task. "
            "Higher anchor_pull_average indicates stronger attraction of numerical estimates toward presented anchors. "
            "Higher framing_reference_sensitivity indicates greater sensitivity to equivalent gain/loss framing. "
            "Higher vaccine_acceptance indicates greater willingness to accept a described vaccine. "
            "Higher public_support_estimate_mean indicates the person believed a larger share of other respondents would agree with them on selected consensus-estimation questions."
        )
        return "\n".join(lines)

    def _render_pricing_consumer(self, pid: str) -> str | None:
        metric = SCORE_SECTION_METRICS["pricing_consumer"][0]
        percentile = self._metric_percentile(pid, metric)
        rendered_value = self._metric_rendered_value(pid, metric)
        if rendered_value is None or percentile is None:
            return None
        return "\n".join(
            [
                SECTION_SPEC_BY_ID["pricing_consumer"].title,
                f"{metric.display_name} = {rendered_value} ({_ordinal(percentile)} percentile)",
                (
                    "This is the share of pricing and consumer-choice vignettes in which the person chose to purchase the offered product. "
                    "Higher values indicate a greater willingness to buy across varied price and product configurations."
                ),
            ]
        )

    def render_section(self, pid: str, section_id: str) -> str | None:
        explanations = {
            "big5": (
                "Openness reflects curiosity and receptiveness to new experiences, Conscientiousness indicates self-discipline and "
                "goal-directed behavior, Extraversion measures sociability and assertiveness, Agreeableness reflects compassion and "
                "cooperativeness, and Neuroticism captures emotional instability and susceptibility to negative emotions. Each score ranges "
                "from 1 to 5, and a higher score indicates a greater display of the associated traits."
            ),
            "need_for_cognition": (
                "Need for cognition is a personality trait that reflects an individual's tendency to seek out, engage in, and enjoy complex cognitive tasks. "
                "The score ranges from 1 to 5, and a higher score indicates a higher need for cognition."
            ),
            "agency_communion": (
                "Agency is the meta-concept associated with self-advancement in social hierarchies; communion is the partner concept associated with maintenance of positive relationships. "
                "Each score ranges from 1 to 9, and higher values indicate higher propensity for these constructs."
            ),
            "minimalism": "The score ranges from 1 to 5, and a higher score indicates a higher preference for minimalism.",
            "basic_empathy": "The score ranges from 1 to 5, and a higher score indicates more empathy.",
            "green": "The score ranges from 1 to 5, and higher scores indicate a higher affinity for environmentalism.",
            "crt": (
                'The score ranges from 0 to 4, and a higher score indicates a greater ability to suppress an intuitive and spontaneous ("system 1") wrong answer in favor of a reflective and deliberative ("system 2") right answer.'
            ),
            "fluid_crystallized": (
                "Fluid intelligence is the capacity to reason, solve novel problems, and adapt to new situations independent of prior knowledge, while crystallized intelligence is the accumulation of knowledge, facts, and skills acquired through experience and education. "
                "The fluid score ranges from 0 to 6, and the crystallized score ranges from 0 to 20; higher scores indicate better performance."
            ),
            "syllogism": (
                'The score ranges from 0 to 12, and a higher score indicates a greater ability to solve verbal reasoning problems like "All A are B" and "All B are C" implying "All A are C."'
            ),
            "total_intelligence": (
                "The total intelligence score is simply the sum of the person's performances on the aforementioned logic / intelligence questions (ranging from 0 to 42 total correct). "
                "The person's overconfidence is the difference between their prediction of their own performance and their actual performance. "
                "The person's overplacement is the difference between their prediction of their own performance and their prediction of other respondents' performance."
            ),
            "ultimatum": (
                "The sender score is the percent of $5 the person chose to offer another person in the ultimatum game. "
                "The receiver score is the percent of offers they accepted, out of a total of 6 offers made in $1 increments from $5 to $0, when acting as the receiver in the game."
            ),
            "mental_accounting": (
                "The score ranges from 0 to 100 percent, and higher scores indicate a greater adherence to the principles of mental accounting proposed by Thaler: segregate gains, integrate losses, segregate a small gain from a large loss, and integrate a small loss with a large gain."
            ),
            "social_desirability": (
                "The score ranges from 0 to 13, and higher scores indicate a greater tendency to respond to questions in a socially desirable way rather than in a truthful way."
            ),
            "secondary_conscientiousness": (
                "This score was computed using a different questionnaire than the Big 5 conscientiousness score reported above, and it ranges from 0 to 8, but it is otherwise similar. "
                "Higher scores indicate a greater propensity for conscientiousness."
            ),
            "beck_anxiety": "The score ranges from 0 to 63, and higher scores indicate a higher tendency for anxiety.",
            "individualism_collectivism": (
                "The Horizontal Individualism (HI) score reflects a person's preference for autonomy and equality, the Horizontal Collectivism (HC) score captures a preference for interdependence and equality, "
                "the Vertical Individualism (VI) score indicates a drive for personal achievement and acceptance of hierarchical inequality, and the Vertical Collectivism (VC) score denotes a focus on group loyalty combined with acceptance of hierarchical structures. "
                "Each score ranges from 1 to 5, and a higher score indicates a greater preference."
            ),
            "financial_literacy": (
                "The score ranges from 0 to 8, and a higher score indicates the person correctly answered more questions related to general financial literacy."
            ),
            "numeracy": "The score ranges from 0 to 8, and a higher score indicates the person correctly answered more questions related to numeracy.",
            "deductive_certainty": (
                "The score ranges from 0 to 4, and a higher score indicates the person correctly answered more modus ponens questions."
            ),
            "discount_presentbias": (
                "These are implied rates computed from the person's time-value of money preferences. Higher values of the discount rate imply greater impatience. "
                "Higher values of present bias imply greater departure from normative economic behavior."
            ),
            "risk_aversion": (
                "Higher scores indicate a greater tendency for risk aversion in a choice between a sure-amount and lottery payout."
            ),
            "loss_aversion": (
                "Higher scores indicate a greater tendency for loss aversion in a choice between a sure-amount and a lottery payout."
            ),
            "regulatory_focus": (
                "The score ranges from 1 to 7. Higher scores indicate a stronger orientation toward either promotion or prevention focus, meaning individuals would be more driven by promotional aspirations or more motivated by avoiding losses."
            ),
            "tightwad_spendthrift": (
                "The score ranges from 4 to 26. Lower scores (4-11) indicate difficulty spending money, while higher scores (19-26) indicate difficulty controlling spending."
            ),
            "beck_depression": (
                "The score ranges from 0 to 61, and a higher score indicates more depressive behaviors."
            ),
            "need_for_uniqueness": (
                "The score ranges from 1 to 5, and higher scores indicate a higher need for uniqueness."
            ),
            "self_monitoring": (
                "The score ranges from 0 to 5, and higher scores indicate a higher ability to monitor one's own behavior."
            ),
            "self_concept_clarity": (
                "The score ranges from 1 to 5, and higher scores indicate a greater certainty about the person's own self-concept."
            ),
            "need_for_closure": (
                "The score ranges from 1 to 5, and higher scores indicate a greater desire for certainty over ambiguity."
            ),
            "maximization": (
                "The score ranges from 1 to 5, and higher scores indicate a tendency to optimize rather than satisfice when making decisions."
            ),
            "wason": "The score ranges from 0 to 4, and higher scores indicate better performance on the Wason selection task.",
        }
        if section_id == "demographics":
            return self._render_demographics(pid)
        if section_id == "forward_flow":
            return self._render_forward_flow(pid)
        if section_id == "trust_game":
            return self._render_trust_game(pid)
        if section_id == "dictator":
            return self._render_dictator(pid)
        if section_id == "qualitative_self":
            return self._render_qualitative_self(pid)
        if section_id == "heuristics_biases":
            return self._render_heuristics_biases(pid)
        if section_id == "pricing_consumer":
            return self._render_pricing_consumer(pid)
        return self._render_score_section(pid, section_id, explanations[section_id])

    def render_summary(self, pid: str, *, variant: str = VARIANT_RELEASED_STYLE_FULL) -> str:
        if pid not in self.available_pids:
            raise KeyError(f"Unknown or incomplete Twin pid: {pid}")
        if variant == VARIANT_RELEASED_STYLE_FULL:
            included_sections = [spec.section_id for spec in SECTION_SPECS if spec.released_style]
        else:
            categories = VARIANT_TO_CATEGORIES[variant]
            included_sections = [spec.section_id for spec in SECTION_SPECS if spec.category in categories]

        chunks = [INTRO_STUB]
        for section_id in included_sections:
            section_text = self.render_section(pid, section_id)
            if section_text:
                chunks.append(section_text)
        return "\n\n".join(chunks).strip()

    def build_variant_rows(self, variant: str) -> list[dict[str, Any]]:
        return [{"pid": pid, "persona_summary": self.render_summary(pid, variant=variant)} for pid in self.available_pids]

    def audit_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in SECTION_SPECS:
            metrics = SCORE_SECTION_METRICS.get(spec.section_id, [])
            source_descriptions = [
                f"{metric.source_kind}:{metric.source_group}:{metric.source_name}" for metric in metrics
            ]
            if spec.section_id == "demographics":
                source_descriptions = [f"background:{name}" for name, _ in DEMOGRAPHIC_LABELS]
            elif spec.section_id == "trust_game":
                source_descriptions.extend(["responses:QID271_*", "responses:QID272_*"])
            elif spec.section_id == "dictator":
                source_descriptions.extend(["responses:QID275_*"])
            elif spec.section_id == "forward_flow":
                source_descriptions.extend(["responses:QID10_*"])
            elif spec.section_id == "qualitative_self":
                source_descriptions = ["responses:QID268_TEXT", "responses:QID269_TEXT", "responses:QID270_TEXT"]
            rows.append(
                {
                    "section_id": spec.section_id,
                    "category": spec.category,
                    "released_style": int(spec.released_style),
                    "recoverability": spec.recoverability,
                    "title": spec.title,
                    "source_descriptions": "|".join(source_descriptions),
                    "notes": spec.notes,
                }
            )
        return rows

    def _extract_released_metrics(self, summary_text: str) -> dict[str, tuple[str, int]]:
        extracted: dict[str, tuple[str, int]] = {}
        for match in RELEASED_METRIC_PATTERN.finditer(summary_text):
            extracted[match.group("name")] = (match.group("value"), int(match.group("pct")))
        return extracted

    def metric_validation_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pid, released_summary in self.released_summary_map.items():
            if pid not in self.available_pids:
                continue
            released_metrics = self._extract_released_metrics(released_summary)
            for section_id, metrics in SCORE_SECTION_METRICS.items():
                if section_id in {"heuristics_biases", "pricing_consumer"}:
                    continue
                for metric in metrics:
                    reconstructed_value = self._metric_value(pid, metric)
                    reconstructed_pct = self._metric_percentile(pid, metric)
                    released = released_metrics.get(metric.display_name)
                    rows.append(
                        {
                            "pid": pid,
                            "section_id": section_id,
                            "metric_name": metric.display_name,
                            "released_present": int(released is not None),
                            "reconstructed_present": int(
                                reconstructed_value is not None and reconstructed_pct is not None
                            ),
                            "released_value": released[0] if released else "",
                            "reconstructed_value": self._metric_rendered_value(pid, metric) or "",
                            "value_match": int(
                                released is not None
                                and self._metric_rendered_value(pid, metric) is not None
                                and released[0] == self._metric_rendered_value(pid, metric)
                            ),
                            "released_percentile": released[1] if released else "",
                            "reconstructed_percentile": reconstructed_pct if reconstructed_pct is not None else "",
                            "percentile_match": int(
                                released is not None
                                and reconstructed_pct is not None
                                and released[1] == reconstructed_pct
                            ),
                        }
                    )
        return rows

    def validation_summary(self) -> dict[str, Any]:
        metric_rows = self.metric_validation_rows()
        by_metric: dict[str, dict[str, int]] = {}
        for row in metric_rows:
            metric_name = str(row["metric_name"])
            bucket = by_metric.setdefault(
                metric_name,
                {
                    "total": 0,
                    "released_present": 0,
                    "reconstructed_present": 0,
                    "value_match": 0,
                    "percentile_match": 0,
                },
            )
            bucket["total"] += 1
            bucket["released_present"] += int(row["released_present"])
            bucket["reconstructed_present"] += int(row["reconstructed_present"])
            bucket["value_match"] += int(row["value_match"])
            bucket["percentile_match"] += int(row["percentile_match"])
        return {
            "available_pid_count": len(self.available_pids),
            "released_pid_count": len(self.released_summary_map),
            "validated_pid_count": len({row["pid"] for row in metric_rows}),
            "metrics": by_metric,
        }

    def write_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for variant in ALL_VARIANTS:
            _write_jsonl(output_dir / f"{variant}.jsonl", self.build_variant_rows(variant))

        audit_rows = self.audit_rows()
        metric_validation_rows = self.metric_validation_rows()
        _write_csv(output_dir / "persona_summary_reconstruction_audit.csv", audit_rows)
        _write_csv(output_dir / "persona_summary_metric_validation.csv", metric_validation_rows)
        (output_dir / "persona_summary_validation_summary.json").write_text(
            json.dumps(self.validation_summary(), indent=2),
            encoding="utf-8",
        )

        sample_dir = output_dir / "sample_summaries"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for pid in self.available_pids[:5]:
            bundle = {
                "pid": pid,
                "released_persona_summary": self.released_summary_map.get(pid, ""),
                "reconstructed_released_style_full": self.render_summary(pid, variant=VARIANT_RELEASED_STYLE_FULL),
                "background_only": self.render_summary(pid, variant=VARIANT_BACKGROUND_ONLY),
                "direct_social_only": self.render_summary(pid, variant=VARIANT_DIRECT_SOCIAL_ONLY),
                "self_report_social_only": self.render_summary(pid, variant=VARIANT_SELF_REPORT_SOCIAL_ONLY),
                "non_social_econ_only": self.render_summary(pid, variant=VARIANT_NON_SOCIAL_ECON_ONLY),
                "cognitive_only": self.render_summary(pid, variant=VARIANT_COGNITIVE_ONLY),
                "misc_heuristics_pricing_text_only": self.render_summary(
                    pid, variant=VARIANT_MISC_HEURISTICS_PRICING_TEXT_ONLY
                ),
            }
            (sample_dir / f"{pid}.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    output_dir = SCRIPT_DIR / "ablation" / "output" / "persona_summary_reconstruction"
    reconstructor = PersonaSummaryReconstructor()
    reconstructor.write_artifacts(output_dir)
    print(str(output_dir))


if __name__ == "__main__":
    main()
