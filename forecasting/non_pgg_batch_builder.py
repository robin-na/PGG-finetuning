from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tiktoken


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

TWIN_PROFILES_JSONL = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "task_grounding"
    / "output"
    / "twin_extended_profiles"
    / "twin_extended_profiles.jsonl"
)
TWIN_CARDS_JSONL = (
    REPO_ROOT
    / "non-PGG_generalization"
    / "task_grounding"
    / "output"
    / "twin_extended_profile_cards"
    / "pgg_prompt_min"
    / "twin_extended_profile_cards.jsonl"
)

AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
EDUCATION_ORDER = ["high school", "college/postsecondary", "postgraduate"]
SEX_ORDER = ["male", "female"]

VARIANT_BASELINE = "baseline"
VARIANT_DEMOGRAPHIC_ONLY = "demographic_only_row_resampled_seed_0"
VARIANT_TWIN_CORRECTED = "twin_sampled_seed_0"
VARIANT_TWIN_UNADJUSTED = "twin_sampled_unadjusted_seed_0"
ALL_VARIANTS = [
    VARIANT_BASELINE,
    VARIANT_DEMOGRAPHIC_ONLY,
    VARIANT_TWIN_CORRECTED,
    VARIANT_TWIN_UNADJUSTED,
]

MODEL_SLUGS = {"gpt-5.1": "gpt_5_1", "gpt-5-mini": "gpt_5_mini"}
ALL_MODELS = ["gpt-5.1", "gpt-5-mini"]

TWIN_TRANSFER_CUE_DISPLAY_NAMES = {
    "cooperation_orientation": "Cooperation orientation",
    "conditional_cooperation": "Conditional cooperation",
    "norm_enforcement": "Norm enforcement",
    "generosity_without_return": "Generosity without return",
    "exploitation_caution": "Exploitation caution",
    "communication_coordination": "Communication/coordination",
    "behavioral_stability": "Behavioral stability",
}

TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


@dataclass(frozen=True)
class DatasetBundle:
    dataset_key: str
    display_name: str
    records: pd.DataFrame
    units: pd.DataFrame
    demographic_source: pd.DataFrame
    twin_matching_fields: list[str]


@dataclass(frozen=True)
class SamplingSpec:
    max_records_per_treatment: int | None
    sampling_seed: int


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "data_expired", "consent_revoked"}:
        return None
    return text


def _clean_numeric_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = _clean_text(value)
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return str(round(numeric, 4))


def _age_to_bracket(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        age = float(value)
    except (TypeError, ValueError):
        text = _clean_text(value)
        if text is None:
            return None
        text = text.replace("years", "").strip()
        if text.endswith("+"):
            try:
                age = float(text[:-1])
            except ValueError:
                return None
        else:
            try:
                age = float(text)
            except ValueError:
                return None
    if age < 30:
        return "18-29"
    if age < 50:
        return "30-49"
    if age < 65:
        return "50-64"
    return "65+"


def _canonical_sex(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"male", "man", "m"}:
        return "male"
    if lowered in {"female", "woman", "f"}:
        return "female"
    return None


def _minority_education_to_harmonized(value: Any) -> str | None:
    text = _clean_text(value)
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


def _multi_game_education_to_harmonized(value: Any) -> str | None:
    text = _clean_text(value)
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


def _longitudinal_gender_to_display(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    mapping = {"male": "Male", "female": "Female", "non-binary": "Non-binary"}
    return mapping.get(text.lower(), text)


def _two_stage_gender_to_display(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    mapping = {1: "Male", 2: "Female", 3: "Other", 4: "Other"}
    return mapping.get(code)


def _two_stage_gender_to_match(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    mapping = {1: "male", 2: "female"}
    return mapping.get(code)


def _minority_sex_to_display(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    mapping = {"male": "Male", "female": "Female"}
    return mapping.get(text.lower(), text)


def _multi_game_gender_to_display(value: Any) -> str | None:
    text = _clean_text(value)
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
    text = _clean_text(value)
    if text is None:
        return None
    mapping = {"man": "male", "woman": "female"}
    return mapping.get(text.lower())


def _longitudinal_gender_to_match(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    mapping = {"male": "male", "female": "female"}
    return mapping.get(text.lower())


def _format_bullet_markdown(pairs: list[tuple[str, str | None]]) -> str:
    lines = [f"- {name}: {value}" for name, value in pairs if value is not None]
    return "\n".join(lines)


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
    markdown = _format_bullet_markdown(
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


def _simple_demographic_summary(
    *,
    age: str | None,
    sex_or_gender: str | None,
    education: str | None = None,
) -> tuple[str, str]:
    parts: list[str] = []
    if age is not None and sex_or_gender is not None:
        parts.append(f"{age}-year-old, {sex_or_gender.lower()}")
    elif age is not None:
        parts.append(f"{age}-year-old")
    elif sex_or_gender is not None:
        parts.append(sex_or_gender.lower())
    if education is not None:
        parts.append(education)
    summary = ", ".join(parts).strip()
    if summary:
        summary = summary[0].upper() + summary[1:] + "."
    markdown = _format_bullet_markdown(
        [
            ("Age", age),
            ("Sex/gender", sex_or_gender),
            ("Education", education),
        ]
    )
    return summary, markdown


def _extract_harmonized_feature_map(profile: dict[str, Any]) -> dict[str, Any]:
    features = profile.get("background_context", {}).get("harmonized_features", [])
    out: dict[str, Any] = {}
    for feature in features:
        name = feature.get("name")
        value = feature.get("value", {})
        raw = value.get("raw") if isinstance(value, dict) else value
        if name:
            out[str(name)] = raw
    return out


def _load_twin_cards(cards_path: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    with cards_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            participant = row.get("participant", {})
            pid = str(participant.get("pid") or row.get("profile_id") or "").strip()
            if pid:
                cards[pid] = row
    return cards


def _load_twin_personas(profiles_path: Path, cards_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    cards_by_pid = _load_twin_cards(cards_path)
    rows: list[dict[str, Any]] = []
    with profiles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profile = json.loads(line)
            participant = profile.get("participant", {})
            pid = str(participant.get("pid") or "").strip()
            if not pid or pid not in cards_by_pid:
                continue
            features = _extract_harmonized_feature_map(profile)
            age = features.get("age_bracket")
            education = features.get("education_completed_harmonized")
            sex = _canonical_sex(features.get("sex_assigned_at_birth"))
            if age not in AGE_ORDER or education not in EDUCATION_ORDER or sex not in SEX_ORDER:
                continue
            rows.append(
                {
                    "twin_pid": pid,
                    "age_bracket": age,
                    "education_harmonized": education,
                    "sex_assigned_at_birth": sex,
                    "matching_age_bracket": age,
                    "matching_education": education,
                    "matching_sex": sex,
                    "headline": str(cards_by_pid[pid].get("headline", "")).strip(),
                    "summary": str(cards_by_pid[pid].get("summary", "")).strip(),
                    "background_summary": str(
                        (cards_by_pid[pid].get("background") or {}).get("summary", "")
                    ).strip(),
                }
            )
    personas = pd.DataFrame(rows)
    if personas.empty:
        raise ValueError("No Twin personas were loaded from the shared profile artifacts.")
    return personas, cards_by_pid


def _build_candidate_maps(
    personas: pd.DataFrame,
    match_fields: list[str],
) -> dict[tuple[str, ...], dict[tuple[Any, ...], list[str]]]:
    candidate_maps: dict[tuple[str, ...], dict[tuple[Any, ...], list[str]]] = {}
    for size in range(1, len(match_fields) + 1):
        for subset in combinations(match_fields, size):
            subset_df = personas.dropna(subset=list(subset)).copy()
            grouped = subset_df.groupby(list(subset), dropna=False)["twin_pid"].apply(list)
            candidate_maps[subset] = {
                key if isinstance(key, tuple) else (key,): list(value)
                for key, value in grouped.to_dict().items()
            }
    return candidate_maps


def _choose_pid(
    candidate_pids: list[str],
    reuse_counter: Counter[str],
    rng: np.random.Generator,
    weighted_by_inverse_reuse: bool,
) -> str:
    if not weighted_by_inverse_reuse:
        return str(candidate_pids[int(rng.integers(len(candidate_pids)))])
    weights = np.array(
        [1.0 / (1.0 + float(reuse_counter.get(pid, 0))) for pid in candidate_pids],
        dtype=float,
    )
    probs = weights / weights.sum()
    return str(candidate_pids[int(rng.choice(len(candidate_pids), p=probs))])


def _sample_demographic_profiles(
    units: pd.DataFrame,
    demographic_source: pd.DataFrame,
    seed: int,
    output_dir: Path,
    dataset_key: str,
) -> tuple[dict[str, dict[str, Any]], Path]:
    rng = np.random.default_rng(seed)
    if demographic_source.empty:
        raise ValueError(f"{dataset_key}: demographic source rows are empty.")
    sampled_positions = rng.choice(demographic_source.index.to_numpy(), size=len(units), replace=True)
    sampled = demographic_source.loc[sampled_positions].reset_index(drop=True)

    assignments: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    unit_to_card: dict[str, dict[str, Any]] = {}

    for idx, unit_row in enumerate(units.itertuples(index=False), start=1):
        source_row = sampled.iloc[idx - 1].to_dict()
        profile_id = f"{dataset_key}_demo_profile_{idx:05d}"
        card = {
            "profile_card_version": f"{dataset_key}_demographic_only_v1",
            "profile_type": f"{dataset_key}_demographic_only",
            "profile_id": profile_id,
            "headline": "",
            "summary": source_row["summary"],
            "markdown": source_row["markdown"],
            "matching_age_bracket": source_row.get("matching_age_bracket"),
            "matching_sex": source_row.get("matching_sex"),
            "matching_education": source_row.get("matching_education"),
            "source": {
                "sampling_mode": "row_resampled_dataset_rows",
                "seed": seed,
                "source_row_id": source_row.get("source_row_id"),
            },
        }
        assignment = {
            "unit_id": str(unit_row.unit_id),
            "profile_id": profile_id,
            "summary": source_row["summary"],
            "source_row_id": source_row.get("source_row_id"),
            "matching_age_bracket": source_row.get("matching_age_bracket"),
            "matching_sex": source_row.get("matching_sex"),
            "matching_education": source_row.get("matching_education"),
        }
        cards.append(card)
        assignments.append(assignment)
        unit_to_card[str(unit_row.unit_id)] = card

    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "demographic_profile_cards.jsonl"
    assignments_path = output_dir / "unit_assignments.jsonl"
    _write_jsonl(cards_path, cards)
    _write_jsonl(assignments_path, assignments)
    pd.DataFrame(assignments).to_csv(output_dir / "unit_assignments.csv", index=False)
    summary = {
        "dataset_key": dataset_key,
        "variant": VARIANT_DEMOGRAPHIC_ONLY,
        "seed": seed,
        "unit_count": len(units),
        "source_row_count": int(len(demographic_source)),
        "unique_source_rows_used": int(len(set(sampled["source_row_id"].tolist()))),
        "cards_file": str(cards_path),
        "assignments_file": str(assignments_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return unit_to_card, assignments_path


def _sample_twin_profiles(
    *,
    units: pd.DataFrame,
    demographic_source: pd.DataFrame,
    twin_personas: pd.DataFrame,
    match_fields: list[str],
    seed: int,
    output_dir: Path,
    dataset_key: str,
    corrected: bool,
) -> tuple[dict[str, dict[str, Any]], Path]:
    rng = np.random.default_rng(seed)
    all_pids = twin_personas["twin_pid"].tolist()
    persona_by_pid = {
        str(row["twin_pid"]): row
        for row in twin_personas.to_dict(orient="records")
    }
    candidate_maps = _build_candidate_maps(twin_personas, match_fields) if corrected else {}
    reuse_counter: Counter[str] = Counter()

    sampled_targets = (
        demographic_source.loc[
            rng.choice(demographic_source.index.to_numpy(), size=len(units), replace=True)
        ].reset_index(drop=True)
        if corrected
        else None
    )

    assignments: list[dict[str, Any]] = []
    unit_to_assignment: dict[str, dict[str, Any]] = {}

    for idx, unit_row in enumerate(units.itertuples(index=False), start=1):
        match_level = "full_pool_unadjusted" if not corrected else "full_pool"
        target_row = sampled_targets.iloc[idx - 1].to_dict() if corrected else None
        chosen_pid: str | None = None

        if corrected and target_row is not None:
            available_fields = [
                field for field in match_fields if _clean_text(target_row.get(field)) is not None
            ]
            for size in range(len(available_fields), 0, -1):
                found = False
                for subset in combinations(available_fields, size):
                    key = tuple(target_row[field] for field in subset)
                    candidates = candidate_maps.get(tuple(subset), {}).get(key, [])
                    if not candidates:
                        continue
                    chosen_pid = _choose_pid(
                        candidates,
                        reuse_counter,
                        rng,
                        weighted_by_inverse_reuse=True,
                    )
                    match_level = "+".join(subset)
                    found = True
                    break
                if found:
                    break

        if chosen_pid is None:
            chosen_pid = _choose_pid(
                all_pids,
                reuse_counter,
                rng,
                weighted_by_inverse_reuse=corrected,
            )

        reuse_before = int(reuse_counter.get(chosen_pid, 0))
        reuse_counter[chosen_pid] += 1
        persona = persona_by_pid[chosen_pid]
        assignment = {
            "unit_id": str(unit_row.unit_id),
            "twin_pid": chosen_pid,
            "headline": persona["headline"],
            "summary": persona["summary"],
            "background_summary": persona["background_summary"],
            "age_bracket": persona["age_bracket"],
            "sex_assigned_at_birth": persona["sex_assigned_at_birth"],
            "education_harmonized": persona["education_harmonized"],
            "match_level": match_level,
            "reuse_count_before": reuse_before,
            "reuse_count_after": int(reuse_counter[chosen_pid]),
        }
        if target_row is not None:
            assignment["sampled_target_source_row_id"] = target_row.get("source_row_id")
            for field in match_fields:
                assignment[f"target_{field}"] = target_row.get(field)
        assignments.append(assignment)
        unit_to_assignment[str(unit_row.unit_id)] = assignment

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "unit_assignments.jsonl"
    _write_jsonl(assignments_path, assignments)
    pd.DataFrame(assignments).to_csv(output_dir / "unit_assignments.csv", index=False)
    summary = {
        "dataset_key": dataset_key,
        "variant": VARIANT_TWIN_CORRECTED if corrected else VARIANT_TWIN_UNADJUSTED,
        "seed": seed,
        "unit_count": len(units),
        "twin_pool_size": int(len(twin_personas)),
        "unique_twin_profiles_used": int(len(set(a["twin_pid"] for a in assignments))),
        "mean_reuse_count": float(np.mean([a["reuse_count_after"] for a in assignments])),
        "match_level_counts": dict(Counter(a["match_level"] for a in assignments)),
        "assignments_file": str(assignments_path),
        "source_profile_cards_file": str(TWIN_CARDS_JSONL),
        "source_profiles_file": str(TWIN_PROFILES_JSONL),
    }
    if corrected:
        summary["source_row_count"] = int(len(demographic_source))
        summary["matching_fields"] = list(match_fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return unit_to_assignment, assignments_path


def _common_shared_note_lines() -> list[str]:
    return [
        "# Shared Prompt Notes",
        "",
        "## General Interpretation Note",
        "",
        "- These profiles summarize prior survey and behavioral-task evidence about each participant.",
        "- Treat the cues as relative tendencies, not deterministic predictions for any single decision in this study.",
        "- Unless a player-specific limit is listed, shared methodological caveats apply to all profiles.",
        "",
        "## Cue Glossary",
        "",
        "### Cooperation orientation",
        "- Meaning: blend of one-shot sharing behavior and cooperation/prosocial self-report.",
        "- Built from: trust-game sending, dictator giving, cooperation/competition items, agreeableness/helpfulness items, and prosocial values.",
        "",
        "### Conditional cooperation",
        "- Meaning: reciprocity and fairness-threshold sensitivity rather than a repeated-game reaction function.",
        "- Built from: trust-game return behavior and ultimatum acceptance-threshold signals.",
        "",
        "### Norm enforcement",
        "- Meaning: resistance to unfair splits and revenge/low-forgiveness cues in ultimatum-like contexts.",
        "- Built from: ultimatum minimum acceptable amounts plus revenge/forgiveness self-report.",
        "",
        "### Generosity without return",
        "- Meaning: willingness to give when repayment incentives are weak or absent.",
        "- Built from: dictator giving, trust-game sending, and prosocial/helpfulness cues.",
        "",
        "### Exploitation caution",
        "- Meaning: guardedness against being taken advantage of.",
        "- Built from: lower trustingness, stricter acceptance thresholds, uncertainty aversion, self-reliance, and revenge tendency.",
        "",
        "### Communication/coordination",
        "- Meaning: indirect cue for likely social expressiveness and coordination readiness.",
        "- Built from: empathy, social-sensitivity/self-monitoring, and extraversion-related self-report.",
        "",
        "### Behavioral stability",
        "- Meaning: rule-like internal consistency across self-regulation items.",
        "- Built from: conscientiousness-related items, self-concept clarity, and lower volatility-related personality items.",
    ]


def _dataset_specific_caveats(dataset_key: str) -> list[str]:
    if dataset_key == "minority_game_bret_njzas":
        return [
            "- Twin does not directly observe repeated minority-game switching, herding, or BRET-style box collection.",
            "- Trust, ultimatum, dictator, uncertainty-aversion, and self-regulation evidence may still transfer as broad priors about cooperation, guardedness, and consistency.",
            "- Communication/coordination is indirect: the source tasks do not directly observe repeated strategic group play.",
        ]
    if dataset_key == "longitudinal_trust_game_ht863":
        return [
            "- Twin includes direct one-shot trust-game evidence, which is relevant here, but this benchmark asks for repeated 1-9 willingness-to-play ratings rather than a single binary trust choice.",
            "- Norm-enforcement cues are secondary in this task because the focal decision is whether to enter a trust interaction, not whether to punish unfairness.",
            "- The repeated ten-session format is not directly observed in Twin, so use these cues as priors for the participant rather than as a literal trajectory template.",
        ]
    if dataset_key == "two_stage_trust_punishment_y2hgu":
        return [
            "- Twin has relevant trust, fairness, dictator, empathy, and uncertainty-aversion evidence, but it does not directly observe checking a cost or impact before acting.",
            "- Twin also does not directly observe deliberation-speed signaling, so fast versus slow should be treated as a coarse behavioral style inference rather than a measured trait.",
            "- Norm-enforcement and generosity cues are especially relevant here, but they are still indirect priors rather than exact predictions for punishment or helping in this design.",
        ]
    if dataset_key == "multi_game_llm_fvk2c":
        return [
            "- Twin overlaps closely with trust, ultimatum, and dictator-style evidence, but it does not directly observe AI delegation, stag-hunt choice, or five-option coordination choice.",
            "- AI-use decisions in this benchmark are therefore transfer tasks from broader social and decision-style evidence, not direct matches to Twin items.",
            "- Communication/coordination remains indirect because Twin does not directly observe interactive AI-supported social play.",
        ]
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def _write_shared_notes_file(dataset_key: str, output_path: Path) -> None:
    lines = _common_shared_note_lines()
    caveats = _dataset_specific_caveats(dataset_key)
    insert_at = lines.index("## Cue Glossary")
    lines = [
        *lines[:insert_at],
        "## Shared Caveats",
        "",
        *[f"- {line[2:]}" if line.startswith("- ") else line for line in caveats],
        "",
        *lines[insert_at:],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _render_demographic_profile_block(card: dict[str, Any]) -> str:
    lines = [
        "# PARTICIPANT PROFILE",
        "Use this sampled profile as lightweight background context. It was sampled from the study population's observed demographic distribution and is not the real participant.",
        "",
        f"Summary: {card['summary']}",
    ]
    markdown = str(card.get("markdown", "")).strip()
    if markdown:
        lines.extend(["Details:", markdown])
    return "\n".join(lines)


def _render_twin_profile_block(
    assignment: dict[str, Any],
    card: dict[str, Any],
    shared_notes: str,
    corrected: bool,
) -> str:
    lines = [
        "# PARTICIPANT PERSONA",
        (
            "Use this sampled persona as a prior about likely tendencies. "
            "It was sampled from Twin to match the target study's observed demographic distribution and is not the real participant."
            if corrected
            else "Use this sampled persona as a prior about likely tendencies. It was sampled from the Twin pool without demographic correction and is not the real participant."
        ),
        "",
        shared_notes.strip(),
        "",
        f"Headline: {card.get('headline', assignment.get('headline', ''))}",
        f"Summary: {card.get('summary', assignment.get('summary', ''))}",
    ]
    background_summary = str((card.get("background") or {}).get("summary", "")).strip()
    if background_summary:
        lines.append(f"Background: {background_summary}")

    behavioral_signature = card.get("behavioral_signature", [])
    if behavioral_signature:
        lines.append("Behavioral Signature:")
        for item in behavioral_signature:
            lines.append(f"- {item}")

    observed_anchors = card.get("observed_anchors", [])
    if observed_anchors:
        lines.append("Observed Anchors:")
        for item in observed_anchors:
            title = str(item.get("title", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if title and detail:
                lines.append(f"- {title}: {detail}")

    transfer_relevance = card.get("transfer_relevance", [])
    if transfer_relevance:
        lines.append("Transfer-Relevant Cues:")
        for item in transfer_relevance:
            cue = str(item.get("cue", "")).strip()
            cue_name = TWIN_TRANSFER_CUE_DISPLAY_NAMES.get(
                cue, cue.replace("_", " ").strip().title()
            )
            label = str(item.get("label", "")).replace("_", " ").strip()
            score = item.get("score_0_to_100", "")
            confidence = str(item.get("confidence", "")).strip()
            lines.append(f"- {cue_name}: {label} ({score}), confidence {confidence}")

    limits = card.get("limits", [])
    if limits:
        lines.append("Limits:")
        for item in limits:
            topic = str(item.get("topic", "")).strip()
            note = str(item.get("note", "")).strip()
            if topic and note:
                lines.append(f"- {topic}: {note}")
            elif note:
                lines.append(f"- {note}")

    return "\n".join(lines)


def _estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    tokens = 3
    for message in messages:
        tokens += 3
        tokens += len(TOKEN_ENCODING.encode(message.get("role", "")))
        tokens += len(TOKEN_ENCODING.encode(message.get("content", "")))
    return int(tokens)


def _batch_entry(custom_id: str, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _maybe_sample_bundle(
    bundle: DatasetBundle,
    *,
    max_records_per_treatment: int | None,
    sampling_seed: int,
) -> DatasetBundle:
    if max_records_per_treatment is None:
        return bundle
    if max_records_per_treatment <= 0:
        raise ValueError("max_records_per_treatment must be positive when provided.")

    sampled_groups: list[pd.DataFrame] = []
    for _, group in bundle.records.groupby("treatment_name", sort=True):
        if len(group) <= max_records_per_treatment:
            sampled = group.copy()
        else:
            sampled = group.sample(n=max_records_per_treatment, random_state=sampling_seed)
        sampled_groups.append(sampled)

    sampled_records = (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["treatment_name", "record_id"])
        .reset_index(drop=True)
    )
    sampled_unit_ids = set(sampled_records["unit_id"].astype(str))
    sampled_units = (
        bundle.units[bundle.units["unit_id"].astype(str).isin(sampled_unit_ids)]
        .copy()
        .sort_values("unit_id")
        .reset_index(drop=True)
    )
    return DatasetBundle(
        dataset_key=bundle.dataset_key,
        display_name=bundle.display_name,
        records=sampled_records,
        units=sampled_units,
        demographic_source=bundle.demographic_source,
        twin_matching_fields=bundle.twin_matching_fields,
    )


def _build_minority_bundle(repo_root: Path) -> DatasetBundle:
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
        age = _clean_numeric_string(row.get("Age"))
        sex = _minority_sex_to_display(row.get("Sex"))
        education = _clean_text(row.get("Highest education level completed"))
        ethnicity = _clean_text(row.get("Ethnicity simplified")) or _clean_text(row.get("Ethnicity_simplified"))
        country_of_birth = _clean_text(row.get("Country of birth")) or _clean_text(row.get("Country_of_birth"))
        country_of_residence = _clean_text(row.get("Country of residence")) or _clean_text(row.get("Country_of_residence"))
        nationality = _clean_text(row.get("Nationality"))
        student_status = _clean_text(row.get("Student status")) or _clean_text(row.get("Student_status"))
        employment = _clean_text(row.get("Employment status")) or _clean_text(row.get("Employment_status"))
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
                "matching_age_bracket": _age_to_bracket(age),
                "matching_sex": _canonical_sex(sex),
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


def _build_longitudinal_bundle(repo_root: Path) -> DatasetBundle:
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

    demo_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(day10_demo.itertuples(index=False), start=1):
        age = _clean_numeric_string(row.Q50)
        sex = _longitudinal_gender_to_display(row.Q49)
        summary, markdown = _simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"longitudinal_demo_source_{idx:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": _age_to_bracket(age),
                "matching_sex": _longitudinal_gender_to_match(row.Q49),
                "matching_education": None,
            }
        )
    demographic_source = pd.DataFrame(demo_rows)

    record_rows: list[dict[str, Any]] = []
    units = [{"unit_id": pid} for pid in complete_pids]
    for pid in complete_pids:
        person = full[full["Q52"] == pid].copy().sort_values("day")
        days_payload: list[dict[str, Any]] = []
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


def _make_two_stage_record(
    *,
    record_id: str,
    unit_id: str,
    treatment_name: str,
    experiment_code: str,
    role: str,
    visibility: str,
    schema_type: str,
    action_context: str,
    deliberation_dimension: str,
    gold_target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "unit_id": unit_id,
        "treatment_name": treatment_name,
        "experiment_code": experiment_code,
        "role": role,
        "visibility": visibility,
        "schema_type": schema_type,
        "action_context": action_context,
        "deliberation_dimension": deliberation_dimension,
        "gold_target_json": json.dumps(gold_target),
    }


def _yes_no(value: Any) -> str:
    return "YES" if int(value) == 1 else "NO"


def _first_present_value(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(row, dict):
            if key in row:
                return row[key]
        else:
            if key in row.index:
                return row[key]
    raise KeyError(f"None of the requested keys were present: {keys}")


def _build_two_stage_bundle(repo_root: Path) -> DatasetBundle:
    data_dir = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "two_stage_trust_punishment_y2hgu"
        / "Data Files"
    )
    config_map = {
        "helpcostcheckE1.csv": ("E1", "help", "cost"),
        "puncostcheckE2.csv": ("E2", "punish", "cost"),
        "helpimpactcheckE4.csv": ("E4", "help", "impact"),
        "punimpactcheckE5.csv": ("E5", "punish", "impact"),
    }

    record_rows: list[dict[str, Any]] = []
    demo_rows: list[dict[str, Any]] = []
    demo_counter = 0

    for filename, (experiment_code, action_context, deliberation_dimension) in config_map.items():
        df = pd.read_csv(data_dir / filename, sep=";")
        for row_index, row in df.iterrows():
            demo_counter += 1
            age = _clean_numeric_string(row.get("age"))
            sex = _two_stage_gender_to_display(row.get("gender"))
            summary, markdown = _simple_demographic_summary(age=age, sex_or_gender=sex)
            demo_rows.append(
                {
                    "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                    "summary": summary,
                    "markdown": markdown,
                    "matching_age_bracket": _age_to_bracket(age),
                    "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                    "matching_education": None,
                }
            )

            condition = int(row["Condition"])
            pid = str(row["PID"])
            if condition in {3, 4}:
                visibility = "observable" if condition == 3 else "hidden"
                check = int(row["checkObs"] if condition == 3 else row["checkHid"])
                action_col_checked = "calcHelp" if action_context == "help" else "calcPun"
                action_col_unchecked = "uncalcHelp" if action_context == "help" else "uncalcPun"
                acted = int(row[action_col_checked] if check == 1 else row[action_col_unchecked])
                target = {"check": _yes_no(check), "act": _yes_no(acted), "return_pct": int(row["return"])}
                treatment_name = f"{experiment_code}_ROLE_A_{visibility.upper()}"
                record_rows.append(
                    _make_two_stage_record(
                        record_id=f"{experiment_code}__A__{pid}",
                        unit_id=f"{experiment_code}__A__{pid}",
                        treatment_name=treatment_name,
                        experiment_code=experiment_code,
                        role="A",
                        visibility=visibility,
                        schema_type="role_a_check",
                        action_context=action_context,
                        deliberation_dimension=deliberation_dimension,
                        gold_target=target,
                    )
                )
            elif condition in {5, 6}:
                visibility = "observable" if condition == 5 else "hidden"
                if visibility == "observable":
                    if action_context == "help":
                        target = {
                            "send_if_act_without_check": int(row["helpUncalc"]),
                            "send_if_act_after_check": int(row["helpCalc"]),
                            "send_if_no_act_without_check": int(row["noUncalc"]),
                            "send_if_no_act_after_check": int(row["noCalc"]),
                        }
                    else:
                        target = {
                            "send_if_act_without_check": int(row["punUncalc"]),
                            "send_if_act_after_check": int(row["punCalc"]),
                            "send_if_no_act_without_check": int(row["noUncalc"]),
                            "send_if_no_act_after_check": int(row["noCalc"]),
                        }
                    schema_type = "role_b_observable_check"
                else:
                    if action_context == "help":
                        target = {
                            "send_if_act": int(_first_present_value(row, "helpedHidB", "helpHid")),
                            "send_if_no_act": int(_first_present_value(row, "noHelpdHidB", "noHelpHid")),
                        }
                    else:
                        target = {
                            "send_if_act": int(row["punHid"]),
                            "send_if_no_act": int(row["noPunHid"]),
                        }
                    schema_type = "role_b_hidden_check"
                treatment_name = f"{experiment_code}_ROLE_B_{visibility.upper()}"
                record_rows.append(
                    _make_two_stage_record(
                        record_id=f"{experiment_code}__B__{pid}",
                        unit_id=f"{experiment_code}__B__{pid}",
                        treatment_name=treatment_name,
                        experiment_code=experiment_code,
                        role="B",
                        visibility=visibility,
                        schema_type=schema_type,
                        action_context=action_context,
                        deliberation_dimension=deliberation_dimension,
                        gold_target=target,
                    )
                )

    # E3 role A and B
    e3a = pd.read_csv(data_dir / "puntimeE3a.csv", sep=";")
    median_decision_time = float(e3a["decisionT"].median())
    for row_index, row in e3a.iterrows():
        demo_counter += 1
        age = _clean_numeric_string(row.get("age"))
        sex = _two_stage_gender_to_display(row.get("gender"))
        summary, markdown = _simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": _age_to_bracket(age),
                "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                "matching_education": None,
            }
        )
        visibility = "observable" if int(row["Condition"]) == 1 else "hidden"
        target = {
            "decision_time_bucket": "FAST" if float(row["decisionT"]) <= median_decision_time else "SLOW",
            "act": _yes_no(row["punishing"]),
            "return_pct": int(row["return"]),
        }
        pid = str(row["ID"])
        record_rows.append(
            _make_two_stage_record(
                record_id=f"E3A__A__{pid}",
                unit_id=f"E3A__A__{pid}",
                treatment_name=f"E3A_ROLE_A_{visibility.upper()}",
                experiment_code="E3A",
                role="A",
                visibility=visibility,
                schema_type="role_a_time",
                action_context="punish",
                deliberation_dimension="decision_time",
                gold_target=target,
            )
        )

    e3b = pd.read_csv(data_dir / "puntimeE3b.csv", sep=";")
    for row_index, row in e3b.iterrows():
        demo_counter += 1
        age = _clean_numeric_string(row.get("age"))
        sex = _two_stage_gender_to_display(row.get("gender"))
        summary, markdown = _simple_demographic_summary(age=age, sex_or_gender=sex)
        demo_rows.append(
            {
                "source_row_id": f"two_stage_demo_source_{demo_counter:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": _age_to_bracket(age),
                "matching_sex": _two_stage_gender_to_match(row.get("gender")),
                "matching_education": None,
            }
        )
        visibility = "observable" if int(row["Condition"]) == 1 else "hidden"
        if visibility == "observable":
            target = {
                "send_if_act_fast": int(row["punFast"]),
                "send_if_no_act_fast": int(row["noFast"]),
                "send_if_act_slow": int(row["punSlow"]),
                "send_if_no_act_slow": int(row["noSlow"]),
            }
            schema_type = "role_b_observable_time"
        else:
            target = {
                "send_if_act": int(row["punHid"]),
                "send_if_no_act": int(row["noPunHid"]),
            }
            schema_type = "role_b_hidden_time"
        pid = str(row["ID"])
        record_rows.append(
            _make_two_stage_record(
                record_id=f"E3B__B__{pid}",
                unit_id=f"E3B__B__{pid}",
                treatment_name=f"E3B_ROLE_B_{visibility.upper()}",
                experiment_code="E3B",
                role="B",
                visibility=visibility,
                schema_type=schema_type,
                action_context="punish",
                deliberation_dimension="decision_time",
                gold_target=target,
            )
        )

    records = pd.DataFrame(record_rows).sort_values(["treatment_name", "record_id"]).reset_index(drop=True)
    units_df = (
        records[["unit_id"]]
        .drop_duplicates("unit_id")
        .sort_values("unit_id")
        .reset_index(drop=True)
    )
    demographic_source = pd.DataFrame(demo_rows)
    return DatasetBundle(
        dataset_key="two_stage_trust_punishment_y2hgu",
        display_name="Two-Stage Trust / Punishment / Helping",
        records=records,
        units=units_df,
        demographic_source=demographic_source,
        twin_matching_fields=["matching_age_bracket", "matching_sex"],
    )


def _recode_multi_game_numeric_like(value: Any) -> Any:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _multi_game_decision_value(raw_value: Any, game: str) -> Any:
    cleaned = _recode_multi_game_numeric_like(raw_value)
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


def _multi_game_scenario_descriptor(scenario_code: str) -> tuple[str, str, int]:
    if scenario_code == "11":
        return "AISupport", "AgainstHuman", 1
    if scenario_code == "21":
        return "NoAISupport", "AgainstHuman", 2
    if scenario_code == "22":
        return "NoAISupport", "AgainstAI", 3
    if scenario_code == "2":
        return "NoAISupport", "Opaque", 2
    raise ValueError(f"Unsupported multi-game scenario code: {scenario_code}")


def _build_multi_game_bundle(repo_root: Path) -> DatasetBundle:
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
            cols = [
                col
                for col in df.columns
                if f"{game}_{scenario_code}" in col and "JUS" not in col
            ]
            df[f"{game}_{scenario_code}"] = df[cols].apply(
                lambda row: "".join(str(value) for value in row if pd.notna(value)),
                axis=1,
            )
        cols = [
            col
            for col in df.columns
            if col.startswith(f"OD_{game}_2") and "JUS" not in col
        ]
        df[f"{game}_2"] = df[cols].apply(
            lambda row: "".join(str(value) for value in row if pd.notna(value)),
            axis=1,
        )
        if game == "TGSender":
            delegation_name = "TGSender_Delegation"
        else:
            delegation_name = f"{game}_Delegation"
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
        age_value = _clean_numeric_string(row.age)
        sex = _multi_game_gender_to_display(row.gender)
        edu = _clean_text(row.education)
        summary, markdown = _simple_demographic_summary(
            age=age_value,
            sex_or_gender=sex,
            education=edu,
        )
        subject_demo_rows.append(
            {
                "source_row_id": f"multi_game_demo_source_{idx:05d}",
                "summary": summary,
                "markdown": markdown,
                "matching_age_bracket": _age_to_bracket(age_value),
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
        scenario_codes = ["11", "21", "22"] if treatment in {
            "TransparentRandom",
            "TransparentDelegation",
        } else ["11", "2"]
        scenario_outputs: list[dict[str, Any]] = []
        scenario_manifest: list[dict[str, Any]] = []
        for scenario_code in scenario_codes:
            scenario, case, scenario_order = _multi_game_scenario_descriptor(scenario_code)
            scenario_outputs.append(
                {
                    "scenario": scenario,
                    "case": case,
                    "UGProposer_decision": _multi_game_decision_value(getattr(row, f"UGProposer_{scenario_code}"), "UGProposer"),
                    "UGResponder_decision": _multi_game_decision_value(getattr(row, f"UGResponder_{scenario_code}"), "UGResponder"),
                    "TGSender_decision": _multi_game_decision_value(getattr(row, f"TGSender_{scenario_code}"), "TGSender"),
                    "TGReceiver_decision": _multi_game_decision_value(getattr(row, f"TGReceiver_{scenario_code}"), "TGReceiver"),
                    "PD_decision": _multi_game_decision_value(getattr(row, f"PD_{scenario_code}"), "PD"),
                    "SH_decision": _multi_game_decision_value(getattr(row, f"SH_{scenario_code}"), "SH"),
                    "C_decision": _multi_game_decision_value(getattr(row, f"C_{scenario_code}"), "C"),
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

    records = pd.DataFrame(record_rows).sort_values(
        ["treatment_name", "record_id"]
    ).reset_index(drop=True)
    return DatasetBundle(
        dataset_key="multi_game_llm_fvk2c",
        display_name="Multi-Game Battery with LLM Delegation",
        records=records,
        units=units_df,
        demographic_source=demographic_source.drop(columns=["unit_id"]),
        twin_matching_fields=["matching_age_bracket", "matching_sex", "matching_education"],
    )


def _build_prompt_minority(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full response in a repeated minority-game study. "
        "Use the task rules, condition, and any provided profile as priors. Return only valid JSON."
    )
    profile_lines = ["", profile_block] if profile_block else []
    user = "\n".join(
        [
            "Forecast one participant's behavior in this study.",
            "",
            "# TASKS",
            "This participant completed two tasks:",
            "1. An 11-round bonus game.",
            "2. A BRET risk task.",
            "",
            "# BONUS GAME RULES",
            "- On each round, the participant chooses `A` or `B`.",
            "- Round 1 always has pot 0.",
            "- From round 2 onward, the current pot depends only on the previous round's choice.",
            "- If the previous choice was `A`, the next-round pots are: 84, 88, 92, 96, 100, 104, 108, 112, 116, 116.",
            "- If the previous choice was `B`, the next-round pots are: 20, 40, 60, 80, 100, 120, 140, 160, 180, 180.",
            "- The crossover is at round 6, where both branches yield 100.",
            "- One round from rounds 2 to 11 is randomly selected for payment.",
            "",
            "# CONDITION",
            (
                "- Deception condition: the participant believed they were playing with real people, although the environment was scripted."
                if int(row["deception_condition"]) == 1
                else "- Non-deception condition: the scripted nature of the environment was not hidden in the same way."
            ),
            *profile_lines,
            "",
            "# BRET RULES",
            "- There are 100 boxes and exactly one hidden bomb.",
            "- The participant chooses how many boxes to collect, from 0 to 100.",
            "- Each collected box earns points, but if the bomb is among the collected boxes, the BRET payoff is 0.",
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "bonus_game_choices": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],',
            '  "bret_boxes": 50',
            "}",
            "- `bonus_game_choices` must contain exactly 11 letters, each either `A` or `B`.",
            "- `bret_boxes` must be an integer from 0 to 100.",
        ]
    )
    return system, user


def _build_prompt_longitudinal(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full response panel in a repeated trust-game study. "
        "Use the task rules, repeated design, and any provided profile as priors. Return only valid JSON."
    )
    profile_lines = ["", profile_block] if profile_block else []
    user = "\n".join(
        [
            "Forecast one participant's full 10-session response panel.",
            "",
            "# TASK RULES",
            "- The same participant completes 10 sessions over 3 weeks.",
            "- Each session contains the same 16 trust-game trials.",
            "- On each trial, the participant sees a partner's past sharing rate and the number of tokens they would need to give.",
            "- The participant responds on a 1 to 9 scale: `1 = Not at all` and `9 = Extremely`.",
            "- If the participant does not play, both sides keep 5 tokens.",
            "- If the participant plays, they give `Y` tokens to the partner and the partner receives `2Y`.",
            "- If the partner shares, both sides end with `5 + Y / 2`.",
            "- If the partner keeps everything, the participant ends with `5 - Y`.",
            "",
            "# TRIAL ORDER",
            "Each day must use this exact 16-trial order:",
            "1.  partner shared 80%, stake 1",
            "2.  partner shared 80%, stake 2",
            "3.  partner shared 80%, stake 4",
            "4.  partner shared 80%, stake 5",
            "5.  partner shared 75%, stake 1",
            "6.  partner shared 75%, stake 2",
            "7.  partner shared 75%, stake 4",
            "8.  partner shared 75%, stake 5",
            "9.  partner shared 70%, stake 1",
            "10. partner shared 70%, stake 2",
            "11. partner shared 70%, stake 4",
            "12. partner shared 70%, stake 5",
            "13. partner shared 65%, stake 1",
            "14. partner shared 65%, stake 2",
            "15. partner shared 65%, stake 4",
            "16. partner shared 65%, stake 5",
            *profile_lines,
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "days": [',
            '    {"day": 1, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 2, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 3, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 4, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 5, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 6, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 7, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 8, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 9, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 10, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}',
            "  ]",
            "}",
            "- There must be exactly 10 day objects, for days 1 through 10.",
            "- Each `ratings` list must contain exactly 16 integers.",
            "- Every rating must be an integer from 1 to 9.",
        ]
    )
    return system, user


def _build_prompt_two_stage(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's structured responses in a two-stage trustworthiness experiment. "
        "Use the experiment rules, role, condition, and any provided profile as priors. Return only valid JSON."
    )
    experiment_code = str(row["experiment_code"])
    role = str(row["role"])
    visibility = str(row["visibility"])
    action_context = str(row["action_context"])
    deliberation_dimension = str(row["deliberation_dimension"])
    schema_type = str(row["schema_type"])

    if experiment_code in {"E1", "E4"}:
        action_phrase = "help a victim of unfair treatment"
        no_action_phrase = "not help"
    else:
        action_phrase = "punish the unfair player"
        no_action_phrase = "not punish"

    if deliberation_dimension == "cost":
        stage1_note = (
            "Before acting, Player A can either check the personal cost or decide without checking. "
            "The unknown personal cost lies between 0p and 10p, and if checked it is revealed to be 5p."
        )
    elif deliberation_dimension == "impact":
        stage1_note = (
            "Before acting, Player A can either check the impact on the target or decide without checking. "
            "The personal cost is fixed at 5p, the unknown impact lies between 1p and 30p, and if checked it is revealed to be 15p."
        )
    else:
        stage1_note = (
            "Player A decides whether to punish after the 5p cost is already known. "
            "The relevant process cue is whether the decision is fast or slow."
        )

    visibility_line = (
        "- Process observable: Player B can see how Player A arrived at the stage-1 decision."
        if visibility == "observable"
        else "- Process hidden: Player B sees only whether Player A acted, not the process by which that choice was made."
    )

    if role == "A":
        output_lines = []
        if schema_type == "role_a_check":
            output_lines = [
                "Return only JSON in this exact schema:",
                "{",
                '  "check": "YES",',
                '  "act": "YES",',
                '  "return_pct": 50',
                "}",
                "- `check` must be `YES` or `NO`.",
                f"- `act` must be `YES` or `NO`, meaning whether Player A chooses to {action_phrase}.",
                "- `return_pct` must be an integer from 0 to 100.",
            ]
        else:
            output_lines = [
                "Return only JSON in this exact schema:",
                "{",
                '  "decision_time_bucket": "FAST",',
                '  "act": "YES",',
                '  "return_pct": 50',
                "}",
                "- `decision_time_bucket` must be `FAST` or `SLOW`.",
                f"- `act` must be `YES` or `NO`, meaning whether Player A chooses to {action_phrase}.",
                "- `return_pct` must be an integer from 0 to 100.",
            ]
        user = "\n".join(
            [
                f"Forecast one participant's decisions in {experiment_code} as Player A.",
                "",
                "# STAGE 1 BACKGROUND",
                "- Player 1 trusted Player 2 with 10p.",
                "- The amount tripled to 30p.",
                "- Player 2 returned 0p.",
                f"- Player A has 10p and can choose whether to {action_phrase}.",
                f"- If Player A acts, the effect on the target is 15p in the checking variants.",
                stage1_note,
                "",
                "# STAGE 2 TRUST GAME",
                "- Player B has 10p and decides how much to send to Player A.",
                "- Any amount sent is tripled.",
                "- Player A returns some percentage of the tripled amount.",
                visibility_line,
                *([ "", profile_block ] if profile_block else []),
                "",
                "# OUTPUT",
                *output_lines,
            ]
        )
        return system, user

    if schema_type == "role_b_observable_check":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act_without_check": 5,',
            '  "send_if_act_after_check": 5,',
            '  "send_if_no_act_without_check": 5,',
            '  "send_if_no_act_after_check": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    elif schema_type == "role_b_hidden_check":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act": 5,',
            '  "send_if_no_act": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    elif schema_type == "role_b_observable_time":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act_fast": 5,',
            '  "send_if_no_act_fast": 5,',
            '  "send_if_act_slow": 5,',
            '  "send_if_no_act_slow": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    else:
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act": 5,',
            '  "send_if_no_act": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]

    role_b_stage1 = [
        f"- Player A previously had the option to {action_phrase} after an unfair interaction.",
        stage1_note,
        visibility_line,
        "- Player B now plays a trust game with Player A using the strategy method.",
        "- Player B has 10p and chooses how much to send to Player A.",
        "- Any amount sent is tripled.",
        "- Player A then returns some percentage of the tripled amount.",
    ]

    user = "\n".join(
        [
            f"Forecast one participant's conditional trust decisions in {experiment_code} as Player B.",
            "",
            "# TASK",
            *role_b_stage1,
            *([ "", profile_block ] if profile_block else []),
            "",
            "# OUTPUT",
            *output_lines,
        ]
    )
    return system, user


def _build_prompt_multi_game(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full decision battery in a multi-game experiment with possible AI delegation. "
        "Use the treatment description, scenario structure, and any provided profile as priors. Return only valid JSON."
    )
    treatment = str(row["Treatment"])
    treatment_code = str(row["TreatmentCode"])
    personalized = int(row["PersonalizedTreatment"])
    scenario_manifest = json.loads(str(row["scenario_manifest_json"]))

    treatment_lines: list[str] = []
    if treatment == "TransparentRandom":
        treatment_lines.extend(
            [
                "- TransparentRandom: AI support or replacement is introduced transparently and is not chosen by the participant.",
                "- The participant does not make voluntary delegation decisions in this treatment.",
            ]
        )
    elif treatment == "TransparentDelegation":
        treatment_lines.extend(
            [
                "- TransparentDelegation: the participant can voluntarily delegate each role to AI.",
                "- If the participant delegates a role, the counterpart knows that delegation occurred.",
            ]
        )
    else:
        treatment_lines.extend(
            [
                "- OpaqueDelegation: the participant can voluntarily delegate each role to AI.",
                "- If the participant delegates a role, the counterpart does not know whether AI was used.",
            ]
        )
    treatment_lines.append(
        "- The AI support is personalized to the participant."
        if personalized == 1
        else "- The AI support is generic rather than personalized."
    )

    scenario_lines = [
        "- Predict the participant's outputs for every scenario below in the listed order.",
    ]
    for item in scenario_manifest:
        scenario_text = str(item["scenario"])
        case_text = str(item["case"])
        if scenario_text == "AISupport":
            support_line = "AI support is present."
        else:
            support_line = "AI support is not present."
        if case_text == "AgainstHuman":
            case_line = "The counterpart is human."
        elif case_text == "AgainstAI":
            case_line = "The counterpart is AI."
        else:
            case_line = "The interaction is opaque with respect to whether the counterpart used AI."
        scenario_lines.append(
            f"- {item['order']}. scenario `{scenario_text}`, case `{case_text}`. {support_line} {case_line}"
        )

    user = "\n".join(
        [
            "Forecast one participant's full decision battery for this experimental run.",
            "",
            "# TREATMENT",
            f"- Treatment code: {treatment_code}",
            *treatment_lines,
            "",
            "# SCENARIOS TO PREDICT",
            *scenario_lines,
            "",
            "# GAME RULES",
            "- Ultimatum Game proposer: choose an offer from 0 to 10.",
            "- Ultimatum Game responder: choose a minimum acceptable offer from 0 to 10.",
            "- Trust Game sender: choose `YES` or `NO` for sending 2 ECU. If sent, it becomes 6 ECU for the receiver.",
            "- Trust Game receiver: if the sender sends, choose how much of 6 ECU to return, from 0 to 6.",
            "- Prisoner's Dilemma: choose `A` or `B` with payoffs (A,A)=5/5, (A,B)=1/8, (B,A)=8/1, (B,B)=3/3.",
            "- Stag Hunt: choose `X` or `Y` with payoffs (X,X)=8/8, (X,Y)=1/5, (Y,X)=5/1, (Y,Y)=4/4.",
            "- Coordination Game: choose one of Mercury, Venus, Earth, Mars, or Saturn. Matching the counterpart yields 5 each; mismatch yields 2 each.",
            *([ "", profile_block ] if profile_block else []),
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "UGProposer_delegated": 0,',
            '  "UGResponder_delegated": 0,',
            '  "TGSender_delegated": 0,',
            '  "TGReceiver_delegated": 0,',
            '  "PD_delegated": 0,',
            '  "SH_delegated": 0,',
            '  "C_delegated": 0,',
            '  "scenario_outputs": [',
            '    {',
            '      "scenario": "AISupport",',
            '      "case": "AgainstHuman",',
            '      "UGProposer_decision": 5,',
            '      "UGResponder_decision": 3,',
            '      "TGSender_decision": "YES",',
            '      "TGReceiver_decision": 2,',
            '      "PD_decision": "A",',
            '      "SH_decision": "X",',
            '      "C_decision": "Earth"',
            "    }",
            "  ]",
            "}",
            "- In voluntary-delegation treatments, each `*_delegated` field records whether the participant delegated that role in the AI-support version of the treatment and must be `0` or `1`.",
            "- In `TransparentRandom`, every `*_delegated` field must be `null` because delegation is not the participant's choice.",
            f"- `scenario_outputs` must contain exactly {int(row['num_scenarios'])} objects in the order listed under `# SCENARIOS TO PREDICT`.",
            "- Each scenario object must repeat the exact `scenario` and `case` labels it is predicting.",
            "- In `AISupport` scenarios, if a role is delegated, set the corresponding `*_decision` field to `null`.",
            "- In `NoAISupport` scenarios, direct decisions may still be non-null even when the corresponding `*_delegated` field is `1`, because the row asks for the participant's own no-AI choice in that scenario.",
            "- `UGProposer_decision` and `UGResponder_decision` must be integers from 0 to 10 or `null`.",
            "- `TGSender_decision` must be `YES`, `NO`, or `null`.",
            "- `TGReceiver_decision` must be an integer from 0 to 6 or `null`.",
            "- `PD_decision` must be `A`, `B`, or `null`.",
            "- `SH_decision` must be `X`, `Y`, or `null`.",
            "- `C_decision` must be one of Mercury, Venus, Earth, Mars, Saturn, or `null`.",
        ]
    )
    return system, user


def _prompt_builder_for_dataset(dataset_key: str):
    mapping = {
        "minority_game_bret_njzas": _build_prompt_minority,
        "longitudinal_trust_game_ht863": _build_prompt_longitudinal,
        "two_stage_trust_punishment_y2hgu": _build_prompt_two_stage,
        "multi_game_llm_fvk2c": _build_prompt_multi_game,
    }
    if dataset_key not in mapping:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    return mapping[dataset_key]


def _bundle_for_dataset(dataset_key: str, repo_root: Path) -> DatasetBundle:
    mapping = {
        "minority_game_bret_njzas": _build_minority_bundle,
        "longitudinal_trust_game_ht863": _build_longitudinal_bundle,
        "two_stage_trust_punishment_y2hgu": _build_two_stage_bundle,
        "multi_game_llm_fvk2c": _build_multi_game_bundle,
    }
    if dataset_key not in mapping:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    return mapping[dataset_key](repo_root)


def _prepare_profile_assets(
    bundle: DatasetBundle,
    forecasting_root: Path,
    seed: int,
) -> dict[str, dict[str, Any]]:
    profile_sampling_root = forecasting_root / "profile_sampling"
    twin_notes_path = profile_sampling_root / "twin_prompt_assets" / "shared_prompt_notes.md"
    _write_shared_notes_file(bundle.dataset_key, twin_notes_path)
    shared_notes = twin_notes_path.read_text(encoding="utf-8").strip()

    twin_personas, twin_cards_by_pid = _load_twin_personas(TWIN_PROFILES_JSONL, TWIN_CARDS_JSONL)

    demographic_cards_by_unit, demographic_assignments_path = _sample_demographic_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_DEMOGRAPHIC_ONLY / "seed_0",
        dataset_key=bundle.dataset_key,
    )
    twin_corrected_assignments_by_unit, twin_corrected_assignments_path = _sample_twin_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        twin_personas=twin_personas,
        match_fields=bundle.twin_matching_fields,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_TWIN_CORRECTED / "seed_0",
        dataset_key=bundle.dataset_key,
        corrected=True,
    )
    twin_unadjusted_assignments_by_unit, twin_unadjusted_assignments_path = _sample_twin_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        twin_personas=twin_personas,
        match_fields=bundle.twin_matching_fields,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_TWIN_UNADJUSTED / "seed_0",
        dataset_key=bundle.dataset_key,
        corrected=False,
    )

    demographic_blocks = {
        unit_id: _render_demographic_profile_block(card)
        for unit_id, card in demographic_cards_by_unit.items()
    }
    twin_corrected_blocks = {
        unit_id: _render_twin_profile_block(
            assignment=twin_corrected_assignments_by_unit[unit_id],
            card=twin_cards_by_pid[twin_corrected_assignments_by_unit[unit_id]["twin_pid"]],
            shared_notes=shared_notes,
            corrected=True,
        )
        for unit_id in twin_corrected_assignments_by_unit
    }
    twin_unadjusted_blocks = {
        unit_id: _render_twin_profile_block(
            assignment=twin_unadjusted_assignments_by_unit[unit_id],
            card=twin_cards_by_pid[twin_unadjusted_assignments_by_unit[unit_id]["twin_pid"]],
            shared_notes=shared_notes,
            corrected=False,
        )
        for unit_id in twin_unadjusted_assignments_by_unit
    }

    return {
        VARIANT_BASELINE: {
            "profile_block_by_unit": {},
            "assignment_file": None,
            "cards_file": None,
            "shared_notes_file": None,
        },
        VARIANT_DEMOGRAPHIC_ONLY: {
            "profile_block_by_unit": demographic_blocks,
            "assignment_file": str(demographic_assignments_path),
            "cards_file": str(
                profile_sampling_root
                / VARIANT_DEMOGRAPHIC_ONLY
                / "seed_0"
                / "demographic_profile_cards.jsonl"
            ),
            "shared_notes_file": None,
        },
        VARIANT_TWIN_CORRECTED: {
            "profile_block_by_unit": twin_corrected_blocks,
            "assignment_file": str(twin_corrected_assignments_path),
            "cards_file": str(TWIN_CARDS_JSONL),
            "shared_notes_file": str(twin_notes_path),
        },
        VARIANT_TWIN_UNADJUSTED: {
            "profile_block_by_unit": twin_unadjusted_blocks,
            "assignment_file": str(twin_unadjusted_assignments_path),
            "cards_file": str(TWIN_CARDS_JSONL),
            "shared_notes_file": str(twin_notes_path),
        },
    }


def _build_run(
    *,
    bundle: DatasetBundle,
    forecasting_root: Path,
    variant: str,
    model: str,
    profile_assets: dict[str, dict[str, Any]],
    sampling_spec: SamplingSpec,
) -> None:
    model_slug = MODEL_SLUGS[model]
    run_name = f"{variant}_{model_slug}"

    batch_input_dir = forecasting_root / "batch_input"
    batch_output_dir = forecasting_root / "batch_output"
    metadata_root = forecasting_root / "metadata"
    results_root = forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata_dir = metadata_root / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = metadata_dir / "sample_prompts"
    sample_dir.mkdir(parents=True, exist_ok=True)

    prompt_builder = _prompt_builder_for_dataset(bundle.dataset_key)
    profile_block_by_unit = profile_assets[variant]["profile_block_by_unit"]

    batch_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    request_manifest_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    sample_written = False
    for row in bundle.records.itertuples(index=False):
        record_series = pd.Series(row._asdict())
        profile_block = profile_block_by_unit.get(str(row.unit_id))
        system_prompt, user_prompt = prompt_builder(record_series, profile_block)
        custom_id = f"{bundle.dataset_key}__{_sanitize_token(str(row.record_id))}"
        batch_row = _batch_entry(custom_id, model, system_prompt, user_prompt)
        batch_rows.append(batch_row)
        gold_rows.append(
            {
                "custom_id": custom_id,
                "record_id": str(row.record_id),
                "unit_id": str(row.unit_id),
                "treatment_name": str(row.treatment_name),
                "gold_target": json.loads(str(row.gold_target_json)),
            }
        )
        manifest_row = {
            "custom_id": custom_id,
            "record_id": str(row.record_id),
            "unit_id": str(row.unit_id),
            "treatment_name": str(row.treatment_name),
            "variant": variant,
            "model": model,
        }
        for column in bundle.records.columns:
            if column in {"record_id", "unit_id", "treatment_name", "gold_target_json"}:
                continue
            manifest_row[column] = record_series[column]
        request_manifest_rows.append(manifest_row)
        token_rows.append(
            {
                "custom_id": custom_id,
                "input_token_estimate": _estimate_input_tokens(batch_row["body"]["messages"]),
            }
        )

        if not sample_written:
            sample_text = "\n\n".join(
                [
                    "=== SYSTEM MESSAGE ===",
                    system_prompt,
                    "=== USER MESSAGE ===",
                    user_prompt,
                ]
            )
            (sample_dir / "sample_prompt.txt").write_text(sample_text, encoding="utf-8")
            sample_written = True

    batch_path = batch_input_dir / f"{run_name}.jsonl"
    _write_jsonl(batch_path, batch_rows)
    _write_jsonl(metadata_dir / "gold_targets.jsonl", gold_rows)
    bundle.records.to_csv(metadata_dir / "selected_records.csv", index=False)
    bundle.units.to_csv(metadata_dir / "selected_units.csv", index=False)
    pd.DataFrame(request_manifest_rows).to_csv(metadata_dir / "request_manifest.csv", index=False)
    token_df = pd.DataFrame(token_rows)
    token_df.to_csv(metadata_dir / "request_token_estimates.csv", index=False)
    token_summary = {
        "request_file": str(batch_path),
        "num_requests": int(len(token_df)),
        "total_input_tokens_estimate": int(token_df["input_token_estimate"].sum()),
        "mean_input_tokens_estimate": float(token_df["input_token_estimate"].mean()),
        "median_input_tokens_estimate": float(token_df["input_token_estimate"].median()),
        "max_input_tokens_estimate": int(token_df["input_token_estimate"].max()),
        "min_input_tokens_estimate": int(token_df["input_token_estimate"].min()),
    }
    (metadata_dir / "request_token_estimates.json").write_text(
        json.dumps(token_summary, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "dataset_key": bundle.dataset_key,
        "display_name": bundle.display_name,
        "run_name": run_name,
        "variant_name": variant,
        "model": model,
        "record_count": int(len(bundle.records)),
        "unit_count": int(len(bundle.units)),
        "max_records_per_treatment": sampling_spec.max_records_per_treatment,
        "sampling_seed": sampling_spec.sampling_seed,
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
        "metadata_dir": str(metadata_dir),
        "profile_assignment_file": profile_assets[variant]["assignment_file"],
        "profile_cards_file": profile_assets[variant]["cards_file"],
        "profile_shared_notes_file": profile_assets[variant]["shared_notes_file"],
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_dataset_runs(
    *,
    dataset_key: str,
    forecasting_root: Path,
    repo_root: Path,
    models: list[str],
    variants: list[str],
    seed: int,
    max_records_per_treatment: int | None,
) -> None:
    sampling_spec = SamplingSpec(
        max_records_per_treatment=max_records_per_treatment,
        sampling_seed=seed,
    )
    bundle = _bundle_for_dataset(dataset_key, repo_root)
    bundle = _maybe_sample_bundle(
        bundle,
        max_records_per_treatment=sampling_spec.max_records_per_treatment,
        sampling_seed=sampling_spec.sampling_seed,
    )
    profile_assets = _prepare_profile_assets(bundle, forecasting_root, seed)
    for variant in variants:
        for model in models:
            _build_run(
                bundle=bundle,
                forecasting_root=forecasting_root,
                variant=variant,
                model=model,
                profile_assets=profile_assets,
                sampling_spec=sampling_spec,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-PGG forecasting batch inputs with baseline, demographic-only, and Twin-sampled variants."
    )
    parser.add_argument(
        "--dataset-key",
        required=True,
        choices=[
            "minority_game_bret_njzas",
            "longitudinal_trust_game_ht863",
            "two_stage_trust_punishment_y2hgu",
            "multi_game_llm_fvk2c",
        ],
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    parser.add_argument("--max-records-per-treatment", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown_models = [model for model in models if model not in MODEL_SLUGS]
    if unknown_models:
        raise ValueError(f"Unsupported model names: {unknown_models}")
    for variant in variants:
        if variant not in ALL_VARIANTS:
            raise ValueError(f"Unsupported variant name: {variant}")
    build_dataset_runs(
        dataset_key=args.dataset_key,
        forecasting_root=args.forecasting_root,
        repo_root=args.repo_root,
        models=models,
        variants=variants,
        seed=args.seed,
        max_records_per_treatment=args.max_records_per_treatment,
    )


if __name__ == "__main__":
    main()
