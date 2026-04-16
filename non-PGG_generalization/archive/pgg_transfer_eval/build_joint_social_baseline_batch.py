#!/usr/bin/env python3
"""Build OpenAI Batch JSONL requests for the joint social-game baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download


REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
CONFIG = "wave_split"
QUESTION_CATALOG_FILE = "question_catalog_and_human_response_csv/question_catalog.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_CSV = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "task_grounding"
    / "twin_question_inventory.csv"
)
LOCAL_WAVE_SPLIT = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "wave_split_dataset"
)
LOCAL_QUESTION_CATALOG = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / QUESTION_CATALOG_FILE
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_baseline"
)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_VARIANT = "baseline_no_retrieval"

TARGET_FAMILIES = ["trust", "ultimatum", "dictator"]
ALLOWED_INPUT_FAMILIES = [
    "demographics",
    "personality",
    "cognitive_tests",
    "mental_accounting",
    "time_preference",
    "risk_preference_gain",
    "risk_preference_loss",
]
STRUCTURED_ALLOWED_INPUT_FAMILIES = [
    "personality",
    "mental_accounting",
    "time_preference",
    "risk_preference_gain",
    "risk_preference_loss",
]
PROFILE_FAMILY_ORDER = [
    "demographics",
    "personality",
    "cognitive_tests",
    "mental_accounting",
    "time_preference",
    "risk_preference_gain",
    "risk_preference_loss",
]
TARGET_FAMILY_ORDER = ["trust", "ultimatum", "dictator"]

PROMPT_VARIANTS = [
    "baseline_no_retrieval",
    "relevant_structured_summary",
    "relevant_structured_summary_tuned",
]

STRUCTURED_Q25_ROWS = [
    ("helpful_unselfish", "Is helpful and unselfish with others"),
    ("forgiving", "Has a forgiving nature"),
    ("generally_trusting", "Is generally trusting"),
    ("cold_aloof", "Can be cold and aloof"),
    ("considerate_kind", "Is considerate and kind to almost everyone"),
    ("starts_quarrels", "Starts quarrels with others"),
    ("rude_to_others", "Is sometimes rude to others"),
    ("likes_to_cooperate", "Likes to cooperate with others"),
]
STRUCTURED_Q27_ROWS = [
    ("resentful_when_denied", "I sometimes feel resentful when I don't get my way"),
    ("took_advantage", "There have been occasions when I took advantage of someone"),
    ("get_even", "I sometimes try to get even rather than forgive and forget"),
    ("jealous_of_others", "There have been times when I was quite jealous of the good fortune of others"),
    ("irritated_by_favors", "I am sometimes irritated by people who ask favors of me"),
    ("always_courteous", "I am always courteous, even to people who are disagreeable"),
]
STRUCTURED_Q29_ROWS = [
    ("trust_value", "TRUST (being true to one’s word, assuming good in others)"),
    ("humility_value", "HUMILITY (appreciating others, being modest about oneself)"),
    ("altruism_value", "ALTRUISM (helping others in need)"),
    ("loyalty_value", "LOYALTY (being faithful to friends, family, and group)"),
    ("politeness_value", "POLITENESS (courtesy, good manners)"),
    ("harmony_value", "HARMONY (good relations, balance, wholeness)"),
    ("honesty_value", "HONESTY (being genuine, sincere)"),
    ("compassion_value", "COMPASSION (caring for others, displaying kindness)"),
    ("equality_value", "EQUALITY (human rights and equal opportunity for all)"),
    ("power_value", "POWER (control over others, dominance)"),
    ("superiority_value", "SUPERIORITY (defeating the competition, standing on top)"),
]
STRUCTURED_Q232_ROWS = [
    ("feel_friend_sadness", "After being with a friend who is sad about something, I usually feel sad."),
    ("understand_friend_happiness", "I can understand my friend’s happiness when she/he does well at something."),
    ("caught_up_in_others_feelings", "I get caught up in other people’s feelings easily."),
    ("understand_when_down", "When someone is feeling down I can usually understand how they feel."),
    ("infer_feelings_before_told", "I can often understand how people are feeling even before they tell me."),
    ("realize_friend_angry", "I can usually realize quickly when a friend is angry."),
]
STRUCTURED_Q233_ROWS = [
    ("depend_on_self", "I'd rather depend on myself than others"),
    ("winning_is_everything", "Winning is everything"),
    ("competition_is_natural", "Competition is the law of nature"),
    ("coworker_prize_pride", "If a co-worker gets a prize, I would feel proud"),
    ("coworker_wellbeing", "The well-being of my coworkers is important to me"),
    ("pleasure_with_others", "To me, pleasure is spending time with others"),
    ("cooperate_feels_good", "I feel good when I cooperate with others"),
]
STRUCTURED_Q236_ROWS = [
    ("adjust_behavior", "In social situations, I have the ability to alter my behavior if I feel that something else is called for."),
    ("read_true_emotions", "I am often able to read people's true emotions correctly through their eyes."),
    ("sensitive_to_expression", "In conversations, I am sensitive to even the slightest change in the facial expression of the person I'm conversing with."),
    ("understand_emotions_motives", "My powers of intuition are quite good when it comes to understanding others' emotions and motives."),
    ("detect_inappropriate", "I can usually tell when I've said something inappropriate by reading it in the listener's eyes."),
    ("detect_lying", "If someone is lying to me, I usually know it at once from that person's manner of expression."),
]
STRUCTURED_Q238_ROWS = [
    ("dislike_uncertainty", "I don’t like situations that are uncertain."),
    ("prefer_order", "I find that a well ordered life with regular hours suits my temperament."),
    ("relieved_after_decision", "When I have made a decision, I feel relieved."),
    ("reach_solution_quickly", "When I am confronted with a problem, I’m dying to reach a solution very quickly."),
    ("dislike_unpredictable", "I dislike unpredictable situations."),
]
MENTAL_ACCOUNTING_QIDS = ["QID149", "QID150", "QID151", "QID152"]
TIME_PREFERENCE_QIDS = ["QID84", "QID244", "QID245", "QID246", "QID247", "QID248"]
RISK_GAIN_QIDS = ["QID250", "QID251", "QID252"]
RISK_LOSS_QIDS = ["QID276", "QID277", "QID278", "QID279"]

TARGET_ROLE_METADATA = {
    "QID117": ("Trust sender", False),
    "QID118": ("Trust receiver", False),
    "QID119": ("Trust receiver", False),
    "QID120": ("Trust receiver", False),
    "QID121": ("Trust receiver", False),
    "QID122": ("Trust receiver", False),
    "QID224": ("Ultimatum proposer", False),
    "QID225": ("Ultimatum receiver", True),
    "QID226": ("Ultimatum receiver", True),
    "QID227": ("Ultimatum receiver", True),
    "QID228": ("Ultimatum receiver", True),
    "QID229": ("Ultimatum receiver", True),
    "QID230": ("Ultimatum receiver", True),
    "QID231": ("Dictator allocator", False),
}


def load_inventory(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def load_wave_split():
    try:
        return load_dataset(REPO_ID, CONFIG)["data"]
    except Exception:
        if not LOCAL_WAVE_SPLIT.exists():
            raise
        return load_from_disk(str(LOCAL_WAVE_SPLIT))["data"]


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def shorten(text: str, limit: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def ref_for_parts(block_name: str, question_id: str) -> str:
    return f"{block_name}::{question_id}"


def ref_for_row(row: Dict[str, str]) -> str:
    return ref_for_parts(row["block_name"], row["question_id"])


def build_source_by_ref(catalog: Iterable[Dict]) -> Dict[str, str]:
    source_by_ref: Dict[str, str] = {}
    for q in catalog:
        ref = ref_for_parts(str(q.get("BlockName", "")), str(q.get("QuestionID", "")))
        source_by_ref[ref] = str(q.get("source", ""))
    return source_by_ref


def question_sort_key(question_id: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in question_id if ch.isdigit())
    if digits:
        return int(digits), question_id
    return 10**9, question_id


def inventory_sort_key(row: Dict[str, str], family_order: Sequence[str]) -> Tuple[int, int, str, str]:
    try:
        family_idx = family_order.index(row["family"])
    except ValueError:
        family_idx = len(family_order)
    q_num, q_raw = question_sort_key(row["question_id"])
    return family_idx, q_num, q_raw, row["block_name"]


def select_allowed_and_target_refs(
    inventory_rows: Sequence[Dict[str, str]],
    source_by_ref: Dict[str, str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    allowed_rows: List[Dict[str, str]] = []
    target_rows: List[Dict[str, str]] = []

    for row in inventory_rows:
        ref = ref_for_row(row)
        if source_by_ref.get(ref) != "wave1_3_persona_json":
            continue
        if row["question_type"] == "DB":
            continue
        if int(row["n_csv_columns"] or 0) <= 0:
            continue

        if row["family"] in ALLOWED_INPUT_FAMILIES:
            allowed_rows.append(row)
        elif row["family"] in TARGET_FAMILIES and row["question_type"] != "TE":
            target_rows.append(row)

    allowed_rows.sort(key=lambda row: inventory_sort_key(row, PROFILE_FAMILY_ORDER))
    target_rows.sort(key=lambda row: inventory_sort_key(row, TARGET_FAMILY_ORDER))
    return allowed_rows, target_rows


def encode_answer_value(q: Dict) -> Optional[object]:
    answers = q.get("Answers", {})
    qtype = q.get("QuestionType")
    if qtype == "MC":
        value = answers.get("SelectedByPosition")
        if isinstance(value, list):
            if not value:
                return None
            value = next((item for item in value if item is not None), None)
        return int(value) if value is not None else None
    if qtype == "Matrix":
        values = answers.get("SelectedByPosition", [])
        if isinstance(values, list):
            out = []
            for value in values:
                if value is None:
                    out.append(None)
                else:
                    out.append(int(value))
            return out
        return None
    if qtype == "TE":
        text_rows = answers.get("Text", [])
        flat: List[str] = []
        if isinstance(text_rows, list):
            for row in text_rows:
                if isinstance(row, dict):
                    for value in row.values():
                        value = normalize_whitespace(str(value))
                        if value:
                            flat.append(value)
        return flat
    return None


def render_profile_item(q: Dict) -> Optional[str]:
    qid = q.get("QuestionID", "")
    qtype = q.get("QuestionType")
    question_text = normalize_whitespace(q.get("QuestionText", ""))
    answers = q.get("Answers", {})

    if qtype == "MC":
        pos = answers.get("SelectedByPosition")
        if isinstance(pos, list):
            if not pos:
                return None
            pos = next((item for item in pos if item is not None), None)
        text = normalize_whitespace(str(answers.get("SelectedText", "")))
        if pos is None:
            return None
        return f"- [{qid}] {question_text} => {int(pos)} | {text}"

    if qtype == "Matrix":
        rows = q.get("Rows", []) or []
        selected = answers.get("SelectedByPosition", []) or []
        parts: List[str] = []
        for row_text, value in zip(rows, selected):
            if value is None:
                continue
            parts.append(f"{normalize_whitespace(str(row_text))}={int(value)}")
        if not parts:
            return None
        return f"- [{qid}] {question_text} => " + " ; ".join(parts)

    if qtype == "TE":
        texts = encode_answer_value(q) or []
        if not texts:
            return None
        compact = " | ".join(normalize_whitespace(str(text)) for text in texts)
        return f"- [{qid}] {question_text} => {compact}"

    return None


def render_target_question(q: Dict) -> str:
    qid = q.get("QuestionID", "")
    question_text = normalize_whitespace(q.get("QuestionText", ""))
    options = q.get("Options", []) or []
    role_name, is_binary_receiver = TARGET_ROLE_METADATA.get(
        qid,
        ("Unknown role", False),
    )
    lines = [f"### {qid}", f"Role: {role_name}", "", question_text, ""]
    for idx, option in enumerate(options, start=1):
        lines.append(f"{idx}. {normalize_whitespace(str(option))}")
    lines.append("")
    if is_binary_receiver:
        lines.append(f"Respond for {qid} with a single integer: 1 or 2 only. Do not output 3 or higher.")
    else:
        lines.append(f"Respond for {qid} with a single integer from 1 to {len(options)}.")
    return "\n".join(lines)


def build_matrix_answer_map(q: Optional[Dict]) -> Dict[str, int]:
    if not q or q.get("QuestionType") != "Matrix":
        return {}
    rows = q.get("Rows", []) or []
    selected = q.get("Answers", {}).get("SelectedByPosition", []) or []
    out: Dict[str, int] = {}
    for row_text, value in zip(rows, selected):
        if value is None:
            continue
        out[normalize_whitespace(str(row_text))] = int(value)
    return out


def render_named_matrix_slice(
    q: Optional[Dict],
    named_rows: Sequence[Tuple[str, str]],
) -> Tuple[List[str], bool]:
    answer_map = build_matrix_answer_map(q)
    parts: List[str] = []
    for short_name, row_text in named_rows:
        value = answer_map.get(normalize_whitespace(row_text))
        if value is None:
            continue
        parts.append(f"{short_name}={value}")
    return parts, bool(parts)


def summarize_matrix_switch_behavior(
    q: Optional[Dict],
    left_choice_name: str,
    right_choice_name: str,
) -> Optional[str]:
    if not q or q.get("QuestionType") != "Matrix":
        return None
    values = q.get("Answers", {}).get("SelectedByPosition", []) or []
    if not values:
        return None
    clean_values = [int(v) for v in values if v is not None]
    if not clean_values:
        return None
    left_count = sum(1 for v in clean_values if v == 1)
    right_count = sum(1 for v in clean_values if v == 2)
    first_right = next((idx for idx, v in enumerate(clean_values, start=1) if v == 2), None)
    if first_right is None:
        switch_text = f"never switches to {right_choice_name}"
    else:
        switch_text = f"switches to {right_choice_name} at row {first_right}/{len(clean_values)}"
    return (
        f"{q.get('QuestionID')}: {left_choice_name} on {left_count}/{len(clean_values)} rows, "
        f"{right_choice_name} on {right_count}/{len(clean_values)} rows, {switch_text}"
    )


def render_mc_answer_summary(q: Optional[Dict]) -> Optional[str]:
    if not q or q.get("QuestionType") != "MC":
        return None
    pos = q.get("Answers", {}).get("SelectedByPosition")
    text = normalize_whitespace(str(q.get("Answers", {}).get("SelectedText", "")))
    if pos is None:
        return None
    question_text = normalize_whitespace(q.get("QuestionText", ""))
    return f"{q.get('QuestionID')}: {question_text} => {int(pos)} | {text}"


def render_structured_profile_text(
    ref_to_question: Dict[str, Dict],
) -> Tuple[str, int, List[str]]:
    qid_to_question = {
        str(question.get("QuestionID", "")): question
        for question in ref_to_question.values()
    }
    lines: List[str] = []
    component_count = 0
    used_refs: List[str] = []

    def add_section(title: str, section_lines: Sequence[str], used_qids: Sequence[str]) -> None:
        nonlocal component_count
        clean_lines = [line for line in section_lines if line]
        if not clean_lines:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {line}" for line in clean_lines)
        lines.append("")
        component_count += len(clean_lines)
        used_refs.extend(
            ref_for_parts(qid_to_question[qid].get("BlockName", ""), qid)
            for qid in used_qids
            if qid in qid_to_question
        )

    social_lines: List[str] = []
    q25_parts, q25_used = render_named_matrix_slice(qid_to_question.get("QID25"), STRUCTURED_Q25_ROWS)
    if q25_used:
        social_lines.append("Cooperation / trust markers: " + " ; ".join(q25_parts))
    q27_parts, q27_used = render_named_matrix_slice(qid_to_question.get("QID27"), STRUCTURED_Q27_ROWS)
    if q27_used:
        social_lines.append("Resentment / retaliation markers: " + " ; ".join(q27_parts))
    q29_parts, q29_used = render_named_matrix_slice(qid_to_question.get("QID29"), STRUCTURED_Q29_ROWS)
    if q29_used:
        social_lines.append("Values: " + " ; ".join(q29_parts))
    q232_parts, q232_used = render_named_matrix_slice(qid_to_question.get("QID232"), STRUCTURED_Q232_ROWS)
    if q232_used:
        social_lines.append("Empathy markers: " + " ; ".join(q232_parts))
    q233_parts, q233_used = render_named_matrix_slice(qid_to_question.get("QID233"), STRUCTURED_Q233_ROWS)
    if q233_used:
        social_lines.append("Independence / competition markers: " + " ; ".join(q233_parts))
    q236_parts, q236_used = render_named_matrix_slice(qid_to_question.get("QID236"), STRUCTURED_Q236_ROWS)
    if q236_used:
        social_lines.append("Social-perception markers: " + " ; ".join(q236_parts))
    q238_parts, q238_used = render_named_matrix_slice(qid_to_question.get("QID238"), STRUCTURED_Q238_ROWS)
    if q238_used:
        social_lines.append("Need-for-certainty markers: " + " ; ".join(q238_parts))
    add_section(
        "Relevant Social Profile",
        social_lines,
        ["QID25", "QID27", "QID29", "QID232", "QID233", "QID236", "QID238"],
    )

    mental_lines = [render_mc_answer_summary(qid_to_question.get(qid)) for qid in MENTAL_ACCOUNTING_QIDS]
    add_section("Mental Accounting", [line for line in mental_lines if line], MENTAL_ACCOUNTING_QIDS)

    time_lines = [
        summarize_matrix_switch_behavior(
            qid_to_question.get(qid),
            left_choice_name="later reward",
            right_choice_name="sooner reward",
        )
        for qid in TIME_PREFERENCE_QIDS
    ]
    add_section("Time Preference", [line for line in time_lines if line], TIME_PREFERENCE_QIDS)

    risk_gain_lines = [
        summarize_matrix_switch_behavior(
            qid_to_question.get(qid),
            left_choice_name="lottery",
            right_choice_name="sure amount",
        )
        for qid in RISK_GAIN_QIDS
    ]
    add_section("Risk Preference (Gains)", [line for line in risk_gain_lines if line], RISK_GAIN_QIDS)

    risk_loss_lines = [
        summarize_matrix_switch_behavior(
            qid_to_question.get(qid),
            left_choice_name="lottery",
            right_choice_name="sure outcome",
        )
        for qid in RISK_LOSS_QIDS
    ]
    add_section("Risk Preference (Losses)", [line for line in risk_loss_lines if line], RISK_LOSS_QIDS)

    if lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip(), component_count, sorted(set(used_refs))


def find_question_map(example: Dict) -> Dict[str, Dict]:
    blocks = json.loads(example["wave1_3_persona_json"])
    ref_to_question: Dict[str, Dict] = {}
    for block in blocks:
        block_name = block.get("BlockName", "")
        for q in block.get("Questions", []):
            ref = ref_for_parts(block_name, q.get("QuestionID", ""))
            ref_to_question[ref] = q
    return ref_to_question


def render_profile_text(
    ref_to_question: Dict[str, Dict],
    allowed_ref_entries: Sequence[Dict[str, str]],
) -> Tuple[str, int]:
    grouped: Dict[str, List[str]] = {family: [] for family in PROFILE_FAMILY_ORDER}

    for ref_entry in allowed_ref_entries:
        ref = ref_entry["ref"]
        question = ref_to_question.get(ref)
        if not question:
            continue
        rendered = render_profile_item(question)
        if not rendered:
            continue
        grouped.setdefault(ref_entry["family"], []).append(rendered)

    lines: List[str] = []
    rendered_items = 0
    for family in PROFILE_FAMILY_ORDER:
        items = grouped.get(family, [])
        if not items:
            continue
        lines.append(f"## {family.replace('_', ' ').title()}")
        lines.extend(items)
        lines.append("")
        rendered_items += len(items)

    return "\n".join(lines).strip(), rendered_items


def group_target_questions(target_questions: Sequence[Dict], target_ref_entries: Sequence[Dict[str, str]]) -> List[Tuple[str, List[Dict]]]:
    q_by_qid = {
        str(q.get("QuestionID", "")): q
        for q in target_questions
    }
    grouped: Dict[str, List[Dict]] = {family: [] for family in TARGET_FAMILY_ORDER}
    for entry in target_ref_entries:
        question = q_by_qid.get(entry["question_id"])
        if question is None:
            continue
        grouped.setdefault(entry["family"], []).append(question)
    return [(family, grouped.get(family, [])) for family in TARGET_FAMILY_ORDER if grouped.get(family)]


def build_messages(
    profile_text: str,
    target_questions: Sequence[Dict],
    target_ref_entries: Sequence[Dict[str, str]],
    include_reasoning: bool,
    prompt_variant: str,
) -> List[Dict[str, str]]:
    system_lines = [
        "You are predicting how a specific Twin participant would answer held-out social decision questions.",
        "The participant profile contains waves 1-3 information with all trust, ultimatum, and dictator items removed.",
        "Infer the answers only from the remaining participant profile.",
        "Return JSON only and do not include markdown.",
    ]
    if prompt_variant in {"relevant_structured_summary", "relevant_structured_summary_tuned"}:
        system_lines.extend(
            [
                "The profile is a compact structured summary built from the non-target information most relevant to social-game prediction.",
                "Use the concrete behavioral-economics evidence and the social-value indicators together.",
                "Do not default to the most generous option just because the participant sounds prosocial.",
                "Calibrate to the actual answer scale and the specific game mechanics.",
            ]
        )
    if prompt_variant == "relevant_structured_summary_tuned":
        system_lines.extend(
            [
                "For each QID, identify the exact game and role before deciding on an answer.",
                "Keep the games distinct: trust return is not ultimatum acceptance, and dictator allocation is not ultimatum offering.",
                "Respect the option list shown for each QID exactly.",
                "For binary ultimatum-receiver questions, the answer must be 1 or 2 only.",
            ]
        )
    if include_reasoning:
        if prompt_variant == "relevant_structured_summary_tuned":
            system_lines.append(
                'Return JSON with keys "reasoning" and "answers", in that order. '
                '"reasoning" must map each QID to a short explanation that begins with the game role. '
                '"answers" must map each QID to an integer option number. '
                "Reason through each QID first, then give the final prediction."
            )
        else:
            system_lines.append(
                'Return JSON with keys "reasoning" and "answers", in that order. '
                '"reasoning" must map each QID to a short explanation. '
                '"answers" must map each QID to an integer option number. '
                "Reason through each QID first, then give the final prediction."
            )
    else:
        system_lines.append(
            'Return JSON with exactly one top-level key, "answers", mapping each QID to an integer option number.'
        )

    qid_list = [q["QuestionID"] for q in target_questions]
    if include_reasoning:
        response_shape = {
            "reasoning": {qid: "short explanation" for qid in qid_list},
            "answers": {qid: "integer option number" for qid in qid_list},
        }
    else:
        response_shape = {"answers": {qid: "integer option number" for qid in qid_list}}

    user_lines = [
        "# Participant Profile",
        "",
        profile_text,
        "",
        "# Held-Out Social Game Questions",
        "",
        "The following questions were removed from the participant profile.",
        "Predict how this participant would answer each question.",
        "",
    ]
    for family, questions in group_target_questions(target_questions, target_ref_entries):
        user_lines.append(f"## {family.replace('_', ' ').title()}")
        user_lines.append("")
        for q in questions:
            user_lines.append(render_target_question(q))
            user_lines.append("")
    user_lines.extend(
        [
            "# Output Format",
            "",
            json.dumps(response_shape, ensure_ascii=False, indent=2),
            "",
            "Use the real predicted option number for each QID.",
        ]
    )
    if prompt_variant == "relevant_structured_summary_tuned" and include_reasoning:
        user_lines.extend(
            [
                "Each reasoning string should explicitly name the role before the explanation.",
                'Example style: "Trust receiver; ..." or "Ultimatum receiver; ...".',
            ]
        )

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model), "model"
    except Exception:
        pass
    for encoding_name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(encoding_name), encoding_name
        except Exception:
            continue
    return None, "char_div4_fallback"


def count_text_tokens(text: str, encoding) -> int:
    if encoding is None:
        return int(math.ceil(len(text) / 4))
    return len(encoding.encode(text))


def estimate_chat_tokens(messages: Sequence[Dict[str, str]], model: str) -> int:
    encoding, _ = get_encoding(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                continue
            total += count_text_tokens(value, encoding)
            if key == "name":
                total += tokens_per_name
    total += 3
    return total


def percentile(sorted_values: Sequence[int], frac: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * frac))))
    return int(sorted_values[idx])


def filter_target_entries(
    target_ref_entries: Sequence[Dict[str, str]],
    target_qids: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    if not target_qids:
        return list(target_ref_entries)
    wanted = {qid.strip() for qid in target_qids if qid.strip()}
    filtered = [entry for entry in target_ref_entries if entry["question_id"] in wanted]
    found = {entry["question_id"] for entry in filtered}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown or unavailable target QIDs for joint baseline: {missing}")
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build OpenAI Batch requests for the joint social-game no-retrieval baseline."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=PROMPT_VARIANTS,
        default=DEFAULT_PROMPT_VARIANT,
        help="Prompt construction variant.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional override for max_completion_tokens. Omit to leave unset.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Ask the model to return a short explanation per target question.",
    )
    parser.add_argument(
        "--limit-participants",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional random sample fraction of participants to keep, e.g. 0.1 for 10%%.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used when --sample-fraction is set.",
    )
    parser.add_argument(
        "--target-qids",
        type=str,
        default=None,
        help="Optional comma-separated subset of target QIDs.",
    )
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        default=None,
        help="Optional manifest whose participant IDs should be reused exactly, in order.",
    )
    args = parser.parse_args()

    inventory_rows = load_inventory(INVENTORY_CSV)
    catalog = load_question_catalog()
    source_by_ref = build_source_by_ref(catalog)
    allowed_rows, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    if not allowed_rows or not target_rows:
        raise ValueError("Failed to build allowed or target ref lists for the joint social baseline.")

    allowed_ref_entries = [
        {
            "ref": ref_for_row(row),
            "block_name": row["block_name"],
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "family": row["family"],
            "question_text_short": row["question_text_short"],
        }
        for row in allowed_rows
    ]
    target_ref_entries = [
        {
            "ref": ref_for_row(row),
            "block_name": row["block_name"],
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "family": row["family"],
            "question_text_short": row["question_text_short"],
        }
        for row in target_rows
    ]
    selected_target_qids = (
        [qid.strip() for qid in args.target_qids.split(",")]
        if args.target_qids
        else None
    )
    target_ref_entries = filter_target_entries(target_ref_entries, selected_target_qids)
    selected_target_families = [
        family
        for family in TARGET_FAMILY_ORDER
        if any(entry["family"] == family for entry in target_ref_entries)
    ]

    ds = load_wave_split()
    n_source_participants = len(ds)
    if args.reuse_manifest is not None:
        manifest_rows = load_jsonl(args.reuse_manifest)
        requested_pids = [str(row["pid"]) for row in manifest_rows if row.get("pid") is not None]
        by_pid = {str(example["pid"]): example for example in ds}
        ds = [by_pid[pid] for pid in requested_pids if pid in by_pid]
    elif args.sample_fraction is not None:
        if not (0 < args.sample_fraction <= 1):
            raise ValueError("--sample-fraction must be in the interval (0, 1].")
        sample_size = max(1, math.ceil(len(ds) * args.sample_fraction))
        ds = ds.shuffle(seed=args.random_seed).select(range(sample_size))
    if args.limit_participants:
        if isinstance(ds, list):
            ds = ds[: min(args.limit_participants, len(ds))]
        else:
            ds = ds.select(range(min(args.limit_participants, len(ds))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.output_dir / f"requests_joint_social_baseline_{args.model}.jsonl"
    manifest_path = args.output_dir / "manifest_joint_social_baseline.jsonl"
    token_path = args.output_dir / f"token_estimate_joint_social_baseline_{args.model}.json"
    preview_path = args.output_dir / f"preview_joint_social_baseline_{args.model}.json"

    token_counts: List[int] = []
    request_count = 0
    first_preview: Optional[Dict] = None
    _, tokenizer_source = get_encoding(args.model)

    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_f:
        for example in ds:
            pid = str(example["pid"])
            ref_to_question = find_question_map(example)

            if args.prompt_variant in {"relevant_structured_summary", "relevant_structured_summary_tuned"}:
                profile_text, rendered_items, used_input_refs = render_structured_profile_text(
                    ref_to_question=ref_to_question,
                )
                used_input_families = STRUCTURED_ALLOWED_INPUT_FAMILIES
            else:
                profile_text, rendered_items = render_profile_text(
                    ref_to_question=ref_to_question,
                    allowed_ref_entries=allowed_ref_entries,
                )
                used_input_refs = [entry["ref"] for entry in allowed_ref_entries]
                used_input_families = ALLOWED_INPUT_FAMILIES
            if not profile_text:
                continue

            target_questions: List[Dict] = []
            ground_truth: Dict[str, int] = {}
            target_family_to_qids: Dict[str, List[str]] = {family: [] for family in selected_target_families}
            for entry in target_ref_entries:
                question = ref_to_question.get(entry["ref"])
                if not question:
                    continue
                target_questions.append(question)
                target_family_to_qids.setdefault(entry["family"], []).append(question["QuestionID"])
                value = encode_answer_value(question)
                if value is not None:
                    ground_truth[question["QuestionID"]] = int(value)

            if len(target_questions) != len(target_ref_entries):
                continue

            messages = build_messages(
                profile_text=profile_text,
                target_questions=target_questions,
                target_ref_entries=target_ref_entries,
                include_reasoning=args.include_reasoning,
                prompt_variant=args.prompt_variant,
            )
            approx_prompt_tokens = estimate_chat_tokens(messages, args.model)
            token_counts.append(approx_prompt_tokens)

            custom_id = f"joint_social_baseline__pid_{pid}"
            body = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "response_format": {"type": "json_object"},
            }
            if args.max_completion_tokens is not None:
                body["max_completion_tokens"] = args.max_completion_tokens
            request_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            req_f.write(json.dumps(request_row, ensure_ascii=False) + "\n")

            manifest_row = {
                "custom_id": custom_id,
                "pid": pid,
                "condition": "joint_social_baseline",
                "prompt_variant": args.prompt_variant,
                "target_family": "joint_social_block",
                "target_families": selected_target_families,
                "target_family_to_qids": target_family_to_qids,
                "model": args.model,
                "include_reasoning": args.include_reasoning,
                "target_question_ids": [q["QuestionID"] for q in target_questions],
                "ground_truth_answers": ground_truth,
                "allowed_input_families": used_input_families,
                "allowed_input_refs": used_input_refs,
                "excluded_target_refs": [entry["ref"] for entry in target_ref_entries],
                "approx_prompt_tokens": approx_prompt_tokens,
                "profile_rendered_item_count": rendered_items,
                "profile_char_count": len(profile_text),
            }
            manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

            if first_preview is None:
                first_preview = {
                    "custom_id": custom_id,
                    "messages": messages,
                    "ground_truth_answers": ground_truth,
                    "approx_prompt_tokens": approx_prompt_tokens,
                }
            request_count += 1

    if first_preview is not None:
        preview_path.write_text(
            json.dumps(first_preview, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    token_counts_sorted = sorted(token_counts)
    token_summary = {
        "model": args.model,
        "tokenizer_source": tokenizer_source,
        "condition": "joint_social_baseline",
        "prompt_variant": args.prompt_variant,
        "include_reasoning": args.include_reasoning,
        "n_source_participants": n_source_participants,
        "n_participants": len(ds),
        "sample_fraction": args.sample_fraction,
        "reuse_manifest": str(args.reuse_manifest) if args.reuse_manifest is not None else None,
        "random_seed": args.random_seed if args.sample_fraction is not None else None,
        "n_requests": request_count,
        "target_families": selected_target_families,
        "target_question_count": len(target_ref_entries),
        "allowed_input_families": STRUCTURED_ALLOWED_INPUT_FAMILIES if args.prompt_variant in {"relevant_structured_summary", "relevant_structured_summary_tuned"} else ALLOWED_INPUT_FAMILIES,
        "total_prompt_tokens": int(sum(token_counts)),
        "mean_prompt_tokens": float(round(statistics.mean(token_counts), 2)) if token_counts else 0.0,
        "median_prompt_tokens": int(statistics.median(token_counts_sorted)) if token_counts else 0,
        "p95_prompt_tokens": percentile(token_counts_sorted, 0.95),
        "requests_jsonl": str(requests_path),
        "manifest_jsonl": str(manifest_path),
        "preview_json": str(preview_path),
    }
    token_path.write_text(json.dumps(token_summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {requests_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {token_path}")
    print(json.dumps(token_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
