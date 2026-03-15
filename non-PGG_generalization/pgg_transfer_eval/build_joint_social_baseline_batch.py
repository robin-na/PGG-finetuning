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


def load_inventory(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    question_text = shorten(q.get("QuestionText", ""), 140)
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
            parts.append(f"{shorten(str(row_text), 72)}={int(value)}")
        if not parts:
            return None
        return f"- [{qid}] {question_text} => " + " ; ".join(parts)

    if qtype == "TE":
        texts = encode_answer_value(q) or []
        if not texts:
            return None
        compact = " | ".join(shorten(text, 72) for text in texts[:6])
        return f"- [{qid}] {question_text} => {compact}"

    return None


def render_target_question(q: Dict) -> str:
    qid = q.get("QuestionID", "")
    question_text = normalize_whitespace(q.get("QuestionText", ""))
    options = q.get("Options", []) or []
    lines = [f"### {qid}", question_text, ""]
    for idx, option in enumerate(options, start=1):
        lines.append(f"{idx}. {normalize_whitespace(str(option))}")
    lines.append("")
    lines.append(f"Respond for {qid} with a single integer from 1 to {len(options)}.")
    return "\n".join(lines)


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
    q_by_ref = {
        ref_for_parts(q.get("BlockName", ""), q.get("QuestionID", "")): q
        for q in target_questions
    }
    grouped: Dict[str, List[Dict]] = {family: [] for family in TARGET_FAMILY_ORDER}
    for entry in target_ref_entries:
        question = q_by_ref.get(entry["ref"])
        if question is None:
            continue
        grouped.setdefault(entry["family"], []).append(question)
    return [(family, grouped.get(family, [])) for family in TARGET_FAMILY_ORDER if grouped.get(family)]


def build_messages(
    profile_text: str,
    target_questions: Sequence[Dict],
    target_ref_entries: Sequence[Dict[str, str]],
    include_reasoning: bool,
) -> List[Dict[str, str]]:
    system_lines = [
        "You are predicting how a specific Twin participant would answer held-out social decision questions.",
        "The participant profile contains waves 1-3 information with all trust, ultimatum, and dictator items removed.",
        "Infer the answers only from the remaining participant profile.",
        "Return JSON only and do not include markdown.",
    ]
    if include_reasoning:
        system_lines.append(
            'Return JSON with keys "answers" and "reasoning". '
            '"answers" must map each QID to an integer option number. '
            '"reasoning" must map each QID to a short explanation.'
        )
    else:
        system_lines.append(
            'Return JSON with exactly one top-level key, "answers", mapping each QID to an integer option number.'
        )

    qid_list = [q["QuestionID"] for q in target_questions]
    if include_reasoning:
        response_shape = {
            "answers": {qid: 1 for qid in qid_list},
            "reasoning": {qid: "short explanation" for qid in qid_list},
        }
    else:
        response_shape = {"answers": {qid: 1 for qid in qid_list}}

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
        "--target-qids",
        type=str,
        default=None,
        help="Optional comma-separated subset of target QIDs.",
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
    if args.limit_participants:
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

            profile_text, rendered_items = render_profile_text(
                ref_to_question=ref_to_question,
                allowed_ref_entries=allowed_ref_entries,
            )
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
                "prompt_variant": "baseline_no_retrieval",
                "target_family": "joint_social_block",
                "target_families": selected_target_families,
                "target_family_to_qids": target_family_to_qids,
                "model": args.model,
                "include_reasoning": args.include_reasoning,
                "target_question_ids": [q["QuestionID"] for q in target_questions],
                "ground_truth_answers": ground_truth,
                "allowed_input_families": ALLOWED_INPUT_FAMILIES,
                "allowed_input_refs": [entry["ref"] for entry in allowed_ref_entries],
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
        "prompt_variant": "baseline_no_retrieval",
        "include_reasoning": args.include_reasoning,
        "n_participants": len(ds),
        "n_requests": request_count,
        "target_families": selected_target_families,
        "target_question_count": len(target_ref_entries),
        "allowed_input_families": ALLOWED_INPUT_FAMILIES,
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
