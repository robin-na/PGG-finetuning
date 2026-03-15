#!/usr/bin/env python3
"""Build OpenAI Batch JSONL requests for the main PGG-transfer benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken
from datasets import load_dataset, load_from_disk


REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
CONFIG = "wave_split"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "task_grounding"
    / "pgg_transfer_benchmark_spec.json"
)
LOCAL_WAVE_SPLIT = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "wave_split_dataset"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "main"
)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0
FAMILY_ORDER = [
    "demographics",
    "personality",
    "cognitive_tests",
    "trust",
    "ultimatum",
    "dictator",
    "mental_accounting",
    "time_preference",
    "risk_preference_gain",
    "risk_preference_loss",
]


def load_spec(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
            ref = f"{block_name}::{q.get('QuestionID', '')}"
            ref_to_question[ref] = q
    return ref_to_question


def family_rank(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def render_profile_text(
    ref_to_question: Dict[str, Dict],
    allowed_ref_entries: Sequence[Dict],
    allowed_families: Sequence[str],
) -> Tuple[str, int]:
    ref_to_family = {entry["ref"]: entry["family"] for entry in allowed_ref_entries}
    grouped: Dict[str, List[str]] = {family: [] for family in allowed_families}

    for ref_entry in allowed_ref_entries:
        ref = ref_entry["ref"]
        question = ref_to_question.get(ref)
        if not question:
            continue
        rendered = render_profile_item(question)
        if not rendered:
            continue
        family = ref_to_family[ref]
        grouped.setdefault(family, []).append(rendered)

    lines: List[str] = []
    rendered_items = 0
    for family in sorted(grouped, key=family_rank):
        items = grouped[family]
        if not items:
            continue
        lines.append(f"## {family.replace('_', ' ').title()}")
        lines.extend(items)
        lines.append("")
        rendered_items += len(items)

    return "\n".join(lines).strip(), rendered_items


def build_messages(
    profile_text: str,
    target_questions: Sequence[Dict],
    include_reasoning: bool,
) -> List[Dict[str, str]]:
    system_lines = [
        "You are predicting how a specific Twin participant would answer held-out economic-game questions.",
        "Use only the participant profile from earlier Twin waves.",
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
        "# Held-Out Economic Questions",
        "",
    ]
    for q in target_questions:
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


def select_cells(spec: Dict, condition: str, target_families: Sequence[str]) -> Dict[str, Dict]:
    wanted = set(target_families)
    cells: Dict[str, Dict] = {}
    for cell in spec["benchmark_cells"]:
        if cell["condition"] != condition:
            continue
        if cell["target_family"] not in wanted:
            continue
        cells[cell["target_family"]] = cell
    missing = wanted - set(cells)
    if missing:
        raise ValueError(f"Missing benchmark cells for target families: {sorted(missing)}")
    return cells


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model), "model"
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base"), "cl100k_base"
        except Exception:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI Batch requests for the main PGG-transfer eval.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition", type=str, default="main")
    parser.add_argument(
        "--target-families",
        type=str,
        default="trust,ultimatum,dictator",
        help="Comma-separated target families.",
    )
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
    args = parser.parse_args()

    spec = load_spec(args.spec)
    target_families = [family.strip() for family in args.target_families.split(",") if family.strip()]
    cells = select_cells(spec, args.condition, target_families)
    ds = load_wave_split()
    if args.limit_participants:
        ds = ds.select(range(min(args.limit_participants, len(ds))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.output_dir / f"requests_{args.condition}_{args.model}.jsonl"
    manifest_path = args.output_dir / f"manifest_{args.condition}.jsonl"
    token_path = args.output_dir / f"token_estimate_{args.condition}_{args.model}.json"
    preview_path = args.output_dir / f"preview_{args.condition}_{args.model}.json"

    token_counts: List[int] = []
    family_token_counts: Dict[str, List[int]] = {family: [] for family in target_families}
    request_count = 0
    first_preview: Optional[Dict] = None
    _, tokenizer_source = get_encoding(args.model)

    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_f:
        for example in ds:
            pid = str(example["pid"])
            ref_to_question = find_question_map(example)

            for target_family in target_families:
                cell = cells[target_family]
                allowed_ref_entries = cell["allowed_input_refs"]
                target_ref_entries = cell["target_choice_refs"]
                allowed_families = cell["allowed_input_families"]

                profile_text, rendered_items = render_profile_text(
                    ref_to_question=ref_to_question,
                    allowed_ref_entries=allowed_ref_entries,
                    allowed_families=allowed_families,
                )
                if not profile_text:
                    continue

                target_questions: List[Dict] = []
                ground_truth: Dict[str, int] = {}
                for entry in target_ref_entries:
                    question = ref_to_question.get(entry["ref"])
                    if not question:
                        continue
                    target_questions.append(question)
                    value = encode_answer_value(question)
                    if value is not None:
                        ground_truth[question["QuestionID"]] = int(value)

                if len(target_questions) != len(target_ref_entries):
                    continue

                messages = build_messages(
                    profile_text=profile_text,
                    target_questions=target_questions,
                    include_reasoning=args.include_reasoning,
                )
                approx_prompt_tokens = estimate_chat_tokens(messages, args.model)
                token_counts.append(approx_prompt_tokens)
                family_token_counts[target_family].append(approx_prompt_tokens)

                custom_id = f"{args.condition}__{target_family}__pid_{pid}"
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
                    "condition": args.condition,
                    "target_family": target_family,
                    "model": args.model,
                    "include_reasoning": args.include_reasoning,
                    "target_question_ids": [q["QuestionID"] for q in target_questions],
                    "ground_truth_answers": ground_truth,
                    "allowed_input_families": allowed_families,
                    "allowed_input_refs": [entry["ref"] for entry in allowed_ref_entries],
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
    family_summary = {}
    for family, counts in family_token_counts.items():
        if not counts:
            continue
        ordered = sorted(counts)
        family_summary[family] = {
            "n_requests": len(counts),
            "total_prompt_tokens": int(sum(counts)),
            "mean_prompt_tokens": float(round(statistics.mean(counts), 2)),
            "median_prompt_tokens": int(statistics.median(ordered)),
            "p95_prompt_tokens": percentile(ordered, 0.95),
        }

    token_summary = {
        "model": args.model,
        "tokenizer_source": tokenizer_source,
        "condition": args.condition,
        "include_reasoning": args.include_reasoning,
        "n_participants": len(ds),
        "n_requests": request_count,
        "total_prompt_tokens": int(sum(token_counts)),
        "mean_prompt_tokens": float(round(statistics.mean(token_counts), 2)) if token_counts else 0.0,
        "median_prompt_tokens": int(statistics.median(token_counts_sorted)) if token_counts else 0,
        "p95_prompt_tokens": percentile(token_counts_sorted, 0.95),
        "family_breakdown": family_summary,
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
