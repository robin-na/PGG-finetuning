#!/usr/bin/env python3
"""Build OpenAI Batch requests that write oracle-library search queries for Twin participants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_joint_social_baseline_batch import (  # type: ignore
    INVENTORY_CSV,
    STRUCTURED_ALLOWED_INPUT_FAMILIES,
    build_source_by_ref,
    estimate_chat_tokens,
    filter_target_entries,
    find_question_map,
    get_encoding,
    load_inventory,
    load_question_catalog,
    load_jsonl,
    load_wave_split,
    ref_for_parts,
    ref_for_row,
    render_structured_profile_text,
    select_allowed_and_target_refs,
)


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_oracle_query"
)

SYSTEM_PROMPT = """You write retrieval queries for a library of oracle archetype cards from repeated public-goods games.

Each library card contains:
- demographics,
- the exact repeated-PGG rules under which behavior was observed,
- an oracle archetype describing contribution style, reciprocity, norm enforcement, communication, and end-game response.

Your job is to convert the Twin participant profile into a compact search query that will retrieve analogous PGG cases useful for predicting held-out trust, ultimatum, and dictator behavior.

Rules:
- Use demographics as a weak prior, not the main criterion.
- Focus on latent transferable traits: generosity, reciprocity, fairness sensitivity, norm enforcement, exploitation caution, conditionality, and behavioral stability.
- Mention that the target use case is held-out trust, ultimatum, and dictator prediction.
- Do not assume the retrieved PGG case is identical to the Twin participant.
- Return JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional override for max_completion_tokens. Omit to leave unset.",
    )
    parser.add_argument("--limit-participants", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--reuse-manifest", type=Path, default=None)
    return parser.parse_args()


def selected_text(question: Optional[Dict]) -> Optional[str]:
    if not question:
        return None
    text = question.get("Answers", {}).get("SelectedText")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())
    return None


def render_demographics(ref_to_question: Dict[str, Dict]) -> str:
    qid_map = {
        str(question.get("QuestionID", "")): question
        for question in ref_to_question.values()
    }
    sex = selected_text(qid_map.get("QID12")) or "unknown"
    age = selected_text(qid_map.get("QID13")) or "unknown"
    education = selected_text(qid_map.get("QID14")) or "unknown"
    return "\n".join(
        [
            "## Demographics",
            f"- Sex assigned at birth: {sex}",
            f"- Age bucket: {age}",
            f"- Education: {education}",
        ]
    )


def build_messages(profile_text: str, demographics_text: str) -> List[Dict[str, str]]:
    response_shape = {
        "search_query": "one compact paragraph for vector-store search",
        "match_cues": [
            "short cue 1",
            "short cue 2",
            "short cue 3",
        ],
    }
    user_lines = [
        "# Twin Participant",
        "",
        demographics_text,
        "",
        "## Known non-target profile",
        profile_text,
        "",
        "# Retrieval Objective",
        "Retrieve repeated-public-goods-game oracle profiles that could help predict this participant's held-out social-game behavior.",
        "The held-out target block combines:",
        "- trust sender and trust receiver choices",
        "- ultimatum proposer and ultimatum receiver choices",
        "- dictator allocation choice",
        "",
        "# Output Format",
        json.dumps(response_shape, ensure_ascii=False, indent=2),
        "",
        "Write the search query as a single paragraph, not a list.",
        "The query should mention demographics as a weak prior and emphasize the transferable behavioral cues.",
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def main() -> int:
    args = parse_args()
    inventory_rows = load_inventory(INVENTORY_CSV)
    catalog = load_question_catalog()
    source_by_ref = build_source_by_ref(catalog)
    allowed_rows, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    if not allowed_rows or not target_rows:
        raise ValueError("Failed to build allowed or target ref lists for oracle query generation.")

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
    target_ref_entries = filter_target_entries(target_ref_entries, None)

    ds = load_wave_split()
    source_participants = len(ds)
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
    requests_path = args.output_dir / f"requests_joint_social_oracle_query_{args.model}.jsonl"
    manifest_path = args.output_dir / "manifest_joint_social_oracle_query.jsonl"
    token_path = args.output_dir / f"token_estimate_joint_social_oracle_query_{args.model}.json"
    preview_path = args.output_dir / f"preview_joint_social_oracle_query_{args.model}.json"

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
            profile_text, rendered_items, used_input_refs = render_structured_profile_text(
                ref_to_question=ref_to_question,
            )
            if not profile_text:
                continue
            demographics_text = render_demographics(ref_to_question)
            messages = build_messages(profile_text=profile_text, demographics_text=demographics_text)
            approx_prompt_tokens = estimate_chat_tokens(messages, args.model)
            token_counts.append(approx_prompt_tokens)

            custom_id = f"joint_social_oracle_query__pid_{pid}"
            body = {
                "model": args.model,
                "messages": messages,
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
                "condition": "joint_social_oracle_query",
                "model": args.model,
                "allowed_input_families": STRUCTURED_ALLOWED_INPUT_FAMILIES + ["demographics"],
                "allowed_input_refs": used_input_refs + [
                    ref_for_parts("Demographics", "QID12"),
                    ref_for_parts("Demographics", "QID13"),
                    ref_for_parts("Demographics", "QID14"),
                ],
                "target_question_ids": [entry["question_id"] for entry in target_ref_entries],
                "approx_prompt_tokens": approx_prompt_tokens,
                "profile_rendered_item_count": rendered_items,
                "profile_char_count": len(profile_text),
                "demographics_text": demographics_text,
            }
            manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

            if first_preview is None:
                first_preview = {
                    "custom_id": custom_id,
                    "messages": messages,
                    "approx_prompt_tokens": approx_prompt_tokens,
                }
            request_count += 1

    if first_preview is not None:
        preview_path.write_text(json.dumps(first_preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    token_counts_sorted = sorted(token_counts)
    token_summary = {
        "model": args.model,
        "tokenizer_source": tokenizer_source,
        "condition": "joint_social_oracle_query",
        "source_participants": source_participants,
        "requests": request_count,
        "approx_prompt_tokens_total": int(sum(token_counts)),
        "approx_prompt_tokens_mean": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0,
        "approx_prompt_tokens_p95": token_counts_sorted[int(round((len(token_counts_sorted) - 1) * 0.95))]
        if token_counts_sorted
        else 0,
    }
    token_path.write_text(json.dumps(token_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(token_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
