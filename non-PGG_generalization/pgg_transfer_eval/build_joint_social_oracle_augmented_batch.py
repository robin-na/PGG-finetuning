#!/usr/bin/env python3
"""Build oracle-augmented joint-social prediction batches from retrieved candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_joint_social_baseline_batch import (  # type: ignore
    INVENTORY_CSV,
    STRUCTURED_ALLOWED_INPUT_FAMILIES,
    build_source_by_ref,
    estimate_chat_tokens,
    find_question_map,
    get_encoding,
    group_target_questions,
    load_inventory,
    load_question_catalog,
    load_wave_split,
    ref_for_parts,
    ref_for_row,
    render_structured_profile_text,
    render_target_question,
    select_allowed_and_target_refs,
)
from build_joint_social_oracle_query_batch import render_demographics  # type: ignore
from evaluate_batch_results import load_jsonl  # type: ignore


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_oracle_augmented"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-jsonl",
        type=Path,
        required=True,
        help="Output from retrieve_oracle_candidates.py",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-participant-demographics", action="store_true", default=True)
    parser.add_argument(
        "--no-include-participant-demographics",
        dest="include_participant_demographics",
        action="store_false",
    )
    return parser.parse_args()


def load_document_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8").strip()


def build_messages(
    profile_text: str,
    retrieved_cards: Sequence[Dict[str, Any]],
    target_questions: Sequence[Dict[str, Any]],
    target_ref_entries: Sequence[Dict[str, str]],
    include_reasoning: bool,
    demographics_text: Optional[str],
) -> List[Dict[str, str]]:
    system_lines = [
        "You are predicting how a specific Twin participant would answer held-out social decision questions.",
        "The participant profile contains waves 1-3 information with all trust, ultimatum, and dictator items removed.",
        "Retrieved oracle cards come from other people observed in repeated public-goods games under the rules shown on each card.",
        "Use the Twin participant profile as primary evidence.",
        "Use the retrieved oracle cards only as analogical evidence.",
        "Do not assume any retrieved oracle card is the same person as the Twin participant.",
        "Do not copy a retrieved card literally into the prediction if it conflicts with the Twin participant profile.",
        "Keep the games distinct: trust return is not ultimatum acceptance, and dictator allocation is not ultimatum offering.",
        "Respect the option list shown for each QID exactly.",
        "For binary ultimatum-receiver questions, the answer must be 1 or 2 only.",
        "Return JSON only and do not include markdown.",
    ]
    if include_reasoning:
        system_lines.append(
            'Return JSON with keys "reasoning" and "answers", in that order. '
            '"reasoning" must map each QID to a short explanation that begins with the game role. '
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
    ]
    if demographics_text:
        user_lines.extend([demographics_text, ""])
    user_lines.extend([profile_text, "", "# Retrieved Analogous PGG Oracle Profiles", ""])
    if not retrieved_cards:
        user_lines.extend(["No oracle cards were retrieved.", ""])
    else:
        for idx, card in enumerate(retrieved_cards, start=1):
            user_lines.append(f"## Retrieved Oracle {idx}")
            user_lines.append("")
            user_lines.append(load_document_text(str(card["doc_path"])))
            user_lines.append("")

    user_lines.extend(
        [
            "# Held-Out Social Game Questions",
            "",
            "The following questions were removed from the participant profile.",
            "Predict how this participant would answer each question.",
            "",
        ]
    )
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
    if include_reasoning:
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


def main() -> int:
    args = parse_args()
    candidate_rows = load_jsonl(args.candidates_jsonl)
    candidate_rows = [row for row in candidate_rows if row.get("pid") is not None]
    if args.limit is not None:
        candidate_rows = candidate_rows[: args.limit]

    inventory_rows = load_inventory(INVENTORY_CSV)
    catalog = load_question_catalog()
    source_by_ref = build_source_by_ref(catalog)
    _, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    if not target_rows:
        raise ValueError("Failed to build target ref list for oracle-augmented joint baseline.")
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

    ds = load_wave_split()
    by_pid = {str(example["pid"]): example for example in ds}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.output_dir / f"requests_joint_social_oracle_augmented_{args.model}.jsonl"
    manifest_path = args.output_dir / "manifest_joint_social_oracle_augmented.jsonl"
    token_path = args.output_dir / f"token_estimate_joint_social_oracle_augmented_{args.model}.json"
    preview_path = args.output_dir / f"preview_joint_social_oracle_augmented_{args.model}.json"

    token_counts: List[int] = []
    request_count = 0
    skipped_missing_pid = 0
    skipped_no_candidates = 0
    first_preview: Optional[Dict[str, Any]] = None
    _, tokenizer_source = get_encoding(args.model)

    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_f:
        for candidate_row in candidate_rows:
            pid = str(candidate_row["pid"])
            example = by_pid.get(pid)
            if example is None:
                skipped_missing_pid += 1
                continue

            candidates = list(candidate_row.get("candidates") or [])[: args.top_k]
            if not candidates:
                skipped_no_candidates += 1
                continue

            ref_to_question = find_question_map(example)
            profile_text, rendered_items, used_input_refs = render_structured_profile_text(
                ref_to_question=ref_to_question,
            )
            if not profile_text:
                continue
            demographics_text = render_demographics(ref_to_question) if args.include_participant_demographics else None

            target_questions: List[Dict[str, Any]] = []
            ground_truth: Dict[str, int] = {}
            target_family_to_qids: Dict[str, List[str]] = {}
            for entry in target_ref_entries:
                question = ref_to_question.get(entry["ref"])
                if question is None:
                    continue
                target_questions.append(question)
                target_family_to_qids.setdefault(entry["family"], []).append(question["QuestionID"])
                answer = question.get("Answers", {}).get("SelectedByPosition")
                if answer is not None:
                    ground_truth[question["QuestionID"]] = int(answer)
            if len(target_questions) != len(target_ref_entries):
                continue

            messages = build_messages(
                profile_text=profile_text,
                retrieved_cards=candidates,
                target_questions=target_questions,
                target_ref_entries=target_ref_entries,
                include_reasoning=args.include_reasoning,
                demographics_text=demographics_text,
            )
            approx_prompt_tokens = estimate_chat_tokens(messages, args.model)
            token_counts.append(approx_prompt_tokens)

            custom_id = f"joint_social_oracle_augmented__pid_{pid}"
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
                "condition": "joint_social_oracle_augmented",
                "model": args.model,
                "include_reasoning": args.include_reasoning,
                "retrieved_top_k": len(candidates),
                "retrieved_custom_ids": [row.get("custom_id") for row in candidates],
                "retrieved_filenames": [row.get("filename") for row in candidates],
                "retrieved_scores": [row.get("score") for row in candidates],
                "search_query": candidate_row.get("search_query"),
                "search_query_obj": candidate_row.get("search_query_obj"),
                "target_family": "joint_social_block",
                "target_family_to_qids": target_family_to_qids,
                "target_question_ids": [q["QuestionID"] for q in target_questions],
                "ground_truth_answers": ground_truth,
                "allowed_input_families": STRUCTURED_ALLOWED_INPUT_FAMILIES
                + (["demographics"] if args.include_participant_demographics else []),
                "allowed_input_refs": used_input_refs
                + (
                    [
                        ref_for_parts("Demographics", "QID12"),
                        ref_for_parts("Demographics", "QID13"),
                        ref_for_parts("Demographics", "QID14"),
                    ]
                    if args.include_participant_demographics
                    else []
                ),
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
        preview_path.write_text(json.dumps(first_preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    token_counts_sorted = sorted(token_counts)
    token_summary = {
        "model": args.model,
        "tokenizer_source": tokenizer_source,
        "condition": "joint_social_oracle_augmented",
        "requests": request_count,
        "skipped_missing_pid": skipped_missing_pid,
        "skipped_no_candidates": skipped_no_candidates,
        "approx_prompt_tokens_total": int(sum(token_counts)),
        "approx_prompt_tokens_mean": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0,
        "approx_prompt_tokens_p95": token_counts_sorted[int(round((len(token_counts_sorted) - 1) * 0.95))]
        if token_counts_sorted
        else 0,
        "include_reasoning": args.include_reasoning,
    }
    token_path.write_text(json.dumps(token_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(token_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
