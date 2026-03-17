#!/usr/bin/env python3
"""Build one-step Responses API batches with hosted file search for oracle augmentation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_joint_social_baseline_batch import (  # type: ignore
    INVENTORY_CSV,
    STRUCTURED_ALLOWED_INPUT_FAMILIES,
    build_source_by_ref,
    estimate_chat_tokens,
    find_question_map,
    filter_target_entries,
    get_encoding,
    group_target_questions,
    load_inventory,
    load_jsonl,
    load_question_catalog,
    load_wave_split,
    ref_for_row,
    render_structured_profile_text,
    render_target_question,
    select_allowed_and_target_refs,
)
from build_joint_social_oracle_query_batch import render_demographics  # type: ignore


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "joint_social_oracle_onestep"
)
TARGET_FAMILY_ORDER = ["trust", "ultimatum", "dictator"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-store-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--limit-participants", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--reuse-manifest", type=Path, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-num-results", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--target-qids",
        type=str,
        default=None,
        help="Optional comma-separated subset of target QIDs.",
    )
    return parser.parse_args()


def build_instructions(include_reasoning: bool) -> str:
    lines = [
        "You are predicting how a specific Twin participant would answer held-out social decision questions.",
        "The participant profile contains waves 1-3 information with all trust, ultimatum, and dictator items removed.",
        "Before answering, use the file_search tool exactly once to retrieve analogous oracle profiles from repeated public-goods games.",
        "The retrieved oracle profiles are analogical evidence only; do not assume any retrieved profile is the same person as the Twin participant.",
        "Use the Twin participant profile as primary evidence and the retrieved oracle profiles as secondary analogical evidence.",
        "Use demographics as a weak prior, not the main criterion.",
        "Search for specific analogical patterns, not generic niceness: baseline cooperation level, conditionality, reciprocity, fairness sensitivity, norm enforcement, exploitation caution, and behavioral stability.",
        "Do not average blindly across retrieved profiles.",
        "Weigh retrieved PGG evidence by how specifically it matches the Twin participant and the target question.",
        "Use retrieved PGG evidence when it adds a concrete analogical pattern that is relevant to the target question.",
        "Calibrate to the actual answer scale and the specific game mechanics.",
        "Keep the games distinct: trust return is not ultimatum acceptance, and dictator allocation is not ultimatum offering.",
        "Respect the option list shown for each QID exactly.",
        "For binary ultimatum-receiver questions, the answer must be 1 or 2 only.",
        "Return JSON only and do not include markdown.",
    ]
    if include_reasoning:
        lines.append(
            'Return JSON with keys "reasoning" and "answers", in that order. '
            '"reasoning" must map each QID to a short explanation that begins with the game role. '
            '"answers" must map each QID to an integer option number. '
            "Reason through each QID first, then give the final prediction."
        )
    else:
        lines.append(
            'Return JSON with exactly one top-level key, "answers", mapping each QID to an integer option number.'
        )
    return "\n".join(lines)


def build_input_text(
    profile_text: str,
    demographics_text: str,
    target_questions: List[Dict[str, Any]],
    target_ref_entries: List[Dict[str, str]],
    include_reasoning: bool,
) -> str:
    qid_list = [q["QuestionID"] for q in target_questions]
    if include_reasoning:
        response_shape = {
            "reasoning": {qid: "short explanation" for qid in qid_list},
            "answers": {qid: "integer option number" for qid in qid_list},
        }
    else:
        response_shape = {"answers": {qid: "integer option number" for qid in qid_list}}

    lines = [
        "# Participant Profile",
        "",
        demographics_text,
        "",
        profile_text,
        "",
        "# Retrieval Goal",
        "Retrieve repeated-public-goods-game oracle profiles that may help predict this participant's held-out trust, ultimatum, and dictator behavior.",
        "Search for analogous latent tendencies such as generosity, reciprocity, fairness sensitivity, norm enforcement, exploitation caution, conditionality, and stability.",
        "Prefer specific behavioral analogies over generic prosocial wording.",
        "Relevant PGG analogies include: always-high vs moderate contribution, rigid vs conditional cooperation, tolerance of free-riding, use or non-use of punishment/reward, and response to exploitation or end-game incentives.",
        "Let the Twin profile lead, and use retrieved oracle profiles to sharpen the judgment when they provide a useful analogy.",
        "",
        "# Held-Out Social Game Questions",
        "",
        "The following questions were removed from the participant profile.",
        "Predict how this participant would answer each question.",
        "",
    ]
    for family, questions in group_target_questions(target_questions, target_ref_entries):
        lines.append(f"## {family.replace('_', ' ').title()}")
        lines.append("")
        for q in questions:
            lines.append(render_target_question(q))
            lines.append("")
    lines.extend(
        [
            "# Output Format",
            "",
            "Your final answer must be valid JSON.",
            json.dumps(response_shape, ensure_ascii=False, indent=2),
            "",
            "Use the real predicted option number for each QID.",
        ]
    )
    if include_reasoning:
        lines.extend(
            [
                "Each reasoning string should explicitly name the role before the explanation.",
                'Example style: "Trust receiver; ..." or "Ultimatum receiver; ...".',
                "If retrieved PGG evidence was useful, mention the specific analogical pattern briefly; if it was not useful, do not force it into the explanation.",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    inventory_rows = load_inventory(INVENTORY_CSV)
    catalog = load_question_catalog()
    source_by_ref = build_source_by_ref(catalog)
    _, target_rows = select_allowed_and_target_refs(inventory_rows, source_by_ref)
    if not target_rows:
        raise ValueError("Failed to build target refs for joint social oracle one-step batch.")
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
        [qid.strip() for qid in args.target_qids.split(",")] if args.target_qids else None
    )
    target_ref_entries = filter_target_entries(target_ref_entries, selected_target_qids)

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
    requests_path = args.output_dir / f"requests_joint_social_oracle_onestep_{args.model}.jsonl"
    manifest_path = args.output_dir / "manifest_joint_social_oracle_onestep.jsonl"
    preview_path = args.output_dir / f"preview_joint_social_oracle_onestep_{args.model}.json"
    token_path = args.output_dir / f"token_estimate_joint_social_oracle_onestep_{args.model}.json"

    token_counts: List[int] = []
    request_count = 0
    first_preview: Optional[Dict[str, Any]] = None
    _, tokenizer_source = get_encoding(args.model)
    instructions = build_instructions(include_reasoning=args.include_reasoning)

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

            target_questions: List[Dict[str, Any]] = []
            ground_truth: Dict[str, int] = {}
            target_family_to_qids: Dict[str, List[str]] = {family: [] for family in TARGET_FAMILY_ORDER}
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

            input_text = build_input_text(
                profile_text=profile_text,
                demographics_text=demographics_text,
                target_questions=target_questions,
                target_ref_entries=target_ref_entries,
                include_reasoning=args.include_reasoning,
            )

            pseudo_messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ]
            approx_prompt_tokens = estimate_chat_tokens(pseudo_messages, args.model)
            token_counts.append(approx_prompt_tokens)

            tools = [
                {
                    "type": "file_search",
                    "vector_store_ids": [args.vector_store_id],
                    "max_num_results": args.max_num_results,
                    "ranking_options": {
                        "ranker": "auto",
                        **(
                            {"score_threshold": args.score_threshold}
                            if args.score_threshold is not None
                            else {}
                        ),
                    },
                }
            ]
            body: Dict[str, Any] = {
                "model": args.model,
                "instructions": instructions,
                "input": input_text,
                "tools": tools,
                "tool_choice": {"type": "file_search"},
                "max_tool_calls": 1,
                "parallel_tool_calls": False,
                "include": ["file_search_call.results"],
                "text": {"format": {"type": "json_object"}},
            }
            if args.max_output_tokens is not None:
                body["max_output_tokens"] = args.max_output_tokens

            custom_id = f"joint_social_oracle_onestep__pid_{pid}"
            request_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            req_f.write(json.dumps(request_row, ensure_ascii=False) + "\n")

            manifest_row = {
                "custom_id": custom_id,
                "pid": pid,
                "condition": "joint_social_oracle_onestep",
                "model": args.model,
                "include_reasoning": args.include_reasoning,
                "vector_store_id": args.vector_store_id,
                "file_search_max_num_results": args.max_num_results,
                "file_search_score_threshold": args.score_threshold,
                "target_family": "joint_social_block",
                "target_families": [family for family in TARGET_FAMILY_ORDER if target_family_to_qids.get(family)],
                "target_family_to_qids": target_family_to_qids,
                "target_question_ids": [q["QuestionID"] for q in target_questions],
                "ground_truth_answers": ground_truth,
                "allowed_input_families": STRUCTURED_ALLOWED_INPUT_FAMILIES + ["demographics"],
                "allowed_input_refs": used_input_refs
                + [
                    "Demographics::QID12",
                    "Demographics::QID13",
                    "Demographics::QID14",
                ],
                "excluded_target_refs": [entry["ref"] for entry in target_ref_entries],
                "approx_prompt_tokens": approx_prompt_tokens,
                "profile_rendered_item_count": rendered_items,
                "profile_char_count": len(profile_text),
            }
            manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

            if first_preview is None:
                first_preview = {
                    "custom_id": custom_id,
                    "instructions": instructions,
                    "input": input_text,
                    "body": body,
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
        "condition": "joint_social_oracle_onestep",
        "source_participants": source_participants,
        "requests": request_count,
        "include_reasoning": args.include_reasoning,
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
