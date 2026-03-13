#!/usr/bin/env python3
"""LLM prediction: given PGG archetype candidates + target question, predict participant response."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import random

from openai import OpenAI

from config import (
    GAME_QIDS,
    HF_CONFIG_WAVE_SPLIT,
    HF_DATASET_NAME,
    OUTPUT_ROOT,
    PILOT_N_PARTICIPANTS,
    PREDICTION_MODEL,
    QUESTION_CATALOG_JSON,
    RANDOM_STATE,
    TOP_K,
)

# ── System prompts ─────────────────────────────────────────────────────────────

_REASONING_SUFFIX = (
    "\n\nFor each question, first briefly explain your reasoning in "
    "<REASONING>...</REASONING> tags, then give your final answer in "
    "<ANSWER>...</ANSWER> tags."
)

SYSTEM_PROMPT = """You are an expert behavioral economist. Your task is to predict how a specific person would answer an economic decision-making question.

You will be given:
1. The person's demographics (age, sex, education)
2. Behavioral profiles from a Public Goods Game (PGG) that describe people with similar demographics. These profiles reveal cooperation tendencies, punishment behavior, fairness concerns, and other traits.
3. A target question about an economic decision.

Use the PGG profiles as personality signals. Extract relevant traits (cooperativeness, fairness concern, risk tolerance, strategic thinking, punishment willingness, patience) and apply them to predict the target question response.

IMPORTANT: You must output your answer in the exact format specified."""

SYSTEM_PROMPT_DEMOGRAPHICS_ONLY = """You are an expert behavioral economist. Your task is to predict how a specific person would answer an economic decision-making question, based solely on their demographics.

You will be given:
1. The person's demographics (age, sex, education)
2. A target question about an economic decision.

Use your knowledge of how demographics correlate with economic behavior to make your prediction.

IMPORTANT: You must output your answer in the exact format specified."""

_CROSS_GAME_SUFFIX = (
    "\n\nYou will also be given this person's actual behavior in other related economic "
    "games. Use these as direct evidence of their fairness preferences, cooperativeness, "
    "and strategic thinking."
)

# ── Game label lookup ──────────────────────────────────────────────────────────

_GAME_LABELS: Dict[str, str] = {
    "QID117": "Trust Game – sender",
    "QID118": "Trust Game – receiver ($5→$15)",
    "QID119": "Trust Game – receiver ($4→$12)",
    "QID120": "Trust Game – receiver ($3→$9)",
    "QID121": "Trust Game – receiver ($2→$6)",
    "QID122": "Trust Game – receiver ($1→$3)",
    "QID224": "Ultimatum Game – proposer",
    "QID225": "Ultimatum Game – responder ($5 offer)",
    "QID226": "Ultimatum Game – responder ($4 offer)",
    "QID227": "Ultimatum Game – responder ($3 offer)",
    "QID228": "Ultimatum Game – responder ($2 offer)",
    "QID229": "Ultimatum Game – responder ($1 offer)",
    "QID230": "Ultimatum Game – responder ($0 offer)",
    "QID231": "Dictator Game",
}

# ── Cross-game behavior helpers ──────────────────────────────────────────────

# Reverse map: QID → game name
_QID_TO_GAME: Dict[str, str] = {
    qid: game
    for game, qids in GAME_QIDS.items()
    for qid in qids
}

_GAME_DISPLAY = {
    "trust": "Trust Game",
    "ultimatum": "Ultimatum Game",
    "dictator": "Dictator Game",
}


def load_cross_game_behavior(pilot_pids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Load wave1_3 economic game answers (Trust/Ultimatum/Dictator) for pilot participants."""
    from datasets import load_dataset

    all_qids = {qid for qids in GAME_QIDS.values() for qid in qids}

    catalog = json.load(QUESTION_CATALOG_JSON.open("r", encoding="utf-8"))
    qid_to_q = {q["QuestionID"]: q for q in catalog}

    print("Loading cross-game ground truth from HuggingFace...")
    ds = load_dataset(HF_DATASET_NAME, HF_CONFIG_WAVE_SPLIT)["data"]

    result: Dict[str, Dict[str, Any]] = {}
    for example in ds:
        pid = str(example["pid"])
        if pid not in pilot_pids:
            continue
        persona = json.loads(example["wave1_3_persona_json"])
        answers: Dict[str, Any] = {}
        for block in persona:
            for q in block.get("Questions", []):
                qid = q.get("QuestionID")
                if qid not in all_qids:
                    continue
                q_def = qid_to_q.get(qid, {})
                q_answers = q.get("Answers", {})
                if q_def.get("QuestionType") == "MC":
                    pos = q_answers.get("SelectedByPosition")
                    if pos is not None:
                        answers[qid] = int(pos)
                elif q_def.get("QuestionType") == "Matrix":
                    csv_cols = q_def.get("csv_columns", [])
                    selected = q_answers.get("SelectedByPosition", [])
                    if isinstance(selected, list):
                        for i, val in enumerate(selected):
                            if i < len(csv_cols) and val is not None:
                                answers[csv_cols[i]] = int(val)
        result[pid] = answers

    print(f"  Loaded cross-game behavior for {len(result)} participants.")
    return result


def _opt_text(qid: str, position: int, questions_by_qid: Dict[str, Dict]) -> str:
    """Resolve option position to human-readable text."""
    q = questions_by_qid.get(qid, {})
    options = q.get("Options", [])
    if 1 <= position <= len(options):
        return options[position - 1]
    return f"option {position}"


def format_cross_game_context(
    pid_answers: Dict[str, Any],
    target_game: str,
    questions_by_qid: Dict[str, Dict],
) -> str:
    """Format participant's actual behavior in the other two games."""
    other_games = [g for g in ["trust", "ultimatum", "dictator"] if g != target_game]
    lines: List[str] = []

    for game in other_games:
        display = _GAME_DISPLAY[game]
        lines.append(f"### {display}")
        found_any = False
        for qid in GAME_QIDS[game]:
            if qid in pid_answers:
                pos = pid_answers[qid]
                opt = _opt_text(qid, pos, questions_by_qid)
                q_short = questions_by_qid.get(qid, {}).get("QuestionText", qid)
                # Trim long question texts
                if len(q_short) > 120:
                    q_short = q_short[:120] + "..."
                lines.append(f"- {q_short}")
                lines.append(f"  → Chose: {opt}")
                found_any = True
        if not found_any:
            lines.append("- (No data available)")
        lines.append("")

    return "\n".join(lines)


# ── Question formatting ───────────────────────────────────────────────────────

def format_mc_question(q: Dict[str, Any]) -> str:
    text = q["QuestionText"]
    options = q.get("Options", [])
    lines = [text, ""]
    for i, opt in enumerate(options, 1):
        lines.append(f"  {i}. {opt}")
    lines.append("")
    lines.append(f"Answer with the option NUMBER (1-{len(options)}).")
    lines.append("")
    lines.append("<ANSWER>option_number</ANSWER>")
    return "\n".join(lines)


def format_matrix_question(q: Dict[str, Any]) -> str:
    text = q["QuestionText"]
    rows = q.get("Rows", [])
    csv_cols = q.get("csv_columns", [])

    lines = [text, ""]
    lines.append("For each row below, choose column 1 or column 2.")
    lines.append("")

    for i, row in enumerate(rows):
        parts = row.split(":")
        if len(parts) == 2:
            lines.append(f"  Row {i+1}: Column 1 = {parts[0].strip()}, Column 2 = {parts[1].strip()}")
        else:
            lines.append(f"  Row {i+1}: {row}")

    lines.append("")
    lines.append(f"Answer with a comma-separated list of {len(rows)} numbers (1 or 2), one for each row.")
    lines.append("Example for 3 rows: <ANSWER>1,2,1</ANSWER>")
    lines.append("")
    lines.append(f"<ANSWER>comma_separated_{len(rows)}_values</ANSWER>")
    return "\n".join(lines)


def format_question(q: Dict[str, Any]) -> str:
    if q["QuestionType"] == "MC":
        return format_mc_question(q)
    elif q["QuestionType"] == "Matrix":
        return format_matrix_question(q)
    else:
        raise ValueError(f"Unsupported question type: {q['QuestionType']}")


# ── Prompt construction ───────────────────────────────────────────────────────

def build_user_message(
    demographics: Dict[str, str],
    candidates: List[Dict[str, Any]],
    question: Dict[str, Any],
    use_archetypes: bool = True,
    reasoning_history: Optional[List[Tuple[str, str]]] = None,
    cross_game_context: Optional[str] = None,
) -> str:
    parts = []

    # Demographics
    parts.append("## Demographics")
    parts.append(f"Age: {demographics.get('age_label', 'Unknown')}")
    parts.append(f"Sex: {demographics.get('sex_label', 'Unknown')}")
    parts.append(f"Education: {demographics.get('education_label', 'Unknown')}")
    parts.append("")

    # Archetype candidates
    if use_archetypes and candidates:
        parts.append("## PGG Behavioral Profiles of People with Similar Demographics")
        parts.append("(These describe how similar people behaved in a Public Goods Game)")
        parts.append("")
        for c in candidates:
            rank = c.get("rank", "?")
            score = c.get("cosine_similarity", 0)
            text = c.get("trait_summary") or c.get("archetype_text", "")
            if len(text) > 1500:
                text = text[:1500] + "..."
            parts.append(f"### Profile {rank} (match score: {score:.2f})")
            parts.append(text)
            parts.append("")

    # Cross-game behavioral evidence
    if cross_game_context:
        parts.append("## This Person's Actual Behavior in Related Economic Games")
        parts.append("(Use this as direct evidence of their preferences and decision style)")
        parts.append("")
        parts.append(cross_game_context)

    # Prior reasoning history (verbalized chain-of-thought)
    if reasoning_history:
        parts.append("## Your Prior Reasoning (this session)")
        for game_label, reasoning_text in reasoning_history:
            parts.append(f"- [{game_label}] You thought: {reasoning_text}")
        parts.append("")

    # Target question
    parts.append("## Question")
    parts.append(format_question(question))

    return "\n".join(parts)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_answer(response_text: str, question: Dict[str, Any]) -> Any:
    match = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", response_text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()

    if question["QuestionType"] == "MC":
        try:
            return int(raw)
        except ValueError:
            nums = re.findall(r"\d+", raw)
            return int(nums[0]) if nums else None

    elif question["QuestionType"] == "Matrix":
        try:
            values = [int(x.strip()) for x in raw.split(",") if x.strip()]
            return values
        except ValueError:
            return None

    return raw


def parse_reasoning(response_text: str) -> Optional[str]:
    m = re.search(r"<REASONING>(.*?)</REASONING>", response_text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Single prediction ─────────────────────────────────────────────────────────

def predict_one(
    client: OpenAI,
    model: str,
    demographics: Dict[str, str],
    candidates: List[Dict[str, Any]],
    question: Dict[str, Any],
    use_archetypes: bool = True,
    with_reasoning: bool = False,
    reasoning_history: Optional[List[Tuple[str, str]]] = None,
    cross_game_context: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPT if use_archetypes else SYSTEM_PROMPT_DEMOGRAPHICS_ONLY
    if cross_game_context:
        system_prompt = system_prompt + _CROSS_GAME_SUFFIX
    if with_reasoning:
        system_prompt = system_prompt + _REASONING_SUFFIX

    user_message = build_user_message(
        demographics, candidates, question, use_archetypes,
        reasoning_history=reasoning_history or [],
        cross_game_context=cross_game_context,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=800 if with_reasoning else 500,
        temperature=0.0,
    )
    response_text = resp.choices[0].message.content
    parsed = parse_answer(response_text, question)
    reasoning = parse_reasoning(response_text) if with_reasoning else None

    return {
        "response_text": response_text,
        "parsed_answer": parsed,
        "reasoning": reasoning,
        "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }


# ── Per-participant job (sequential questions, accumulating reasoning) ─────────

def predict_participant(
    pid: str,
    pid_data: Dict[str, Any],
    questions: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    mode: str,
    with_reasoning: bool,
    completed: Set[Tuple[str, str]],
    cross_game_data: Optional[Dict[str, Any]] = None,
    questions_by_qid: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    demographics = {
        "age_label": pid_data.get("age_label", "Unknown"),
        "sex_label": pid_data.get("sex_label", "Unknown"),
        "education_label": pid_data.get("education_label", "Unknown"),
    }

    # Pre-load this participant's cross-game answers once
    pid_answers: Dict[str, Any] = (cross_game_data or {}).get(pid, {})

    reasoning_history: List[Tuple[str, str]] = []
    results = []

    for q in questions:
        qid = q["QuestionID"]
        if (pid, qid) in completed:
            continue

        # Build cross-game context for this target question (exclude its own game)
        cross_game_context: Optional[str] = None
        if pid_answers and questions_by_qid:
            target_game = _QID_TO_GAME.get(qid)
            if target_game:
                cross_game_context = format_cross_game_context(
                    pid_answers, target_game, questions_by_qid
                )

        try:
            result = predict_one(
                client=client,
                model=model,
                demographics=demographics,
                candidates=candidates,
                question=q,
                use_archetypes=(mode != "demographics_only"),
                with_reasoning=with_reasoning,
                reasoning_history=reasoning_history if with_reasoning else [],
                cross_game_context=cross_game_context,
            )

            if with_reasoning and result.get("reasoning"):
                label = _GAME_LABELS.get(qid, qid)
                reasoning_history.append((label, result["reasoning"]))

            results.append({
                "pid": pid,
                "question_id": qid,
                "question_type": q["QuestionType"],
                "mode": mode,
                "parsed_answer": result["parsed_answer"],
                "response_text": result["response_text"],
                "reasoning": result.get("reasoning"),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            })

        except Exception as e:
            print(f"  ERROR pid={pid} qid={qid}: {e}")
            time.sleep(1)

    return results


# ── Data loading ──────────────────────────────────────────────────────────────

def load_economic_questions(
    catalog_path: Path,
    game_filter: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    catalog = json.load(catalog_path.open("r", encoding="utf-8"))
    questions = [
        q for q in catalog
        if "Economic preferences" in q.get("BlockName", "")
        and not q.get("is_descriptive", False)
        and q["QuestionType"] in ("MC", "Matrix")
    ]
    if game_filter:
        questions = [q for q in questions if q["QuestionID"] in game_filter]

    # Sort: trust → ultimatum → dictator → others, then by QID
    game_order: Dict[str, int] = {}
    for i, game_qids in enumerate([GAME_QIDS["trust"], GAME_QIDS["ultimatum"], GAME_QIDS["dictator"]]):
        for qid in game_qids:
            game_order[qid] = i
    questions.sort(key=lambda q: (game_order.get(q["QuestionID"], 99), q["QuestionID"]))

    return questions


def load_candidates_with_summaries(
    candidates_jsonl: Path,
    summaries_jsonl: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    summary_by_idx: Dict[int, str] = {}
    if summaries_jsonl and summaries_jsonl.exists():
        with summaries_jsonl.open("r") as f:
            for line in f:
                s = line.strip()
                if s:
                    row = json.loads(s)
                    summary_by_idx[row["idx"]] = row["trait_summary"]

    by_pid: Dict[str, Dict[str, Any]] = {}
    with candidates_jsonl.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                row = json.loads(s)
                if summary_by_idx:
                    for c in row.get("candidates", []):
                        idx = c.get("bank_idx")
                        if idx is not None and idx in summary_by_idx:
                            c["trait_summary"] = summary_by_idx[idx]
                by_pid[str(row["pid"])] = row
    return by_pid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM prediction with PGG archetype transfer.")
    parser.add_argument(
        "--candidates-jsonl",
        type=Path,
        default=OUTPUT_ROOT / "candidate_archetypes.jsonl",
    )
    parser.add_argument(
        "--summaries-jsonl",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_trait_summaries.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "predictions",
    )
    parser.add_argument("--model", type=str, default=PREDICTION_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--pilot-n", type=int, default=PILOT_N_PARTICIPANTS)
    parser.add_argument(
        "--mode",
        choices=["archetype", "demographics_only", "random_archetype"],
        default="archetype",
    )
    parser.add_argument(
        "--games",
        type=str,
        default="all",
        help="Comma-separated games to predict: trust,ultimatum,dictator,all",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Enable verbalized chain-of-thought: model outputs <REASONING> injected into subsequent questions",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for API calls (parallelism across participants)",
    )
    parser.add_argument(
        "--with-cross-game",
        action="store_true",
        help="Include participant's actual behavior in the other two games as prompt context",
    )
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    # Parse game filter
    game_filter: Optional[Set[str]] = None
    if args.games and args.games.lower() != "all":
        game_names = [g.strip().lower() for g in args.games.split(",")]
        for g in game_names:
            if g not in GAME_QIDS:
                print(f"Unknown game '{g}'. Available: {list(GAME_QIDS.keys())}")
                sys.exit(1)
        game_filter = set()
        for g in game_names:
            game_filter.update(GAME_QIDS[g])

    # Load questions
    questions = load_economic_questions(QUESTION_CATALOG_JSON, game_filter)
    print(f"Loaded {len(questions)} questions: {[q['QuestionID'] for q in questions]}")

    # Load candidates
    print("Loading candidate archetypes...")
    candidates_by_pid = load_candidates_with_summaries(
        args.candidates_jsonl,
        args.summaries_jsonl if args.mode != "demographics_only" else None,
    )
    all_pids = list(candidates_by_pid.keys())
    print(f"Loaded candidates for {len(all_pids)} participants.")

    # Sample pilot participants
    rng = random.Random(RANDOM_STATE)
    pilot_pids = rng.sample(all_pids, min(args.pilot_n, len(all_pids))) if args.pilot_n else all_pids
    print(f"Pilot: {len(pilot_pids)} participants × {len(questions)} questions = {len(pilot_pids) * len(questions)} calls")
    if args.with_reasoning:
        print("Verbalized reasoning: ON")
    if args.with_cross_game:
        print("Cross-game context: ON")
    print(f"Workers: {args.workers}")

    # Load cross-game behavior if requested (loads from HuggingFace once upfront)
    cross_game_data: Optional[Dict[str, Dict[str, Any]]] = None
    questions_by_qid: Optional[Dict[str, Dict]] = None
    if args.with_cross_game:
        catalog = json.load(QUESTION_CATALOG_JSON.open("r", encoding="utf-8"))
        questions_by_qid = {q["QuestionID"]: q for q in catalog}
        cross_game_data = load_cross_game_behavior(set(pilot_pids))

    # For random archetype mode
    all_candidates_flat = []
    if args.mode == "random_archetype":
        for pid_data in candidates_by_pid.values():
            all_candidates_flat.extend(pid_data.get("candidates", []))

    # Output setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"predictions_{args.mode}.jsonl"

    # Load existing completed (pid, qid) pairs for resume
    completed: Set[Tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                s = line.strip()
                if s:
                    row = json.loads(s)
                    completed.add((str(row["pid"]), row["question_id"]))
        if completed:
            print(f"Resuming: {len(completed)} predictions already done.")

    client = OpenAI()
    write_lock = threading.Lock()
    total = len(pilot_pids) * len(questions)
    done_count = len(completed)
    done_lock = threading.Lock()

    def write_results(rows: List[Dict[str, Any]]) -> None:
        nonlocal done_count
        with write_lock:
            with output_path.open("a", encoding="utf-8") as f_out:
                for row in rows:
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            with done_lock:
                done_count += len(rows)
                if done_count % 50 < len(rows):
                    print(f"  [{done_count}/{total}] predictions written")

    def make_candidates(pid: str) -> List[Dict[str, Any]]:
        if args.mode == "archetype":
            return candidates_by_pid[pid].get("candidates", [])[:args.top_k]
        elif args.mode == "random_archetype":
            return rng.sample(all_candidates_flat, min(args.top_k, len(all_candidates_flat)))
        else:
            return []

    # Filter pids that still have work to do
    pending_pids = [
        pid for pid in pilot_pids
        if any((pid, q["QuestionID"]) not in completed for q in questions)
    ]
    print(f"Participants with remaining work: {len(pending_pids)}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                predict_participant,
                pid,
                candidates_by_pid[pid],
                questions,
                make_candidates(pid),
                client,
                args.model,
                args.mode,
                args.with_reasoning,
                completed,
                cross_game_data,
                questions_by_qid,
            ): pid
            for pid in pending_pids
        }

        for future in as_completed(futures):
            pid = futures[future]
            try:
                rows = future.result()
                if rows:
                    write_results(rows)
            except Exception as e:
                print(f"  FATAL ERROR for pid={pid}: {e}")

    print(f"\nDone. Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
