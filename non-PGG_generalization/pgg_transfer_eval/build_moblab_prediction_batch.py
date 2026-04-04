#!/usr/bin/env python3
"""Build OpenAI Batch requests for MobLab prediction tasks under multiple LLM baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluate_batch_results import extract_content, parse_json_object  # type: ignore
from moblab_llm_utils import (
    DEFAULT_OUTPUT_ROOT,
    build_prediction_messages,
    load_jsonl,
    normalize_whitespace,
)


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "prediction_batch"
BASELINES = ("direct", "persona", "meta_persona", "retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=600)
    parser.add_argument(
        "--persona-responses-jsonl",
        type=Path,
        default=None,
        help="Required for baseline persona or meta_persona.",
    )
    parser.add_argument(
        "--retrieval-candidates-jsonl",
        type=Path,
        default=None,
        help="Required for baseline retrieval.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def persona_text_by_instance(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    result: Dict[str, str] = {}
    for row in load_jsonl(path):
        content, error, _ = extract_content(row)
        if error is not None or not content:
            continue
        obj = parse_json_object(content)
        if not isinstance(obj, dict):
            continue
        persona = obj.get("persona_summary")
        decision_style = obj.get("decision_style")
        parts = []
        if isinstance(persona, str) and persona.strip():
            parts.append(persona.strip())
        if isinstance(decision_style, str) and decision_style.strip():
            parts.append(decision_style.strip())
        if not parts:
            continue
        instance_id = str(row.get("custom_id", "")).split("moblab_persona::", 1)[-1]
        result[instance_id] = "\n".join(parts)
    return result


def candidates_by_instance(path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        result[str(row["instance_id"])] = list(row.get("candidates") or [])
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks_jsonl)
    task_stem = args.tasks_jsonl.stem
    persona_by_instance = persona_text_by_instance(args.persona_responses_jsonl)
    retrieval_by_instance = candidates_by_instance(args.retrieval_candidates_jsonl)

    if args.baseline in {"persona", "meta_persona"} and not persona_by_instance:
        raise ValueError("--persona-responses-jsonl is required for baseline persona or meta_persona.")
    if args.baseline == "retrieval" and not retrieval_by_instance:
        raise ValueError("--retrieval-candidates-jsonl is required for baseline retrieval.")

    requests_path = args.output_dir / f"requests_{task_stem}_{args.baseline}_{args.model}.jsonl"
    manifest_path = args.output_dir / f"manifest_{task_stem}_{args.baseline}.jsonl"
    preview_path = args.output_dir / f"preview_{task_stem}_{args.baseline}_{args.model}.json"

    count = 0
    first_preview: Optional[Dict[str, Any]] = None
    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open("w", encoding="utf-8") as man_f:
        for instance in tasks:
            instance_id = str(instance["instance_id"])
            persona_text = persona_by_instance.get(instance_id)
            retrieved_cards = retrieval_by_instance.get(instance_id, [])[: args.top_k]
            custom_id = f"moblab_predict::{args.baseline}::{instance_id}"
            messages = build_prediction_messages(
                instance=instance,
                baseline=args.baseline,
                persona_text=persona_text,
                retrieved_cards=retrieved_cards,
            )
            request_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": args.max_completion_tokens,
                },
            }
            manifest_row = {
                "custom_id": custom_id,
                "baseline": args.baseline,
                "instance_id": instance_id,
                "task_type": instance["task_type"],
                "prediction_mode": instance.get("prediction_mode", "scalar"),
                "target_measure": instance["target_measure"],
                "user_id": instance["user_id"],
                "session_id": instance.get("session_id"),
                "model": args.model,
                "used_persona": bool(persona_text),
                "used_retrieval": bool(retrieved_cards),
                "retrieved_candidate_ids": [normalize_whitespace(str(card.get("custom_id", ""))) for card in retrieved_cards],
                "gold_share_percent": instance.get("gold_share_percent", instance.get("gold_future_share_percent")),
                "gold_future_rounds_share_percent": instance.get("gold_future_rounds_share_percent"),
                "persistence_share_percent": instance.get("persistence_share_percent"),
                "persistence_future_rounds_share_percent": instance.get("persistence_future_rounds_share_percent"),
            }
            req_f.write(json.dumps(request_row, ensure_ascii=False) + "\n")
            man_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            if first_preview is None:
                first_preview = {"custom_id": custom_id, "messages": messages}
            count += 1

    if first_preview is not None:
        preview_path.write_text(json.dumps(first_preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"count": count, "requests_path": str(requests_path), "manifest_path": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
