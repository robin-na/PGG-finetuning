#!/usr/bin/env python3
"""Build OpenAI Batch requests that generate PGG-library retrieval queries for MobLab tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from evaluate_batch_results import extract_content, parse_json_object  # type: ignore
from moblab_llm_utils import (
    DEFAULT_OUTPUT_ROOT,
    build_retrieval_query_messages,
    load_jsonl,
)


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "retrieval_query_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument(
        "--persona-responses-jsonl",
        type=Path,
        default=None,
        help="Optional batch output JSONL from build_moblab_persona_batch.py. If provided, persona text is added to the query-writing prompt.",
    )
    return parser.parse_args()


def persona_text_by_instance(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    result: Dict[str, str] = {}
    for row in load_jsonl(path):
        custom_id = str(row.get("custom_id", ""))
        content, error, _ = extract_content(row)
        if error is not None or not content:
            continue
        obj = parse_json_object(content)
        if not isinstance(obj, dict):
            continue
        persona = obj.get("persona_summary")
        decision_style = obj.get("decision_style")
        text_parts = []
        if isinstance(persona, str) and persona.strip():
            text_parts.append(persona.strip())
        if isinstance(decision_style, str) and decision_style.strip():
            text_parts.append(decision_style.strip())
        if not text_parts:
            continue
        instance_id = custom_id.split("moblab_persona::", 1)[-1]
        result[instance_id] = "\n".join(text_parts)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks_jsonl)
    task_stem = args.tasks_jsonl.stem
    persona_by_instance = persona_text_by_instance(args.persona_responses_jsonl)

    requests_path = args.output_dir / f"requests_{task_stem}_retrieval_query_{args.model}.jsonl"
    manifest_path = args.output_dir / f"manifest_{task_stem}_retrieval_query.jsonl"
    preview_path = args.output_dir / f"preview_{task_stem}_retrieval_query_{args.model}.json"

    count = 0
    first_preview: Optional[Dict[str, Any]] = None
    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open("w", encoding="utf-8") as man_f:
        for instance in tasks:
            custom_id = f"moblab_retrieval_query::{instance['instance_id']}"
            persona_text = persona_by_instance.get(instance["instance_id"])
            messages = build_retrieval_query_messages(instance, persona_text=persona_text)
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
                "instance_id": instance["instance_id"],
                "task_type": instance["task_type"],
                "prediction_mode": instance.get("prediction_mode", "scalar"),
                "target_measure": instance["target_measure"],
                "user_id": instance["user_id"],
                "session_id": instance.get("session_id"),
                "used_persona": bool(persona_text),
                "model": args.model,
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
