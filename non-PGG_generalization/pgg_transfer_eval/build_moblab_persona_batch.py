#!/usr/bin/env python3
"""Build OpenAI Batch requests that generate compact personas for MobLab tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from moblab_llm_utils import (
    DEFAULT_OUTPUT_ROOT,
    build_persona_messages,
    load_jsonl,
)


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "persona_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-completion-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks_jsonl)
    task_stem = args.tasks_jsonl.stem
    if args.limit is not None:
        tasks = tasks[: args.limit]

    requests_path = args.output_dir / f"requests_{task_stem}_persona_{args.model}.jsonl"
    manifest_path = args.output_dir / f"manifest_{task_stem}_persona.jsonl"
    preview_path = args.output_dir / f"preview_{task_stem}_persona_{args.model}.json"

    first_preview: Optional[Dict[str, Any]] = None
    count = 0
    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open("w", encoding="utf-8") as man_f:
        for instance in tasks:
            custom_id = f"moblab_persona::{instance['instance_id']}"
            messages = build_persona_messages(instance)
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
