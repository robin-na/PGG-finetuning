#!/usr/bin/env python3
"""Submit a local JSONL file to the OpenAI Batch API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions")
    parser.add_argument("--completion-window", type=str, default="24h")
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI()
    metadata = None
    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    with args.requests_jsonl.open("rb") as handle:
        upload = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata,
    )
    payload = batch.model_dump() if hasattr(batch, "model_dump") else {}
    print(json.dumps({"input_file_id": upload.id, "batch": payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
