#!/usr/bin/env python3
"""Download output or error files for an OpenAI Batch job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--which", choices=("output", "error"), default="output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    payload = batch.model_dump() if hasattr(batch, "model_dump") else {}
    file_id = payload.get("output_file_id") if args.which == "output" else payload.get("error_file_id")
    if not file_id:
        raise ValueError(f"Batch {args.batch_id} does not have a {args.which}_file_id yet.")
    content = client.files.content(file_id)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(content.read())
    print(json.dumps({"batch_id": args.batch_id, "file_id": file_id, "output_path": str(args.output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
