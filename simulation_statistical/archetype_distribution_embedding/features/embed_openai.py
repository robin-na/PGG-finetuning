#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
for path in (SCRIPT_DIR, PACKAGE_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openai import OpenAI

from simulation_statistical.archetype_distribution_embedding.utils.io_utils import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for archetype input JSONL.")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-retries", type=int, default=6)
    return parser


def _load_processed_row_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed = set()
    for row in read_jsonl(path):
        row_id = row.get("row_id")
        if row_id:
            processed.add(str(row_id))
    return processed


def _batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def _embed_batch(
    client: OpenAI,
    batch: list[dict[str, Any]],
    model: str,
    max_retries: int,
) -> list[dict[str, Any]]:
    texts = [row["text"] for row in batch]
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            rows = []
            for meta, embedding_row in zip(batch, response.data):
                item = dict(meta)
                item["embedding"] = embedding_row.embedding
                rows.append(item)
            return rows
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            sleep_seconds = min(2 ** attempt, 30)
            time.sleep(sleep_seconds)
    raise RuntimeError("Unreachable retry state")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set. Export your API key before running."
        )

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    input_rows = read_jsonl(input_path)
    processed_row_ids = _load_processed_row_ids(output_path)
    pending_rows = [row for row in input_rows if str(row.get("row_id")) not in processed_row_ids]

    print(
        f"Embedding {len(pending_rows)} pending rows out of {len(input_rows)} total "
        f"(resume found {len(processed_row_ids)} completed rows).",
        flush=True,
    )
    if not pending_rows:
        return

    client = OpenAI(api_key=api_key)
    batches = _batched(pending_rows, args.batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        embedded_rows = _embed_batch(client=client, batch=batch, model=args.model, max_retries=args.max_retries)
        write_jsonl(output_path, embedded_rows, append=True)
        print(f"Completed batch {batch_index}/{len(batches)}", flush=True)


if __name__ == "__main__":
    main()
