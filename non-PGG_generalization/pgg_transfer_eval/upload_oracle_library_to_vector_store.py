#!/usr/bin/env python3
"""Upload the local oracle-library documents into an OpenAI vector store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LIBRARY_DIR = (
    PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "oracle_library"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-manifest",
        type=Path,
        default=DEFAULT_LIBRARY_DIR / "oracle_library_manifest.jsonl",
    )
    parser.add_argument(
        "--upload-manifest",
        type=Path,
        default=DEFAULT_LIBRARY_DIR / "oracle_vector_store_uploads.jsonl",
    )
    parser.add_argument("--vector-store-id", type=str, default=None)
    parser.add_argument("--vector-store-name", type=str, default="PGG Oracle Library")
    parser.add_argument(
        "--mode",
        choices=("single", "batch"),
        default="single",
        help="Upload one file at a time with per-file attributes, or use fast batch upload without per-file attributes.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-interval-ms", type=int, default=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many files to include in each batch upload when --mode batch is used.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="OpenAI SDK upload concurrency for --mode batch.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunked(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main() -> int:
    args = parse_args()
    manifest_rows = load_jsonl(args.library_manifest)
    if not manifest_rows:
        raise ValueError(f"No library rows found in {args.library_manifest}")

    client = OpenAI()
    vector_store_id = args.vector_store_id
    if not vector_store_id:
        vector_store = client.vector_stores.create(name=args.vector_store_name)
        vector_store_id = vector_store.id

    uploaded_filenames: Set[str] = set()
    if args.resume:
        for row in load_jsonl(args.upload_manifest):
            if row.get("filename"):
                uploaded_filenames.add(str(row["filename"]))

    pending_rows = [row for row in manifest_rows if str(row["filename"]) not in uploaded_filenames]
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]

    args.upload_manifest.parent.mkdir(parents=True, exist_ok=True)
    uploaded_count = 0
    with args.upload_manifest.open("a", encoding="utf-8") as out_f:
        if args.mode == "single":
            for row in pending_rows:
                filename = str(row["filename"])
                doc_path = Path(str(row["doc_path"]))
                if not doc_path.exists():
                    raise FileNotFoundError(f"Missing library document: {doc_path}")

                with doc_path.open("rb") as file_obj:
                    uploaded = client.vector_stores.files.upload_and_poll(
                        vector_store_id=vector_store_id,
                        file=file_obj,
                        attributes=row.get("attributes") or None,
                        poll_interval_ms=args.poll_interval_ms,
                    )

                payload = uploaded.model_dump() if hasattr(uploaded, "model_dump") else {}
                upload_row = {
                    "vector_store_id": vector_store_id,
                    "upload_mode": "single",
                    "custom_id": row["custom_id"],
                    "filename": filename,
                    "doc_path": str(doc_path),
                    "attributes": row.get("attributes") or {},
                    "vector_store_file": payload,
                }
                out_f.write(json.dumps(upload_row, ensure_ascii=False) + "\n")
                uploaded_count += 1
        else:
            for batch_index, rows in enumerate(chunked(pending_rows, args.batch_size), start=1):
                file_handles = []
                try:
                    for row in rows:
                        doc_path = Path(str(row["doc_path"]))
                        if not doc_path.exists():
                            raise FileNotFoundError(f"Missing library document: {doc_path}")
                        file_handles.append(doc_path.open("rb"))

                    uploaded = client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store_id,
                        files=file_handles,
                        max_concurrency=args.max_concurrency,
                        poll_interval_ms=args.poll_interval_ms,
                    )
                    payload = uploaded.model_dump() if hasattr(uploaded, "model_dump") else {}
                    batch_id = payload.get("id")
                    batch_status = payload.get("status")
                    for row in rows:
                        upload_row = {
                            "vector_store_id": vector_store_id,
                            "upload_mode": "batch",
                            "batch_index": batch_index,
                            "vector_store_file_batch_id": batch_id,
                            "vector_store_file_batch_status": batch_status,
                            "custom_id": row["custom_id"],
                            "filename": row["filename"],
                            "doc_path": row["doc_path"],
                            "attributes": row.get("attributes") or {},
                        }
                        out_f.write(json.dumps(upload_row, ensure_ascii=False) + "\n")
                        uploaded_count += 1
                finally:
                    for handle in file_handles:
                        handle.close()

    print(
        json.dumps(
            {
                "vector_store_id": vector_store_id,
                "upload_mode": args.mode,
                "uploaded_count": uploaded_count,
                "upload_manifest": str(args.upload_manifest),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
