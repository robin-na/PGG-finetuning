#!/usr/bin/env python3
"""Run OpenAI chat-completion requests from a batch-style JSONL file synchronously."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI, RateLimitError


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_custom_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            custom_id = row.get("custom_id")
            if custom_id:
                done.add(str(custom_id))
    return done


def error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        body = None
        try:
            body = exc.response.json()
        except Exception:
            body = getattr(exc, "body", None) or str(exc)
        return {
            "status_code": status,
            "body": body,
        }
    return {
        "status_code": None,
        "body": {"type": exc.__class__.__name__, "message": str(exc)},
    }


def call_one(client: OpenAI, request_row: Dict[str, Any], max_retries: int) -> Dict[str, Any]:
    custom_id = str(request_row["custom_id"])
    body = dict(request_row["body"])
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**body)
            payload = resp.model_dump() if hasattr(resp, "model_dump") else {}
            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": payload,
                },
            }
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            if attempt >= max_retries:
                return {"custom_id": custom_id, "error": {"type": exc.__class__.__name__, "detail": str(exc)}}
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS or attempt >= max_retries:
                payload = error_payload(exc)
                return {
                    "custom_id": custom_id,
                    "response": payload,
                }
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                return {"custom_id": custom_id, "error": {"type": exc.__class__.__name__, "detail": str(exc)}}
        time.sleep(delay)
        delay = min(delay * 2.0, 30.0)
    return {"custom_id": custom_id, "error": {"type": "unknown", "detail": "unreachable retry loop exit"}}


def main() -> int:
    args = parse_args()
    client = OpenAI()
    request_rows = load_jsonl(args.requests_jsonl)
    done_ids = existing_custom_ids(args.output_jsonl) if args.resume else set()
    pending = [row for row in request_rows if str(row.get("custom_id")) not in done_ids]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = 0
    total = len(pending)

    with args.output_jsonl.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(call_one, client, row, args.max_retries): str(row["custom_id"])
                for row in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                with lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(json.dumps({"completed": completed, "total": total, "output_jsonl": str(args.output_jsonl)}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
