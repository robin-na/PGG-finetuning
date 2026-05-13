from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from repo_env import require_env_var


OPENAI_API_BASE = "https://api.openai.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small OpenAI Chat Completions JSONL request file synchronously and write "
            "Batch-like output JSONL. Intended as a fallback when Batch rejects a model/logprobs setup."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--api-base", type=str, default=OPENAI_API_BASE)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def request_chat_completion(
    *,
    api_key: str,
    api_base: str,
    request_row: dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    retry_base_sec: float,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{request_row['url']}"
    payload = json.dumps(request_row["body"]).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=payload, headers=headers, method=request_row["method"])
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
                return {
                    "custom_id": request_row["custom_id"],
                    "response": {
                        "status_code": response.status,
                        "request_id": response.headers.get("x-request-id"),
                        "body": body,
                    },
                    "error": None,
                }
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed_body: Any = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_body = raw_body
            last_error = {
                "status_code": exc.code,
                "request_id": exc.headers.get("x-request-id"),
                "body": parsed_body,
            }
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= max_retries:
                return {
                    "custom_id": request_row["custom_id"],
                    "response": last_error,
                    "error": parsed_body.get("error") if isinstance(parsed_body, dict) else parsed_body,
                }
        except urllib.error.URLError as exc:
            last_error = {
                "status_code": None,
                "request_id": None,
                "body": {"error": {"message": str(exc), "type": "url_error"}},
            }
            if attempt >= max_retries:
                return {
                    "custom_id": request_row["custom_id"],
                    "response": last_error,
                    "error": last_error["body"]["error"],
                }
        time.sleep(retry_base_sec * (2**attempt))

    return {
        "custom_id": request_row["custom_id"],
        "response": last_error or {"status_code": None, "request_id": None, "body": {}},
        "error": "exhausted retries",
    }


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output_jsonl}. Pass --force to overwrite.")
    rows = read_jsonl(args.input_jsonl)
    api_key = require_env_var("OPENAI_API_KEY")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    started_at = time.time()
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            result = request_chat_completion(
                api_key=api_key,
                api_base=args.api_base,
                request_row=row,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                retry_base_sec=args.retry_base_sec,
            )
            status_code = (result.get("response") or {}).get("status_code")
            if status_code == 200:
                successes += 1
            else:
                failures += 1
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(rows),
                            "successes": successes,
                            "failures": failures,
                            "elapsed_seconds": round(time.time() - started_at, 1),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
