from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "repo_env.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from repo_env import require_env_var


API_BASE = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit an OpenAI Batch JSONL file without requiring the openai Python package."
    )
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--state-json", type=Path, default=None)
    return parser.parse_args()


def request_json(api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return open_request(request)


def multipart_body(
    *,
    fields: dict[str, str],
    files: dict[str, Path],
    boundary: str,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def upload_file(api_key: str, path: Path, *, purpose: str) -> dict[str, Any]:
    boundary = f"----codex-{uuid.uuid4().hex}"
    body = multipart_body(fields={"purpose": purpose}, files={"file": path}, boundary=boundary)
    request = urllib.request.Request(
        f"{API_BASE}/files",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    return open_request(request)


def open_request(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {error.code}: {error_body}") from error
    return json.loads(response_body)


def main() -> None:
    args = parse_args()
    requests_jsonl = args.requests_jsonl.expanduser().resolve()
    if not requests_jsonl.is_file():
        raise FileNotFoundError(f"Requests JSONL not found: {requests_jsonl}")

    metadata = None
    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))

    api_key = require_env_var("OPENAI_API_KEY")
    upload = upload_file(api_key, requests_jsonl, purpose="batch")
    batch_payload: dict[str, Any] = {
        "input_file_id": upload["id"],
        "endpoint": args.endpoint,
        "completion_window": args.completion_window,
    }
    if metadata is not None:
        batch_payload["metadata"] = metadata
    batch = request_json(api_key, "/batches", batch_payload)
    state = {"input_file_id": upload["id"], "file": upload, "batch": batch}

    if args.state_json is not None:
        state_path = args.state_json.expanduser().resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
