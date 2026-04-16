from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from repo_env import require_env_var


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
BATCH_STATE_FILENAME = "openai_batch_state.json"
ERROR_OUTPUT_FILENAME = "openai_batch_error.jsonl"


@dataclass(frozen=True)
class BatchPaths:
    manifest_json: Path
    metadata_dir: Path
    requests_jsonl: Path
    output_jsonl: Path
    state_json: Path
    run_name: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_json is not None:
        return args.manifest_json.expanduser().resolve()
    if not args.run_name:
        raise ValueError("Provide either --manifest-json or --run-name.")
    forecasting_root = args.forecasting_root.expanduser().resolve()
    return forecasting_root / "metadata" / args.run_name / "manifest.json"


def _resolve_paths(args: argparse.Namespace) -> tuple[BatchPaths, dict[str, Any]]:
    manifest_json = _resolve_manifest_path(args)
    if not manifest_json.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_json}")

    manifest = _load_json(manifest_json)
    metadata_dir = Path(manifest.get("metadata_dir") or manifest_json.parent).expanduser().resolve()
    requests_jsonl = Path(
        getattr(args, "requests_jsonl", None) or manifest["batch_input_file"]
    ).expanduser().resolve()
    output_jsonl = Path(
        getattr(args, "output_jsonl", None) or manifest["expected_batch_output_file"]
    ).expanduser().resolve()
    state_json = Path(
        getattr(args, "state_json", None) or (metadata_dir / BATCH_STATE_FILENAME)
    ).expanduser().resolve()
    run_name = str(manifest.get("run_name") or requests_jsonl.stem)

    return (
        BatchPaths(
            manifest_json=manifest_json,
            metadata_dir=metadata_dir,
            requests_jsonl=requests_jsonl,
            output_jsonl=output_jsonl,
            state_json=state_json,
            run_name=run_name,
        ),
        manifest,
    )


def _detect_endpoint(requests_jsonl: Path) -> str:
    with requests_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            endpoint = str(record.get("url") or "").strip()
            if not endpoint:
                raise ValueError(f"Could not detect request URL from {requests_jsonl}")
            return endpoint
    raise ValueError(f"No requests found in {requests_jsonl}")


def _make_client() -> OpenAI:
    api_key = require_env_var("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _batch_to_dict(batch: Any) -> dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    if hasattr(batch, "model_dump"):
        return dict(batch.model_dump())
    if hasattr(batch, "model_dump_json"):
        return json.loads(batch.model_dump_json())
    raise TypeError(f"Unsupported batch payload type: {type(batch)!r}")


def _state_batch_snapshot(batch_payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "status",
        "endpoint",
        "completion_window",
        "input_file_id",
        "output_file_id",
        "error_file_id",
        "request_counts",
        "created_at",
        "in_progress_at",
        "finalizing_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "cancelled_at",
        "metadata",
    ]
    return {key: batch_payload.get(key) for key in keys if key in batch_payload}


def _save_state(
    paths: BatchPaths,
    state: dict[str, Any],
    *,
    batch_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(state)
    state["manifest_json"] = str(paths.manifest_json)
    state["requests_jsonl"] = str(paths.requests_jsonl)
    state["output_jsonl"] = str(paths.output_jsonl)
    state["run_name"] = paths.run_name
    state["last_updated_at"] = _utc_now_iso()
    if batch_payload is not None:
        snapshot = _state_batch_snapshot(batch_payload)
        state["batch"] = snapshot
        if snapshot.get("id"):
            state["batch_id"] = snapshot["id"]
        if snapshot.get("input_file_id"):
            state["input_file_id"] = snapshot["input_file_id"]
        if snapshot.get("output_file_id") is not None:
            state["output_file_id"] = snapshot["output_file_id"]
        if snapshot.get("error_file_id") is not None:
            state["error_file_id"] = snapshot["error_file_id"]
        if snapshot.get("status"):
            state["status"] = snapshot["status"]
    _write_json(paths.state_json, state)
    return state


def _resolve_batch_id(args: argparse.Namespace, state: dict[str, Any]) -> str:
    batch_id = getattr(args, "batch_id", None) or state.get("batch_id")
    if not batch_id:
        raise ValueError("No batch id found. Submit first or pass --batch-id.")
    return str(batch_id)


def _default_error_output_path(paths: BatchPaths) -> Path:
    return paths.metadata_dir / ERROR_OUTPUT_FILENAME


def _download_file(client: OpenAI, file_id: str, output_path: Path) -> Path:
    content = client.files.content(file_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content.read())
    return output_path


def _submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    paths, manifest = _resolve_paths(args)
    state = _load_state(paths.state_json)
    if state.get("batch_id") and not args.force_new:
        raise ValueError(
            f"State file already has batch_id={state['batch_id']}. Pass --force-new to submit another batch."
        )

    endpoint = args.endpoint or _detect_endpoint(paths.requests_jsonl)
    client = _make_client()
    with paths.requests_jsonl.open("rb") as handle:
        upload = client.files.create(file=handle, purpose="batch")

    metadata = {
        "run_name": paths.run_name,
        "source": "forecasting.pgg.manage_openai_batch",
        "manifest_json": str(paths.manifest_json),
        "model": str(manifest.get("model") or ""),
    }
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=endpoint,
        completion_window=args.completion_window,
        metadata=metadata,
    )
    batch_payload = _batch_to_dict(batch)
    state = _save_state(
        paths,
        {
            "submitted_at": _utc_now_iso(),
            "completion_window": args.completion_window,
            "endpoint": endpoint,
            "input_file_id": upload.id,
        },
        batch_payload=batch_payload,
    )
    return {
        "action": "submit",
        "run_name": paths.run_name,
        "state_json": str(paths.state_json),
        "requests_jsonl": str(paths.requests_jsonl),
        "output_jsonl": str(paths.output_jsonl),
        "batch_id": state.get("batch_id"),
        "input_file_id": state.get("input_file_id"),
        "status": state.get("status"),
    }


def _refresh_status(args: argparse.Namespace) -> dict[str, Any]:
    paths, _ = _resolve_paths(args)
    state = _load_state(paths.state_json)
    batch_id = _resolve_batch_id(args, state)
    client = _make_client()
    batch_payload = _batch_to_dict(client.batches.retrieve(batch_id))
    state = _save_state(paths, state, batch_payload=batch_payload)
    return {
        "action": "status",
        "run_name": paths.run_name,
        "state_json": str(paths.state_json),
        "batch_id": batch_id,
        "status": state.get("status"),
        "output_file_id": state.get("output_file_id"),
        "error_file_id": state.get("error_file_id"),
        "request_counts": state.get("batch", {}).get("request_counts"),
    }


def _download_batch_file(args: argparse.Namespace) -> dict[str, Any]:
    paths, _ = _resolve_paths(args)
    state = _load_state(paths.state_json)
    batch_id = _resolve_batch_id(args, state)
    client = _make_client()
    batch_payload = _batch_to_dict(client.batches.retrieve(batch_id))
    state = _save_state(paths, state, batch_payload=batch_payload)

    which = args.which
    file_id = state.get("output_file_id") if which == "output" else state.get("error_file_id")
    if not file_id:
        raise ValueError(f"Batch {batch_id} does not have a ready {which}_file_id yet.")

    if args.output_jsonl is not None:
        output_path = args.output_jsonl.expanduser().resolve()
    elif which == "output":
        output_path = paths.output_jsonl
    else:
        output_path = _default_error_output_path(paths)

    written_path = _download_file(client, str(file_id), output_path)
    state_key = "downloaded_output_jsonl" if which == "output" else "downloaded_error_jsonl"
    state[state_key] = str(written_path)
    state[f"{which}_downloaded_at"] = _utc_now_iso()
    _save_state(paths, state)
    return {
        "action": "download",
        "run_name": paths.run_name,
        "batch_id": batch_id,
        "which": which,
        "file_id": file_id,
        "output_path": str(written_path),
        "state_json": str(paths.state_json),
    }


def _sync_batch(args: argparse.Namespace) -> dict[str, Any]:
    paths, _ = _resolve_paths(args)
    state = _load_state(paths.state_json)
    client = _make_client()

    if args.force_new or not state.get("batch_id"):
        submit_result = _submit_batch(args)
        state = _load_state(paths.state_json)
    else:
        submit_result = None

    batch_id = _resolve_batch_id(args, state)
    deadline = time.time() + args.timeout_sec if args.wait and args.timeout_sec > 0 else None

    while True:
        batch_payload = _batch_to_dict(client.batches.retrieve(batch_id))
        state = _save_state(paths, state, batch_payload=batch_payload)
        status = str(state.get("status") or "")

        if status in TERMINAL_BATCH_STATUSES or not args.wait:
            break
        if deadline is not None and time.time() >= deadline:
            break
        time.sleep(args.poll_interval_sec)

    downloads: list[dict[str, Any]] = []
    status = str(state.get("status") or "")
    if status == "completed" and state.get("output_file_id"):
        output_path = _download_file(client, str(state["output_file_id"]), paths.output_jsonl)
        state["downloaded_output_jsonl"] = str(output_path)
        state["output_downloaded_at"] = _utc_now_iso()
        state = _save_state(paths, state)
        downloads.append({"which": "output", "output_path": str(output_path)})
    elif status in TERMINAL_BATCH_STATUSES and args.download_error_if_ready and state.get("error_file_id"):
        error_path = _download_file(
            client,
            str(state["error_file_id"]),
            _default_error_output_path(paths),
        )
        state["downloaded_error_jsonl"] = str(error_path)
        state["error_downloaded_at"] = _utc_now_iso()
        state = _save_state(paths, state)
        downloads.append({"which": "error", "output_path": str(error_path)})

    return {
        "action": "sync",
        "run_name": paths.run_name,
        "state_json": str(paths.state_json),
        "batch_id": batch_id,
        "status": state.get("status"),
        "submitted": submit_result is not None,
        "downloads": downloads,
        "request_counts": state.get("batch", {}).get("request_counts"),
    }


def _add_common_resolution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--forecasting-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder that contains metadata/, batch_input/, and batch_output/ for the run.",
    )
    parser.add_argument("--requests-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--batch-id", type=str, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit, poll, and download OpenAI Batch jobs for forecasting manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Upload the request JSONL and create a batch job.")
    _add_common_resolution_args(submit_parser)
    submit_parser.add_argument("--endpoint", type=str, default=None)
    submit_parser.add_argument("--completion-window", type=str, default="24h")
    submit_parser.add_argument("--force-new", action="store_true")

    status_parser = subparsers.add_parser("status", help="Refresh and print the current batch status.")
    _add_common_resolution_args(status_parser)

    download_parser = subparsers.add_parser("download", help="Download the output or error JSONL for a batch.")
    _add_common_resolution_args(download_parser)
    download_parser.add_argument("--which", choices=("output", "error"), default="output")

    sync_parser = subparsers.add_parser(
        "sync",
        help="Submit if needed, refresh status, and download outputs when the batch is ready.",
    )
    _add_common_resolution_args(sync_parser)
    sync_parser.add_argument("--endpoint", type=str, default=None)
    sync_parser.add_argument("--completion-window", type=str, default="24h")
    sync_parser.add_argument("--force-new", action="store_true")
    sync_parser.add_argument("--wait", action="store_true")
    sync_parser.add_argument("--poll-interval-sec", type=int, default=30)
    sync_parser.add_argument("--timeout-sec", type=int, default=0)
    sync_parser.add_argument("--download-error-if-ready", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "submit":
        payload = _submit_batch(args)
    elif args.command == "status":
        payload = _refresh_status(args)
    elif args.command == "download":
        payload = _download_batch_file(args)
    elif args.command == "sync":
        payload = _sync_batch(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
