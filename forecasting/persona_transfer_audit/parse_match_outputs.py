from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["custom_id"]: row for row in csv.DictReader(handle)}


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("output_text"), str):
                    parts.append(item["output_text"])
        return "\n".join(parts)
    return str(content)


def _extract_content(record: dict[str, Any]) -> str:
    response = record.get("response") or {}
    body = response.get("body") or record.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("No choices found in response record.")
    message = choices[0].get("message") or {}
    return _flatten_message_content(message.get("content"))


def parse_outputs(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    manifest = _load_manifest(metadata_dir / "request_manifest.csv")
    parsed_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in _read_jsonl(output_jsonl):
        custom_id = str(record.get("custom_id", ""))
        base = dict(manifest.get(custom_id, {}))
        base["custom_id"] = custom_id
        try:
            content = _extract_content(record)
            parsed = json.loads(content)
            parsed_rows.append({**base, **parsed, "raw_content": content})
        except Exception as exc:
            errors.append({**base, "error": str(exc), "record": record})

    _write_jsonl(metadata_dir / "parsed_matches.jsonl", parsed_rows)
    _write_jsonl(metadata_dir / "parse_errors.jsonl", errors)
    summary = {
        "output_jsonl": str(output_jsonl),
        "parsed": len(parsed_rows),
        "errors": len(errors),
    }
    (metadata_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse persona transfer match batch outputs.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    parse_outputs(parse_args())


if __name__ == "__main__":
    main()

