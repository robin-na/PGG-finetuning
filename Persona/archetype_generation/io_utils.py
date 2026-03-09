from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            if value != value:
                return ""
        except Exception:
            pass
    return str(value)


def coerce_bool(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return None


def token_estimate_from_chars(text: str) -> int:
    # Simple heuristic that is stable and cheap.
    return max(1, int(round(len(text) / 4.0)))


def strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def parse_json_from_text(text: str) -> Dict[str, Any]:
    s = strip_code_fences(text)
    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find largest JSON object span.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        sub = s[start : end + 1]
        obj = json.loads(sub)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse a JSON object from model output text.")


def extract_batch_content(batch_row: Dict[str, Any]) -> str:
    """Extract textual model output from OpenAI batch output row.

    Supports common chat-completions and responses API output layouts.
    """
    body = batch_row.get("response", {}).get("body", {})

    # chat.completions shape
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts)

    # responses API: output_text shortcut
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # responses API: output list
    output = body.get("output")
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt)
        if texts:
            return "\n".join(texts)

    raise ValueError("Could not extract model content from batch row.")


def header_signature(text: str) -> List[str]:
    tags = re.findall(r"^<([A-Z_]+)>\s*$", text or "", flags=re.M)
    seen = []
    seen_set = set()
    for tag in tags:
        if tag in seen_set:
            continue
        seen.append(tag)
        seen_set.add(tag)
    return seen


def unknown_ratio_by_section(text: str) -> float:
    tags = list(re.finditer(r"^<([A-Z_]+)>\s*$", text or "", flags=re.M))
    if not tags:
        low = (text or "").lower()
        return 1.0 if "unknown" in low else 0.0

    unknown_count = 0
    total = 0
    for i, m in enumerate(tags):
        start = m.end()
        end = tags[i + 1].start() if i + 1 < len(tags) else len(text)
        body = (text[start:end] or "").strip().lower()
        total += 1
        if (
            "unknown" in body
            or body.startswith("insufficient information")
            or body.startswith("not enough information")
        ):
            unknown_count += 1

    return float(unknown_count) / float(total or 1)
