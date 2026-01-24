from __future__ import annotations

import ast
import json
import re
from typing import List, Optional, Tuple

from utils import log


def extract_answer_tagged(s: str, tag: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    pattern = rf"Answer:\s*<\s*{re.escape(tag)}\s*>(.*?)</\s*{re.escape(tag)}\s*>"
    match = re.search(pattern, s, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1)


def first_int(s: str, tag: Optional[str] = None) -> Tuple[int, bool]:
    if not isinstance(s, str):
        log("[parse] expected string for integer output; defaulting to 0")
        return 0, False
    target = s
    if tag:
        tagged = extract_answer_tagged(s, tag)
        if tagged is not None:
            target = tagged
    if "Answer:" in target:
        target = target.split("Answer:", 1)[1]
    m = re.search(r"-?\d+", target)
    if not m:
        log(f"[parse] no integer found for tag={tag or 'raw'}; defaulting to 0")
        return 0, False
    return int(m.group(0)), True


def parse_first_int_array(s: str, tag: Optional[str] = None) -> Tuple[Optional[List[int]], bool]:
    if not isinstance(s, str):
        log("[parse] expected string for array output; defaulting to []")
        return None, False

    target = s
    if tag:
        tagged = extract_answer_tagged(s, tag)
        if tagged is not None:
            target = tagged
    if "Answer:" in target:
        target = target.split("Answer:", 1)[1]

    cleaned = target.strip()
    if cleaned.startswith("<<") and ">>" in cleaned:
        cleaned = cleaned[2:cleaned.find(">>")].strip()

    lb = cleaned.find("[")
    rb = cleaned.find("]", lb + 1) if lb != -1 else -1
    if lb == -1 or rb == -1 or rb <= lb:
        if lb != -1:
            tail = cleaned[lb:] + "]"
            try:
                arr = json.loads(tail)
                if isinstance(arr, list):
                    return [int(x) for x in arr], True
            except Exception:
                try:
                    arr = ast.literal_eval(tail)
                    if isinstance(arr, list):
                        return [int(x) for x in arr], True
                except Exception:
                    log(f"[parse] malformed array for tag={tag or 'raw'}; defaulting to []")
                    return None, False
        log(f"[parse] no array brackets found for tag={tag or 'raw'}; defaulting to []")
        return None, False

    chunk = cleaned[lb:rb + 1]

    for loader in (json.loads, ast.literal_eval):
        try:
            arr = loader(chunk)
            if isinstance(arr, list):
                return [int(x) for x in arr], True
        except Exception:
            pass
    log(f"[parse] failed to parse array for tag={tag or 'raw'}; defaulting to []")
    return None, False


def extract_tag_content(s: str, tag: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    tagged = extract_answer_tagged(s, tag)
    if tagged is not None:
        return tagged
    pattern = rf"<\s*{re.escape(tag)}\s*>(.*?)</\s*{re.escape(tag)}\s*>"
    match = re.search(pattern, s, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1)


def parse_chat_message(s: str) -> Tuple[str, bool]:
    if not isinstance(s, str):
        log("[parse] expected string for chat output; defaulting to silence")
        return "", False
    content = extract_tag_content(s, "CHAT")
    if content is None:
        content = s
    if "Answer:" in content:
        content = content.split("Answer:", 1)[1]
    msg = content.strip()
    if msg in {"", "...", "SILENT", "silence", "NONE", "none"}:
        return "", True
    return msg, True
