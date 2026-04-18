from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten_message_content(content: Any) -> str:
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


def extract_text_from_response_record(record: dict[str, Any]) -> str:
    if "text" in record:
        return str(record["text"])
    if "response" in record:
        response = record.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return flatten_message_content(message.get("content"))
    if "body" in record:
        body = record.get("body") or {}
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return flatten_message_content(message.get("content"))
    raise ValueError("Could not locate assistant text in the provided record.")


def load_request_manifest_df(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in [
        "custom_id",
        "run_name",
        "simbench_split",
        "simbench_row_id",
        "dataset_name",
        "variant",
        "model",
        "twin_pid",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def load_parsed_outputs_df(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "custom_id" in frame.columns:
        frame["custom_id"] = frame["custom_id"].astype(str)
    return frame


def normalize_distribution(dist: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in dist.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        cleaned[str(key)] = numeric
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 0.0 for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def total_variation_distance(
    predicted: dict[str, float],
    gold: dict[str, float],
    option_labels: list[str],
) -> float:
    return 0.5 * sum(abs(float(predicted.get(label, 0.0)) - float(gold.get(label, 0.0))) for label in option_labels)


def shannon_entropy(dist: dict[str, float], option_labels: list[str]) -> float:
    entropy = 0.0
    for label in option_labels:
        probability = float(dist.get(label, 0.0))
        if probability > 0:
            entropy -= probability * math.log(probability)
    return float(entropy)


def jensen_shannon_divergence(
    predicted: dict[str, float],
    gold: dict[str, float],
    option_labels: list[str],
) -> float:
    p = np.array([float(predicted.get(label, 0.0)) for label in option_labels], dtype=float)
    q = np.array([float(gold.get(label, 0.0)) for label in option_labels], dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def modal_label(dist: dict[str, float], option_labels: list[str]) -> str | None:
    if not option_labels:
        return None
    best_label: str | None = None
    best_value = float("-inf")
    for label in option_labels:
        value = float(dist.get(label, 0.0))
        if value > best_value:
            best_value = value
            best_label = label
    return best_label


NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def _normalize_choice_token(value: str) -> str:
    return NON_ALNUM_RE.sub("", value.strip().upper())


def parse_choice_label(text: str, option_labels: list[str]) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if not text.strip():
        return None, ["Response text is empty."]

    normalized_map = {
        _normalize_choice_token(label): str(label)
        for label in option_labels
        if _normalize_choice_token(label)
    }
    normalized_candidates = list(normalized_map.keys())
    upper_text = text.strip().upper()
    stripped_variants = [
        upper_text,
        upper_text.strip("()[]{}"),
        upper_text.replace("OPTION ", "").strip(),
        upper_text.replace("ANSWER: ", "").strip(),
    ]
    for variant in stripped_variants:
        normalized = _normalize_choice_token(variant)
        if normalized in normalized_map:
            return normalized_map[normalized], []

    matches: list[str] = []
    for normalized_label, original_label in normalized_map.items():
        pattern = rf"(?<![A-Z0-9]){re.escape(normalized_label)}(?![A-Z0-9])"
        if re.search(pattern, _normalize_choice_token(" " + upper_text + " ")):
            matches.append(original_label)
            continue
        raw_pattern = rf"(?<![A-Z0-9]){re.escape(str(original_label).upper())}(?![A-Z0-9])"
        if re.search(raw_pattern, upper_text):
            matches.append(original_label)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0], []
    if len(unique_matches) > 1:
        errors.append(f"Multiple option labels detected in response: {unique_matches}")
    else:
        errors.append(f"Could not identify a valid option label from supported labels: {option_labels}")
    return None, errors
