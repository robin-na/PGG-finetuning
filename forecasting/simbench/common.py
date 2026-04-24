from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


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


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Response text is empty.")

    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    fenced = CODE_FENCE_RE.findall(stripped)
    for block in fenced:
        candidate = block.strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue

    start = stripped.find("{")
    if start == -1:
        raise ValueError("Could not find a JSON object in the response text.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    raise ValueError("Could not find a balanced JSON object in the response text.")


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


def uniform_distribution(option_labels: list[str]) -> dict[str, float]:
    if not option_labels:
        return {}
    probability = 1.0 / float(len(option_labels))
    return {str(label): probability for label in option_labels}


def uniform_baseline_tvd(
    gold: dict[str, float],
    option_labels: list[str],
) -> float:
    return total_variation_distance(uniform_distribution(option_labels), gold, option_labels)


def simbench_score(
    predicted: dict[str, float],
    gold: dict[str, float],
    option_labels: list[str],
    *,
    scale: float = 100.0,
) -> float:
    baseline_tvd = uniform_baseline_tvd(gold, option_labels)
    predicted_tvd = total_variation_distance(predicted, gold, option_labels)
    if baseline_tvd <= 0:
        if predicted_tvd <= 0:
            return float(scale)
        return float("nan")
    return float(scale * (1.0 - (predicted_tvd / baseline_tvd)))


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


def _coerce_nonnegative_float(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, "Value must be numeric, not boolean."
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    else:
        text = str(value).strip().replace("%", "")
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return None, "Value must be numeric."
    if not math.isfinite(numeric):
        return None, "Value must be finite."
    if numeric < 0:
        return None, "Value must be non-negative."
    return float(numeric), None


def _option_alias_maps(
    option_labels: list[str],
    option_text_map: dict[str, Any] | None = None,
) -> tuple[dict[str, str], set[str]]:
    alias_to_label: dict[str, str] = {}
    ambiguous_aliases: set[str] = set()

    def register(alias: str, label: str) -> None:
        normalized = _normalize_choice_token(alias)
        if not normalized:
            return
        existing = alias_to_label.get(normalized)
        if existing is None:
            alias_to_label[normalized] = label
            return
        if existing != label:
            ambiguous_aliases.add(normalized)

    option_text_map = option_text_map or {}
    for label in option_labels:
        register(str(label), str(label))
        option_text = option_text_map.get(label)
        if option_text is not None:
            register(str(option_text), str(label))

    for alias in ambiguous_aliases:
        alias_to_label.pop(alias, None)
    return alias_to_label, ambiguous_aliases


def normalize_option_distribution(
    raw: Any,
    option_labels: list[str],
    option_text_map: dict[str, Any] | None = None,
) -> tuple[dict[str, float] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["Distribution must be a JSON object."]

    alias_to_label, ambiguous_aliases = _option_alias_maps(option_labels, option_text_map)
    errors: list[str] = []
    canonical_values: dict[str, float] = {}

    for raw_key, raw_value in raw.items():
        canonical_key = _normalize_choice_token(str(raw_key))
        if canonical_key in ambiguous_aliases:
            errors.append(f"Option key {raw_key!r} is ambiguous.")
            continue
        label = alias_to_label.get(canonical_key)
        if label is None:
            errors.append(f"Unrecognized option key {raw_key!r}; expected one of {option_labels}.")
            continue
        numeric_value, value_error = _coerce_nonnegative_float(raw_value)
        if value_error:
            errors.append(f"Option {raw_key!r}: {value_error}")
            continue
        if label in canonical_values:
            errors.append(f"Duplicate values provided for option label {label!r}; summing them.")
            canonical_values[label] += float(numeric_value)
        else:
            canonical_values[label] = float(numeric_value)

    missing_labels = [label for label in option_labels if label not in canonical_values]
    if missing_labels:
        errors.append(f"Missing option labels: {missing_labels}")
        for label in missing_labels:
            canonical_values[label] = 0.0

    total = sum(float(canonical_values.get(label, 0.0)) for label in option_labels)
    if total <= 0:
        errors.append("Distribution has zero total mass after cleaning.")
        return None, errors

    if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"Distribution totals {total:.6f}, not 100; renormalizing.")

    normalized = {
        str(label): float(canonical_values.get(label, 0.0)) / total
        for label in option_labels
    }
    return normalized, errors


def validate_batched_prediction_payload(
    payload: Any,
    question_manifest: list[dict[str, Any]],
    response_schema: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, list[str]]]:
    if not isinstance(payload, dict):
        return None, ["Top-level JSON value must be an object."], {}

    errors: list[str] = []
    question_errors: dict[str, list[str]] = {}
    explanation = ""

    if response_schema == "batched_explanation_plus_answers":
        top_level_keys = set(payload.keys())
        required_keys = {"explanation", "answers"}
        missing_keys = sorted(required_keys - top_level_keys)
        extra_keys = sorted(top_level_keys - required_keys)
        if missing_keys:
            errors.append(f"Missing top-level keys: {missing_keys}")
        if extra_keys:
            errors.append(f"Unexpected top-level keys: {extra_keys}")

        explanation_value = payload.get("explanation", "")
        if isinstance(explanation_value, str):
            explanation = explanation_value.strip()
        elif explanation_value in (None, ""):
            explanation = ""
            errors.append("Top-level key 'explanation' is empty or missing.")
        else:
            explanation = str(explanation_value).strip()
            errors.append("Top-level key 'explanation' must be a string; coercing to string.")

        answers_payload = payload.get("answers")
        if not isinstance(answers_payload, dict):
            errors.append("Top-level key 'answers' must be an object.")
            answers_payload = {}
    elif response_schema == "batched_distribution_only":
        answers_payload = payload
    else:
        return None, [f"Unsupported response schema: {response_schema}"], {}

    expected_question_ids = {str(item["question_id"]) for item in question_manifest}
    provided_question_ids = {str(key) for key in answers_payload.keys()}
    extra_question_ids = sorted(provided_question_ids - expected_question_ids)
    if extra_question_ids:
        errors.append(f"Unexpected question ids in answers: {extra_question_ids}")

    normalized_answers: dict[str, dict[str, float]] = {}
    for item in question_manifest:
        question_id = str(item["question_id"])
        option_labels = [str(label) for label in item.get("option_labels", [])]
        option_text_map = item.get("option_text_map") or {}
        raw_answer = answers_payload.get(question_id)
        item_errors: list[str] = []

        if raw_answer is None:
            item_errors.append("Missing answer for question id.")
        else:
            normalized_answer, dist_errors = normalize_option_distribution(
                raw_answer,
                option_labels,
                option_text_map=option_text_map,
            )
            item_errors.extend(dist_errors)
            if normalized_answer is not None:
                normalized_answers[question_id] = normalized_answer

        if item_errors:
            question_errors[question_id] = item_errors

    normalized_payload = {
        "explanation": explanation,
        "answers": normalized_answers,
        "expected_question_count": int(len(question_manifest)),
        "valid_question_count": int(len(normalized_answers)),
    }
    return normalized_payload, errors, question_errors


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
