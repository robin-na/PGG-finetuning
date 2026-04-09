from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_A_CHECK = "role_a_check"
ROLE_A_TIME = "role_a_time"
ROLE_B_OBSERVABLE_CHECK = "role_b_observable_check"
ROLE_B_HIDDEN_CHECK = "role_b_hidden_check"
ROLE_B_OBSERVABLE_TIME = "role_b_observable_time"
ROLE_B_HIDDEN_TIME = "role_b_hidden_time"

ALL_TARGET_FIELDS = [
    "check",
    "act",
    "decision_time_bucket",
    "return_pct",
    "send_if_act_without_check",
    "send_if_act_after_check",
    "send_if_no_act_without_check",
    "send_if_no_act_after_check",
    "send_if_act_fast",
    "send_if_no_act_fast",
    "send_if_act_slow",
    "send_if_no_act_slow",
    "send_if_act",
    "send_if_no_act",
]

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


def _normalize_yes_no(value: Any, field_name: str) -> tuple[str | None, str | None]:
    if isinstance(value, bool):
        return ("YES" if value else "NO"), None
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return ("YES" if int(value) == 1 else "NO"), None
    text = str(value).strip().upper()
    if text in {"YES", "NO"}:
        return text, None
    if text in {"1", "TRUE"}:
        return "YES", None
    if text in {"0", "FALSE"}:
        return "NO", None
    return None, f"{field_name} must be YES or NO."


def _normalize_fast_slow(value: Any, field_name: str) -> tuple[str | None, str | None]:
    text = str(value).strip().upper()
    if text in {"FAST", "SLOW"}:
        return text, None
    return None, f"{field_name} must be FAST or SLOW."


def _normalize_bounded_int(
    value: Any,
    field_name: str,
    *,
    min_value: int,
    max_value: int,
) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{field_name} must be an integer between {min_value} and {max_value}."
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} must be an integer between {min_value} and {max_value}."
    if not numeric.is_integer():
        return None, f"{field_name} must be an integer between {min_value} and {max_value}."
    integer = int(numeric)
    if integer < min_value or integer > max_value:
        return None, f"{field_name} must be between {min_value} and {max_value}."
    return integer, None


def validate_prediction_payload(
    payload: Any,
    schema_type: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Top-level JSON value must be an object."]

    errors: list[str] = []
    normalized: dict[str, Any] = {}

    def expect_yes_no(field_name: str) -> None:
        value, error = _normalize_yes_no(payload.get(field_name), field_name)
        if error:
            errors.append(error)
        else:
            normalized[field_name] = value

    def expect_fast_slow(field_name: str) -> None:
        value, error = _normalize_fast_slow(payload.get(field_name), field_name)
        if error:
            errors.append(error)
        else:
            normalized[field_name] = value

    def expect_int(field_name: str, min_value: int, max_value: int) -> None:
        value, error = _normalize_bounded_int(
            payload.get(field_name),
            field_name,
            min_value=min_value,
            max_value=max_value,
        )
        if error:
            errors.append(error)
        else:
            normalized[field_name] = value

    if schema_type == ROLE_A_CHECK:
        expect_yes_no("check")
        expect_yes_no("act")
        expect_int("return_pct", 0, 100)
        expected_keys = {"check", "act", "return_pct"}
    elif schema_type == ROLE_A_TIME:
        expect_fast_slow("decision_time_bucket")
        expect_yes_no("act")
        expect_int("return_pct", 0, 100)
        expected_keys = {"decision_time_bucket", "act", "return_pct"}
    elif schema_type == ROLE_B_OBSERVABLE_CHECK:
        for field_name in [
            "send_if_act_without_check",
            "send_if_act_after_check",
            "send_if_no_act_without_check",
            "send_if_no_act_after_check",
        ]:
            expect_int(field_name, 0, 10)
        expected_keys = {
            "send_if_act_without_check",
            "send_if_act_after_check",
            "send_if_no_act_without_check",
            "send_if_no_act_after_check",
        }
    elif schema_type == ROLE_B_HIDDEN_CHECK:
        for field_name in ["send_if_act", "send_if_no_act"]:
            expect_int(field_name, 0, 10)
        expected_keys = {"send_if_act", "send_if_no_act"}
    elif schema_type == ROLE_B_OBSERVABLE_TIME:
        for field_name in [
            "send_if_act_fast",
            "send_if_no_act_fast",
            "send_if_act_slow",
            "send_if_no_act_slow",
        ]:
            expect_int(field_name, 0, 10)
        expected_keys = {
            "send_if_act_fast",
            "send_if_no_act_fast",
            "send_if_act_slow",
            "send_if_no_act_slow",
        }
    elif schema_type == ROLE_B_HIDDEN_TIME:
        for field_name in ["send_if_act", "send_if_no_act"]:
            expect_int(field_name, 0, 10)
        expected_keys = {"send_if_act", "send_if_no_act"}
    else:
        return None, [f"Unsupported schema_type: {schema_type}"]

    extra_keys = sorted(set(payload.keys()) - expected_keys)
    if extra_keys:
        errors.append(f"Unexpected keys present: {extra_keys}")

    if errors:
        return None, errors
    return normalized, []


def load_request_manifest_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_parsed_outputs_df(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _flatten_target_dict(schema_type: str, target: dict[str, Any]) -> dict[str, Any]:
    row = {field_name: None for field_name in ALL_TARGET_FIELDS}
    row.update(target)
    row["process_label"] = None
    row["process_binary"] = np.nan
    row["act_binary"] = np.nan
    row["pattern_label"] = None
    row["action_premium"] = np.nan
    row["action_premium_without_check"] = np.nan
    row["action_premium_after_check"] = np.nan
    row["action_premium_fast"] = np.nan
    row["action_premium_slow"] = np.nan
    row["process_premium_if_act"] = np.nan
    row["process_premium_if_no_act"] = np.nan

    if schema_type == ROLE_A_CHECK:
        row["process_label"] = row["check"]
        row["process_binary"] = 1.0 if row["check"] == "YES" else 0.0
        row["act_binary"] = 1.0 if row["act"] == "YES" else 0.0
        row["pattern_label"] = (
            ("checked" if row["check"] == "YES" else "unchecked")
            + "_"
            + ("act" if row["act"] == "YES" else "no_act")
        )
    elif schema_type == ROLE_A_TIME:
        row["process_label"] = row["decision_time_bucket"]
        row["process_binary"] = 1.0 if row["decision_time_bucket"] == "FAST" else 0.0
        row["act_binary"] = 1.0 if row["act"] == "YES" else 0.0
        row["pattern_label"] = (
            ("fast" if row["decision_time_bucket"] == "FAST" else "slow")
            + "_"
            + ("act" if row["act"] == "YES" else "no_act")
        )
    elif schema_type == ROLE_B_OBSERVABLE_CHECK:
        row["action_premium_without_check"] = (
            row["send_if_act_without_check"] - row["send_if_no_act_without_check"]
        )
        row["action_premium_after_check"] = (
            row["send_if_act_after_check"] - row["send_if_no_act_after_check"]
        )
        row["process_premium_if_act"] = (
            row["send_if_act_without_check"] - row["send_if_act_after_check"]
        )
        row["process_premium_if_no_act"] = (
            row["send_if_no_act_without_check"] - row["send_if_no_act_after_check"]
        )
    elif schema_type == ROLE_B_OBSERVABLE_TIME:
        row["action_premium_fast"] = row["send_if_act_fast"] - row["send_if_no_act_fast"]
        row["action_premium_slow"] = row["send_if_act_slow"] - row["send_if_no_act_slow"]
        row["process_premium_if_act"] = row["send_if_act_fast"] - row["send_if_act_slow"]
        row["process_premium_if_no_act"] = (
            row["send_if_no_act_fast"] - row["send_if_no_act_slow"]
        )
    elif schema_type in {ROLE_B_HIDDEN_CHECK, ROLE_B_HIDDEN_TIME}:
        row["action_premium"] = row["send_if_act"] - row["send_if_no_act"]
    return row


def build_human_records_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest = load_request_manifest_df(request_manifest_csv)
    gold_rows = pd.DataFrame(read_jsonl(gold_targets_jsonl))
    if gold_rows.empty:
        return pd.DataFrame()
    merged = gold_rows.merge(
        manifest,
        on=["custom_id", "record_id", "unit_id", "treatment_name"],
        how="left",
        validate="one_to_one",
    )
    flat_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        target = row.get("gold_target") or {}
        flat_rows.append(
            {
                **row,
                "data_source": "human",
                **_flatten_target_dict(str(row["schema_type"]), dict(target)),
            }
        )
    return pd.DataFrame(flat_rows)


def build_generated_records_df(
    *,
    parsed_output_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest = load_request_manifest_df(request_manifest_csv)
    parsed_df = load_parsed_outputs_df(parsed_output_jsonl)
    if parsed_df.empty:
        return pd.DataFrame()
    merged = parsed_df.merge(manifest, on="custom_id", how="left", validate="many_to_one")
    merged = merged[merged["parse_success"].astype(bool)].copy()
    flat_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        target = row.get("parsed_target") or {}
        flat_rows.append(
            {
                **row,
                "data_source": "generated",
                **_flatten_target_dict(str(row["schema_type"]), dict(target)),
            }
        )
    return pd.DataFrame(flat_rows)


def wasserstein_distance_1d(x_values: pd.Series, y_values: pd.Series) -> float:
    x = np.sort(pd.Series(x_values).dropna().astype(float).to_numpy())
    y = np.sort(pd.Series(y_values).dropna().astype(float).to_numpy())
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size == 1 and y.size == 1:
        return float(abs(x[0] - y[0]))
    support = np.sort(np.concatenate([x, y]))
    deltas = np.diff(support)
    if deltas.size == 0:
        return 0.0
    x_cdf = np.searchsorted(x, support[:-1], side="right") / float(x.size)
    y_cdf = np.searchsorted(y, support[:-1], side="right") / float(y.size)
    return float(np.sum(np.abs(x_cdf - y_cdf) * deltas))


def total_variation_distance(
    x_labels: pd.Series,
    y_labels: pd.Series,
    support: list[str],
) -> float:
    x_probs = x_labels.value_counts(normalize=True)
    y_probs = y_labels.value_counts(normalize=True)
    return float(
        0.5
        * sum(abs(float(x_probs.get(label, 0.0)) - float(y_probs.get(label, 0.0))) for label in support)
    )


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def exact_match_rate(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def mean_value(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
