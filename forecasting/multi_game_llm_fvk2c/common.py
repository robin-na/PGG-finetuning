from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DELEGATION_FIELDS = [
    "UGProposer_delegated",
    "UGResponder_delegated",
    "TGSender_delegated",
    "TGReceiver_delegated",
    "PD_delegated",
    "SH_delegated",
    "C_delegated",
]

SCENARIO_FIELDS = [
    "UGProposer_decision",
    "UGResponder_decision",
    "TGSender_decision",
    "TGReceiver_decision",
    "PD_decision",
    "SH_decision",
    "C_decision",
]

ROLE_TO_DELEGATION_FIELD = {
    "UGProposer_decision": "UGProposer_delegated",
    "UGResponder_decision": "UGResponder_delegated",
    "TGSender_decision": "TGSender_delegated",
    "TGReceiver_decision": "TGReceiver_delegated",
    "PD_decision": "PD_delegated",
    "SH_decision": "SH_delegated",
    "C_decision": "C_delegated",
}

NUMERIC_SCENARIO_FIELDS = {
    "UGProposer_decision": (0, 10),
    "UGResponder_decision": (0, 10),
    "TGReceiver_decision": (0, 6),
}

CHOICE_SUPPORT = {
    "TGSender_decision": ["YES", "NO"],
    "PD_decision": ["A", "B"],
    "SH_decision": ["X", "Y"],
    "C_decision": ["Mercury", "Venus", "Earth", "Mars", "Saturn"],
}

STATE_SUPPORT = {
    "UGProposer_decision": [str(value) for value in range(0, 11)] + ["NULL"],
    "UGResponder_decision": [str(value) for value in range(0, 11)] + ["NULL"],
    "TGSender_decision": ["YES", "NO", "NULL"],
    "TGReceiver_decision": [str(value) for value in range(0, 7)] + ["NULL"],
    "PD_decision": ["A", "B", "NULL"],
    "SH_decision": ["X", "Y", "NULL"],
    "C_decision": ["Mercury", "Venus", "Earth", "Mars", "Saturn", "NULL"],
}

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
    for column in ["custom_id", "record_id", "unit_id", "treatment_name", "TreatmentCode", "Treatment"]:
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


def _normalize_nullable_binary(value: Any, field_name: str) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return int(value), None
    text = str(value).strip().upper()
    if text in {"0", "1"}:
        return int(text), None
    if text in {"TRUE", "FALSE"}:
        return (1 if text == "TRUE" else 0), None
    return None, f"{field_name} must be 0, 1, or null."


def _normalize_nullable_int(
    value: Any,
    field_name: str,
    *,
    min_value: int,
    max_value: int,
) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{field_name} must be an integer between {min_value} and {max_value} or null."
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} must be an integer between {min_value} and {max_value} or null."
    if not numeric.is_integer():
        return None, f"{field_name} must be an integer between {min_value} and {max_value} or null."
    integer = int(numeric)
    if integer < min_value or integer > max_value:
        return None, f"{field_name} must be between {min_value} and {max_value}."
    return integer, None


def _normalize_nullable_choice(
    value: Any,
    field_name: str,
    *,
    choices: list[str],
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if field_name in {"TGSender_decision", "PD_decision", "SH_decision"}:
        text = text.upper()
    choice_map = {choice.upper(): choice for choice in choices}
    mapped = choice_map.get(text.upper())
    if mapped is None:
        return None, f"{field_name} must be one of {choices} or null."
    return mapped, None


def validate_prediction_payload(
    payload: Any,
    *,
    treatment: str,
    num_scenarios: int,
    scenario_manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Top-level JSON value must be an object."]

    errors: list[str] = []
    normalized: dict[str, Any] = {}
    expected_top_level_keys = set(DELEGATION_FIELDS + ["scenario_outputs"])
    extra_top_level = sorted(set(payload.keys()) - expected_top_level_keys)
    if extra_top_level:
        errors.append(f"Unexpected top-level keys present: {extra_top_level}")

    for field_name in DELEGATION_FIELDS:
        value, error = _normalize_nullable_binary(payload.get(field_name), field_name)
        if error:
            errors.append(error)
            continue
        if treatment == "TransparentRandom":
            if value is not None:
                errors.append(f"{field_name} must be null in TransparentRandom.")
            normalized[field_name] = None
        else:
            if value is None:
                errors.append(f"{field_name} must be 0 or 1 in voluntary-delegation treatments.")
            else:
                normalized[field_name] = value

    scenario_outputs = payload.get("scenario_outputs")
    if not isinstance(scenario_outputs, list):
        errors.append("scenario_outputs must be a list.")
        scenario_outputs = []
    if len(scenario_outputs) != num_scenarios:
        errors.append(f"scenario_outputs must contain exactly {num_scenarios} objects.")

    normalized_scenarios: list[dict[str, Any]] = []
    for index, scenario_payload in enumerate(scenario_outputs):
        if not isinstance(scenario_payload, dict):
            errors.append(f"scenario_outputs[{index}] must be an object.")
            continue

        manifest_item = scenario_manifest[index] if index < len(scenario_manifest) else {}
        expected_scenario = str(manifest_item.get("scenario", ""))
        expected_case = str(manifest_item.get("case", ""))
        scenario_obj: dict[str, Any] = {}
        allowed_keys = {"scenario", "case", *SCENARIO_FIELDS}
        extra_keys = sorted(set(scenario_payload.keys()) - allowed_keys)
        if extra_keys:
            errors.append(f"scenario_outputs[{index}] has unexpected keys: {extra_keys}")

        scenario_value = str(scenario_payload.get("scenario"))
        case_value = str(scenario_payload.get("case"))
        if scenario_value != expected_scenario:
            errors.append(
                f"scenario_outputs[{index}].scenario must be {expected_scenario!r}, got {scenario_value!r}."
            )
        if case_value != expected_case:
            errors.append(
                f"scenario_outputs[{index}].case must be {expected_case!r}, got {case_value!r}."
            )
        scenario_obj["scenario"] = expected_scenario
        scenario_obj["case"] = expected_case

        for field_name, bounds in NUMERIC_SCENARIO_FIELDS.items():
            value, error = _normalize_nullable_int(
                scenario_payload.get(field_name),
                field_name,
                min_value=bounds[0],
                max_value=bounds[1],
            )
            if error:
                errors.append(f"scenario_outputs[{index}]: {error}")
            else:
                scenario_obj[field_name] = value

        for field_name, choices in CHOICE_SUPPORT.items():
            value, error = _normalize_nullable_choice(
                scenario_payload.get(field_name),
                field_name,
                choices=choices,
            )
            if error:
                errors.append(f"scenario_outputs[{index}]: {error}")
            else:
                scenario_obj[field_name] = value

        if expected_scenario == "NoAISupport":
            for field_name in SCENARIO_FIELDS:
                if scenario_obj.get(field_name) is None:
                    errors.append(
                        f"scenario_outputs[{index}].{field_name} must be non-null in NoAISupport scenarios."
                    )
        elif treatment == "TransparentRandom":
            for field_name in SCENARIO_FIELDS:
                if scenario_obj.get(field_name) is None:
                    errors.append(
                        f"scenario_outputs[{index}].{field_name} must be non-null in TransparentRandom AISupport scenarios."
                    )
        else:
            for field_name in SCENARIO_FIELDS:
                delegation_field = ROLE_TO_DELEGATION_FIELD[field_name]
                delegated = normalized.get(delegation_field)
                value = scenario_obj.get(field_name)
                if delegated == 1 and value is not None:
                    errors.append(
                        f"scenario_outputs[{index}].{field_name} must be null when {delegation_field}=1 in AISupport."
                    )
                if delegated == 0 and value is None:
                    errors.append(
                        f"scenario_outputs[{index}].{field_name} must be non-null when {delegation_field}=0 in AISupport."
                    )

        normalized_scenarios.append(scenario_obj)

    if errors:
        return None, errors
    normalized["scenario_outputs"] = normalized_scenarios
    return normalized, []


def _state_label(value: Any) -> str:
    return "NULL" if value is None else str(value)


def _explode_session_rows(
    merged_rows: pd.DataFrame,
    *,
    target_column: str,
    data_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in merged_rows.to_dict(orient="records"):
        target = row.get(target_column) or {}
        scenario_manifest = json.loads(str(row["scenario_manifest_json"]))
        scenario_outputs = target.get("scenario_outputs") or []
        for order_index, scenario_obj in enumerate(scenario_outputs):
            manifest_item = scenario_manifest[order_index] if order_index < len(scenario_manifest) else {}
            scenario = str(scenario_obj.get("scenario", manifest_item.get("scenario", "")))
            case = str(scenario_obj.get("case", manifest_item.get("case", "")))
            cell_name = f"{row['treatment_name']}__{scenario}__{case}"
            flat_row = {
                "custom_id": row["custom_id"],
                "record_id": row["record_id"],
                "unit_id": row["unit_id"],
                "treatment_name": row["treatment_name"],
                "TreatmentCode": row["TreatmentCode"],
                "Treatment": row["Treatment"],
                "PersonalizedTreatment": row["PersonalizedTreatment"],
                "scenario": scenario,
                "case": case,
                "scenario_order": int(manifest_item.get("order", order_index + 1)),
                "cell_name": cell_name,
                "data_source": data_source,
            }
            for delegation_field in DELEGATION_FIELDS:
                flat_row[delegation_field] = target.get(delegation_field)
            for field_name in SCENARIO_FIELDS:
                value = scenario_obj.get(field_name)
                flat_row[field_name] = value
                flat_row[f"{field_name}_nonnull"] = 0 if value is None else 1
                flat_row[f"{field_name}_state"] = _state_label(value)
            rows.append(flat_row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def explode_session_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    data_source: str,
) -> pd.DataFrame:
    return _explode_session_rows(frame, target_column=target_column, data_source=data_source)


def build_human_sessions_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest = load_request_manifest_df(request_manifest_csv)
    gold_rows = pd.DataFrame(read_jsonl(gold_targets_jsonl))
    if gold_rows.empty:
        return pd.DataFrame()
    for column in ["custom_id", "record_id", "unit_id", "treatment_name"]:
        if column in gold_rows.columns:
            gold_rows[column] = gold_rows[column].astype(str)
    merged = gold_rows.merge(
        manifest,
        on=["custom_id", "record_id", "unit_id", "treatment_name"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        target = row.get("gold_target") or {}
        rows.append(
            {
                **row,
                "data_source": "human",
                **{field_name: target.get(field_name) for field_name in DELEGATION_FIELDS},
            }
        )
    return pd.DataFrame(rows)


def build_generated_sessions_df(
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
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        target = row.get("parsed_target") or {}
        rows.append(
            {
                **row,
                "data_source": "generated",
                **{field_name: target.get(field_name) for field_name in DELEGATION_FIELDS},
            }
        )
    return pd.DataFrame(rows)


def build_human_scenarios_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest = load_request_manifest_df(request_manifest_csv)
    gold_rows = pd.DataFrame(read_jsonl(gold_targets_jsonl))
    if gold_rows.empty:
        return pd.DataFrame()
    for column in ["custom_id", "record_id", "unit_id", "treatment_name"]:
        if column in gold_rows.columns:
            gold_rows[column] = gold_rows[column].astype(str)
    merged = gold_rows.merge(
        manifest,
        on=["custom_id", "record_id", "unit_id", "treatment_name"],
        how="left",
        validate="one_to_one",
    )
    return _explode_session_rows(merged, target_column="gold_target", data_source="human")


def build_generated_scenarios_df(
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
    return _explode_session_rows(merged, target_column="parsed_target", data_source="generated")


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
