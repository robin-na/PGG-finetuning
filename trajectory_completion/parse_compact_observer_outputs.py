from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


ROUND_BEGIN_RE = re.compile(r"^#{2,3}\s*ROUND (\d+) BEGINS\s*$")
ROUND_EXPLANATION_RE = re.compile(r"^### ROUND (\d+) EXPLANATION\s*$")
ROUND_SUMMARY_RE = re.compile(r"^#{2,3}\s*ROUND (\d+) SUMMARY SHOWN TO PLAYERS\s*$")
CONTRIBUTIONS_RE = re.compile(r"^### CONTRIBUTION(?:S)?:\s*<<(.*)>>\s*$")
INTERACTIONS_RE = re.compile(r"^### (PUNISHMENT/REWARD|PUNISHMENT|REWARD):\s*<<(.*)>>\s*$")
CHAT_RE = re.compile(r"^CHAT(?: from |/)([^:]+):\s*(.*)$")
OVERALL_REFLECTION_RE = re.compile(r"^### OVERALL REFLECTION\s*$")
INTERACTION_TUPLE_RE = re.compile(
    r"\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,\s*([A-Za-z][A-Za-z0-9_]*)\s*,\s*([+-]?\d+)\s*\)"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            avatars = ast.literal_eval(row["avatars"]) if row.get("avatars") else []
            manifest[str(row["custom_id"])] = {
                "k": int(row["k"]) if row.get("k") else None,
                "num_rounds": int(row["num_rounds"]) if row.get("num_rounds") else None,
                "num_players": int(row["num_players"]) if row.get("num_players") else None,
                "avatars": avatars,
                "chat_enabled": _as_bool(row.get("chat_enabled")),
                "punishment_enabled": _as_bool(row.get("punishment_enabled")),
                "reward_enabled": _as_bool(row.get("reward_enabled")),
            }
    return manifest


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _extract_text_from_response_record(record: dict[str, Any]) -> str:
    if "gold_continuation_text" in record:
        return str(record["gold_continuation_text"])
    if "text" in record:
        return str(record["text"])
    if "response" in record:
        response = record.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            return _flatten_message_content(content)
    if "body" in record:
        body = record.get("body") or {}
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            return _flatten_message_content(content)
    raise ValueError("Could not locate assistant text in the provided record.")


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


def _parse_contributions(inner: str) -> list[int]:
    parsed = ast.literal_eval(inner)
    if not isinstance(parsed, list):
        raise ValueError("Contribution block is not a list.")
    values: list[int] = []
    for item in parsed:
        if not isinstance(item, int):
            raise ValueError("Contribution block contains a non-integer value.")
        values.append(item)
    return values


def _parse_interactions(inner: str) -> list[list[Any]]:
    inner = inner.strip()
    if inner == "[]":
        return []
    if not (inner.startswith("[") and inner.endswith("]")):
        raise ValueError("Interaction block must be bracketed.")
    body = inner[1:-1]
    results: list[list[Any]] = []
    position = 0
    for match in INTERACTION_TUPLE_RE.finditer(body):
        separator = body[position: match.start()].strip()
        if separator not in {"", ","}:
            raise ValueError(f"Unexpected interaction separator: {separator!r}")
        source = match.group(1).strip()
        target = match.group(2).strip()
        unit = int(match.group(3))
        results.append([source, target, unit])
        position = match.end()
    trailing = body[position:].strip()
    if trailing not in {"", ","}:
        raise ValueError(f"Unparsed interaction content remains: {trailing!r}")
    return results


def parse_compact_observer_text(text: str) -> dict[str, Any]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    reflection_lines: list[str] = []
    reflection_marker_seen = False
    rounds: list[dict[str, Any]] = []
    errors: list[str] = []

    current_round: dict[str, Any] | None = None
    current_stage = "reflection"
    current_explanation_lines: list[str] | None = None

    def finalize_round() -> None:
        nonlocal current_round, current_stage, current_explanation_lines
        if current_round is None:
            return
        if current_explanation_lines is not None:
            current_round["explanation"] = "\n".join(current_explanation_lines).strip()
            current_explanation_lines = None
        current_round.pop("summary_marker_seen", None)
        rounds.append(current_round)
        current_round = None
        current_stage = "reflection"

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if current_explanation_lines is not None:
                current_explanation_lines.append("")
            elif current_round is None:
                reflection_lines.append("")
            continue

        round_begin_match = ROUND_BEGIN_RE.match(stripped)
        if round_begin_match:
            finalize_round()
            current_round = {
                "round_number": int(round_begin_match.group(1)),
                "explanation": "",
                "contributions": None,
                "interactions": None,
                "messages": [],
                "summary_marker_seen": False,
            }
            current_stage = "contribution"
            continue

        if current_round is None:
            if OVERALL_REFLECTION_RE.match(stripped):
                reflection_marker_seen = True
            elif stripped in {"---", "## Next rounds", "### Next rounds", "## Predicted continuation", "### Predicted continuation"}:
                continue
            else:
                reflection_lines.append(line)
            continue

        explanation_match = ROUND_EXPLANATION_RE.match(stripped)
        if explanation_match:
            explanation_round_number = int(explanation_match.group(1))
            if explanation_round_number != current_round["round_number"]:
                errors.append(
                    f"Line {line_number}: explanation round {explanation_round_number} does not match current round {current_round['round_number']}."
                )
            current_explanation_lines = []
            continue

        contributions_match = CONTRIBUTIONS_RE.match(stripped)
        if contributions_match:
            if current_explanation_lines is not None:
                current_round["explanation"] = "\n".join(current_explanation_lines).strip()
                current_explanation_lines = None
            try:
                current_round["contributions"] = _parse_contributions(contributions_match.group(1))
            except Exception as exc:
                errors.append(f"Line {line_number}: could not parse contributions: {exc}")
            current_stage = "outcome"
            continue

        interactions_match = INTERACTIONS_RE.match(stripped)
        if interactions_match:
            if current_explanation_lines is not None:
                current_round["explanation"] = "\n".join(current_explanation_lines).strip()
                current_explanation_lines = None
            try:
                current_round["interactions"] = _parse_interactions(interactions_match.group(2))
            except Exception as exc:
                errors.append(f"Line {line_number}: could not parse interactions: {exc}")
            continue

        summary_match = ROUND_SUMMARY_RE.match(stripped)
        if summary_match:
            if current_explanation_lines is not None:
                current_round["explanation"] = "\n".join(current_explanation_lines).strip()
                current_explanation_lines = None
            summary_round_number = int(summary_match.group(1))
            if summary_round_number != current_round["round_number"]:
                errors.append(
                    f"Line {line_number}: summary round {summary_round_number} does not match current round {current_round['round_number']}."
                )
            current_round["summary_marker_seen"] = True
            current_stage = "summary"
            continue

        chat_match = CHAT_RE.match(stripped)
        if chat_match:
            if current_explanation_lines is not None:
                current_round["explanation"] = "\n".join(current_explanation_lines).strip()
                current_explanation_lines = None
            current_round["messages"].append(
                {
                    "speaker": chat_match.group(1).strip(),
                    "text": chat_match.group(2),
                    "phase": current_stage,
                }
            )
            continue

        if stripped in {"---", "## Next rounds", "### Next rounds", "## Predicted continuation", "### Predicted continuation"}:
            continue

        if current_explanation_lines is not None:
            current_explanation_lines.append(line)
        else:
            errors.append(f"Line {line_number}: unrecognized content inside round: {line}")

    finalize_round()

    reflection = "\n".join(reflection_lines).strip()
    return {
        "reflection": reflection,
        "overall_reflection_marker_seen": reflection_marker_seen,
        "predicted_rounds": rounds,
        "parse_errors": errors,
    }


def _sanitize_parsed_output(parsed: dict[str, Any], expectations: dict[str, Any] | None) -> dict[str, Any]:
    if not expectations:
        return parsed

    expected_players = set(expectations.get("avatars") or [])
    sanitized_rounds: list[dict[str, Any]] = []
    for round_payload in parsed["predicted_rounds"]:
        sanitized_round = dict(round_payload)
        interactions = []
        for interaction in round_payload.get("interactions") or []:
            if not isinstance(interaction, list) or len(interaction) != 3:
                continue
            source, target, unit = interaction
            if unit == 0:
                continue
            if expected_players:
                if source not in expected_players or target not in expected_players:
                    continue
            if source == target:
                continue
            interactions.append(interaction)
        sanitized_round["interactions"] = interactions if round_payload.get("interactions") is not None else None

        messages = []
        for message in round_payload.get("messages", []):
            if expected_players and message.get("speaker") not in expected_players:
                continue
            messages.append(message)
        sanitized_round["messages"] = messages
        sanitized_rounds.append(sanitized_round)

    return {
        **parsed,
        "predicted_rounds": sanitized_rounds,
    }


def _validate_parsed_output(parsed: dict[str, Any], expectations: dict[str, Any] | None) -> list[str]:
    if not expectations:
        return []

    errors: list[str] = []
    expected_k = expectations.get("k")
    expected_num_rounds = expectations.get("num_rounds")
    expected_players = expectations.get("avatars") or []
    expected_num_players = expectations.get("num_players")
    interactions_enabled = bool(expectations.get("punishment_enabled") or expectations.get("reward_enabled"))

    predicted_rounds = parsed["predicted_rounds"]
    if expected_k is not None and expected_num_rounds is not None:
        expected_round_numbers = list(range(expected_k + 1, expected_num_rounds + 1))
        actual_round_numbers = [round_payload["round_number"] for round_payload in predicted_rounds]
        if actual_round_numbers != expected_round_numbers:
            errors.append(
                f"Expected round numbers {expected_round_numbers} but parsed {actual_round_numbers}."
            )

    for round_payload in predicted_rounds:
        contributions = round_payload.get("contributions")
        if contributions is None:
            errors.append(f"Round {round_payload['round_number']}: missing contributions line.")
        elif expected_num_players is not None and len(contributions) != expected_num_players:
            errors.append(
                f"Round {round_payload['round_number']}: expected {expected_num_players} contributions but parsed {len(contributions)}."
            )

        interactions = round_payload.get("interactions")
        if interactions_enabled and interactions is None:
            errors.append(
                f"Round {round_payload['round_number']}: missing punishment/reward line even though interactions are enabled."
            )
        if (not interactions_enabled) and interactions is not None:
            errors.append(
                f"Round {round_payload['round_number']}: parsed punishment/reward line even though interactions are disabled."
            )

        for interaction in interactions or []:
            source, target, unit = interaction
            if expected_players:
                if source not in expected_players:
                    errors.append(
                        f"Round {round_payload['round_number']}: unknown interaction source {source}."
                    )
                if target not in expected_players:
                    errors.append(
                        f"Round {round_payload['round_number']}: unknown interaction target {target}."
                    )
            if source == target:
                errors.append(
                    f"Round {round_payload['round_number']}: self-targeting interaction {source}->{target}."
                )
            if unit == 0:
                errors.append(f"Round {round_payload['round_number']}: zero-unit interaction parsed.")

    return errors


def _build_output_row(
    record: dict[str, Any],
    parsed: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    custom_id = str(record.get("custom_id", ""))
    output = {
        "custom_id": custom_id,
        "reflection": parsed["reflection"],
        "overall_reflection_marker_seen": parsed["overall_reflection_marker_seen"],
        "predicted_rounds": parsed["predicted_rounds"],
        "parse_errors": parsed["parse_errors"],
        "validation_errors": validation_errors,
        "parse_success": not parsed["parse_errors"] and not validation_errors,
    }
    if "game_id" in record:
        output["game_id"] = record["game_id"]
    if "treatment_name" in record:
        output["treatment_name"] = record["treatment_name"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse compact observer transcript outputs into structured round predictions."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    args = parser.parse_args()

    records = _read_jsonl(args.input_jsonl)
    expectations_by_custom_id = _load_manifest(args.request_manifest_csv)

    output_rows: list[dict[str, Any]] = []
    parse_success_count = 0
    parse_error_count = 0

    for record in records:
        custom_id = str(record.get("custom_id", ""))
        try:
            text = _extract_text_from_response_record(record)
            parsed = parse_compact_observer_text(text)
        except Exception as exc:
            parsed = {
                "reflection": "",
                "overall_reflection_marker_seen": False,
                "predicted_rounds": [],
                "parse_errors": [f"Top-level extraction failure: {exc}"],
            }
        parsed = _sanitize_parsed_output(parsed, expectations_by_custom_id.get(custom_id))
        validation_errors = _validate_parsed_output(parsed, expectations_by_custom_id.get(custom_id))
        output_row = _build_output_row(record, parsed, validation_errors)
        output_rows.append(output_row)
        if output_row["parse_success"]:
            parse_success_count += 1
        else:
            parse_error_count += 1

    _write_jsonl(args.output_jsonl, output_rows)

    summary = {
        "input_path": str(args.input_jsonl),
        "output_path": str(args.output_jsonl),
        "total_records": len(output_rows),
        "parse_success_count": parse_success_count,
        "parse_error_count": parse_error_count,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
