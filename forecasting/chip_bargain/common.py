from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
STATUS_SUPPORT = {"ACCEPTED", "DECLINED"}
TOP_LEVEL_KEYS = {"rounds", "game_explanation"}
NUMERIC_QUOTE_RE = re.compile(r'(:\s*-?\d+(?:\.\d+)?)"(?=\s*[,}\]])')
TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")


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

    candidate_texts = [stripped]
    repaired = NUMERIC_QUOTE_RE.sub(r"\1", stripped)
    if repaired != stripped:
        candidate_texts.append(repaired)
    without_trailing_commas = TRAILING_COMMA_RE.sub("", stripped)
    if without_trailing_commas != stripped:
        candidate_texts.append(without_trailing_commas)
    if repaired != stripped:
        repaired_without_trailing_commas = TRAILING_COMMA_RE.sub("", repaired)
        if repaired_without_trailing_commas != repaired:
            candidate_texts.append(repaired_without_trailing_commas)

    for candidate_text in candidate_texts:
        try:
            json.loads(candidate_text)
            return candidate_text
        except Exception:
            pass

    fenced = CODE_FENCE_RE.findall(stripped)
    for block in fenced:
        candidate = block.strip()
        repaired_candidates = [candidate]
        repaired_block = NUMERIC_QUOTE_RE.sub(r"\1", candidate)
        if repaired_block != candidate:
            repaired_candidates.append(repaired_block)
        block_without_trailing_commas = TRAILING_COMMA_RE.sub("", candidate)
        if block_without_trailing_commas != candidate:
            repaired_candidates.append(block_without_trailing_commas)
        if repaired_block != candidate:
            repaired_block_without_trailing_commas = TRAILING_COMMA_RE.sub("", repaired_block)
            if repaired_block_without_trailing_commas != repaired_block:
                repaired_candidates.append(repaired_block_without_trailing_commas)
        for repaired_candidate in repaired_candidates:
            try:
                json.loads(repaired_candidate)
                return repaired_candidate
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
                candidate = stripped[start : index + 1]
                repaired_candidates = [candidate]
                repaired_candidate = NUMERIC_QUOTE_RE.sub(r"\1", candidate)
                if repaired_candidate != candidate:
                    repaired_candidates.append(repaired_candidate)
                candidate_without_trailing_commas = TRAILING_COMMA_RE.sub("", candidate)
                if candidate_without_trailing_commas != candidate:
                    repaired_candidates.append(candidate_without_trailing_commas)
                if repaired_candidate != candidate:
                    repaired_candidate_without_trailing_commas = TRAILING_COMMA_RE.sub("", repaired_candidate)
                    if repaired_candidate_without_trailing_commas != repaired_candidate:
                        repaired_candidates.append(repaired_candidate_without_trailing_commas)
                for repaired_version in repaired_candidates:
                    try:
                        json.loads(repaired_version)
                        return repaired_version
                    except Exception:
                        continue
                return candidate

    raise ValueError("Could not find a balanced JSON object in the response text.")


def load_request_manifest_df(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in [
        "custom_id",
        "record_id",
        "unit_id",
        "treatment_name",
        "chip_family",
        "chip_family_display",
        "cohort_name",
        "stage_name",
        "stage_code",
        "experiment_name",
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


def _compute_player_welfare(
    holdings: dict[str, int],
    valuations: dict[str, float],
) -> float:
    return float(
        sum(float(valuations.get(chip_id, 0.0)) * int(quantity) for chip_id, quantity in holdings.items())
    )


def _sorted_chip_ids(chip_definitions: list[dict[str, Any]]) -> list[str]:
    return [str(chip["id"]).upper() for chip in chip_definitions]


def _normalize_quantity_map(
    value: Any,
    *,
    field_name: str,
    valid_chip_ids: set[str],
) -> tuple[dict[str, int] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, [f"{field_name} must be an object with exactly one chip color."]

    normalized: dict[str, int] = {}
    for raw_chip_id, raw_quantity in value.items():
        chip_id = str(raw_chip_id).strip().upper()
        if chip_id not in valid_chip_ids:
            errors.append(f"{field_name} uses invalid chip color `{chip_id}`.")
            continue
        if isinstance(raw_quantity, bool):
            errors.append(f"{field_name}.{chip_id} must be a positive integer.")
            continue
        try:
            numeric = float(raw_quantity)
        except (TypeError, ValueError):
            errors.append(f"{field_name}.{chip_id} must be a positive integer.")
            continue
        if not numeric.is_integer() or int(numeric) <= 0:
            errors.append(f"{field_name}.{chip_id} must be a positive integer.")
            continue
        normalized[chip_id] = int(numeric)

    if len(normalized) != 1:
        errors.append(f"{field_name} must contain exactly one chip color with a positive integer quantity.")
    return (normalized if len(normalized) == 1 else None), errors


def _normalize_response_entry(value: Any, player_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    def _payload(accepted_value: bool, reasoning_value: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"accepted": bool(accepted_value)}
        if reasoning_value is not None:
            payload["reasoning"] = reasoning_value
        return payload

    if isinstance(value, dict):
        accepted = value.get("accepted")
        reasoning = value.get("reasoning")
        normalized_reasoning: str | None = None
        if reasoning is not None:
            if not isinstance(reasoning, str):
                return None, [f"responses.{player_id}.reasoning must be a string when present."]
            normalized_reasoning = reasoning.strip()
        if isinstance(accepted, bool):
            return _payload(bool(accepted), normalized_reasoning), []
        if isinstance(accepted, (int, np.integer)) and int(accepted) in {0, 1}:
            return _payload(bool(int(accepted)), normalized_reasoning), []
        if isinstance(accepted, str) and accepted.strip().upper() in {"TRUE", "FALSE", "YES", "NO", "0", "1"}:
            mapped = accepted.strip().upper() in {"TRUE", "YES", "1"}
            return _payload(mapped, normalized_reasoning), []
    if isinstance(value, bool):
        return _payload(bool(value)), []
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return _payload(bool(int(value))), []
    if isinstance(value, str) and value.strip().upper() in {"TRUE", "FALSE", "YES", "NO", "0", "1"}:
        mapped = value.strip().upper() in {"TRUE", "YES", "1"}
        return _payload(mapped), []
    return None, [f"responses.{player_id} must be an object with an accepted boolean."]


def _normalize_optional_text(value: Any, *, field_name: str) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return None, [f"{field_name} must be a string when present."]
    return value.strip(), []


def attempt_repair_prediction_payload(
    payload: Any,
    *,
    players: list[str],
    turn_order: list[str],
    round_turn_orders: list[list[str]] | None = None,
    chip_definitions: list[dict[str, Any]],
    initial_chip_holdings: dict[str, dict[str, int]],
    participant_chip_values: dict[str, dict[str, float]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Top-level JSON value must be an object for repair."]
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        return None, ["`rounds` must be a list for repair."]

    repaired = deepcopy(payload)
    players = [str(player_id) for player_id in players]
    player_set = set(players)
    chip_ids = _sorted_chip_ids(chip_definitions)
    chip_id_set = set(chip_ids)
    normalized_round_turn_orders = (
        [[str(player_id) for player_id in round_order] for round_order in round_turn_orders]
        if round_turn_orders is not None
        else [list(turn_order) for _ in range(len(rounds))]
    )
    holdings: dict[str, dict[str, int]] = {
        player_id: {chip_id: int(quantity) for chip_id, quantity in chip_map.items()}
        for player_id, chip_map in initial_chip_holdings.items()
    }

    repair_notes: list[str] = []

    for round_index, round_value in enumerate(repaired.get("rounds", [])):
        if not isinstance(round_value, dict):
            return None, repair_notes
        turns = round_value.get("turns")
        if not isinstance(turns, list):
            return None, repair_notes
        expected_round_order = (
            normalized_round_turn_orders[round_index]
            if round_index < len(normalized_round_turn_orders)
            else list(turn_order)
        )

        for turn_index, turn_value in enumerate(turns):
            if not isinstance(turn_value, dict):
                return None, repair_notes

            sender_id = str(turn_value.get("sender_id", ""))
            if sender_id not in player_set:
                return None, repair_notes
            if turn_index < len(expected_round_order) and sender_id != expected_round_order[turn_index]:
                return None, repair_notes

            buy_map, buy_errors = _normalize_quantity_map(
                turn_value.get("buy"),
                field_name=f"rounds[{round_index}].turns[{turn_index}].buy",
                valid_chip_ids=chip_id_set,
            )
            sell_map, sell_errors = _normalize_quantity_map(
                turn_value.get("sell"),
                field_name=f"rounds[{round_index}].turns[{turn_index}].sell",
                valid_chip_ids=chip_id_set,
            )
            if buy_errors or sell_errors or buy_map is None or sell_map is None:
                return None, repair_notes

            buy_color, buy_quantity = next(iter(buy_map.items()))
            sell_color, sell_quantity = next(iter(sell_map.items()))
            if buy_color == sell_color:
                return None, repair_notes

            available_sell = int(holdings[sender_id].get(sell_color, 0))
            if available_sell <= 0:
                return None, repair_notes
            if sell_quantity > available_sell:
                turn_value["sell"] = {sell_color: available_sell}
                sell_quantity = available_sell
                repair_notes.append(
                    f"rounds[{round_index}].turns[{turn_index}]: clipped sell quantity to sender inventory."
                )

            expected_response_players = [player_id for player_id in players if player_id != sender_id]
            responses_raw = turn_value.get("responses")
            if not isinstance(responses_raw, dict):
                responses_raw = {}
            repaired_responses: dict[str, dict[str, bool]] = {}
            valid_acceptors: list[str] = []

            for player_id in expected_response_players:
                raw_response = responses_raw.get(player_id, {"accepted": False})
                normalized_response, _ = _normalize_response_entry(raw_response, player_id)
                if normalized_response is None:
                    normalized_response = {"accepted": False}
                    repair_notes.append(
                        f"rounds[{round_index}].turns[{turn_index}].responses[{player_id}]: replaced invalid response with decline."
                    )
                if normalized_response["accepted"] and holdings[player_id].get(buy_color, 0) < buy_quantity:
                    normalized_response = {**normalized_response, "accepted": False}
                    repair_notes.append(
                        f"rounds[{round_index}].turns[{turn_index}].responses[{player_id}]: removed impossible acceptance."
                    )
                repaired_responses[player_id] = normalized_response
                if normalized_response["accepted"]:
                    valid_acceptors.append(player_id)

            turn_value["responses"] = repaired_responses

            raw_status = str(turn_value.get("status", "")).strip().upper()
            recipient_id_value = turn_value.get("recipient_id")
            recipient_id = str(recipient_id_value) if recipient_id_value is not None else None

            if raw_status == "DECLINED":
                if recipient_id is not None:
                    turn_value["recipient_id"] = None
                    repair_notes.append(
                        f"rounds[{round_index}].turns[{turn_index}]: cleared recipient_id on declined trade."
                    )
                for player_id, response in repaired_responses.items():
                    if response["accepted"]:
                        repaired_responses[player_id] = {**response, "accepted": False}
                valid_acceptors = []
            else:
                if raw_status != "ACCEPTED":
                    raw_status = "ACCEPTED" if valid_acceptors else "DECLINED"
                    turn_value["status"] = raw_status
                    repair_notes.append(
                        f"rounds[{round_index}].turns[{turn_index}]: normalized invalid status."
                    )

                if raw_status == "ACCEPTED":
                    if not valid_acceptors:
                        turn_value["status"] = "DECLINED"
                        turn_value["recipient_id"] = None
                        repair_notes.append(
                            f"rounds[{round_index}].turns[{turn_index}]: downgraded accepted trade to decline because no feasible accepter remained."
                        )
                    else:
                        if recipient_id not in valid_acceptors:
                            turn_value["recipient_id"] = valid_acceptors[0]
                            recipient_id = valid_acceptors[0]
                            repair_notes.append(
                                f"rounds[{round_index}].turns[{turn_index}]: reassigned recipient to a feasible accepter."
                            )
                        holdings[sender_id][sell_color] -= sell_quantity
                        holdings[sender_id][buy_color] += buy_quantity
                        holdings[recipient_id][buy_color] -= buy_quantity
                        holdings[recipient_id][sell_color] += sell_quantity
                else:
                    turn_value["recipient_id"] = None

    return repaired, repair_notes


def _minimal_output_target(full_target: dict[str, Any]) -> dict[str, Any]:
    return {
        "rounds": [
            {
                "round": int(round_row["round"]),
                "turns": [
                    {
                        "turn_index": int(turn_row["turn_index"]),
                        "sender_id": str(turn_row["sender_id"]),
                        "buy": {str(k).upper(): int(v) for k, v in (turn_row.get("buy") or {}).items()},
                        "sell": {str(k).upper(): int(v) for k, v in (turn_row.get("sell") or {}).items()},
                        "status": str(turn_row["status"]).upper(),
                        "recipient_id": (
                            str(turn_row["recipient_id"]) if turn_row.get("recipient_id") is not None else None
                        ),
                        "responses": {
                            str(player_id): {"accepted": bool((response or {}).get("accepted", False))}
                            for player_id, response in (turn_row.get("responses") or {}).items()
                        },
                    }
                    for turn_row in round_row["turns"]
                ],
            }
            for round_row in full_target["rounds"]
        ]
    }


def compute_optimal_total_welfare(
    *,
    players: list[str],
    chip_ids: list[str],
    participant_chip_values: dict[str, dict[str, float]],
    initial_chip_holdings: dict[str, dict[str, int]],
) -> float:
    player_count = len(players)
    chip_count = len(chip_ids)
    var_count = player_count * chip_count

    c: list[float] = []
    for player_id in players:
        for chip_id in chip_ids:
            c.append(-float(participant_chip_values[player_id][chip_id]))

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for chip_index, chip_id in enumerate(chip_ids):
        row = [0.0] * var_count
        for player_index in range(player_count):
            row[player_index * chip_count + chip_index] = 1.0
        a_eq.append(row)
        b_eq.append(
            float(sum(int(initial_chip_holdings[player_id][chip_id]) for player_id in players))
        )

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for player_index, player_id in enumerate(players):
        row = [0.0] * var_count
        for chip_index, chip_id in enumerate(chip_ids):
            row[player_index * chip_count + chip_index] = -float(participant_chip_values[player_id][chip_id])
        initial_welfare = _compute_player_welfare(
            initial_chip_holdings[player_id],
            participant_chip_values[player_id],
        )
        a_ub.append(row)
        b_ub.append(-initial_welfare)

    result = linprog(
        c=np.asarray(c, dtype=float),
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(a_eq, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=[(0.0, None)] * var_count,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Failed to solve Pareto-optimal welfare LP: {result.message}")
    return float(-result.fun)


def validate_prediction_payload(
    payload: Any,
    *,
    players: list[str],
    turn_order: list[str],
    round_turn_orders: list[list[str]] | None = None,
    chip_definitions: list[dict[str, Any]],
    initial_chip_holdings: dict[str, dict[str, int]],
    participant_chip_values: dict[str, dict[str, float]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Top-level JSON value must be an object."]
    if not set(payload.keys()).issubset(TOP_LEVEL_KEYS) or "rounds" not in payload:
        return None, ["Top-level JSON object must contain `rounds` and may optionally include `game_explanation`."]
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        return None, ["`rounds` must be a list."]

    players = [str(player_id) for player_id in players]
    turn_order = [str(player_id) for player_id in turn_order]
    normalized_round_turn_orders = (
        [[str(player_id) for player_id in round_order] for round_order in round_turn_orders]
        if round_turn_orders is not None
        else None
    )
    player_set = set(players)
    chip_ids = _sorted_chip_ids(chip_definitions)
    chip_id_set = set(chip_ids)
    holdings: dict[str, dict[str, int]] = {
        player_id: {chip_id: int(quantity) for chip_id, quantity in chip_map.items()}
        for player_id, chip_map in initial_chip_holdings.items()
    }
    valuations: dict[str, dict[str, float]] = {
        player_id: {chip_id.upper(): float(value) for chip_id, value in chip_map.items()}
        for player_id, chip_map in participant_chip_values.items()
    }

    initial_player_welfare = {
        player_id: _compute_player_welfare(holdings[player_id], valuations[player_id])
        for player_id in players
    }
    initial_total_welfare = float(sum(initial_player_welfare.values()))
    optimal_total_welfare = compute_optimal_total_welfare(
        players=players,
        chip_ids=chip_ids,
        participant_chip_values=valuations,
        initial_chip_holdings=holdings,
    )
    optimal_total_surplus = float(optimal_total_welfare - initial_total_welfare)

    errors: list[str] = []
    normalized_rounds: list[dict[str, Any]] = []
    flat_turn_rows: list[dict[str, Any]] = []

    game_explanation, game_explanation_errors = _normalize_optional_text(
        payload.get("game_explanation"),
        field_name="game_explanation",
    )
    errors.extend(game_explanation_errors)

    if len(rounds) != 3:
        errors.append("`rounds` must contain exactly 3 round objects.")
        return None, errors

    absolute_turn_index = 0
    for expected_round_index, round_value in enumerate(rounds):
        if not isinstance(round_value, dict):
            errors.append(f"rounds[{expected_round_index}] must be an object.")
            continue
        if int(round_value.get("round", -1)) != expected_round_index:
            errors.append(f"rounds[{expected_round_index}].round must equal {expected_round_index}.")
        round_explanation, round_explanation_errors = _normalize_optional_text(
            round_value.get("round_explanation"),
            field_name=f"rounds[{expected_round_index}].round_explanation",
        )
        errors.extend(round_explanation_errors)
        turns = round_value.get("turns")
        if not isinstance(turns, list) or len(turns) != len(turn_order):
            errors.append(
                f"rounds[{expected_round_index}].turns must contain exactly {len(turn_order)} turn objects."
            )
            continue

        normalized_turns: list[dict[str, Any]] = []
        round_sender_ids: list[str] = []
        for expected_turn_index, turn_value in enumerate(turns):
            location = f"rounds[{expected_round_index}].turns[{expected_turn_index}]"
            if not isinstance(turn_value, dict):
                errors.append(f"{location} must be an object.")
                continue

            before_states = {
                player_id: {
                    "chips": deepcopy(holdings[player_id]),
                    "payout": _compute_player_welfare(holdings[player_id], valuations[player_id]),
                }
                for player_id in players
            }
            group_total_welfare_before = float(sum(state["payout"] for state in before_states.values()))
            group_surplus_before = float(group_total_welfare_before - initial_total_welfare)
            group_surplus_ratio_before = (
                float(group_surplus_before / optimal_total_surplus) if optimal_total_surplus > 0 else float("nan")
            )

            if int(turn_value.get("turn_index", -1)) != expected_turn_index:
                errors.append(f"{location}.turn_index must equal {expected_turn_index}.")
            sender_id = str(turn_value.get("sender_id", ""))
            if sender_id not in player_set:
                errors.append(f"{location}.sender_id must be one of the known players.")
                continue
            if normalized_round_turn_orders is not None:
                if expected_round_index >= len(normalized_round_turn_orders):
                    errors.append("round_turn_orders is missing a round schedule required for validation.")
                    continue
                expected_round_order = normalized_round_turn_orders[expected_round_index]
                if expected_turn_index >= len(expected_round_order):
                    errors.append(
                        f"round_turn_orders[{expected_round_index}] is missing sender position {expected_turn_index}."
                    )
                    continue
                expected_sender_id = expected_round_order[expected_turn_index]
                if sender_id != expected_sender_id:
                    errors.append(
                        f"{location}.sender_id must equal `{expected_sender_id}` for this round schedule."
                    )
                    continue
            round_sender_ids.append(sender_id)

            proposer_reasoning, proposer_reasoning_errors = _normalize_optional_text(
                turn_value.get("proposer_reasoning"),
                field_name=f"{location}.proposer_reasoning",
            )
            errors.extend(proposer_reasoning_errors)

            buy_map, buy_errors = _normalize_quantity_map(
                turn_value.get("buy"),
                field_name=f"{location}.buy",
                valid_chip_ids=chip_id_set,
            )
            sell_map, sell_errors = _normalize_quantity_map(
                turn_value.get("sell"),
                field_name=f"{location}.sell",
                valid_chip_ids=chip_id_set,
            )
            errors.extend(buy_errors)
            errors.extend(sell_errors)
            if buy_map is None or sell_map is None:
                continue

            buy_color, buy_quantity = next(iter(buy_map.items()))
            sell_color, sell_quantity = next(iter(sell_map.items()))
            if buy_color == sell_color:
                errors.append(f"{location}.buy and {location}.sell must use different chip colors.")
                continue
            if holdings[sender_id].get(sell_color, 0) < sell_quantity:
                errors.append(
                    f"{location}.sender_id does not hold enough `{sell_color}` chips to sell {sell_quantity}."
                )
                continue

            responses_raw = turn_value.get("responses")
            if not isinstance(responses_raw, dict):
                errors.append(f"{location}.responses must be an object.")
                continue
            expected_response_players = [player_id for player_id in players if player_id != sender_id]
            actual_response_players = sorted(str(player_id) for player_id in responses_raw.keys())
            if actual_response_players != sorted(expected_response_players):
                errors.append(
                    f"{location}.responses must contain exactly the non-sender players: {expected_response_players}."
                )
                continue

            normalized_responses: dict[str, dict[str, bool]] = {}
            accepted_players: list[str] = []
            for player_id in expected_response_players:
                normalized_response, response_errors = _normalize_response_entry(
                    responses_raw[player_id],
                    player_id,
                )
                errors.extend(response_errors)
                if normalized_response is None:
                    continue
                normalized_responses[player_id] = normalized_response
                if normalized_response["accepted"]:
                    accepted_players.append(player_id)

            status = str(turn_value.get("status", "")).strip().upper()
            if status not in STATUS_SUPPORT:
                errors.append(f"{location}.status must be one of {sorted(STATUS_SUPPORT)}.")
                continue

            recipient_id_value = turn_value.get("recipient_id")
            recipient_id = str(recipient_id_value) if recipient_id_value is not None else None
            if status == "DECLINED":
                if recipient_id is not None:
                    errors.append(f"{location}.recipient_id must be null when status is DECLINED.")
                if accepted_players:
                    errors.append(f"{location}.responses cannot include accepted players when status is DECLINED.")
            else:
                if recipient_id is None:
                    errors.append(f"{location}.recipient_id must be a player id when status is ACCEPTED.")
                    continue
                if recipient_id not in accepted_players:
                    errors.append(
                        f"{location}.recipient_id must be one of the players who accepted the trade."
                    )
                    continue
                if holdings[recipient_id].get(buy_color, 0) < buy_quantity:
                    errors.append(
                        f"{location}.recipient_id does not hold enough `{buy_color}` chips to satisfy the request."
                    )
                    continue
                for accepting_player in accepted_players:
                    if holdings[accepting_player].get(buy_color, 0) < buy_quantity:
                        errors.append(
                            f"{location}.responses[{accepting_player}] accepted a trade they cannot fulfill."
                        )
                holdings[sender_id][sell_color] -= sell_quantity
                holdings[sender_id][buy_color] += buy_quantity
                holdings[recipient_id][buy_color] -= buy_quantity
                holdings[recipient_id][sell_color] += sell_quantity

            after_states = {
                player_id: {
                    "chips": deepcopy(holdings[player_id]),
                    "payout": _compute_player_welfare(holdings[player_id], valuations[player_id]),
                }
                for player_id in players
            }
            group_total_welfare_after = float(sum(state["payout"] for state in after_states.values()))
            group_surplus_after = float(group_total_welfare_after - initial_total_welfare)
            group_surplus_ratio_after = (
                float(group_surplus_after / optimal_total_surplus) if optimal_total_surplus > 0 else float("nan")
            )
            proposer_net_surplus = float(
                valuations[sender_id][buy_color] * buy_quantity - valuations[sender_id][sell_color] * sell_quantity
            )
            normalized_turn = {
                "turn_index": expected_turn_index,
                "sender_id": sender_id,
                "proposer_reasoning": proposer_reasoning,
                "buy": buy_map,
                "sell": sell_map,
                "status": status,
                "recipient_id": recipient_id,
                "responses": normalized_responses,
                "player_states_before": before_states,
                "player_states_after": after_states,
            }
            normalized_turns.append(normalized_turn)
            flat_turn_rows.append(
                {
                    "round": expected_round_index,
                    "turn_index": expected_turn_index,
                    "absolute_turn_index": absolute_turn_index,
                    "sender_id": sender_id,
                    "buy_color": buy_color,
                    "buy_quantity": int(buy_quantity),
                    "sell_color": sell_color,
                    "sell_quantity": int(sell_quantity),
                    "status": status,
                    "recipient_id": recipient_id,
                    "accepted_binary": int(status == "ACCEPTED"),
                    "num_acceptors": int(len(accepted_players)),
                    "multiple_acceptors_binary": int(len(accepted_players) > 1),
                    "proposer_net_surplus": proposer_net_surplus,
                    "trade_ratio": float(sell_quantity / buy_quantity),
                    "group_total_welfare_before": group_total_welfare_before,
                    "group_total_welfare_after": group_total_welfare_after,
                    "group_surplus_before": group_surplus_before,
                    "group_surplus_after": group_surplus_after,
                    "group_surplus_ratio_before": group_surplus_ratio_before,
                    "group_surplus_ratio_after": group_surplus_ratio_after,
                }
            )
            absolute_turn_index += 1

        if sorted(round_sender_ids) != sorted(players):
            errors.append(
                f"rounds[{expected_round_index}] must contain exactly one turn from each player."
            )
        normalized_rounds.append(
            {
                "round": expected_round_index,
                "round_explanation": round_explanation,
                "turns": normalized_turns,
            }
        )

    if errors:
        return None, errors

    final_turn = flat_turn_rows[-1]
    final_holdings = {player_id: deepcopy(holdings[player_id]) for player_id in players}
    final_player_welfare = {
        player_id: _compute_player_welfare(final_holdings[player_id], valuations[player_id])
        for player_id in players
    }
    final_total_welfare = float(sum(final_player_welfare.values()))
    final_total_surplus = float(final_total_welfare - initial_total_welfare)
    final_surplus_ratio = (
        float(final_total_surplus / optimal_total_surplus) if optimal_total_surplus > 0 else float("nan")
    )

    normalized_payload = {
        "game_explanation": game_explanation,
        "rounds": normalized_rounds,
        "_derived": {
            "players": players,
            "turn_order": turn_order,
            "round_turn_orders": normalized_round_turn_orders or [turn_order for _ in range(len(rounds))],
            "initial_total_welfare": initial_total_welfare,
            "optimal_total_welfare": optimal_total_welfare,
            "optimal_total_surplus": optimal_total_surplus,
            "final_total_welfare": final_total_welfare,
            "final_total_surplus": final_total_surplus,
            "final_surplus_ratio": final_surplus_ratio,
            "initial_player_welfare": initial_player_welfare,
            "final_player_welfare": final_player_welfare,
            "final_holdings": final_holdings,
            "flat_turn_rows": flat_turn_rows,
            "final_group_surplus_ratio_after_turn": final_turn["group_surplus_ratio_after"],
        },
    }
    return normalized_payload, []


def _manifest_row_lookup(manifest_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["custom_id"]): row
        for row in manifest_df.to_dict(orient="records")
    }


def _manifest_metadata(manifest_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": str(manifest_row["custom_id"]),
        "record_id": str(manifest_row["record_id"]),
        "unit_id": str(manifest_row["unit_id"]),
        "treatment_name": str(manifest_row["treatment_name"]),
        "chip_family": str(manifest_row["chip_family"]),
        "chip_family_display": str(manifest_row["chip_family_display"]),
        "cohort_name": str(manifest_row["cohort_name"]),
        "stage_name": str(manifest_row["stage_name"]),
        "stage_code": str(manifest_row["stage_code"]),
        "experiment_name": str(manifest_row["experiment_name"]),
    }


def _metadata_to_validation_inputs(manifest_row: dict[str, Any]) -> dict[str, Any]:
    inputs = {
        "players": json.loads(str(manifest_row["players_json"])),
        "turn_order": json.loads(str(manifest_row["turn_order_json"])),
        "chip_definitions": json.loads(str(manifest_row["chip_definitions_json"])),
        "initial_chip_holdings": json.loads(str(manifest_row["initial_chip_holdings_json"])),
        "participant_chip_values": json.loads(str(manifest_row["participant_chip_values_json"])),
    }
    if "round_turn_orders_json" in manifest_row and pd.notna(manifest_row["round_turn_orders_json"]):
        inputs["round_turn_orders"] = json.loads(str(manifest_row["round_turn_orders_json"]))
    return inputs


def sample_random_normalized_payload_from_manifest_row(
    manifest_row: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    inputs = _metadata_to_validation_inputs(manifest_row)
    players = [str(player_id) for player_id in inputs["players"]]
    chip_definitions = list(inputs["chip_definitions"])
    chip_ids = _sorted_chip_ids(chip_definitions)
    round_turn_orders = inputs.get("round_turn_orders") or [list(inputs["turn_order"]) for _ in range(3)]
    holdings = {
        str(player_id): {
            str(chip_id).upper(): int(quantity)
            for chip_id, quantity in chip_map.items()
        }
        for player_id, chip_map in inputs["initial_chip_holdings"].items()
    }
    can_buy = {
        str(chip["id"]).upper(): bool(chip.get("can_buy", True))
        for chip in chip_definitions
    }
    can_sell = {
        str(chip["id"]).upper(): bool(chip.get("can_sell", True))
        for chip in chip_definitions
    }

    rounds: list[dict[str, Any]] = []
    for round_index, round_order in enumerate(round_turn_orders):
        turns: list[dict[str, Any]] = []
        for turn_index, sender_id in enumerate(round_order):
            sell_color_options = [
                chip_id
                for chip_id in chip_ids
                if can_sell.get(chip_id, True) and holdings[sender_id].get(chip_id, 0) > 0
            ]
            if not sell_color_options:
                raise ValueError(
                    f"Random baseline could not find a sellable chip for sender {sender_id} in {manifest_row['custom_id']}."
                )
            sell_color = sell_color_options[int(rng.integers(len(sell_color_options)))]
            sell_quantity = int(rng.integers(1, holdings[sender_id][sell_color] + 1))

            requestable_quantities = {
                chip_id: max(holdings[player_id].get(chip_id, 0) for player_id in players if player_id != sender_id)
                for chip_id in chip_ids
                if chip_id != sell_color and can_buy.get(chip_id, True)
            }
            requestable_colors = [chip_id for chip_id, max_qty in requestable_quantities.items() if max_qty > 0]
            if requestable_colors:
                buy_color = requestable_colors[int(rng.integers(len(requestable_colors)))]
                buy_quantity = int(rng.integers(1, requestable_quantities[buy_color] + 1))
            else:
                fallback_buy_colors = [
                    chip_id for chip_id in chip_ids if chip_id != sell_color and can_buy.get(chip_id, True)
                ]
                if not fallback_buy_colors:
                    raise ValueError(
                        f"Random baseline could not find a buyable chip for sender {sender_id} in {manifest_row['custom_id']}."
                    )
                buy_color = fallback_buy_colors[int(rng.integers(len(fallback_buy_colors)))]
                buy_quantity = 1

            responses: dict[str, dict[str, bool]] = {}
            accepted_players: list[str] = []
            for player_id in players:
                if player_id == sender_id:
                    continue
                can_accept = holdings[player_id].get(buy_color, 0) >= buy_quantity
                accepted = bool(rng.integers(0, 2)) if can_accept else False
                responses[player_id] = {"accepted": accepted}
                if accepted:
                    accepted_players.append(player_id)

            if accepted_players:
                recipient_id = accepted_players[int(rng.integers(len(accepted_players)))]
                status = "ACCEPTED"
                holdings[sender_id][sell_color] -= sell_quantity
                holdings[sender_id][buy_color] += buy_quantity
                holdings[recipient_id][buy_color] -= buy_quantity
                holdings[recipient_id][sell_color] += sell_quantity
            else:
                recipient_id = None
                status = "DECLINED"

            turns.append(
                {
                    "turn_index": int(turn_index),
                    "sender_id": str(sender_id),
                    "buy": {str(buy_color): int(buy_quantity)},
                    "sell": {str(sell_color): int(sell_quantity)},
                    "status": status,
                    "recipient_id": recipient_id,
                    "responses": responses,
                }
            )
        rounds.append({"round": int(round_index), "turns": turns})

    normalized_payload, errors = validate_prediction_payload(
        {"rounds": rounds},
        **inputs,
    )
    if errors or normalized_payload is None:
        raise ValueError(
            f"Random baseline payload failed validation for {manifest_row['custom_id']}: {errors}"
        )
    return normalized_payload


def build_random_record_frames(
    *,
    request_manifest_csv: Path,
    sample_count_map: dict[str, int],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_df = load_request_manifest_df(request_manifest_csv)
    game_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    for treatment_name, requested_n in sorted(sample_count_map.items()):
        if requested_n <= 0:
            continue
        treatment_df = manifest_df[manifest_df["treatment_name"] == str(treatment_name)].copy()
        if treatment_df.empty:
            continue
        replace = requested_n > len(treatment_df)
        sampled_df = treatment_df.sample(
            n=requested_n,
            replace=replace,
            random_state=int(rng.integers(2**31 - 1)),
        )
        for manifest_row in sampled_df.to_dict(orient="records"):
            normalized_payload = sample_random_normalized_payload_from_manifest_row(
                manifest_row,
                rng=rng,
            )
            game_rows.append(
                _game_summary_row(
                    manifest_row=manifest_row,
                    normalized_payload=normalized_payload,
                    data_source="random_uniform",
                )
            )
            player_rows.extend(
                _player_rows(
                    manifest_row=manifest_row,
                    normalized_payload=normalized_payload,
                    data_source="random_uniform",
                )
            )
            round_rows.extend(
                _round_rows(
                    manifest_row=manifest_row,
                    normalized_payload=normalized_payload,
                    data_source="random_uniform",
                )
            )

    game_df = (
        pd.DataFrame(game_rows).sort_values(["treatment_name", "custom_id"]).reset_index(drop=True)
        if game_rows
        else pd.DataFrame()
    )
    player_df = (
        pd.DataFrame(player_rows).sort_values(["treatment_name", "custom_id", "player_id"]).reset_index(drop=True)
        if player_rows
        else pd.DataFrame()
    )
    round_df = (
        pd.DataFrame(round_rows).sort_values(["treatment_name", "custom_id", "round_number"]).reset_index(drop=True)
        if round_rows
        else pd.DataFrame()
    )
    return game_df, player_df, round_df


def _game_summary_row(
    *,
    manifest_row: dict[str, Any],
    normalized_payload: dict[str, Any],
    data_source: str,
) -> dict[str, Any]:
    derived = normalized_payload["_derived"]
    flat_turn_df = pd.DataFrame(derived["flat_turn_rows"])
    return {
        **_manifest_metadata(manifest_row),
        "data_source": data_source,
        "n_players": int(len(derived["players"])),
        "n_rounds": int(len(normalized_payload["rounds"])),
        "n_turns": int(len(derived["flat_turn_rows"])),
        "initial_total_welfare": float(derived["initial_total_welfare"]),
        "optimal_total_welfare": float(derived["optimal_total_welfare"]),
        "optimal_total_surplus": float(derived["optimal_total_surplus"]),
        "final_total_welfare": float(derived["final_total_welfare"]),
        "final_total_surplus": float(derived["final_total_surplus"]),
        "final_surplus_ratio": float(derived["final_surplus_ratio"]),
        "final_group_surplus_ratio_after_turn": float(derived["final_group_surplus_ratio_after_turn"]),
        "mean_trade_ratio": float(flat_turn_df["trade_ratio"].mean()),
        "mean_acceptance_rate": float(flat_turn_df["accepted_binary"].mean()),
        "mean_proposer_net_surplus": float(flat_turn_df["proposer_net_surplus"].mean()),
        "mean_player_final_surplus": float(
            derived["final_total_surplus"] / max(len(derived["players"]), 1)
        ),
    }


def _turn_rows(
    *,
    manifest_row: dict[str, Any],
    normalized_payload: dict[str, Any],
    data_source: str,
) -> list[dict[str, Any]]:
    base = _manifest_metadata(manifest_row)
    rows: list[dict[str, Any]] = []
    for turn_row in normalized_payload["_derived"]["flat_turn_rows"]:
        rows.append(
            {
                **base,
                "data_source": data_source,
                **turn_row,
            }
        )
    return rows


def _player_rows(
    *,
    manifest_row: dict[str, Any],
    normalized_payload: dict[str, Any],
    data_source: str,
) -> list[dict[str, Any]]:
    base = _manifest_metadata(manifest_row)
    derived = normalized_payload["_derived"]
    flat_turn_df = pd.DataFrame(derived["flat_turn_rows"])
    rows: list[dict[str, Any]] = []
    for player_id in derived["players"]:
        proposer_turns = flat_turn_df[flat_turn_df["sender_id"] == player_id]
        final_surplus = float(
            derived["final_player_welfare"][player_id] - derived["initial_player_welfare"][player_id]
        )
        rows.append(
            {
                **base,
                "data_source": data_source,
                "player_id": str(player_id),
                "final_welfare": float(derived["final_player_welfare"][player_id]),
                "initial_welfare": float(derived["initial_player_welfare"][player_id]),
                "final_surplus": final_surplus,
                "proposer_turn_count": int(len(proposer_turns)),
                "proposer_mean_trade_ratio": float(proposer_turns["trade_ratio"].mean()),
                "proposer_acceptance_rate": float(proposer_turns["accepted_binary"].mean()),
                "proposer_mean_net_surplus": float(proposer_turns["proposer_net_surplus"].mean()),
            }
        )
    return rows


def _round_rows(
    *,
    manifest_row: dict[str, Any],
    normalized_payload: dict[str, Any],
    data_source: str,
) -> list[dict[str, Any]]:
    base = _manifest_metadata(manifest_row)
    flat_turn_df = pd.DataFrame(normalized_payload["_derived"]["flat_turn_rows"])
    rows: list[dict[str, Any]] = []
    for round_number, group in flat_turn_df.groupby("round", sort=True):
        ordered = group.sort_values("turn_index")
        last_turn = ordered.iloc[-1]
        rows.append(
            {
                **base,
                "data_source": data_source,
                "round_number": int(round_number),
                "round_mean_trade_ratio": float(ordered["trade_ratio"].mean()),
                "round_acceptance_rate": float(ordered["accepted_binary"].mean()),
                "round_mean_proposer_net_surplus": float(ordered["proposer_net_surplus"].mean()),
                "round_end_total_surplus": float(last_turn["group_surplus_after"]),
                "round_end_surplus_ratio": float(last_turn["group_surplus_ratio_after"]),
            }
        )
    return rows


def build_human_game_records_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    game_rows: list[dict[str, Any]] = []
    for row in read_jsonl(gold_targets_jsonl):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        minimal_payload = _minimal_output_target(row["gold_target"])
        normalized_payload, errors = validate_prediction_payload(
            minimal_payload,
            **_metadata_to_validation_inputs(manifest_row),
        )
        if errors or normalized_payload is None:
            raise ValueError(f"Gold target failed validation for {custom_id}: {errors}")
        game_rows.append(
            _game_summary_row(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="human_gold",
            )
        )
    return pd.DataFrame(game_rows).sort_values(["treatment_name", "custom_id"]).reset_index(drop=True)


def build_human_turn_records_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    turn_rows: list[dict[str, Any]] = []
    for row in read_jsonl(gold_targets_jsonl):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        minimal_payload = _minimal_output_target(row["gold_target"])
        normalized_payload, errors = validate_prediction_payload(
            minimal_payload,
            **_metadata_to_validation_inputs(manifest_row),
        )
        if errors or normalized_payload is None:
            raise ValueError(f"Gold target failed validation for {custom_id}: {errors}")
        turn_rows.extend(
            _turn_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="human_gold",
            )
        )
    return pd.DataFrame(turn_rows).sort_values(
        ["treatment_name", "custom_id", "absolute_turn_index"]
    ).reset_index(drop=True)


def build_human_player_records_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    player_rows: list[dict[str, Any]] = []
    for row in read_jsonl(gold_targets_jsonl):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        minimal_payload = _minimal_output_target(row["gold_target"])
        normalized_payload, errors = validate_prediction_payload(
            minimal_payload,
            **_metadata_to_validation_inputs(manifest_row),
        )
        if errors or normalized_payload is None:
            raise ValueError(f"Gold target failed validation for {custom_id}: {errors}")
        player_rows.extend(
            _player_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="human_gold",
            )
        )
    return pd.DataFrame(player_rows).sort_values(["treatment_name", "custom_id", "player_id"]).reset_index(drop=True)


def build_human_round_records_df(
    *,
    gold_targets_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    round_rows: list[dict[str, Any]] = []
    for row in read_jsonl(gold_targets_jsonl):
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        minimal_payload = _minimal_output_target(row["gold_target"])
        normalized_payload, errors = validate_prediction_payload(
            minimal_payload,
            **_metadata_to_validation_inputs(manifest_row),
        )
        if errors or normalized_payload is None:
            raise ValueError(f"Gold target failed validation for {custom_id}: {errors}")
        round_rows.extend(
            _round_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="human_gold",
            )
        )
    return pd.DataFrame(round_rows).sort_values(
        ["treatment_name", "custom_id", "round_number"]
    ).reset_index(drop=True)


def build_generated_game_records_df(
    *,
    parsed_output_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    parsed_df = load_parsed_outputs_df(parsed_output_jsonl)
    if parsed_df.empty:
        return pd.DataFrame()
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    game_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        if not bool(row.get("parse_success")):
            continue
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        normalized_payload = row.get("parsed_target")
        if not isinstance(normalized_payload, dict):
            continue
        game_rows.append(
            _game_summary_row(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="generated",
            )
        )
    if not game_rows:
        return pd.DataFrame()
    return pd.DataFrame(game_rows).sort_values(["treatment_name", "custom_id"]).reset_index(drop=True)


def build_generated_turn_records_df(
    *,
    parsed_output_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    parsed_df = load_parsed_outputs_df(parsed_output_jsonl)
    if parsed_df.empty:
        return pd.DataFrame()
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    turn_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        if not bool(row.get("parse_success")):
            continue
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        normalized_payload = row.get("parsed_target")
        if not isinstance(normalized_payload, dict):
            continue
        turn_rows.extend(
            _turn_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="generated",
            )
        )
    if not turn_rows:
        return pd.DataFrame()
    return pd.DataFrame(turn_rows).sort_values(
        ["treatment_name", "custom_id", "absolute_turn_index"]
    ).reset_index(drop=True)


def build_generated_player_records_df(
    *,
    parsed_output_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    parsed_df = load_parsed_outputs_df(parsed_output_jsonl)
    if parsed_df.empty:
        return pd.DataFrame()
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    player_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        if not bool(row.get("parse_success")):
            continue
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        normalized_payload = row.get("parsed_target")
        if not isinstance(normalized_payload, dict):
            continue
        player_rows.extend(
            _player_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="generated",
            )
        )
    if not player_rows:
        return pd.DataFrame()
    return pd.DataFrame(player_rows).sort_values(
        ["treatment_name", "custom_id", "player_id"]
    ).reset_index(drop=True)


def build_generated_round_records_df(
    *,
    parsed_output_jsonl: Path,
    request_manifest_csv: Path,
) -> pd.DataFrame:
    parsed_df = load_parsed_outputs_df(parsed_output_jsonl)
    if parsed_df.empty:
        return pd.DataFrame()
    manifest_lookup = _manifest_row_lookup(load_request_manifest_df(request_manifest_csv))
    round_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        if not bool(row.get("parse_success")):
            continue
        custom_id = str(row["custom_id"])
        manifest_row = manifest_lookup[custom_id]
        normalized_payload = row.get("parsed_target")
        if not isinstance(normalized_payload, dict):
            continue
        round_rows.extend(
            _round_rows(
                manifest_row=manifest_row,
                normalized_payload=normalized_payload,
                data_source="generated",
            )
        )
    if not round_rows:
        return pd.DataFrame()
    return pd.DataFrame(round_rows).sort_values(
        ["treatment_name", "custom_id", "round_number"]
    ).reset_index(drop=True)


def wasserstein_distance_1d(x: pd.Series | list[float], y: pd.Series | list[float]) -> float:
    x_arr = np.asarray(pd.Series(x).dropna().astype(float))
    y_arr = np.asarray(pd.Series(y).dropna().astype(float))
    if x_arr.size == 0 or y_arr.size == 0:
        return float("nan")
    x_arr.sort()
    y_arr.sort()
    values = np.sort(np.unique(np.concatenate([x_arr, y_arr])))
    if values.size == 1:
        return 0.0
    x_cdf = np.searchsorted(x_arr, values, side="right") / x_arr.size
    y_cdf = np.searchsorted(y_arr, values, side="right") / y_arr.size
    deltas = np.diff(values)
    return float(np.sum(np.abs(x_cdf[:-1] - y_cdf[:-1]) * deltas))
