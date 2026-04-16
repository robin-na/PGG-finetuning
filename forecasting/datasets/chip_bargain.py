from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import DatasetBundle


CHIP_DATASETS = [
    ("chip2", "2-Chip Bargaining"),
    ("chip3", "3-Chip Bargaining"),
    ("chip4", "4-Chip Bargaining"),
]


def _stage_code(stage_name: str) -> str:
    if stage_name.startswith("First negotiation game"):
        return "GAME_1"
    if stage_name.startswith("Second negotiation game"):
        return "GAME_2_ALT_PROFILE"
    return stage_name.upper().replace(" ", "_")


def _slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _normalize_offer_map(raw: dict[str, Any] | None) -> dict[str, int]:
    if not raw:
        return {}
    return {str(color): int(quantity) for color, quantity in raw.items() if int(quantity) != 0}


def _build_player_state_map(
    *,
    sender_data: dict[str, Any],
    response_data: dict[str, Any],
    after: bool,
) -> dict[str, dict[str, Any]]:
    suffix = "AfterTurn" if after else "BeforeTurn"
    payout_key = f"payout{suffix}"
    chips_key = f"chips{suffix}"

    out: dict[str, dict[str, Any]] = {
        str(sender_data["participantId"]): {
            "chips": {
                str(color): int(quantity)
                for color, quantity in sender_data[chips_key].items()
            },
            "payout": float(sender_data[payout_key]),
        }
    }
    for participant_id, participant_data in response_data.items():
        out[str(participant_id)] = {
            "chips": {
                str(color): int(quantity)
                for color, quantity in participant_data[chips_key].items()
            },
            "payout": float(participant_data[payout_key]),
        }
    return out


def _build_rounds(
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[list[str]]]:
    rounds: list[dict[str, Any]] = []
    turn_order: list[str] = []
    round_turn_orders: list[list[str]] = []
    for round_data in history:
        turns: list[dict[str, Any]] = []
        round_order: list[str] = []
        for turn_index, turn_data in enumerate(round_data["turns"]):
            transaction = turn_data["transaction"]
            offer = transaction["offer"]
            sender_data = turn_data["senderData"]
            response_data = turn_data["responseData"]
            sender_id = str(sender_data["participantId"])
            round_order.append(sender_id)
            if len(turn_order) <= turn_index:
                turn_order.append(sender_id)

            turns.append(
                {
                    "turn_index": int(turn_index),
                    "sender_id": sender_id,
                    "buy": _normalize_offer_map(offer.get("buy")),
                    "sell": _normalize_offer_map(offer.get("sell")),
                    "status": str(transaction["status"]),
                    "recipient_id": (
                        str(transaction["recipientId"])
                        if transaction.get("recipientId") is not None
                        else None
                    ),
                    "responses": {
                        str(participant_id): {
                            "accepted": bool(participant_data["offerResponse"]),
                            "selected_as_recipient": bool(
                                participant_data["selectedAsRecipient"]
                            ),
                        }
                        for participant_id, participant_data in response_data.items()
                    },
                    "player_states_before": _build_player_state_map(
                        sender_data=sender_data,
                        response_data=response_data,
                        after=False,
                    ),
                    "player_states_after": _build_player_state_map(
                        sender_data=sender_data,
                        response_data=response_data,
                        after=True,
                    ),
                }
            )
        rounds.append({"round": int(round_data["round"]), "turns": turns})
        round_turn_orders.append(round_order)
    return rounds, turn_order, round_turn_orders


def _empty_demographic_source() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_row_id",
            "summary",
            "markdown",
            "matching_age_bracket",
            "matching_sex",
            "matching_education",
        ]
    )


def build_bundle(repo_root: Path) -> DatasetBundle:
    root = repo_root / "non-PGG_generalization" / "data" / "chip_bargain"
    record_rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []

    for chip_key, display_name in CHIP_DATASETS:
        human_path = root / f"{chip_key}_data" / f"{chip_key}_human_data"
        parsed_path = human_path / f"parsed_{chip_key}_human_data_FIXED_n=24.json"
        payload = json.loads(parsed_path.read_text(encoding="utf-8"))

        for game in payload["games"]:
            stage_name = str(game["stageName"])
            stage_code = _stage_code(stage_name)
            cohort_name = str(game["cohortName"])
            metadata = game["data"]["metadata"]
            rounds, turn_order, round_turn_orders = _build_rounds(game["data"]["history"])

            players = [str(player_id) for player_id in metadata["players"]]
            chip_definitions = [
                {
                    "id": str(chip["id"]),
                    "name": str(chip["name"]),
                    "starting_quantity": int(chip["startingQuantity"]),
                    "can_buy": bool(chip["canBuy"]),
                    "can_sell": bool(chip["canSell"]),
                    "lower_value": float(chip["lowerValue"]),
                    "upper_value": float(chip["upperValue"]),
                }
                for chip in metadata["chips"]
            ]
            participant_chip_values = {
                str(participant_id): {
                    str(color): float(value) for color, value in values.items()
                }
                for participant_id, values in metadata["participantChipValueMap"].items()
            }
            initial_chip_holdings = {
                player_id: {
                    str(chip["id"]): int(chip["starting_quantity"]) for chip in chip_definitions
                }
                for player_id in players
            }

            target = {
                "experiment_name": str(game["experimentName"]),
                "cohort_name": cohort_name,
                "stage_name": stage_name,
                "chip_family": chip_key,
                "players": players,
                "chip_definitions": chip_definitions,
                "participant_chip_values": participant_chip_values,
                "initial_chip_holdings": initial_chip_holdings,
                "turn_order": turn_order,
                "round_turn_orders": round_turn_orders,
                "rounds": rounds,
            }

            record_id = f"{chip_key}__{_slug(cohort_name)}__{_slug(stage_code)}"
            units.append({"unit_id": record_id})
            record_rows.append(
                {
                    "record_id": record_id,
                    "unit_id": record_id,
                    "treatment_name": f"{chip_key.upper()}__{stage_code}",
                    "chip_family": chip_key,
                    "chip_family_display": display_name,
                    "cohort_name": cohort_name,
                    "stage_name": stage_name,
                    "stage_code": stage_code,
                    "experiment_name": str(game["experimentName"]),
                    "players_json": json.dumps(players),
                    "chip_definitions_json": json.dumps(chip_definitions),
                    "participant_chip_values_json": json.dumps(participant_chip_values),
                    "initial_chip_holdings_json": json.dumps(initial_chip_holdings),
                    "turn_order_json": json.dumps(turn_order),
                    "round_turn_orders_json": json.dumps(round_turn_orders),
                    "gold_target_json": json.dumps(target),
                }
            )

    records = pd.DataFrame(record_rows).sort_values(["treatment_name", "record_id"]).reset_index(
        drop=True
    )
    units_df = pd.DataFrame(units).drop_duplicates("unit_id").sort_values("unit_id").reset_index(
        drop=True
    )
    return DatasetBundle(
        dataset_key="chip_bargain",
        display_name="Chip Bargaining",
        records=records,
        units=units_df,
        demographic_source=_empty_demographic_source(),
        twin_matching_fields=[],
        supported_variants=["baseline"],
    )
