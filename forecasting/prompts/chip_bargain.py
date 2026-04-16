from __future__ import annotations

import json

import pandas as pd


def _format_chip_catalog(chip_definitions: list[dict[str, object]]) -> str:
    lines = []
    for chip in chip_definitions:
        lines.append(
            "- {name} (`{id}`): start with {qty}; common public range ${low:.2f} to ${high:.2f}".format(
                name=chip["name"],
                id=chip["id"],
                qty=chip["starting_quantity"],
                low=float(chip["lower_value"]),
                high=float(chip["upper_value"]),
            )
        )
    return "\n".join(lines)


def _format_player_values(
    players: list[str],
    participant_chip_values: dict[str, dict[str, float]],
) -> str:
    lines = []
    for player_id in players:
        values = participant_chip_values[player_id]
        value_text = ", ".join(
            f"{color}=${float(value):.2f}" for color, value in sorted(values.items())
        )
        lines.append(f"- `{player_id}`: {value_text}")
    return "\n".join(lines)


def _response_example(
    players: list[str],
    sender_id: str,
    accepted_player_id: str | None,
) -> str:
    response_parts = []
    for player_id in players:
        if player_id == sender_id:
            continue
        accepted = accepted_player_id == player_id
        response_parts.append(f'"{player_id}": {{"accepted": {str(accepted).lower()}}}')
    return "{%s}" % ", ".join(response_parts)


def _format_round_schedule(round_turn_orders: list[list[str]]) -> str:
    lines = []
    for round_index, round_order in enumerate(round_turn_orders):
        lines.append(
            f"- Round {round_index}: {', '.join(f'`{player_id}`' for player_id in round_order)}"
        )
    return "\n".join(lines)


def _output_example(players: list[str], round_turn_orders: list[list[str]]) -> str:
    first_round_order = round_turn_orders[0]
    sender_a = first_round_order[0]
    sender_b = first_round_order[1]
    accepted_player = next(player_id for player_id in players if player_id not in {sender_a, sender_b})
    return "\n".join(
        [
            "{",
            '  "game_explanation": "<explain the likely bargaining dynamics, incentives, and uncertainty for the full game>",',
            '  "rounds": [',
            "    {",
            '      "round": 0,',
            '      "round_explanation": "<explain the likely dynamics and strategic considerations for round 0>",',
            '      "turns": [',
            "        {",
            f'          "turn_index": 0,',
            f'          "sender_id": "{sender_a}",',
            '          "buy": {"GREEN": 2},',
            '          "sell": {"RED": 1},',
            '          "status": "DECLINED",',
            '          "recipient_id": null,',
            f'          "responses": {_response_example(players, sender_a, None)}',
            "        },",
            "        {",
            f'          "turn_index": 1,',
            f'          "sender_id": "{sender_b}",',
            '          "buy": {"RED": 3},',
            '          "sell": {"GREEN": 2},',
            '          "status": "ACCEPTED",',
            f'          "recipient_id": "{accepted_player}",',
            f'          "responses": {_response_example(players, sender_b, accepted_player)}',
            "        }",
            "      ]",
            "    }",
            "  ]",
            "}",
        ]
    )


def build_prompt(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    players = json.loads(str(row["players_json"]))
    chip_definitions = json.loads(str(row["chip_definitions_json"]))
    participant_chip_values = json.loads(str(row["participant_chip_values_json"]))
    if "round_turn_orders_json" in row and pd.notna(row["round_turn_orders_json"]):
        round_turn_orders = json.loads(str(row["round_turn_orders_json"]))
    else:
        turn_order = json.loads(str(row["turn_order_json"]))
        round_turn_orders = [turn_order, turn_order, turn_order]

    system = (
        "You forecast one complete three-player bargaining game from the human chip-trading study. "
        "Start with a game-level explanation of the likely bargaining dynamics, incentives, and uncertainty. "
        "Then represent the full game in JSON, including one round explanation before the turns for each round. "
        "Use the game rules and the specific player valuations as analyst-side priors. "
        "Return only valid JSON."
    )
    user = "\n".join(
        [
            "Forecast one complete bargaining game from the chip-bargain study.",
            "",
            "# SESSION",
            f"- Chip family: {row['chip_family_display']} ({row['chip_family']})",
            f"- Cohort: {row['cohort_name']}",
            f"- Stage: {row['stage_name']}",
            f"- Experiment label: {row['experiment_name']}",
            "",
            "# PLAYERS",
            "\n".join(f"- `{player_id}`" for player_id in players),
            "",
            "# CHIP TYPES",
            _format_chip_catalog(chip_definitions),
            "",
            "# PLAYER-SPECIFIC CHIP VALUES",
            "Green chips are common-value. Red/blue/purple values differ by player as shown below.",
            _format_player_values(players, participant_chip_values),
            "",
            "# RULES",
            "- This is one standalone game. Treat it independently from any other game.",
            "- There are 3 rounds.",
            "- Each round has one proposal turn per player.",
            "- On a proposal turn, the proposer requests a positive quantity of one chip color and offers a positive quantity of a different chip color.",
            "- The proposer cannot offer more chips than they currently hold.",
            "- After the proposal is shown, all non-sender players respond simultaneously and privately with accept or decline.",
            "- If no one accepts, the trade is declined and holdings stay the same.",
            "- If exactly one player accepts, that player is the recipient and the trade executes.",
            "- If multiple players accept, one accepting player is chosen at random as the realized recipient; the proposer does not choose the partner.",
            "- After each turn, chip holdings update immediately before the next proposer acts.",
            "- Players observe the public trade history and current chip holdings, but each player knows only their own private chip values during play.",
            "- Each player's behavioral goal is to maximize their own total chip value, not group surplus.",
            "",
            "# PROPOSAL SCHEDULE FOR THIS GAME",
            "Follow this proposer schedule exactly. It is the exogenous order of proposal turns in this game:",
            _format_round_schedule(round_turn_orders),
            "",
            "# INTERPRETATION",
            "- You are writing an observer-view reconstruction of the full game.",
            "- Use the provided private values to infer what each player is likely to propose or accept, while respecting that the players themselves only know their own values during play.",
            "- Start with a top-level `game_explanation` describing the likely bargaining dynamics, incentives, and uncertainty for the full game.",
            "- For each round, include a `round_explanation` before the turns for that round.",
            *(["", profile_block] if profile_block else []),
            "",
            "# OUTPUT",
            "Return only JSON with this schema:",
            _output_example(players, round_turn_orders),
            "",
            "Requirements:",
            '- Include a top-level `"game_explanation"` string.',
            '- Include exactly 3 round objects with `"round"` equal to 0, 1, and 2.',
            '- Each round must include a `"round_explanation"` string.',
            '- Each round must contain exactly 3 turns with `"turn_index"` equal to 0, 1, and 2.',
            '- `"sender_id"` must follow the provided round-by-round proposer schedule exactly.',
            '- `"buy"` and `"sell"` must each contain at most one chip color and a positive integer quantity when present.',
            '- `"status"` must be either `"ACCEPTED"` or `"DECLINED"`.',
            '- `"recipient_id"` must be null for declined trades and a player id for accepted trades.',
            '- `"responses"` must include one entry for each non-sender player with an `"accepted"` boolean.',
            "- Do not include explanations or markdown outside the JSON.",
        ]
    )
    return system, user
