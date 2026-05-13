"""Build persona-to-player matching batches for Twin -> chip bargaining."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.datasets.chip_bargain import build_bundle


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "forecasting/persona_transfer_audit"
DEFAULT_PERSONA_SUMMARIES = (
    PROJECT_ROOT / "forecasting/simbench/cache/twin_persona_summary_cache.jsonl"
)

SYSTEM_PROMPT = (
    "You are behaving as a person with the given profile. Identify which player in the "
    "provided social interaction matches most closely with your personality."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trim_persona_summary(text: str, max_chars: int) -> str:
    text = re.sub(
        r"^\s*The following is a description of a person\.\s*",
        "",
        text,
        count=1,
    )
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED FOR PILOT PROMPT]"


def _sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[index] for index in indices]


def _load_exclusion_values(path: Path | None, column: str) -> set[str]:
    if path is None:
        return set()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Exclusion manifest not found: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return {str(row[column]) for row in csv.DictReader(handle) if row.get(column)}


def _sample_games(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    mode: str,
    games_per_treatment: int,
) -> list[dict[str, Any]]:
    if mode == "random":
        return _sample_rows(rows, n, seed)
    if mode not in {"one_per_treatment", "n_per_treatment"}:
        raise ValueError(f"Unsupported game sampling mode: {mode}")
    by_treatment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_treatment.setdefault(str(row["treatment_name"]), []).append(row)
    rng = random.Random(seed)
    selected = []
    for group in by_treatment.values():
        sorted_group = sorted(group, key=lambda item: str(item["record_id"]))
        if mode == "one_per_treatment":
            selected.append(rng.choice(sorted_group))
        else:
            if games_per_treatment <= 0:
                raise ValueError("--games-per-treatment must be positive for n_per_treatment mode.")
            if games_per_treatment > len(sorted_group):
                raise ValueError(
                    f"Requested {games_per_treatment} games from treatment "
                    f"{sorted_group[0]['treatment_name']}, but only {len(sorted_group)} are available."
                )
            selected.extend(rng.sample(sorted_group, games_per_treatment))
    selected = sorted(selected, key=lambda item: (str(item["treatment_name"]), str(item["record_id"])))
    return selected if n <= 0 else selected[:n]


def _estimate_tokens_from_chars(char_count: int) -> int:
    return int(round(char_count / 4))


def _request_input_char_count(request: dict[str, Any]) -> int:
    body = request["body"]
    messages = body.get("messages", [])
    message_chars = sum(len(str(message.get("content", ""))) for message in messages)
    schema_chars = len(json.dumps(body.get("response_format", {}), ensure_ascii=False))
    return message_chars + schema_chars


def _format_chip_catalog(chip_definitions: list[dict[str, Any]]) -> str:
    lines = []
    for chip in chip_definitions:
        lines.append(
            "- {name} (`{id}`): each player starts with {qty}; public value range ${low:.2f} to ${high:.2f}".format(
                name=chip["name"],
                id=chip["id"],
                qty=int(chip["starting_quantity"]),
                low=float(chip["lower_value"]),
                high=float(chip["upper_value"]),
            )
        )
    return "\n".join(lines)


def _format_player_values(players: list[str], values: dict[str, dict[str, float]]) -> str:
    lines = []
    for player_id in players:
        value_text = ", ".join(
            f"{chip_id}=${float(value):.2f}" for chip_id, value in sorted(values[player_id].items())
        )
        lines.append(f"- `{player_id}`: {value_text}")
    return "\n".join(lines)


def _format_map(value: dict[str, int]) -> str:
    if not value:
        return "{}"
    return "{" + ", ".join(f"{color}: {quantity}" for color, quantity in sorted(value.items())) + "}"


def _format_responses(turn: dict[str, Any]) -> str:
    parts = []
    for player_id, response in sorted(turn["responses"].items()):
        accepted = "accepted" if bool(response["accepted"]) else "declined"
        selected = " selected as trade partner" if bool(response["selected_as_recipient"]) else ""
        parts.append(f"{player_id}: {accepted}{selected}")
    return "; ".join(parts)


def _format_state_change(turn: dict[str, Any], players: list[str]) -> str:
    before = turn["player_states_before"]
    after = turn["player_states_after"]
    pieces = []
    for player_id in players:
        before_payout = float(before[player_id]["payout"])
        after_payout = float(after[player_id]["payout"])
        delta = after_payout - before_payout
        if abs(delta) > 1e-9:
            pieces.append(f"{player_id}: payout {before_payout:.2f} -> {after_payout:.2f} ({delta:+.2f})")
    return "; ".join(pieces) if pieces else "No payout changes."


def _render_transcript(target: dict[str, Any]) -> str:
    players = [str(player_id) for player_id in target["players"]]
    lines = ["Below is the observed interaction."]
    for round_data in target["rounds"]:
        lines.append("")
        lines.append(f"ROUND {int(round_data['round'])}")
        for turn in round_data["turns"]:
            status = str(turn["status"]).upper()
            recipient = turn["recipient_id"] if turn["recipient_id"] is not None else "none"
            lines.extend(
                [
                    f"TURN {int(turn['turn_index'])} PROPOSER: {turn['sender_id']}",
                    f"Proposal: proposer offers {_format_map(turn['sell'])} to buy {_format_map(turn['buy'])}.",
                    f"Responses: {_format_responses(turn)}.",
                    f"Outcome: {status}; realized trade partner: {recipient}.",
                    f"Payout change from this turn: {_format_state_change(turn, players)}",
                ]
            )
    return "\n".join(lines)


def _render_game_context(target: dict[str, Any]) -> str:
    players = [str(player_id) for player_id in target["players"]]
    return "\n".join(
        [
            "# SOCIAL INTERACTION SCRIPT",
            "This is an online chip-bargaining game.",
            "Three players bargain by proposing chip trades over 3 rounds.",
            "Each round has one proposal turn per player, following the observed proposer schedule.",
            "On a proposal turn, the proposer requests a positive quantity of one chip color and offers a positive quantity of a different chip color.",
            "All non-proposers privately accept or decline. If exactly one player accepts, the trade executes with that player. If multiple players accept, one accepting player is randomly selected as the trade partner. If no one accepts, no trade occurs.",
            "Players observe the trade history and current chip holdings, but during the game each player knows only their own private chip values. The values below are shown to you so you can interpret the incentives behind each revealed proposal and response.",
            "",
            "# CHIP TYPES",
            _format_chip_catalog(target["chip_definitions"]),
            "",
            "# PLAYER-SPECIFIC CHIP VALUES",
            _format_player_values(players, target["participant_chip_values"]),
            "",
            "# OBSERVED PLAYERS",
            "\n".join(f"- `{player_id}`" for player_id in players),
        ]
    )


def _render_user_prompt(
    persona: dict[str, Any],
    target: dict[str, Any],
    max_persona_chars: int,
    top_k: int,
) -> str:
    players = list(target["players"])
    rendered_top_k = min(top_k, len(players))
    persona_summary = _trim_persona_summary(str(persona["persona_summary"]).strip(), max_persona_chars)
    return "\n\n".join(
        [
            "Below is information about yourself.",
            persona_summary,
            _render_game_context(target),
            _render_transcript(target),
            (
                f"Rank up to {rendered_top_k} observed players by how closely their revealed behavior matches "
                "your personality, the behavior you would most plausibly have produced in this game."
            ),
            (
                "Return only valid JSON. Put the closest match first. Assign each listed player a probability "
                "from 0 to 1, and make the probabilities across the listed players sum to exactly 1. "
                "Any player not listed is treated as having probability 0."
            ),
        ]
    )


def _response_schema(players: list[str], top_k: int) -> dict[str, Any]:
    rendered_top_k = min(top_k, len(players))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chip_bargain_persona_match",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "top_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "player": {"type": "string", "enum": players},
                                "probability": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["player", "probability"],
                        },
                        "minItems": 1,
                        "maxItems": rendered_top_k,
                    },
                    "behavioral_basis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                    "uncertainty_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 3,
                    },
                },
                "required": ["top_matches", "behavioral_basis", "uncertainty_notes"],
            },
        },
    }


def _load_chip_records(repo_root: Path) -> list[dict[str, Any]]:
    bundle = build_bundle(repo_root)
    rows = []
    for row in bundle.records.to_dict(orient="records"):
        target = json.loads(str(row["gold_target_json"]))
        rows.append({**row, "target": target})
    return rows


def build_batch(args: argparse.Namespace) -> None:
    run_name = args.run_name
    output_root = args.output_root.expanduser().resolve()
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    exclude_persona_ids = _load_exclusion_values(args.exclude_manifest, "persona_pid")
    persona_pool = [
        row for row in _read_jsonl(args.persona_summaries) if str(row["pid"]) not in exclude_persona_ids
    ]
    personas = _sample_rows(persona_pool, args.num_personas, args.seed)
    chip_rows = _sample_games(
        _load_chip_records(args.repo_root.expanduser().resolve()),
        args.num_games,
        args.seed + 1,
        args.game_sampling_mode,
        args.games_per_treatment,
    )

    requests: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for persona_index, persona in enumerate(personas, start=1):
        for game_index, game in enumerate(chip_rows, start=1):
            target = game["target"]
            custom_id = (
                f"persona_chip_match__pid_{persona['pid']}__record_{game['record_id']}__"
                f"p{persona_index:03d}__g{game_index:03d}"
            )
            user_prompt = _render_user_prompt(persona, target, args.max_persona_chars, args.top_k)
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": _response_schema(list(target["players"]), args.top_k),
                },
            }
            if args.temperature is not None:
                request["body"]["temperature"] = args.temperature
            requests.append(request)
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "persona_pid": str(persona["pid"]),
                    "persona_source": "twin_direct_persona_summary",
                    "record_id": str(game["record_id"]),
                    "unit_id": str(game["unit_id"]),
                    "treatment_name": str(game["treatment_name"]),
                    "chip_family": str(game["chip_family"]),
                    "chip_family_display": str(game["chip_family_display"]),
                    "cohort_name": str(game["cohort_name"]),
                    "stage_name": str(game["stage_name"]),
                    "stage_code": str(game["stage_code"]),
                    "experiment_name": str(game["experiment_name"]),
                    "players": json.dumps(target["players"]),
                    "model": args.model,
                    "max_persona_chars": args.max_persona_chars,
                    "top_k": min(args.top_k, len(target["players"])),
                }
            )
            gold_rows.append(
                {
                    "custom_id": custom_id,
                    "record_id": str(game["record_id"]),
                    "treatment_name": str(game["treatment_name"]),
                    "gold_target": target,
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
    _write_jsonl(metadata_dir / "gold_targets.jsonl", gold_rows)

    token_rows = []
    for request in requests:
        input_chars = _request_input_char_count(request)
        token_rows.append(
            {
                "custom_id": request["custom_id"],
                "estimated_input_chars": input_chars,
                "estimated_input_tokens_char4": _estimate_tokens_from_chars(input_chars),
            }
        )
    _write_csv(metadata_dir / "request_token_estimates.csv", token_rows)
    token_values = [row["estimated_input_tokens_char4"] for row in token_rows]
    (metadata_dir / "request_token_estimates.json").write_text(
        json.dumps(
            {
                "method": "character_count_divided_by_4",
                "num_requests": len(token_values),
                "total_estimated_input_tokens": sum(token_values),
                "mean_estimated_input_tokens": sum(token_values) / len(token_values) if token_values else 0,
                "min_estimated_input_tokens": min(token_values) if token_values else 0,
                "max_estimated_input_tokens": max(token_values) if token_values else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )

    manifest = {
        "dataset_key": "persona_transfer_audit_chip_bargain",
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "num_personas": len(personas),
        "num_games": len(chip_rows),
        "num_requests": len(requests),
        "game_sampling_mode": args.game_sampling_mode,
        "games_per_treatment": args.games_per_treatment,
        "top_k": args.top_k,
        "exclude_manifest": str(args.exclude_manifest.expanduser().resolve())
        if args.exclude_manifest
        else None,
        "persona_summaries": str(args.persona_summaries.expanduser().resolve()),
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "gold_targets": str(metadata_dir / "gold_targets.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="twin_direct_summary_to_chip_bargain_pilot__n8_x6__gpt_5_mini__seed_0",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--num-personas", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=0)
    parser.add_argument(
        "--game-sampling-mode",
        choices=["random", "one_per_treatment", "n_per_treatment"],
        default="one_per_treatment",
    )
    parser.add_argument("--games-per-treatment", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-persona-chars", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--persona-summaries", type=Path, default=DEFAULT_PERSONA_SUMMARIES)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Optional request_manifest.csv whose persona_pid values should be excluded.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_batch(parse_args())
