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

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE_GOLD = (
    PROJECT_ROOT
    / "forecasting/pgg/metadata/twin_sampled_seed_0_gpt_5_mini/gold_continuations_k0.jsonl"
)
DEFAULT_PERSONA_SUMMARIES = (
    PROJECT_ROOT / "forecasting/simbench/cache/twin_persona_summary_cache.jsonl"
)

SYSTEM_PROMPT = """You are auditing how a persona maps onto revealed behavior in a public goods game.

Your task is not to judge which player behaved best. Your task is to identify which observed player is the closest behavioral match for the provided persona, based on the full game transcript."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trim_persona_summary(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED FOR PILOT PROMPT]"


def _sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[index] for index in indices]


def _eligible_games(rows: list[dict[str, Any]], min_rounds: int, max_players: int) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        rounds = row.get("gold_rounds") or []
        players = row.get("players") or []
        if len(rounds) < min_rounds:
            continue
        if max_players > 0 and len(players) > max_players:
            continue
        if not str(row.get("gold_continuation_text", "")).strip():
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError(
            f"No eligible games after filtering for min_rounds={min_rounds}, max_players={max_players}."
        )
    return eligible


def _contribution_rule(gold_row: dict[str, Any]) -> str:
    values = []
    for round_row in gold_row.get("gold_rounds", []):
        values.extend(round_row.get("contributions", []))
    max_contribution = max(values) if values else "the endowment"
    unique_values = sorted({int(value) for value in values if isinstance(value, int)})
    if len(unique_values) <= 2 and unique_values and unique_values[0] == 0:
        return f"Each player chooses a contribution each round; in this game the observed contribution values are {unique_values}."
    return f"Each player chooses an integer contribution each round, usually from 0 to {max_contribution}."


def _interaction_rule(gold_row: dict[str, Any]) -> str:
    interactions = [
        interaction
        for round_row in gold_row.get("gold_rounds", [])
        for interaction in round_row.get("interactions", [])
    ]
    if not interactions:
        return "This observed transcript contains no punishment/reward action lines."
    has_negative = any(len(item) == 3 and int(item[2]) < 0 for item in interactions)
    if has_negative:
        return "After contributions, players may punish or reward other players; negative units are punishment and positive units are reward."
    return "After contributions, players may punish or reward other players when action lines appear in the transcript."


def _messages_rule(gold_row: dict[str, Any]) -> str:
    has_messages = any(round_row.get("messages") for round_row in gold_row.get("gold_rounds", []))
    if has_messages:
        return "Players may send group chat messages; chat lines are part of the revealed behavior."
    return "This observed game has no group chat messages in the transcript."


def _render_game_context(gold_row: dict[str, Any]) -> str:
    players = ", ".join(gold_row["players"])
    return "\n".join(
        [
            "# TARGET GAME CONTEXT",
            f"Game ID: {gold_row['game_id']}",
            f"Treatment: {gold_row['treatment_name']}",
            f"Rounds: {len(gold_row.get('gold_rounds', []))}",
            f"Players: {players}",
            _contribution_rule(gold_row),
            _interaction_rule(gold_row),
            _messages_rule(gold_row),
        ]
    )


def _render_user_prompt(persona: dict[str, Any], gold_row: dict[str, Any], max_persona_chars: int) -> str:
    players = gold_row["players"]
    score_keys = ", ".join(players)
    persona_summary = _trim_persona_summary(str(persona["persona_summary"]).strip(), max_persona_chars)
    return "\n\n".join(
        [
            "# PERSONA",
            f"Persona ID: {persona['pid']}",
            persona_summary,
            _render_game_context(gold_row),
            "# OBSERVED GAME TRANSCRIPT",
            gold_row["gold_continuation_text"],
            "# TASK",
            (
                "Rank the observed players by how closely their revealed behavior matches the persona. "
                "Use contribution behavior, changes over rounds, punishment/reward behavior, responsiveness to others, "
                "and communication style when present. Do not select the player you morally prefer. Select the player "
                "whose behavior this persona would most plausibly have produced in this target game."
            ),
            "# OUTPUT",
            (
                "Return only valid JSON. `player_rankings` must list every player exactly once from most aligned to least aligned. "
                f"`alignment_scores` must contain exactly these player keys: {score_keys}. Scores should be numbers from 0 to 1."
            ),
        ]
    )


def _response_schema(players: list[str]) -> dict[str, Any]:
    score_properties = {player: {"type": "number", "minimum": 0, "maximum": 1} for player in players}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "persona_pgg_alignment",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "most_aligned_player": {"type": "string", "enum": players},
                    "least_aligned_player": {"type": "string", "enum": players},
                    "player_rankings": {
                        "type": "array",
                        "items": {"type": "string", "enum": players},
                        "minItems": len(players),
                        "maxItems": len(players),
                    },
                    "alignment_scores": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": score_properties,
                        "required": players,
                    },
                    "behavioral_basis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 6,
                    },
                    "uncertainty_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 4,
                    },
                },
                "required": [
                    "most_aligned_player",
                    "least_aligned_player",
                    "player_rankings",
                    "alignment_scores",
                    "behavioral_basis",
                    "uncertainty_notes",
                ],
            },
        },
    }


def build_pilot(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    run_name = args.run_name or (
        f"twin_direct_summary_to_pgg_pilot__n{args.num_personas}_x{args.num_games}__"
        f"{_sanitize_token(args.model)}__seed_{args.seed}"
    )
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    personas = _sample_rows(_read_jsonl(args.persona_summaries), args.num_personas, args.seed)
    games = _sample_rows(
        _eligible_games(_read_jsonl(args.source_gold), args.min_rounds, args.max_players),
        args.num_games,
        args.seed + 1,
    )

    requests: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for persona_index, persona in enumerate(personas, start=1):
        for game_index, game in enumerate(games, start=1):
            custom_id = (
                f"persona_match__pid_{persona['pid']}__game_{game['game_id']}__"
                f"p{persona_index:03d}__g{game_index:03d}"
            )
            user_prompt = _render_user_prompt(persona, game, args.max_persona_chars)
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
                    "response_format": _response_schema(game["players"]),
                },
            }
            if args.temperature is not None:
                request["body"]["temperature"] = args.temperature
            requests.append(request)
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "persona_pid": persona["pid"],
                    "persona_source": "twin_direct_persona_summary",
                    "game_id": game["game_id"],
                    "treatment_name": game["treatment_name"],
                    "config_id": game.get("config_id", ""),
                    "num_rounds": len(game.get("gold_rounds", [])),
                    "players": json.dumps(game["players"]),
                    "model": args.model,
                    "max_persona_chars": args.max_persona_chars,
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "audit": "twin_direct_persona_summary_to_pgg_player_matching",
        "model": args.model,
        "seed": args.seed,
        "num_personas": len(personas),
        "num_games": len(games),
        "num_requests": len(requests),
        "min_rounds": args.min_rounds,
        "max_players": args.max_players,
        "persona_summaries": str(args.persona_summaries.expanduser().resolve()),
        "source_gold": str(args.source_gold.expanduser().resolve()),
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(requests)} requests")
    print(f"Batch input: {batch_input_file}")
    print(f"Manifest: {metadata_dir / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Twin direct-summary to PGG matching pilot batch.")
    parser.add_argument("--output-root", type=Path, default=THIS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-personas", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--max-persona-chars", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--persona-summaries", type=Path, default=DEFAULT_PERSONA_SUMMARIES)
    parser.add_argument("--source-gold", type=Path, default=DEFAULT_SOURCE_GOLD)
    return parser.parse_args()


def main() -> None:
    build_pilot(parse_args())


if __name__ == "__main__":
    main()
