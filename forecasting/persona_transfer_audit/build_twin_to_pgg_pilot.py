from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
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

SYSTEM_PROMPT = (
    "You are behaving as a person with the given profile. Identify which player in the "
    "provided social interaction matches most closely with your personality."
)


@dataclass(frozen=True)
class PromptMetadata:
    game_id: str
    num_rounds: int
    endowment: int
    multiplier: float
    all_or_nothing: bool
    chat_enabled: bool
    default_contrib_prop: bool
    punishment_exists: bool
    punishment_cost: int
    punishment_magnitude: int
    reward_exists: bool
    reward_cost: int
    reward_magnitude: int
    show_n_rounds: bool
    show_other_summaries: bool
    show_punishment_id: bool
    show_reward_id: bool


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _format_num(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _coin_phrase(value: int) -> str:
    return f"{value} coin" if int(value) == 1 else f"{value} coins"


def _load_prompt_metadata(processed_path: Path) -> dict[str, PromptMetadata]:
    metadata: dict[str, PromptMetadata] = {}
    with processed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = str(row["gameId"])
            if game_id in metadata:
                continue
            metadata[game_id] = PromptMetadata(
                game_id=game_id,
                num_rounds=int(row["CONFIG_numRounds"]),
                endowment=int(round(float(row["CONFIG_endowment"]))),
                multiplier=float(row["CONFIG_multiplier"]),
                all_or_nothing=_as_bool(row["CONFIG_allOrNothing"]),
                chat_enabled=_as_bool(row["CONFIG_chat"]),
                default_contrib_prop=_as_bool(row["CONFIG_defaultContribProp"]),
                punishment_exists=_as_bool(row["CONFIG_punishmentExists"]),
                punishment_cost=int(round(float(row["CONFIG_punishmentCost"]))),
                punishment_magnitude=int(round(float(row["CONFIG_punishmentMagnitude"]))),
                reward_exists=_as_bool(row["CONFIG_rewardExists"]),
                reward_cost=int(round(float(row["CONFIG_rewardCost"]))),
                reward_magnitude=int(round(float(row["CONFIG_rewardMagnitude"]))),
                show_n_rounds=_as_bool(row["CONFIG_showNRounds"]),
                show_other_summaries=_as_bool(row["CONFIG_showOtherSummaries"]),
                show_punishment_id=_as_bool(row["CONFIG_showPunishmentId"]),
                show_reward_id=_as_bool(row["CONFIG_showRewardId"]),
            )
    return metadata


def _contribution_rule(metadata: PromptMetadata) -> str:
    if metadata.default_contrib_prop:
        if metadata.all_or_nothing:
            return (
                f"Each round, the {metadata.endowment} coins begin in the shared pot. "
                f"Each player chooses how many coins to withdraw for private use, so the resulting contribution must be either 0 or {metadata.endowment}."
            )
        return (
            f"Each round, the {metadata.endowment} coins begin in the shared pot. "
            "Each player chooses how many coins to withdraw for private use, and the remainder becomes their contribution. "
            f"Contributions are integers from 0 to {metadata.endowment}."
        )
    if metadata.all_or_nothing:
        return (
            f"Each player receives {metadata.endowment} coins in private holdings each round and must contribute either 0 or {metadata.endowment} to the shared pot."
        )
    return (
        f"Each player receives {metadata.endowment} coins in private holdings each round and chooses an integer contribution from 0 to {metadata.endowment}."
    )


def _mechanism_rules(metadata: PromptMetadata) -> list[str]:
    lines = []
    if metadata.punishment_exists and metadata.reward_exists:
        lines.append("After contributions are revealed and redistributed, players may punish and reward other players.")
        lines.append(
            f"Each punishment unit costs {_coin_phrase(metadata.punishment_cost)} to the source and deducts {_coin_phrase(metadata.punishment_magnitude)} from the target."
        )
        lines.append(
            f"Each reward unit costs {_coin_phrase(metadata.reward_cost)} to the source and adds {_coin_phrase(metadata.reward_magnitude)} to the target."
        )
    elif metadata.punishment_exists:
        lines.append("After contributions are revealed and redistributed, players may punish other players.")
        lines.append(
            f"Each punishment unit costs {_coin_phrase(metadata.punishment_cost)} to the source and deducts {_coin_phrase(metadata.punishment_magnitude)} from the target."
        )
    elif metadata.reward_exists:
        lines.append("After contributions are revealed and redistributed, players may reward other players.")
        lines.append(
            f"Each reward unit costs {_coin_phrase(metadata.reward_cost)} to the source and adds {_coin_phrase(metadata.reward_magnitude)} to the target."
        )
    return lines


def _visibility_rules(metadata: PromptMetadata) -> list[str]:
    lines = []
    if metadata.chat_enabled:
        lines.append("Players may send messages to the group whenever they choose to do so.")
    else:
        lines.append("Players cannot send group chat messages in this game.")
    if metadata.show_n_rounds:
        lines.append(f"Players know that the game lasts {metadata.num_rounds} rounds.")
    else:
        lines.append("Players do not know the total number of rounds while playing.")
    if metadata.punishment_exists and metadata.reward_exists:
        if metadata.show_punishment_id == metadata.show_reward_id:
            if metadata.show_punishment_id:
                lines.append("Players can identify who punished/rewarded them in their summary information.")
            else:
                lines.append("Players cannot identify who punished/rewarded them in their summary information.")
        else:
            lines.append(
                "Players can identify who punished them in their summary information."
                if metadata.show_punishment_id
                else "Players cannot identify who punished them in their summary information."
            )
            lines.append(
                "Players can identify who rewarded them in their summary information."
                if metadata.show_reward_id
                else "Players cannot identify who rewarded them in their summary information."
            )
    elif metadata.punishment_exists:
        lines.append(
            "Players can identify who punished them in their summary information."
            if metadata.show_punishment_id
            else "Players cannot identify who punished them in their summary information."
        )
    elif metadata.reward_exists:
        lines.append(
            "Players can identify who rewarded them in their summary information."
            if metadata.show_reward_id
            else "Players cannot identify who rewarded them in their summary information."
        )
    if metadata.punishment_exists and metadata.reward_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for punishing and rewarding other players, amount deducted from punishment, and amount received for reward."
        )
    elif metadata.punishment_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for punishing other players and amount deducted from punishment."
        )
    elif metadata.reward_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for rewarding other players and amount received for reward."
        )
    else:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round."
        )
    if metadata.show_other_summaries:
        lines.append("The corresponding information of the peers are also shown.")
    else:
        lines.append("However, the corresponding information of the peers are not shown.")
    return lines


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


def _load_exclusion_values(path: Path | None, column: str) -> set[str]:
    if path is None:
        return set()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Exclusion manifest not found: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return {str(row[column]) for row in csv.DictReader(handle) if row.get(column)}


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


def _estimate_tokens_from_chars(char_count: int) -> int:
    return int(round(char_count / 4))


def _request_input_char_count(request: dict[str, Any]) -> int:
    body = request["body"]
    messages = body.get("messages", [])
    message_chars = sum(len(str(message.get("content", ""))) for message in messages)
    schema_chars = len(json.dumps(body.get("response_format", {}), ensure_ascii=False))
    return message_chars + schema_chars


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


def _sample_games(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    sampling_mode: str,
    prompt_metadata: dict[str, PromptMetadata],
) -> list[dict[str, Any]]:
    if sampling_mode == "random":
        return _sample_rows(rows, n, seed)
    if sampling_mode == "one_per_treatment":
        by_treatment: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_treatment.setdefault(str(row["treatment_name"]), []).append(row)
        rng = random.Random(seed)
        selected: list[dict[str, Any]] = []
        for treatment_name in sorted(by_treatment):
            candidates = sorted(by_treatment[treatment_name], key=lambda row: str(row["game_id"]))
            selected.append(rng.choice(candidates))
        if n > 0 and n < len(selected):
            selected = _sample_rows(selected, n, seed)
        return selected
    if sampling_mode == "one_complete_per_treatment":
        by_treatment = {}
        for row in rows:
            by_treatment.setdefault(str(row["treatment_name"]), []).append(row)
        selected = []
        for treatment_name in sorted(by_treatment):
            candidates = sorted(
                by_treatment[treatment_name],
                key=lambda row: (
                    -int(
                        len(row.get("gold_rounds") or [])
                        == prompt_metadata[str(row["game_id"])].num_rounds
                    ),
                    -len(row.get("gold_rounds") or []),
                    str(row["game_id"]),
                ),
            )
            selected.append(candidates[0])
        if n > 0 and n < len(selected):
            selected = _sample_rows(selected, n, seed)
        return selected
    raise ValueError(f"Unsupported game sampling mode: {sampling_mode}")


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


def _render_game_rules(metadata: PromptMetadata, players: list[str]) -> str:
    interaction_format = []
    if metadata.punishment_exists and metadata.reward_exists:
        interaction_format.append(
            "`### PUNISHMENT/REWARD: <<[(SOURCE, TARGET, UNIT), ...]>>` lists actions after contributions. Positive UNIT means SOURCE rewarded TARGET; negative UNIT means SOURCE punished TARGET."
        )
    elif metadata.punishment_exists:
        interaction_format.append(
            "`### PUNISHMENT: <<[(SOURCE, TARGET, UNIT), ...]>>` lists punishment actions after contributions. SOURCE punished TARGET by UNIT units."
        )
    elif metadata.reward_exists:
        interaction_format.append(
            "`### REWARD: <<[(SOURCE, TARGET, UNIT), ...]>>` lists reward actions after contributions. SOURCE rewarded TARGET by UNIT units."
        )
    return "\n".join(
        [
            "# SOCIAL INTERACTION SCRIPT",
            "This is an online public goods game (PGG).",
            _contribution_rule(metadata),
            "Players do not see others' choices before deciding.",
            f"The shared pot is multiplied by {_format_num(metadata.multiplier)} and split equally among all active players.",
            *(_mechanism_rules(metadata)),
            *(_visibility_rules(metadata)),
            f"The observed players are: {', '.join(players)}.",
            (
                "`### CONTRIBUTIONS: <<[...]>>` gives the players' contributions for that round "
                "in the same order as the observed player list above."
            ),
            *interaction_format,
            "`<<[]>>` means no actions of that type occurred in that round.",
            "",
            "Below is the observed interaction.",
        ]
    )


def _render_user_prompt(
    persona: dict[str, Any],
    gold_row: dict[str, Any],
    metadata: Any,
    max_persona_chars: int,
    top_k: int,
) -> str:
    players = list(gold_row["players"])
    rendered_top_k = min(top_k, len(players))
    persona_summary = _trim_persona_summary(str(persona["persona_summary"]).strip(), max_persona_chars)
    return "\n\n".join(
        [
            "Below is information about yourself.",
            persona_summary,
            _render_game_rules(metadata, players),
            gold_row["gold_continuation_text"],
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
            "name": "persona_pgg_alignment",
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
                    "top_matches",
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

    prompt_metadata = _load_prompt_metadata(PROJECT_ROOT / "data/processed_data/df_analysis_val.csv")
    exclude_persona_ids = _load_exclusion_values(args.exclude_manifest, "persona_pid")
    exclude_game_ids = _load_exclusion_values(args.exclude_manifest, "game_id")
    persona_pool = [
        row for row in _read_jsonl(args.persona_summaries) if str(row["pid"]) not in exclude_persona_ids
    ]
    game_pool = [
        row
        for row in _eligible_games(_read_jsonl(args.source_gold), args.min_rounds, args.max_players)
        if str(row["game_id"]) not in exclude_game_ids
    ]
    personas = _sample_rows(persona_pool, args.num_personas, args.seed)
    games = _sample_games(
        game_pool,
        args.num_games,
        args.seed + 1,
        args.game_sampling_mode,
        prompt_metadata,
    )

    requests: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for persona_index, persona in enumerate(personas, start=1):
        for game_index, game in enumerate(games, start=1):
            custom_id = (
                f"persona_match__pid_{persona['pid']}__game_{game['game_id']}__"
                f"p{persona_index:03d}__g{game_index:03d}"
            )
            metadata = prompt_metadata[str(game["game_id"])]
            user_prompt = _render_user_prompt(
                persona,
                game,
                metadata,
                args.max_persona_chars,
                args.top_k,
            )
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
                    "response_format": _response_schema(game["players"], args.top_k),
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
                    "top_k": min(args.top_k, len(game["players"])),
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
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
    token_summary = {
        "method": "character_count_divided_by_4; tokenizer unavailable locally",
        "num_requests": len(token_values),
        "total_estimated_input_tokens": sum(token_values),
        "mean_estimated_input_tokens": sum(token_values) / len(token_values) if token_values else 0,
        "min_estimated_input_tokens": min(token_values) if token_values else 0,
        "max_estimated_input_tokens": max(token_values) if token_values else 0,
    }
    (metadata_dir / "request_token_estimates.json").write_text(
        json.dumps(token_summary, indent=2),
        encoding="utf-8",
    )
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
        "game_sampling_mode": args.game_sampling_mode,
        "top_k": args.top_k,
        "exclude_manifest": str(args.exclude_manifest.expanduser().resolve())
        if args.exclude_manifest
        else None,
        "persona_summaries": str(args.persona_summaries.expanduser().resolve()),
        "source_gold": str(args.source_gold.expanduser().resolve()),
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
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
    parser.add_argument(
        "--game-sampling-mode",
        choices=["random", "one_per_treatment", "one_complete_per_treatment"],
        default="random",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-persona-chars", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--persona-summaries", type=Path, default=DEFAULT_PERSONA_SUMMARIES)
    parser.add_argument("--source-gold", type=Path, default=DEFAULT_SOURCE_GOLD)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Optional request_manifest.csv whose persona_pid and game_id values should be excluded.",
    )
    return parser.parse_args()


def main() -> None:
    build_pilot(parse_args())


if __name__ == "__main__":
    main()
