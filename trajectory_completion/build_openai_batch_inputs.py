from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import GameTrajectory, RoundRecord, load_wave_games


ROUND_BLOCK_RE = re.compile(r"<ROUND i=\"\d+ of \d+\">.*?</ROUND>", re.DOTALL)

SYSTEM_PROMPT = (
    "You forecast the remaining rounds of an observed public goods game from an external observer "
    "perspective. Return only JSON that matches the provided schema. Do not add explanations."
)


@dataclass(frozen=True)
class TranscriptRecord:
    experiment: str
    participant: str
    perspective: str
    game_finished: bool
    text: str


def _parse_k_values(raw_value: str) -> list[int]:
    values: list[int] = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise ValueError("At least one k value is required.")
    return values


def _load_transcripts(transcript_path: Path) -> dict[str, TranscriptRecord]:
    transcripts: dict[str, TranscriptRecord] = {}
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            record = TranscriptRecord(
                experiment=str(raw["experiment"]),
                participant=str(raw["participant"]),
                perspective=str(raw["perspective"]),
                game_finished=bool(raw["game_finished"]),
                text=str(raw["text"]),
            )
            transcripts[record.experiment] = record
    return transcripts


def _load_game_player_order(repo_root: Path, wave_name: str) -> dict[str, list[str]]:
    games_path = repo_root / f"data/raw_data/{wave_name}/games.csv"
    player_order_by_game: dict[str, list[str]] = {}
    with games_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = str(row["_id"])
            player_ids_raw = str(row.get("playerIds", "")).strip()
            if not player_ids_raw:
                continue
            player_order_by_game[game_id] = [player_id.strip() for player_id in player_ids_raw.split(",") if player_id.strip()]
    return player_order_by_game


def _transcript_player_names(full_text: str) -> list[str]:
    match = re.search(r"<PLAYERS>\s*([^<]+)\s*</PLAYERS>", full_text)
    if match is None:
        raise ValueError("Transcript is missing a <PLAYERS> block.")
    return [player_name.strip() for player_name in match.group(1).split(",") if player_name.strip()]


def _split_transcript_prefix(full_text: str, k: int) -> tuple[str, int]:
    matches = list(ROUND_BLOCK_RE.finditer(full_text))
    if len(matches) < k:
        raise ValueError(f"Transcript has only {len(matches)} rounds, cannot keep prefix of {k}.")
    prefix_end = matches[k - 1].end()
    header = full_text[: matches[0].start()].rstrip()
    prefix_rounds = full_text[matches[0].start() : prefix_end].rstrip()
    total_rounds = len(matches)
    prefix_text = (
        f"{header}\n"
        f"{prefix_rounds}\n"
        f"# OBSERVED PREFIX ENDS AFTER ROUND {k} OF {total_rounds}"
    )
    return prefix_text, total_rounds


def _player_prediction_schema(
    game: GameTrajectory,
    output_players: list[str],
    include_punish: bool,
    include_reward: bool,
) -> dict[str, Any]:
    action_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": "string", "enum": output_players},
            "units": {"type": "integer", "minimum": 1, "maximum": game.config.endowment},
        },
        "required": ["target", "units"],
    }
    if game.config.all_or_nothing:
        contribution_schema: dict[str, Any] = {"type": "integer", "enum": [0, game.config.endowment]}
    else:
        contribution_schema = {"type": "integer", "minimum": 0, "maximum": game.config.endowment}

    punish_schema: dict[str, Any] = {"type": "array", "items": action_item, "maxItems": max(game.num_players - 1, 0)}
    reward_schema: dict[str, Any] = {"type": "array", "items": action_item, "maxItems": max(game.num_players - 1, 0)}
    if not include_punish:
        punish_schema = {"type": "array", "maxItems": 0, "items": action_item}
    if not include_reward:
        reward_schema = {"type": "array", "maxItems": 0, "items": action_item}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "player": {"type": "string", "enum": output_players},
            "contribution": contribution_schema,
            "punish": punish_schema,
            "reward": reward_schema,
        },
        "required": ["player", "contribution", "punish", "reward"],
    }


def _response_schema(game: GameTrajectory, k: int, output_players: list[str]) -> dict[str, Any]:
    remaining_rounds = list(range(k + 1, game.num_rounds + 1))
    player_prediction_schema = _player_prediction_schema(
        game,
        output_players=output_players,
        include_punish=game.config.punishment_exists,
        include_reward=game.config.reward_exists,
    )
    return {
        "name": "trajectory_completion_forecast",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "game_id": {"type": "string", "enum": [game.game_id]},
                "observed_rounds": {"type": "integer", "enum": [k]},
                "predicted_rounds": {
                    "type": "array",
                    "minItems": len(remaining_rounds),
                    "maxItems": len(remaining_rounds),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "round_number": {"type": "integer", "enum": remaining_rounds},
                            "player_predictions": {
                                "type": "array",
                                "minItems": game.num_players,
                                "maxItems": game.num_players,
                                "items": player_prediction_schema,
                            },
                        },
                        "required": ["round_number", "player_predictions"],
                    },
                },
            },
            "required": ["game_id", "observed_rounds", "predicted_rounds"],
        },
    }


def _format_output_contract(game: GameTrajectory, k: int, output_players: list[str]) -> str:
    remaining_rounds = list(range(k + 1, game.num_rounds + 1))
    return "\n".join(
        [
            "TASK",
            f"- Predict every remaining round from round {k + 1} through round {game.num_rounds}.",
            "- Use the observer transcript prefix exactly as given below.",
            "- Do not predict chat messages. Predict only contributions, punishments, and rewards.",
            "",
            "STRICT OUTPUT RULES",
            "- Return JSON only. No markdown. No explanation.",
            f"- Keep `game_id` exactly `{game.game_id}`.",
            f"- Keep `observed_rounds` exactly {k}.",
            f"- Output exactly {len(remaining_rounds)} objects in `predicted_rounds`.",
            f"- `round_number` values must be {remaining_rounds}.",
            f"- In every predicted round, include exactly these players in `player_predictions`: {output_players}.",
            f"- Each contribution must be an integer from 0 to {game.config.endowment}.",
            (
                f"- This is all-or-nothing, so every contribution must be either 0 or {game.config.endowment}."
                if game.config.all_or_nothing
                else "- This is not all-or-nothing."
            ),
            (
                "- Punishment is enabled. `punish` must list outgoing punishment actions by that player."
                if game.config.punishment_exists
                else "- Punishment is disabled. Every `punish` array must be empty."
            ),
            (
                "- Reward is enabled. `reward` must list outgoing reward actions by that player."
                if game.config.reward_exists
                else "- Reward is disabled. Every `reward` array must be empty."
            ),
            "- Never target the acting player themself.",
            "- Use integer `units >= 1` for every listed action.",
            "- If a player sends no punishment or no reward, use an empty array for that field.",
            "- Preserve exact player names.",
        ]
    )


def _build_user_prompt(game: GameTrajectory, transcript_prefix: str, k: int, output_players: list[str]) -> str:
    mechanism_lines = [
        "GAME SNAPSHOT",
        f"- game_id: {game.game_id}",
        f"- total_rounds: {game.num_rounds}",
        f"- observed_rounds: {k}",
        f"- remaining_rounds: {game.num_rounds - k}",
        f"- players: {', '.join(output_players)}",
        f"- endowment_per_round: {game.config.endowment}",
        f"- all_or_nothing: {str(game.config.all_or_nothing).lower()}",
        f"- punishment_enabled: {str(game.config.punishment_exists).lower()}",
        f"- reward_enabled: {str(game.config.reward_exists).lower()}",
        "",
        _format_output_contract(game, k, output_players),
        "",
        "OBSERVED TRANSCRIPT PREFIX",
        transcript_prefix,
    ]
    return "\n".join(mechanism_lines)


def _round_record_to_prediction(
    round_record: RoundRecord,
    raw_player_order: list[str],
    raw_to_display: dict[str, str],
) -> dict[str, Any]:
    player_predictions = []
    for player_id in raw_player_order:
        punish = [
            {"target": raw_to_display[target], "units": units}
            for target, units in sorted(round_record.punished[player_id].items())
        ]
        reward = [
            {"target": raw_to_display[target], "units": units}
            for target, units in sorted(round_record.rewarded[player_id].items())
        ]
        player_predictions.append(
            {
                "player": raw_to_display[player_id],
                "contribution": round_record.contributions[player_id],
                "punish": punish,
                "reward": reward,
            }
        )
    return {
        "round_number": round_record.index + 1,
        "player_predictions": player_predictions,
    }


def _gold_continuation(
    game: GameTrajectory,
    k: int,
    raw_player_order: list[str],
    raw_to_display: dict[str, str],
) -> dict[str, Any]:
    return {
        "game_id": game.game_id,
        "observed_rounds": k,
        "predicted_rounds": [
            _round_record_to_prediction(round_record, raw_player_order, raw_to_display)
            for round_record in game.rounds[k:]
        ],
    }


def _batch_entry(
    *,
    game: GameTrajectory,
    k: int,
    model: str,
    transcript_prefix: str,
    output_players: list[str],
    max_completion_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    custom_id = f"trajectory_completion__{game.game_id}__k{k}"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": _response_schema(game, k, output_players),
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(game, transcript_prefix, k, output_players)},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _split_config(split: str) -> tuple[str, str, Path]:
    if split == "val":
        return (
            "validation_wave",
            "val",
            Path("Persona/transcripts_observer_val.jsonl"),
        )
    if split == "learn":
        return (
            "learning_wave",
            "learn",
            Path("Persona/transcripts_observer_learn.jsonl"),
        )
    raise ValueError(f"Unsupported split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenAI Batch inputs for trajectory completion prompts.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", type=str, choices=["val", "learn"], default="val")
    parser.add_argument("--k-values", type=str, default="1,3,5,8")
    parser.add_argument("--min-num-rounds-exclusive", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=16000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "batch_inputs" / "validation_wave_complete_gt10_k1358_observer",
    )
    parser.add_argument("--limit-games", type=int, default=0)
    args = parser.parse_args()

    wave_name, processed_suffix, transcript_rel_path = _split_config(args.split)
    transcript_path = args.repo_root / transcript_rel_path
    k_values = _parse_k_values(args.k_values)

    games = load_wave_games(
        repo_root=args.repo_root,
        wave_name=wave_name,
        processed_suffix=processed_suffix,
        min_num_rounds_exclusive=args.min_num_rounds_exclusive,
    )
    transcripts = _load_transcripts(transcript_path)
    player_order_by_game = _load_game_player_order(args.repo_root, wave_name)
    eligible_games = [game for game in games if game.game_id in transcripts]
    if args.limit_games > 0:
        eligible_games = eligible_games[: args.limit_games]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "sample_prompts"
    samples_dir.mkdir(parents=True, exist_ok=True)

    all_batch_rows: list[dict[str, Any]] = []
    all_gold_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for k in k_values:
        batch_rows: list[dict[str, Any]] = []
        gold_rows: list[dict[str, Any]] = []
        sample_prompt_written = False

        for game in eligible_games:
            if game.num_rounds <= k:
                continue
            transcript = transcripts[game.game_id]
            raw_player_order = player_order_by_game.get(game.game_id)
            if raw_player_order is None:
                continue
            display_players = _transcript_player_names(transcript.text)
            if len(raw_player_order) != len(display_players):
                continue
            raw_to_display = dict(zip(raw_player_order, display_players))
            transcript_prefix, transcript_rounds = _split_transcript_prefix(transcript.text, k)
            if transcript_rounds != game.num_rounds:
                continue

            batch_row = _batch_entry(
                game=game,
                k=k,
                model=args.model,
                transcript_prefix=transcript_prefix,
                output_players=display_players,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
            )
            gold_row = {
                "custom_id": batch_row["custom_id"],
                "game_id": game.game_id,
                "k": k,
                "gold_continuation": _gold_continuation(game, k, raw_player_order, raw_to_display),
            }
            batch_rows.append(batch_row)
            gold_rows.append(gold_row)
            metadata_rows.append(
                {
                    "custom_id": batch_row["custom_id"],
                    "game_id": game.game_id,
                    "split": args.split,
                    "k": k,
                    "num_rounds": game.num_rounds,
                    "num_players": game.num_players,
                    "prompt_player_names": json.dumps(display_players),
                    "all_or_nothing": game.config.all_or_nothing,
                    "punishment_enabled": game.config.punishment_exists,
                    "reward_enabled": game.config.reward_exists,
                    "game_finished_flag_in_transcript": transcript.game_finished,
                }
            )

            if not sample_prompt_written:
                prompt_preview = "\n\n".join(
                    [
                        "=== SYSTEM MESSAGE ===",
                        SYSTEM_PROMPT,
                        "=== USER MESSAGE ===",
                        batch_row["body"]["messages"][1]["content"],
                    ]
                )
                (samples_dir / f"sample_prompt_k{k}.txt").write_text(prompt_preview, encoding="utf-8")
                sample_prompt_written = True

        _write_jsonl(args.output_dir / f"openai_batch_k{k}.jsonl", batch_rows)
        _write_jsonl(args.output_dir / f"gold_continuations_k{k}.jsonl", gold_rows)
        all_batch_rows.extend(batch_rows)
        all_gold_rows.extend(gold_rows)

    _write_jsonl(args.output_dir / "openai_batch_all_k.jsonl", all_batch_rows)
    _write_jsonl(args.output_dir / "gold_continuations_all_k.jsonl", all_gold_rows)

    metadata_path = args.output_dir / "request_manifest.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        if metadata_rows:
            writer.writeheader()
            writer.writerows(metadata_rows)

    manifest = {
        "repo_root": str(args.repo_root),
        "split": args.split,
        "wave_name": wave_name,
        "processed_suffix": processed_suffix,
        "transcript_path": str(transcript_path),
        "k_values": k_values,
        "min_num_rounds_exclusive": args.min_num_rounds_exclusive,
        "model": args.model,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "games_loaded_after_filters": len(eligible_games),
        "requests_by_k": {
            str(k): int(sum(game.num_rounds > k and game.game_id in transcripts for game in eligible_games))
            for k in k_values
        },
        "total_requests": len(all_batch_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote batch inputs to {args.output_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
