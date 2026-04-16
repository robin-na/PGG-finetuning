from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..build_batch_inputs import (
    _contribution_rule,
    _empty_chat_index,
    _index_chat_log,
    _interaction_tag_name,
    _load_avatar_map,
    _load_game_rows,
    _load_prompt_metadata,
    _mechanism_rules,
    _visibility_rules,
)


def _parse_action_dict(value: Any) -> dict[str, int]:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str):
        try:
            raw = json.loads(value)
        except Exception:
            try:
                raw = ast.literal_eval(value)
            except Exception:
                return {}
    else:
        return {}

    parsed: dict[str, int] = {}
    if not isinstance(raw, dict):
        return parsed
    for key, units in raw.items():
        try:
            unit_value = int(round(float(units)))
        except (TypeError, ValueError):
            continue
        if unit_value > 0:
            parsed[str(key)] = unit_value
    return parsed


def _load_round_rows(repo_root: Path, wave_name: str) -> pd.DataFrame:
    player_rounds = pd.read_csv(
        repo_root / f"data/raw_data/{wave_name}/player-rounds.csv",
        usecols=[
            "playerId",
            "roundId",
            "gameId",
            "data.punished",
            "data.rewarded",
            "data.contribution",
            "data.roundPayoff",
        ],
    ).rename(
        columns={
            "data.punished": "punished_raw",
            "data.rewarded": "rewarded_raw",
            "data.contribution": "contribution",
            "data.roundPayoff": "round_payoff",
        }
    )
    rounds = pd.read_csv(repo_root / f"data/raw_data/{wave_name}/rounds.csv", usecols=["_id", "index"])
    merged = player_rounds.merge(
        rounds,
        left_on="roundId",
        right_on="_id",
        how="left",
        validate="many_to_one",
    )
    return merged.sort_values(["gameId", "index", "playerId"]).reset_index(drop=True)


def _player_avatar_order(
    game_id: str,
    game_rows: dict[str, dict[str, Any]],
    avatar_map: dict[str, str],
    game_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    raw_player_order = list(game_rows.get(game_id, {}).get("player_order", []))
    if not raw_player_order:
        raw_player_order = sorted(game_df["playerId"].astype(str).unique().tolist())
    active_mask = game_df["contribution"].notna() | game_df["round_payoff"].notna()
    active_players = set(game_df.loc[active_mask, "playerId"].astype(str).unique().tolist())
    raw_player_order = [player_id for player_id in raw_player_order if player_id in active_players]
    if not raw_player_order:
        raw_player_order = sorted(active_players)
    avatar_order: list[str] = []
    used_avatars: set[str] = set()
    for idx, player_id in enumerate(raw_player_order, start=1):
        avatar = str(avatar_map.get(player_id, f"PLAYER_{idx}")).strip().upper()
        if avatar in used_avatars:
            suffix = 2
            candidate = f"{avatar}_{suffix}"
            while candidate in used_avatars:
                suffix += 1
                candidate = f"{avatar}_{suffix}"
            avatar = candidate
        used_avatars.add(avatar)
        avatar_order.append(avatar)
    if len(set(avatar_order)) != len(avatar_order):
        raise ValueError(f"Game {game_id} has duplicate avatar names within the selected roster.")
    return raw_player_order, avatar_order


def _format_contribution_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _render_contributions(round_df: pd.DataFrame, raw_player_order: list[str]) -> str:
    by_player = {
        str(row.playerId): row.contribution
        for row in round_df.itertuples(index=False)
    }
    values = [_format_contribution_value(by_player.get(player_id)) for player_id in raw_player_order]
    return f"<<[{', '.join(values)}]>>"


def _render_interactions(
    round_df: pd.DataFrame,
    metadata: Any,
    raw_player_order: list[str],
    raw_to_avatar: dict[str, str],
) -> str:
    rows_by_player = {str(row.playerId): row for row in round_df.itertuples(index=False)}
    tuples: list[str] = []
    for source_id in raw_player_order:
        row = rows_by_player.get(source_id)
        if row is None:
            continue
        punished = _parse_action_dict(getattr(row, "punished_raw"))
        rewarded = _parse_action_dict(getattr(row, "rewarded_raw"))
        for target_id in raw_player_order:
            if target_id == source_id:
                continue
            punish_units = punished.get(target_id, 0)
            reward_units = rewarded.get(target_id, 0)
            if punish_units > 0:
                unit_value = -int(punish_units) if metadata.reward_exists else int(punish_units)
                tuples.append(
                    f"({raw_to_avatar[source_id]}, {raw_to_avatar[target_id]}, {unit_value})"
                )
            if reward_units > 0:
                tuples.append(
                    f"({raw_to_avatar[source_id]}, {raw_to_avatar[target_id]}, {int(reward_units)})"
                )
    return f"<<[{', '.join(tuples)}]>>" if tuples else "<<[]>>"


def _render_chat_lines(
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    round_number: int,
    phase: str,
) -> list[str]:
    return [
        f"CHAT from {avatar}: {text}"
        for avatar, text in chat_index.get(round_number, {}).get(phase, [])
    ]


def _render_round_block(
    *,
    round_number: int,
    round_df: pd.DataFrame,
    metadata: Any,
    raw_player_order: list[str],
    raw_to_avatar: dict[str, str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    chat_enabled: bool,
    interaction_tag: str | None,
) -> str:
    lines = [f"## ROUND {round_number} BEGINS"]
    if chat_enabled:
        lines.extend(_render_chat_lines(chat_index, round_number, "contribution"))
    lines.append(f"### CONTRIBUTIONS: {_render_contributions(round_df, raw_player_order)}")
    if chat_enabled:
        lines.extend(_render_chat_lines(chat_index, round_number, "outcome"))
    if interaction_tag is not None:
        lines.append(
            f"### {interaction_tag}: {_render_interactions(round_df, metadata, raw_player_order, raw_to_avatar)}"
        )
    lines.append(f"### ROUND {round_number} SUMMARY SHOWN TO PLAYERS")
    if chat_enabled:
        lines.extend(_render_chat_lines(chat_index, round_number, "summary"))
    return "\n".join(lines)


def _build_transcript_text(
    *,
    game_df: pd.DataFrame,
    metadata: Any,
    raw_player_order: list[str],
    avatar_order: list[str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
) -> str:
    raw_to_avatar = dict(zip(raw_player_order, avatar_order))
    interaction_tag = _interaction_tag_name(metadata)
    lines = [
        "# GAME RULES",
        "This is an online public goods game (PGG).",
        _contribution_rule(metadata),
        "Players do not see others' choices before deciding.",
        f"The shared pot is multiplied by {metadata.multiplier:g} and split equally among all active players.",
        *(_mechanism_rules(metadata)),
        *(_visibility_rules(metadata)),
        "",
        "# GAME STARTS",
        f"<PLAYERS> {', '.join(avatar_order)} </PLAYERS>",
    ]

    for round_index, round_df in game_df.groupby("index", sort=True):
        lines.append(
            _render_round_block(
                round_number=int(round_index) + 1,
                round_df=round_df.sort_values("playerId").copy(),
                metadata=metadata,
                raw_player_order=raw_player_order,
                raw_to_avatar=raw_to_avatar,
                chat_index=chat_index,
                chat_enabled=metadata.chat_enabled,
                interaction_tag=interaction_tag,
            )
        )
    return "\n".join(lines)


def _game_finished(game_df: pd.DataFrame) -> bool:
    return bool(game_df["round_payoff"].notna().all())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_wave_transcripts(
    *,
    repo_root: Path,
    wave_name: str,
    processed_suffix: str,
    output_path: Path,
) -> dict[str, Any]:
    round_rows = _load_round_rows(repo_root, wave_name)
    prompt_metadata = _load_prompt_metadata(repo_root / f"data/processed_data/df_analysis_{processed_suffix}.csv")
    avatar_map = _load_avatar_map(repo_root / f"data/raw_data/{wave_name}/players.csv")
    game_rows = _load_game_rows(repo_root / f"data/raw_data/{wave_name}/games.csv")

    rows: list[dict[str, Any]] = []
    selected_game_ids = sorted(
        [game_id for game_id in prompt_metadata if game_id in game_rows and game_id in set(round_rows["gameId"].astype(str))],
        key=lambda game_id: (game_rows[game_id]["created_at"], game_id),
    )
    round_rows["gameId"] = round_rows["gameId"].astype(str)

    for game_id in selected_game_ids:
        game_df = round_rows[round_rows["gameId"] == game_id].copy()
        if game_df.empty:
            continue
        metadata = prompt_metadata[game_id]
        raw_player_order, avatar_order = _player_avatar_order(game_id, game_rows, avatar_map, game_df)
        chat_index = _index_chat_log(metadata.chat_log) if metadata.chat_enabled else _empty_chat_index()
        rows.append(
            {
                "experiment": game_id,
                "participant": "OBSERVER",
                "perspective": "observer",
                "game_finished": _game_finished(game_df),
                "text": _build_transcript_text(
                    game_df=game_df,
                    metadata=metadata,
                    raw_player_order=raw_player_order,
                    avatar_order=avatar_order,
                    chat_index=chat_index,
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    return {
        "wave_name": wave_name,
        "processed_suffix": processed_suffix,
        "output_path": str(output_path),
        "num_transcripts": len(rows),
        "game_finished_true": sum(1 for row in rows if row["game_finished"]),
        "game_finished_false": sum(1 for row in rows if not row["game_finished"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact observer transcripts matching the trajectory-completion prompt format."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--learn-output",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Persona/transcripts_observer_compact_learn.jsonl",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Persona/transcripts_observer_compact_val.jsonl",
    )
    args = parser.parse_args()

    learn_manifest = _build_wave_transcripts(
        repo_root=args.repo_root,
        wave_name="learning_wave",
        processed_suffix="learn",
        output_path=args.learn_output,
    )
    val_manifest = _build_wave_transcripts(
        repo_root=args.repo_root,
        wave_name="validation_wave",
        processed_suffix="val",
        output_path=args.val_output,
    )
    print(json.dumps({"learn": learn_manifest, "val": val_manifest}, indent=2))


if __name__ == "__main__":
    main()
