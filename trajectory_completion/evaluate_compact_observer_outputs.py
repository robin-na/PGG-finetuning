from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import simulate_round
from .build_compact_observer_batch_inputs import _load_avatar_map, _load_game_rows
from .data import GameTrajectory, RoundRecord, load_wave_games
from .evaluate import _evaluate_game_rollout, _summarize_game, _summarize_overall


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_request_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest_df = pd.read_csv(path)
    manifest: dict[str, dict[str, Any]] = {}
    for row in manifest_df.to_dict(orient="records"):
        row["k"] = int(row["k"])
        row["num_rounds"] = int(row["num_rounds"])
        row["num_players"] = int(row["num_players"])
        row["avatars"] = ast.literal_eval(row["avatars"]) if isinstance(row.get("avatars"), str) else []
        manifest[str(row["custom_id"])] = row
    return manifest


def _build_avatar_mapping(
    *,
    game_id: str,
    expected_avatars: list[str],
    game_rows: dict[str, dict[str, Any]],
    avatar_map: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    raw_player_order = list(game_rows[game_id]["player_order"])
    actual_avatar_order = [avatar_map[player_id] for player_id in raw_player_order]
    if expected_avatars and actual_avatar_order != expected_avatars:
        raise ValueError(
            f"Avatar order mismatch for game {game_id}: expected {expected_avatars}, found {actual_avatar_order}."
        )
    return raw_player_order, dict(zip(actual_avatar_order, raw_player_order))


def _parsed_rounds_to_round_records(
    *,
    game: GameTrajectory,
    k: int,
    raw_player_order: list[str],
    avatar_to_player: dict[str, str],
    parsed_rounds: list[dict[str, Any]],
) -> list[RoundRecord]:
    payload_by_round_number = {
        int(round_payload["round_number"]): round_payload for round_payload in parsed_rounds
    }
    predicted_rounds: list[RoundRecord] = []

    for actual_round in game.rounds[k:]:
        round_number = actual_round.index + 1
        if round_number not in payload_by_round_number:
            raise ValueError(f"Missing predicted round {round_number}.")
        payload = payload_by_round_number[round_number]
        contribution_values = payload.get("contributions")
        if not isinstance(contribution_values, list):
            raise ValueError(f"Round {round_number} is missing a contribution list.")
        if len(contribution_values) != len(raw_player_order):
            raise ValueError(
                f"Round {round_number} contribution count mismatch: expected {len(raw_player_order)}, "
                f"found {len(contribution_values)}."
            )

        contributions = {
            player_id: int(contribution_values[index])
            for index, player_id in enumerate(raw_player_order)
        }
        punished = {player_id: {} for player_id in game.players}
        rewarded = {player_id: {} for player_id in game.players}

        for interaction in payload.get("interactions") or []:
            if not isinstance(interaction, list) or len(interaction) != 3:
                raise ValueError(f"Round {round_number} contains a malformed interaction: {interaction!r}")
            source_avatar, target_avatar, unit_value = interaction
            if source_avatar not in avatar_to_player:
                raise ValueError(f"Unknown source avatar {source_avatar!r} in round {round_number}.")
            if target_avatar not in avatar_to_player:
                raise ValueError(f"Unknown target avatar {target_avatar!r} in round {round_number}.")
            source_id = avatar_to_player[source_avatar]
            target_id = avatar_to_player[target_avatar]
            unit = int(unit_value)
            if unit > 0:
                rewarded[source_id][target_id] = rewarded[source_id].get(target_id, 0) + unit
            elif unit < 0:
                punished[source_id][target_id] = punished[source_id].get(target_id, 0) + abs(unit)

        predicted_rounds.append(
            simulate_round(
                game,
                round_index=actual_round.index,
                contributions=contributions,
                punished=punished,
                rewarded=rewarded,
            )
        )

    return predicted_rounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parsed compact observer outputs with the same metrics as baseline trajectory completion."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", type=str, choices=["learn", "val"], default="val")
    parser.add_argument("--parsed-output-jsonl", type=Path, required=True)
    parser.add_argument("--request-manifest-csv", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="compact_observer_llm")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    wave_name = "learning_wave" if args.split == "learn" else "validation_wave"
    processed_suffix = "learn" if args.split == "learn" else "val"

    games = load_wave_games(
        repo_root=args.repo_root,
        wave_name=wave_name,
        processed_suffix=processed_suffix,
        min_num_rounds_exclusive=0,
    )
    games_by_id = {game.game_id: game for game in games}

    avatar_map = _load_avatar_map(args.repo_root / f"data/raw_data/{wave_name}/players.csv")
    game_rows = _load_game_rows(args.repo_root / f"data/raw_data/{wave_name}/games.csv")
    request_manifest = _load_request_manifest(args.request_manifest_csv)
    parsed_rows = _read_jsonl(args.parsed_output_jsonl)

    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    request_status_rows: list[dict[str, object]] = []

    for parsed_row in parsed_rows:
        custom_id = str(parsed_row.get("custom_id", ""))
        manifest_row = request_manifest.get(custom_id)
        if manifest_row is None:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "evaluated": False,
                    "reason": "missing_manifest_row",
                    "parse_success": bool(parsed_row.get("parse_success")),
                }
            )
            continue

        game_id = str(manifest_row["game_id"])
        game = games_by_id.get(game_id)
        if game is None:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": "missing_game",
                    "parse_success": bool(parsed_row.get("parse_success")),
                }
            )
            continue

        if not bool(parsed_row.get("parse_success")):
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": "parse_failed",
                    "parse_success": False,
                    "parse_errors": json.dumps(parsed_row.get("parse_errors", []), ensure_ascii=False),
                    "validation_errors": json.dumps(parsed_row.get("validation_errors", []), ensure_ascii=False),
                }
            )
            continue

        try:
            raw_player_order, avatar_to_player = _build_avatar_mapping(
                game_id=game_id,
                expected_avatars=list(manifest_row["avatars"]),
                game_rows=game_rows,
                avatar_map=avatar_map,
            )
            predicted_rounds = _parsed_rounds_to_round_records(
                game=game,
                k=int(manifest_row["k"]),
                raw_player_order=raw_player_order,
                avatar_to_player=avatar_to_player,
                parsed_rounds=list(parsed_row.get("predicted_rounds", [])),
            )
            game_actor_rows, game_round_rows = _evaluate_game_rollout(
                game=game,
                baseline_name=args.baseline_name,
                k=int(manifest_row["k"]),
                predicted_rounds=predicted_rounds,
            )
            actor_rows.extend(game_actor_rows)
            round_rows.extend(game_round_rows)
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": True,
                    "reason": "",
                    "parse_success": True,
                    "k": int(manifest_row["k"]),
                    "num_predicted_rounds": len(predicted_rounds),
                }
            )
        except Exception as exc:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": f"conversion_failed: {exc}",
                    "parse_success": bool(parsed_row.get("parse_success")),
                    "k": int(manifest_row["k"]),
                }
            )

    if not actor_rows or not round_rows:
        raise RuntimeError("No parsed outputs could be evaluated.")

    actor_df = pd.DataFrame(actor_rows)
    round_df = pd.DataFrame(round_rows)
    status_df = pd.DataFrame(request_status_rows)

    game_summary_rows = []
    for (_, k, game_id), group in actor_df.groupby(["baseline", "k", "game_id"], sort=True):
        game_round_group = round_df[
            (round_df["baseline"] == args.baseline_name)
            & (round_df["k"] == k)
            & (round_df["game_id"] == game_id)
        ]
        game_summary_rows.append(_summarize_game(group, game_round_group))

    game_summary_df = pd.DataFrame(game_summary_rows).sort_values(["k", "game_id"]).reset_index(drop=True)
    overall_df = _summarize_overall(actor_df, round_df)
    future_efficiency_df = (
        game_summary_df.groupby(["baseline", "k"], as_index=False)[
            [
                "actual_future_relative_efficiency",
                "predicted_future_relative_efficiency",
                "future_relative_efficiency_abs_error",
                "actual_future_normalized_efficiency",
                "predicted_future_normalized_efficiency",
                "future_normalized_efficiency_abs_error",
            ]
        ]
        .mean()
    )
    overall_df = overall_df.merge(future_efficiency_df, on=["baseline", "k"], how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_df.to_csv(args.output_dir / "actor_level_predictions.csv", index=False)
    round_df.to_csv(args.output_dir / "round_level_predictions.csv", index=False)
    game_summary_df.to_csv(args.output_dir / "game_summary.csv", index=False)
    overall_df.to_csv(args.output_dir / "overall_summary.csv", index=False)
    status_df.to_csv(args.output_dir / "request_status.csv", index=False)

    manifest = {
        "repo_root": str(args.repo_root),
        "split": args.split,
        "wave_name": wave_name,
        "processed_suffix": processed_suffix,
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "baseline_name": args.baseline_name,
        "records_in_input": len(parsed_rows),
        "evaluated_requests": int(status_df["evaluated"].fillna(False).sum()),
        "skipped_requests": int((~status_df["evaluated"].fillna(False)).sum()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(overall_df.to_string(index=False))
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    main()
