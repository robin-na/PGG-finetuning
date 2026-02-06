import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from pgg_prompt_utils import SYSTEM_PROMPT, build_user_prompt, load_config_map


def load_summary(summary_path):
    by_game = defaultdict(list)
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            game_id = str(obj.get("experiment"))
            by_game[game_id].append(obj)
    return by_game


def is_finished(obj):
    return obj.get("game_finished") is True


def filter_rows(rows, strategy):
    if strategy == "drop-games":
        if any(not is_finished(r) for r in rows):
            return None
        return rows
    if strategy == "drop-players":
        kept = [r for r in rows if is_finished(r)]
        return kept
    if strategy == "keep-all":
        return rows
    raise ValueError(f"Unknown strategy: {strategy}")


def build_record(
    game_id,
    rows,
    cfg_map,
    player_count_source,
    sort_participants,
    min_players,
    max_players,
):
    cfg = cfg_map.get(game_id)
    if cfg is None:
        return None

    if sort_participants:
        rows = sorted(rows, key=lambda r: str(r.get("participant", "")))

    types = []
    type_idx = 1
    for row in rows:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            types.append({"id": f"type_{type_idx}", "text": text.strip()})
            type_idx += 1

    if not types:
        return None

    n_players = len(types)
    if n_players < min_players:
        return None
    if max_players > 0 and n_players > max_players:
        return None

    cfg_for_prompt = dict(cfg)
    if player_count_source == "actual":
        cfg_for_prompt["playerCount"] = n_players

    user_prompt = build_user_prompt(cfg_for_prompt, n_players)
    assistant_text = json.dumps({"types": types}, ensure_ascii=True)

    record = {
        "game_id": game_id,
        "n_players": n_players,
        "config_player_count": cfg.get("playerCount"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
    }
    return record


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/summary_gpt51_learn.jsonl",
    )
    parser.add_argument(
        "--config",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/data/processed_data/df_analysis_learn.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Finetuning/data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--incomplete-strategy",
        choices=["drop-games", "drop-players", "keep-all"],
        default="drop-players",
        help="How to handle players with game_finished == False.",
    )
    parser.add_argument(
        "--player-count-source",
        choices=["config", "actual"],
        default="actual",
        help="Whether to use CONFIG_playerCount or the number of kept players in the prompt.",
    )
    parser.add_argument(
        "--sort-participants",
        action="store_true",
        help="Sort players by participant id before building the list of types.",
    )
    parser.add_argument("--min-players", type=int, default=0)
    parser.add_argument(
        "--max-players",
        type=int,
        default=0,
        help="If > 0, drop games with more than this many players.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_map = load_config_map(args.config)
    by_game = load_summary(args.summary)

    total_games = len(by_game)
    kept_games = 0
    skipped_games = 0
    dropped_players = 0

    records = []
    for game_id, rows in by_game.items():
        filtered = filter_rows(rows, args.incomplete_strategy)
        if filtered is None:
            skipped_games += 1
            continue
        if args.incomplete_strategy == "drop-players":
            dropped_players += sum(1 for r in rows if not is_finished(r))
        record = build_record(
            game_id,
            filtered,
            cfg_map,
            args.player_count_source,
            args.sort_participants,
            args.min_players,
            args.max_players,
        )
        if record is None:
            skipped_games += 1
            continue
        kept_games += 1
        records.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    split_idx = int(len(records) * (1 - args.val_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    train_path = output_dir / "persona_type_train.jsonl"
    val_path = output_dir / "persona_type_val.jsonl"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print(f"Total games: {total_games}")
    print(f"Kept games: {kept_games}")
    print(f"Skipped games: {skipped_games}")
    if args.incomplete_strategy == "drop-players":
        print(f"Dropped players: {dropped_players}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Train output: {train_path}")
    print(f"Val output: {val_path}")


if __name__ == "__main__":
    main()
