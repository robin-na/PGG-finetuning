#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a stable subset of games from a manifest CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Source manifest CSV containing a game-id column.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV for the sampled subset.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        required=True,
        help="Maximum number of unique games to keep.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="gameId",
        help="Column used to identify unique games. Default: gameId.",
    )
    parser.add_argument(
        "--salt",
        type=str,
        default="",
        help="Optional salt added before hashing to choose a different stable subset.",
    )
    return parser.parse_args()


def stable_hash(value: str, salt: str) -> str:
    return hashlib.md5(f"{salt}\n{value}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_games <= 0:
        raise ValueError("--max-games must be positive.")

    frame = pd.read_csv(args.input_csv)
    if args.column not in frame.columns:
        raise ValueError(f"Column '{args.column}' not found in {args.input_csv}.")
    if frame.empty:
        raise ValueError(f"Input manifest is empty: {args.input_csv}")

    key_series = frame[args.column].fillna("").astype(str).str.strip()
    key_to_hash = {}
    for row_index, key in enumerate(key_series.tolist()):
        chosen = key if key else f"row-{row_index}"
        key_to_hash.setdefault(chosen, stable_hash(chosen, args.salt))

    ordered_keys = sorted(key_to_hash, key=lambda item: (key_to_hash[item], item))
    selected_keys = set(ordered_keys[: args.max_games])

    selected_rows = []
    for row_index, key in enumerate(key_series.tolist()):
        chosen = key if key else f"row-{row_index}"
        if chosen in selected_keys:
            selected_rows.append(row_index)

    sampled = frame.iloc[selected_rows].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)

    print(f"{args.output_csv}\trows={len(sampled)}\tunique_games={min(len(selected_keys), len(ordered_keys))}")


if __name__ == "__main__":
    main()
