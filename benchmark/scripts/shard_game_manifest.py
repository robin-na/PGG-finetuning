#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a benchmark manifest CSV into stable shards for multi-GPU runs."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Source manifest CSV containing a game-id column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where shard CSV files will be written.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of output shards to write.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="gameId",
        help="Column used to assign rows to shards. Default: gameId.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="manifest",
        help="Filename prefix for output shards.",
    )
    return parser.parse_args()


def stable_shard_index(value: str, num_shards: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_shards


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")

    frame = pd.read_csv(args.input_csv)
    if args.column not in frame.columns:
        raise ValueError(f"Column '{args.column}' not found in {args.input_csv}.")
    if frame.empty:
        raise ValueError(f"Input manifest is empty: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = [[] for _ in range(args.num_shards)]

    for row_index, row in frame.iterrows():
        raw_value = row.get(args.column)
        shard_key = "" if pd.isna(raw_value) else str(raw_value).strip()
        if not shard_key:
            shard_key = f"row-{row_index}"
        shard_rows[stable_shard_index(shard_key, args.num_shards)].append(row_index)

    for shard_idx, row_indices in enumerate(shard_rows):
        shard_frame = frame.iloc[row_indices].copy() if row_indices else frame.iloc[0:0].copy()
        output_path = args.output_dir / f"{args.prefix}_shard_{shard_idx:02d}_of_{args.num_shards:02d}.csv"
        shard_frame.to_csv(output_path, index=False)
        print(f"{output_path}\trows={len(shard_frame)}")


if __name__ == "__main__":
    main()
