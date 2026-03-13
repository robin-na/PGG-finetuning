#!/usr/bin/env python3
"""Embed full PGG archetype texts using OpenAI text-embedding-3-large."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OUTPUT_ROOT,
    PGG_ARCHETYPE_JSONL,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_batch(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> np.ndarray:
    """Embed texts in batches, returning (N, dim) array."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace empty strings to avoid API errors
        batch = [t if t.strip() else "No information available." for t in batch]
        resp = client.embeddings.create(input=batch, model=model)
        batch_embs = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embs)
        if i + batch_size < len(texts):
            print(f"  Embedded {i + batch_size}/{len(texts)}...")
            time.sleep(0.1)  # light rate limit courtesy
    return np.array(all_embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed PGG archetype texts.")
    parser.add_argument(
        "--archetype-jsonl",
        type=Path,
        default=PGG_ARCHETYPE_JSONL,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading archetypes from {args.archetype_jsonl}...")
    archetypes = load_jsonl(args.archetype_jsonl)
    print(f"Loaded {len(archetypes)} archetypes.")

    texts = [a.get("text", "") for a in archetypes]
    metadata = [
        {
            "idx": i,
            "experiment": a.get("experiment", ""),
            "participant": a.get("participant", ""),
            "game_finished": a.get("game_finished"),
            "wave": a.get("_wave", ""),
        }
        for i, a in enumerate(archetypes)
    ]

    print(f"Embedding {len(texts)} texts with {EMBEDDING_MODEL}...")
    client = OpenAI()
    embeddings = embed_batch(client, texts, batch_size=args.batch_size)
    print(f"Embedding shape: {embeddings.shape}")
    assert embeddings.shape == (len(texts), EMBEDDING_DIM), (
        f"Expected ({len(texts)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.output_dir / "archetype_bank_embeddings.npy"
    meta_path = args.output_dir / "archetype_bank_metadata.jsonl"

    np.save(emb_path, embeddings)
    write_jsonl(meta_path, metadata)

    # Also save the full archetype texts for retrieval use
    texts_path = args.output_dir / "archetype_bank_texts.jsonl"
    text_rows = [
        {"idx": i, "text": texts[i]} for i in range(len(texts))
    ]
    write_jsonl(texts_path, text_rows)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Saved texts to {texts_path}")


if __name__ == "__main__":
    main()
