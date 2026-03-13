#!/usr/bin/env python3
"""Retrieve top-K PGG archetype candidates for each Twin-2k-500 participant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from config import D_FEATURE_COLUMNS, OUTPUT_ROOT, TOP_K


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-K archetype candidates.")
    parser.add_argument(
        "--twin-demographics",
        type=Path,
        default=OUTPUT_ROOT / "twin2k_demographics_pgg_compatible.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=OUTPUT_ROOT / "d_only_model" / "d_only_ridge.joblib",
    )
    parser.add_argument(
        "--bank-embeddings",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_embeddings.npy",
    )
    parser.add_argument(
        "--bank-metadata",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_metadata.jsonl",
    )
    parser.add_argument(
        "--bank-texts",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_texts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "candidate_archetypes.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    # Load model
    print("Loading D-only Ridge model...")
    artifact = joblib.load(args.model_path)
    estimator = artifact["estimator"]
    feature_columns = artifact["feature_columns"]

    # Load Twin demographics
    print("Loading Twin-2k-500 demographics...")
    twin_df = pd.read_csv(args.twin_demographics)
    print(f"  {len(twin_df)} participants")

    # Load archetype bank
    print("Loading archetype bank...")
    bank_emb = np.load(args.bank_embeddings)
    bank_meta = load_jsonl(args.bank_metadata)
    bank_texts = load_jsonl(args.bank_texts)
    print(f"  Bank: {bank_emb.shape[0]} archetypes, {bank_emb.shape[1]} dims")

    # Build text lookup
    text_by_idx = {r["idx"]: r["text"] for r in bank_texts}

    # Prepare feature matrix
    X = twin_df[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Predict embeddings for all participants
    print("Predicting embeddings...")
    pred_emb = estimator.predict(X).astype(np.float64)
    # Clamp any inf/nan from numerical overflow
    pred_emb = np.nan_to_num(pred_emb, nan=0.0, posinf=1e6, neginf=-1e6)

    # Compute cosine similarity (batch)
    print("Computing cosine similarities...")
    pred_n = normalize_rows(pred_emb.astype(np.float64))
    bank_n = normalize_rows(bank_emb.astype(np.float64))
    sim_matrix = pred_n @ bank_n.T  # (N_twin, N_bank)
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

    # Extract top-K for each participant
    print(f"Extracting top-{args.top_k} candidates...")
    k = min(args.top_k, bank_emb.shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(len(twin_df)):
            sims = sim_matrix[i]
            if k <= 1:
                top_idx = np.array([np.argmax(sims)])
            else:
                top_idx = np.argpartition(-sims, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]

            candidates = []
            for rank, idx in enumerate(top_idx):
                meta = bank_meta[idx]
                candidates.append({
                    "rank": rank + 1,
                    "cosine_similarity": float(sims[idx]),
                    "bank_idx": int(idx),
                    "experiment": meta.get("experiment", ""),
                    "participant": meta.get("participant", ""),
                    "archetype_text": text_by_idx.get(int(idx), ""),
                })

            row = {
                "pid": int(twin_df.iloc[i]["pid"]),
                "sex_label": str(twin_df.iloc[i].get("sex_label", "")),
                "age_label": str(twin_df.iloc[i].get("age_label", "")),
                "education_label": str(twin_df.iloc[i].get("education_label", "")),
                "candidates": candidates,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved candidates to {args.output}")

    # Summary statistics
    top1_sims = sim_matrix[np.arange(len(twin_df)), np.argmax(sim_matrix, axis=1)]
    print(f"\nTop-1 cosine similarity: mean={np.mean(top1_sims):.4f}, "
          f"min={np.min(top1_sims):.4f}, max={np.max(top1_sims):.4f}")


if __name__ == "__main__":
    main()
