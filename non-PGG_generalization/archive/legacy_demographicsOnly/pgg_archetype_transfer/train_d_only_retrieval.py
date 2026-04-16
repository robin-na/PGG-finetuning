#!/usr/bin/env python3
"""Train a D-only Ridge regression model mapping demographics → archetype embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    D_FEATURE_COLUMNS,
    OUTPUT_ROOT,
    PGG_DEMOGRAPHICS_CSV,
    RANDOM_STATE,
    RIDGE_ALPHA,
)


def normalize_key(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def retrieval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bank_vectors: np.ndarray,
    top_k: int = 5,
) -> Dict[str, float]:
    y_true_n = normalize_rows(y_true)
    y_pred_n = normalize_rows(y_pred)
    bank_n = normalize_rows(bank_vectors)

    sim_pred_bank = y_pred_n @ bank_n.T
    pred_idx = np.argmax(sim_pred_bank, axis=1)

    sim_true_bank = y_true_n @ bank_n.T
    oracle_idx = np.argmax(sim_true_bank, axis=1)

    k = min(top_k, bank_n.shape[0])
    if k <= 1:
        topk_idx = pred_idx.reshape(-1, 1)
    else:
        topk_idx = np.argpartition(-sim_pred_bank, kth=k - 1, axis=1)[:, :k]

    hit_at_1 = float(np.mean(pred_idx == oracle_idx))
    hit_at_k = float(
        np.mean([oracle_idx[i] in topk_idx[i] for i in range(len(oracle_idx))])
    )

    pred_true_cos = np.sum(y_pred_n * y_true_n, axis=1)
    retrieved_true_cos = np.sum(bank_n[pred_idx] * y_true_n, axis=1)

    return {
        "pred_true_cosine_mean": float(np.mean(pred_true_cos)),
        "retrieved_true_cosine_mean": float(np.mean(retrieved_true_cos)),
        "oracle_hit_at_1": hit_at_1,
        f"oracle_hit_at_{k}": hit_at_k,
        "embedding_mse_mean": float(np.mean((y_true - y_pred) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D-only Ridge retrieval model.")
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=PGG_DEMOGRAPHICS_CSV,
    )
    parser.add_argument(
        "--embeddings-npy",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_embeddings.npy",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_metadata.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "d_only_model",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Load demographics
    print("Loading demographics...")
    demo = pd.read_csv(args.demographics_csv)
    demo["gameId"] = demo["gameId"].map(normalize_key)
    demo["playerId"] = demo["playerId"].map(normalize_key)
    print(f"  Demographics: {len(demo)} rows")

    # Load embeddings + metadata
    print("Loading embeddings and metadata...")
    embeddings = np.load(args.embeddings_npy)
    meta_rows: List[Dict[str, Any]] = []
    with args.metadata_jsonl.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                meta_rows.append(json.loads(s))
    print(f"  Embeddings: {embeddings.shape}")
    assert len(meta_rows) == embeddings.shape[0]

    # Build a joined table: metadata + demographics
    meta_df = pd.DataFrame(meta_rows)
    meta_df["gameId"] = meta_df["experiment"].map(normalize_key)
    meta_df["playerId"] = meta_df["participant"].map(normalize_key)

    merged = meta_df.merge(demo, on=["gameId", "playerId"], how="inner")
    print(f"  Merged (inner join): {len(merged)} / {len(meta_df)} archetypes have demographics")

    # Align embeddings to merged rows
    keep_idx = merged["idx"].values
    y = embeddings[keep_idx].astype(np.float32)
    X = merged[D_FEATURE_COLUMNS].copy()
    for col in D_FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    groups = merged["gameId"]

    print(f"\nTraining D-only Ridge (alpha={RIDGE_ALPHA})...")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Unique games: {groups.nunique()}")

    # Cross-validation
    n_splits = min(args.cv_splits, groups.nunique())
    splitter = GroupKFold(n_splits=n_splits)

    cv_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
        ])
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test).astype(np.float32)

        metrics = retrieval_metrics(y_test, y_pred, y_train, top_k=args.top_k)
        metrics["fold"] = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        cv_results.append(metrics)
        print(f"  Fold {fold_idx}: cos={metrics['pred_true_cosine_mean']:.4f} "
              f"hit@1={metrics['oracle_hit_at_1']:.4f}")

    cv_df = pd.DataFrame(cv_results)

    # Also train a mean baseline for comparison
    mean_vec = y.mean(axis=0, keepdims=True)
    mean_preds = np.repeat(mean_vec, len(y), axis=0)
    mean_metrics = retrieval_metrics(y, mean_preds, y, top_k=args.top_k)
    print(f"\n  Mean baseline: cos={mean_metrics['pred_true_cosine_mean']:.4f}")

    # Fit full model on all data
    print("\nFitting full model on all data...")
    full_est = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])
    full_est.fit(X, y)

    # Save artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "d_only_ridge.joblib"
    artifact = {
        "artifact_version": 1,
        "model_name": "ridge",
        "feature_columns": D_FEATURE_COLUMNS,
        "estimator": full_est,
        "ridge_alpha": RIDGE_ALPHA,
        "n_train": len(X),
        "embedding_dim": int(y.shape[1]),
    }
    joblib.dump(artifact, model_path)

    cv_path = args.output_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)

    summary = {
        "n_archetypes_total": len(meta_rows),
        "n_archetypes_with_demographics": len(merged),
        "n_unique_games": int(groups.nunique()),
        "cv_folds": n_splits,
        "cv_mean_pred_true_cosine": float(cv_df["pred_true_cosine_mean"].mean()),
        "cv_mean_hit_at_1": float(cv_df["oracle_hit_at_1"].mean()),
        "mean_baseline_pred_true_cosine": mean_metrics["pred_true_cosine_mean"],
        "mean_baseline_hit_at_1": mean_metrics["oracle_hit_at_1"],
    }
    summary_path = args.output_dir / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\nSaved model to {model_path}")
    print(f"Saved CV results to {cv_path}")
    print(f"Saved summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
