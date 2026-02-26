#!/usr/bin/env python3
"""
Retrieve nearest archetype text for a given D+E input using a trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from retrieval_common import (
    DEMOGRAPHIC_KEY_COLUMNS,
    coerce_feature_frame,
    is_tag_active,
    load_jsonl,
    normalize_key,
    normalize_rows,
)


def load_latest_run(output_root: Path) -> Path:
    latest = output_root / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(
            f"No latest run pointer found at {latest}. Pass --run-dir explicitly."
        )
    return Path(latest.read_text(encoding="utf-8").strip())


def load_feature_row_from_ids(
    game_id: str,
    player_id: str,
    demographics_csv: Path,
    environment_csv: Path,
    feature_columns: List[str],
) -> pd.DataFrame:
    demo = pd.read_csv(demographics_csv)
    env = pd.read_csv(environment_csv)

    demo["gameId"] = demo["gameId"].map(normalize_key)
    demo["playerId"] = demo["playerId"].map(normalize_key)
    env["gameId"] = env["gameId"].map(normalize_key)

    game_id = normalize_key(game_id)
    player_id = normalize_key(player_id)

    demo_row = demo[(demo["gameId"] == game_id) & (demo["playerId"] == player_id)]
    env_row = env[env["gameId"] == game_id]

    if env_row.empty:
        raise ValueError(f"gameId={game_id} not found in {environment_csv}")

    merged = pd.DataFrame([{"gameId": game_id, "playerId": player_id}])
    merged = merged.merge(demo, on=DEMOGRAPHIC_KEY_COLUMNS, how="left")
    merged = merged.merge(env, on="gameId", how="left")

    if demo_row.empty:
        print(
            f"Warning: demographics missing for gameId={game_id}, playerId={player_id}. "
            "Using config-only features where possible."
        )

    X = coerce_feature_frame(merged, feature_columns)
    return X


def load_feature_row_from_json(
    feature_json: Path,
    feature_columns: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    obj = json.loads(feature_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{feature_json} must contain a JSON object.")
    row = pd.DataFrame([obj])
    X = coerce_feature_frame(row, feature_columns)
    return X, obj


def top_k_neighbors(
    query_embedding: np.ndarray,
    bank_embeddings: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    q = normalize_rows(query_embedding.reshape(1, -1))
    b = normalize_rows(bank_embeddings)
    sim = (q @ b.T).reshape(-1)

    k = min(top_k, len(sim))
    if k <= 0:
        return {"indices": [], "scores": []}
    idx = np.argpartition(-sim, kth=k - 1)[:k]
    idx = idx[np.argsort(-sim[idx])]
    return {"indices": idx.tolist(), "scores": sim[idx].tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve nearest archetype text.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/model_runs"),
        help="Used only when --run-dir is omitted.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory. Defaults to latest_run.txt target.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--feature-json",
        type=Path,
        default=None,
        help="JSON file with feature values. Alternative to --game-id/--player-id.",
    )
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("demographics/demographics_numeric_learn.csv"),
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=Path("data/processed_data/df_analysis_learn.csv"),
    )
    parser.add_argument(
        "--allow-inactive-tag",
        action="store_true",
        help="Allow retrieval even when tag activation conditions are not met.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag.strip().upper()
    model_name = args.model.strip().lower()
    run_dir = args.run_dir if args.run_dir is not None else load_latest_run(args.output_root)

    model_path = run_dir / tag / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    feature_columns: List[str] = artifact.get("feature_columns", [])
    if not feature_columns:
        raise ValueError(f"Missing feature_columns in artifact: {model_path}")

    if args.feature_json is not None:
        X, input_row = load_feature_row_from_json(args.feature_json, feature_columns)
    else:
        if args.game_id is None or args.player_id is None:
            raise ValueError(
                "Provide either --feature-json, or both --game-id and --player-id."
            )
        X = load_feature_row_from_ids(
            game_id=args.game_id,
            player_id=args.player_id,
            demographics_csv=args.demographics_csv,
            environment_csv=args.environment_csv,
            feature_columns=feature_columns,
        )
        input_row = X.iloc[0].to_dict()

    if not args.allow_inactive_tag and not is_tag_active(tag, input_row):
        out = {
            "status": "inactive_tag",
            "tag": tag,
            "message": "Tag is inactive for the provided environment/config.",
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if args.output_json is not None:
            args.output_json.write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0

    if model_name == "mean":
        pred = artifact["baseline_mean_vector"].astype(np.float32).reshape(1, -1)
    else:
        est = artifact["estimator"]
        pred = est.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        pred = pred.astype(np.float32, copy=False)

    bank_embeddings = np.load(run_dir / tag / "bank_embeddings.npy")
    bank_metadata = load_jsonl(run_dir / tag / "bank_metadata.jsonl")

    nn = top_k_neighbors(pred[0], bank_embeddings, args.top_k)
    results: List[Dict[str, Any]] = []
    for idx, score in zip(nn["indices"], nn["scores"]):
        meta = bank_metadata[idx]
        results.append(
            {
                "rank": len(results) + 1,
                "cosine_similarity": float(score),
                "gameId": meta.get("gameId"),
                "playerId": meta.get("playerId"),
                "section_text": meta.get("section_text"),
            }
        )

    out = {
        "status": "ok",
        "run_dir": str(run_dir),
        "tag": tag,
        "model": model_name,
        "top_k": args.top_k,
        "results": results,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
