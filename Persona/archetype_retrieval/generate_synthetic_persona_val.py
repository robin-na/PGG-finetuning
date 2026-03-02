#!/usr/bin/env python3
"""
Generate synthetic validation-wave persona summaries using trained ridge models.

Output format matches Persona/summary_gpt51_val.jsonl rows:
- experiment
- participant
- game_finished
- text  (multi-section persona with <TAG> headers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from retrieval_common import coerce_feature_frame, is_tag_active, load_jsonl, normalize_key
from validate_archetype_retrieval import load_latest_run


TAG_OUTPUT_ORDER = [
    "CONTRIBUTION",
    "COMMUNICATION",
    "RESPONSE_TO_END_GAME",
    "PUNISHMENT",
    "RESPONSE_TO_PUNISHER",
    "REWARD",
    "RESPONSE_TO_REWARDER",
    "RESPONSE_TO_OTHERS_OUTCOME",
]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def load_feature_tables(
    demo_csv: Path,
    env_csv: Path,
    base_players: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    demo = pd.read_csv(demo_csv)
    env = pd.read_csv(env_csv)
    demo["gameId"] = demo["gameId"].map(normalize_key)
    demo["playerId"] = demo["playerId"].map(normalize_key)
    env["gameId"] = env["gameId"].map(normalize_key)

    demo = demo.drop_duplicates(subset=["gameId", "playerId"], keep="first")
    env = env.drop_duplicates(subset=["gameId"], keep="first")

    if base_players is None:
        base = demo[["gameId", "playerId"]].copy()
    else:
        base = base_players[["gameId", "playerId"]].copy()
    base = base.drop_duplicates(subset=["gameId", "playerId"], keep="first")

    demo_cols = [c for c in demo.columns if c not in {"gameId", "playerId"}]
    env_cols = [c for c in env.columns if c != "gameId"]

    merged = base.merge(demo[["gameId", "playerId", *demo_cols]], on=["gameId", "playerId"], how="left")
    merged = merged.merge(env[["gameId", *env_cols]], on="gameId", how="left")
    return merged


def load_oracle_rows(summary_jsonl: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with summary_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            rows.append(
                {
                    "gameId": normalize_key(row.get("experiment")),
                    "playerId": normalize_key(row.get("participant")),
                    "game_finished": row.get("game_finished"),
                    "_row_order": len(rows),
                }
            )
    return pd.DataFrame(rows, columns=["gameId", "playerId", "game_finished", "_row_order"])


def build_tag_predictions(
    run_dir: Path,
    players_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Returns:
      {tag: {row_index: {"section_text": str, "source_gameId": str, "source_playerId": str, "cosine": float}}}
    """
    predictions: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for tag in TAG_OUTPUT_ORDER:
        tag_dir = run_dir / tag
        model_path = tag_dir / "models" / f"{model_name}.joblib"
        bank_emb_path = tag_dir / "bank_embeddings.npy"
        bank_meta_path = tag_dir / "bank_metadata.jsonl"
        if not (model_path.exists() and bank_emb_path.exists() and bank_meta_path.exists()):
            continue

        artifact = joblib.load(model_path)
        feature_columns = artifact.get("feature_columns", [])
        if not feature_columns:
            continue

        active_mask = players_df.apply(lambda r: is_tag_active(tag, r.to_dict()), axis=1)
        active_idx = players_df.index[active_mask].to_numpy()
        if len(active_idx) == 0:
            predictions[tag] = {}
            continue

        X = coerce_feature_frame(players_df.loc[active_idx], feature_columns)
        if model_name == "mean":
            y_pred = np.repeat(
                artifact["baseline_mean_vector"].reshape(1, -1), repeats=len(active_idx), axis=0
            ).astype(np.float32)
        else:
            est = artifact["estimator"]
            y_pred = est.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(1, -1)
            y_pred = y_pred.astype(np.float32, copy=False)

        bank = np.load(bank_emb_path)
        bank_meta = load_jsonl(bank_meta_path)
        pred_n = normalize_rows(y_pred)
        bank_n = normalize_rows(bank.astype(np.float32, copy=False))
        sim = pred_n @ bank_n.T
        nn_idx = np.argmax(sim, axis=1)

        tag_pred: Dict[int, Dict[str, Any]] = {}
        for i, row_idx in enumerate(active_idx):
            k = int(nn_idx[i])
            meta = bank_meta[k]
            tag_pred[int(row_idx)] = {
                "section_text": str(meta.get("section_text") or ""),
                "source_gameId": meta.get("gameId"),
                "source_playerId": meta.get("playerId"),
                "cosine": float(sim[i, k]),
            }
        predictions[tag] = tag_pred

    return predictions


def build_persona_text(
    row_idx: int,
    tag_predictions: Dict[str, Dict[int, Dict[str, Any]]],
) -> str:
    parts: List[str] = []
    for tag in TAG_OUTPUT_ORDER:
        pred = tag_predictions.get(tag, {}).get(row_idx)
        if pred is None:
            continue
        section = str(pred.get("section_text") or "").strip()
        if not section:
            section = "unknown."
        parts.append(f"<{tag}>\n{section}")
    return "\n\n".join(parts).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic validation personas from trained retrieval models."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory. If omitted, uses latest from --output-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/model_runs_learning_for_validation"),
        help="Used only when --run-dir is omitted.",
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("demographics/demographics_numeric_val.csv"),
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=Path("data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument(
        "--oracle-summary-jsonl",
        type=Path,
        default=Path("Persona/summary_gpt51_val.jsonl"),
        help="Used to transfer game_finished values by (gameId/playerId).",
    )
    parser.add_argument("--model", default="ridge")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("Persona/archetype_retrieval/validation_wave/synthetic_persona_ridge_val.jsonl"),
    )
    parser.add_argument(
        "--output-trace-jsonl",
        type=Path,
        default=Path("Persona/archetype_retrieval/validation_wave/synthetic_persona_ridge_val_trace.jsonl"),
        help="Optional sidecar with retrieval provenance per section.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else load_latest_run(args.output_root)
    model_name = str(args.model).strip().lower()

    oracle_rows = load_oracle_rows(args.oracle_summary_jsonl)
    players = load_feature_tables(
        args.demographics_csv,
        args.environment_csv,
        base_players=oracle_rows,
    )
    players = oracle_rows.merge(players, on=["gameId", "playerId"], how="left")
    players = players.sort_values("_row_order").reset_index(drop=True)

    tag_preds = build_tag_predictions(
        run_dir=run_dir,
        players_df=players,
        model_name=model_name,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_trace_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    missing_finished = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout, args.output_trace_jsonl.open(
        "w", encoding="utf-8"
    ) as ftrace:
        for idx, row in players.iterrows():
            game_id = normalize_key(row["gameId"])
            player_id = normalize_key(row["playerId"])
            finished = row.get("game_finished")
            if finished is None:
                missing_finished += 1

            text = build_persona_text(int(idx), tag_preds)
            out_row = {
                "experiment": game_id,
                "participant": player_id,
                "game_finished": finished,
                "text": text,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False))
            fout.write("\n")

            trace_row = {
                "gameId": game_id,
                "playerId": player_id,
                "sections": {},
            }
            for tag in TAG_OUTPUT_ORDER:
                pred = tag_preds.get(tag, {}).get(int(idx))
                if pred is not None:
                    trace_row["sections"][tag] = pred
            ftrace.write(json.dumps(trace_row, ensure_ascii=False))
            ftrace.write("\n")
            n_rows += 1

    print(f"Run dir: {to_repo_rel(run_dir)}")
    print(f"Model: {model_name}")
    print(f"Oracle rows: {len(oracle_rows)}")
    print(f"Rows written: {n_rows}")
    print(f"Missing game_finished assignments: {missing_finished}")
    print(f"Synthetic personas: {to_repo_rel(args.output_jsonl)}")
    print(f"Retrieval trace: {to_repo_rel(args.output_trace_jsonl)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
