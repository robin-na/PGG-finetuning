#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from io_utils import read_jsonl, write_json, write_jsonl


def _renorm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(v for v in d.values() if v > 0)
    if s <= 0:
        return {}
    return {k: v / s for k, v in d.items() if v > 0}


def finalize(args: argparse.Namespace) -> None:
    row_meta = {r["row_id"]: r for r in read_jsonl(Path(args.map_row_table_jsonl))}
    local_to_global = {
        r["local_cluster_key"]: r["global_cluster_id"]
        for r in read_jsonl(Path(args.local_to_global_jsonl))
    }

    if not local_to_global:
        raise RuntimeError("No local->global mappings found.")

    by_row: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in read_jsonl(Path(args.map_assignments_jsonl)):
        scope = rec.get("scope")
        if scope != args.scope:
            continue
        row_id = rec.get("row_id")
        if not isinstance(row_id, str):
            continue
        by_row[row_id].append(rec)

    row_global_assignments: List[Dict[str, Any]] = []
    missing_row_meta = 0
    missing_local_keys = Counter()

    for row_id, recs in by_row.items():
        # Average repeated assignment records for the same row if they exist.
        accum = defaultdict(float)
        for rec in recs:
            probs = rec.get("local_cluster_probs", {})
            if not isinstance(probs, dict):
                continue
            for local_key, p in probs.items():
                try:
                    fp = float(p)
                except Exception:
                    continue
                if fp <= 0:
                    continue
                global_id = local_to_global.get(local_key)
                if global_id is None:
                    missing_local_keys[local_key] += 1
                    continue
                accum[global_id] += fp

        # Mean across repeated records, then normalize.
        n = float(len(recs)) if recs else 1.0
        mean_probs = {k: v / n for k, v in accum.items()}
        global_probs = _renorm(mean_probs)
        if not global_probs:
            continue

        primary = max(global_probs.items(), key=lambda kv: kv[1])[0]
        confidence = global_probs[primary]

        rm = row_meta.get(row_id)
        if rm is None:
            missing_row_meta += 1
            game_id = ""
            player_id = ""
            game_design = {}
        else:
            game_id = rm.get("game_id", "")
            player_id = rm.get("player_id", "")
            game_design = rm.get("game_design", {})

        row_global_assignments.append(
            {
                "row_id": row_id,
                "game_id": game_id,
                "player_id": player_id,
                "scope": args.scope,
                "global_cluster_probs": global_probs,
                "primary_global_cluster_id": primary,
                "confidence": confidence,
                "game_design": game_design,
            }
        )

    # Aggregate to game-level distributions.
    by_game = defaultdict(list)
    for rec in row_global_assignments:
        by_game[rec["game_id"]].append(rec)

    game_distributions: List[Dict[str, Any]] = []
    cluster_global_weight = defaultdict(float)

    for game_id, items in sorted(by_game.items()):
        accum = defaultdict(float)
        n = len(items)
        for rec in items:
            for gid, p in rec["global_cluster_probs"].items():
                accum[gid] += p
                cluster_global_weight[gid] += p

        mean_probs = {k: v / n for k, v in accum.items()}
        mean_probs = _renorm(mean_probs)

        game_design = items[0].get("game_design", {})

        game_distributions.append(
            {
                "game_id": game_id,
                "n_rows": n,
                "global_cluster_probs": mean_probs,
                "game_design": game_design,
            }
        )

    total_weight = sum(cluster_global_weight.values())
    cluster_prevalence = []
    for gid, w in sorted(cluster_global_weight.items()):
        cluster_prevalence.append(
            {
                "global_cluster_id": gid,
                "weight": w,
                "share": (w / total_weight) if total_weight > 0 else 0.0,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "row_global_assignments.jsonl", row_global_assignments)
    write_jsonl(out_dir / "game_cluster_distributions.jsonl", game_distributions)
    write_jsonl(out_dir / "global_cluster_prevalence.jsonl", cluster_prevalence)

    # Optional wide table for step-4 modeling.
    all_global_ids = sorted({c["global_cluster_id"] for c in cluster_prevalence})
    table_rows = []
    for g in game_distributions:
        row = {
            "game_id": g["game_id"],
            "n_rows": g["n_rows"],
        }
        gd = g.get("game_design", {})
        if isinstance(gd, dict):
            for k, v in gd.items():
                row[k] = v
        for gid in all_global_ids:
            row[f"p__{gid}"] = g["global_cluster_probs"].get(gid, 0.0)
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_path = out_dir / "game_cluster_distribution_table.csv"
    table_df.to_csv(table_path, index=False)

    summary = {
        "scope": args.scope,
        "n_row_assignments": len(row_global_assignments),
        "n_games": len(game_distributions),
        "n_global_clusters": len(cluster_prevalence),
        "missing_row_meta": missing_row_meta,
        "missing_local_cluster_keys": len(missing_local_keys),
        "top_missing_local_keys": missing_local_keys.most_common(20),
        "files": {
            "row_global_assignments": str(out_dir / "row_global_assignments.jsonl"),
            "game_cluster_distributions": str(out_dir / "game_cluster_distributions.jsonl"),
            "global_cluster_prevalence": str(out_dir / "global_cluster_prevalence.jsonl"),
            "game_cluster_distribution_table": str(table_path),
        },
    }
    write_json(out_dir / "finalize_summary.json", summary)
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize global game-level cluster distributions from parsed map assignments and reduce mapping."
    )
    parser.add_argument(
        "--map-row-table-jsonl",
        default="Persona/archetype_generation/out/map/map_row_table.jsonl",
    )
    parser.add_argument(
        "--map-assignments-jsonl",
        default="Persona/archetype_generation/out/map_parsed/map_assignments.jsonl",
    )
    parser.add_argument(
        "--local-to-global-jsonl",
        default="Persona/archetype_generation/out/reduce_parsed/local_to_global.jsonl",
    )
    parser.add_argument(
        "--scope",
        default="target",
        choices=["target", "anchor"],
        help="Which assignment scope to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        default="Persona/archetype_generation/out/final",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
