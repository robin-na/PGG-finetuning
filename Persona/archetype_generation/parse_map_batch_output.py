#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from io_utils import (
    extract_batch_content,
    parse_json_from_text,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _renorm_probs(probs: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in probs.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv < 0:
            continue
        out[str(k)] = fv
    s = sum(out.values())
    if s <= 0:
        return out
    return {k: (v / s) for k, v in out.items()}


def _parse_assignments(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    assignments = payload.get("assignments", {})
    if isinstance(assignments, list):
        return {"target": assignments, "anchors": []}
    if not isinstance(assignments, dict):
        return {"target": [], "anchors": []}
    target = assignments.get("target", [])
    anchors = assignments.get("anchors", [])
    if not isinstance(target, list):
        target = []
    if not isinstance(anchors, list):
        anchors = []
    return {"target": target, "anchors": anchors}


def parse_map_outputs(args: argparse.Namespace) -> None:
    manifest = {r["custom_id"]: r for r in read_jsonl(Path(args.request_manifest_jsonl))}

    parsed_requests: List[Dict[str, Any]] = []
    clusters_out: List[Dict[str, Any]] = []
    assignments_out: List[Dict[str, Any]] = []
    games_out: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    seen_custom_ids = set()

    for row in read_jsonl(Path(args.batch_output_jsonl)):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str):
            errors.append({"error": "missing_custom_id", "row": row})
            continue

        seen_custom_ids.add(custom_id)
        man = manifest.get(custom_id)
        if man is None:
            errors.append({"custom_id": custom_id, "error": "custom_id_not_in_manifest"})
            continue

        try:
            content = extract_batch_content(row)
            payload = parse_json_from_text(content)
        except Exception as exc:
            errors.append({"custom_id": custom_id, "error": f"parse_failed: {exc}"})
            continue

        local_clusters = payload.get("clusters", [])
        if not isinstance(local_clusters, list):
            local_clusters = []

        local_ids: List[str] = []
        for c in local_clusters:
            if not isinstance(c, dict):
                continue
            local_id = str(c.get("local_cluster_id") or "").strip()
            if not local_id:
                continue
            local_ids.append(local_id)
            local_key = f"{custom_id}::{local_id}"
            clusters_out.append(
                {
                    "request_custom_id": custom_id,
                    "local_cluster_id": local_id,
                    "local_cluster_key": local_key,
                    "name": c.get("name"),
                    "description": c.get("description"),
                    "representative_persona": c.get("representative_persona"),
                    "non_redundant_signal": c.get("non_redundant_signal"),
                }
            )

        assignment_groups = _parse_assignments(payload)

        expected_target = set(man.get("target_row_ids", []))
        expected_anchor = set(man.get("anchor_row_ids", []))
        seen_target = set()
        seen_anchor = set()

        for scope_name in ("target", "anchors"):
            records = assignment_groups[scope_name]
            scope = "anchor" if scope_name == "anchors" else "target"
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                row_id = str(rec.get("row_id") or "").strip()
                if not row_id:
                    continue

                probs = rec.get("cluster_probs", {})
                if not isinstance(probs, dict):
                    probs = {}
                probs = _renorm_probs(probs)

                # Prefix local IDs with request custom_id to avoid collisions.
                prefixed_probs: Dict[str, float] = {}
                for local_id, p in probs.items():
                    local_key = f"{custom_id}::{local_id}"
                    prefixed_probs[local_key] = p

                if prefixed_probs:
                    # Re-normalize after prefix transformation.
                    total = sum(prefixed_probs.values())
                    if total > 0:
                        prefixed_probs = {k: v / total for k, v in prefixed_probs.items()}

                primary_local_id = str(rec.get("primary_cluster_id") or "").strip()
                primary_local_key = (
                    f"{custom_id}::{primary_local_id}" if primary_local_id else ""
                )

                out_rec = {
                    "request_custom_id": custom_id,
                    "scope": scope,
                    "row_id": row_id,
                    "local_cluster_probs": prefixed_probs,
                    "primary_local_cluster_key": primary_local_key,
                    "confidence": rec.get("confidence"),
                }
                assignments_out.append(out_rec)

                if scope == "target":
                    seen_target.add(row_id)
                else:
                    seen_anchor.add(row_id)

        missing_target = sorted(expected_target - seen_target)
        extra_target = sorted(seen_target - expected_target)
        missing_anchor = sorted(expected_anchor - seen_anchor)
        extra_anchor = sorted(seen_anchor - expected_anchor)

        game_distributions = payload.get("game_distributions", [])
        if isinstance(game_distributions, list):
            for g in game_distributions:
                if not isinstance(g, dict):
                    continue
                probs = g.get("cluster_probs", {})
                if not isinstance(probs, dict):
                    probs = {}
                probs = _renorm_probs(probs)
                prefixed_probs = {
                    f"{custom_id}::{local_id}": p for local_id, p in probs.items()
                }
                if prefixed_probs:
                    total = sum(prefixed_probs.values())
                    if total > 0:
                        prefixed_probs = {k: v / total for k, v in prefixed_probs.items()}
                games_out.append(
                    {
                        "request_custom_id": custom_id,
                        "game_id": g.get("game_id"),
                        "n_rows": g.get("n_rows"),
                        "local_cluster_probs": prefixed_probs,
                    }
                )

        parsed_requests.append(
            {
                "custom_id": custom_id,
                "request_id": payload.get("request_id"),
                "n_clusters": len(local_ids),
                "missing_target_rows": missing_target,
                "extra_target_rows": extra_target,
                "missing_anchor_rows": missing_anchor,
                "extra_anchor_rows": extra_anchor,
            }
        )

    # Add explicit missing-output errors for manifest requests not found in output.
    for custom_id in manifest:
        if custom_id not in seen_custom_ids:
            errors.append({"custom_id": custom_id, "error": "missing_batch_output_row"})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "map_parsed_requests.jsonl", parsed_requests)
    write_jsonl(out_dir / "map_clusters.jsonl", clusters_out)
    write_jsonl(out_dir / "map_assignments.jsonl", assignments_out)
    write_jsonl(out_dir / "map_game_distributions_llm.jsonl", games_out)
    write_jsonl(out_dir / "map_parse_errors.jsonl", errors)

    by_scope = defaultdict(int)
    for r in assignments_out:
        by_scope[r["scope"]] += 1

    summary = {
        "batch_output_jsonl": str(args.batch_output_jsonl),
        "request_manifest_jsonl": str(args.request_manifest_jsonl),
        "n_manifest_requests": len(manifest),
        "n_parsed_requests": len(parsed_requests),
        "n_clusters": len(clusters_out),
        "n_assignments": len(assignments_out),
        "n_assignments_by_scope": dict(by_scope),
        "n_llm_game_distributions": len(games_out),
        "n_errors": len(errors),
        "files": {
            "parsed_requests": str(out_dir / "map_parsed_requests.jsonl"),
            "clusters": str(out_dir / "map_clusters.jsonl"),
            "assignments": str(out_dir / "map_assignments.jsonl"),
            "game_distributions_llm": str(out_dir / "map_game_distributions_llm.jsonl"),
            "errors": str(out_dir / "map_parse_errors.jsonl"),
        },
    }
    write_json(out_dir / "map_parse_summary.json", summary)
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse map-stage OpenAI batch outputs into normalized local clusters and assignments."
    )
    parser.add_argument(
        "--batch-output-jsonl",
        required=True,
        help="OpenAI batch result JSONL file.",
    )
    parser.add_argument(
        "--request-manifest-jsonl",
        default="Persona/archetype_generation/out/map/map_request_manifest.jsonl",
        help="Manifest generated by build_map_batch_input.py",
    )
    parser.add_argument(
        "--output-dir",
        default="Persona/archetype_generation/out/map_parsed",
        help="Output directory for parsed map artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parse_map_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
