#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

from io_utils import extract_batch_content, parse_json_from_text, read_jsonl, write_json, write_jsonl


def parse_reduce_output(args: argparse.Namespace) -> None:
    expected_local_keys: Set[str] = set()
    if args.local_cluster_cards_jsonl:
        for r in read_jsonl(Path(args.local_cluster_cards_jsonl)):
            key = str(r.get("local_cluster_key") or "").strip()
            if key:
                expected_local_keys.add(key)

    rows = list(read_jsonl(Path(args.batch_output_jsonl)))
    if not rows:
        raise RuntimeError("Empty reduce batch output.")

    parsed_payload = None
    used_custom_id = None
    errors: List[Dict[str, Any]] = []

    target_custom_id = args.custom_id
    for row in rows:
        cid = row.get("custom_id")
        if target_custom_id and cid != target_custom_id:
            continue
        used_custom_id = cid
        try:
            content = extract_batch_content(row)
            parsed_payload = parse_json_from_text(content)
            break
        except Exception as exc:
            errors.append({"custom_id": cid, "error": f"parse_failed: {exc}"})

    if parsed_payload is None:
        raise RuntimeError("Could not parse reduce output for the requested custom_id.")

    global_clusters_raw = parsed_payload.get("global_clusters", [])
    local_to_global_raw = parsed_payload.get("local_to_global", [])
    redundant_pairs = parsed_payload.get("redundant_pairs", [])

    if not isinstance(global_clusters_raw, list):
        global_clusters_raw = []
    if not isinstance(local_to_global_raw, list):
        local_to_global_raw = []
    if not isinstance(redundant_pairs, list):
        redundant_pairs = []

    global_clusters: List[Dict[str, Any]] = []
    for gc in global_clusters_raw:
        if not isinstance(gc, dict):
            continue
        gid = str(gc.get("global_cluster_id") or "").strip()
        if not gid:
            continue
        merged = gc.get("merged_local_cluster_keys", [])
        if not isinstance(merged, list):
            merged = []
        global_clusters.append(
            {
                "global_cluster_id": gid,
                "name": gc.get("name"),
                "description": gc.get("description"),
                "representative_persona": gc.get("representative_persona"),
                "merged_local_cluster_keys": merged,
            }
        )

    local_to_global: Dict[str, Dict[str, Any]] = {}

    # Explicit mappings.
    for m in local_to_global_raw:
        if not isinstance(m, dict):
            continue
        local_key = str(m.get("local_cluster_key") or "").strip()
        global_id = str(m.get("global_cluster_id") or "").strip()
        if not local_key or not global_id:
            continue
        conf = m.get("confidence")
        local_to_global[local_key] = {
            "local_cluster_key": local_key,
            "global_cluster_id": global_id,
            "confidence": conf,
            "mapping_source": "local_to_global",
        }

    # Derive mappings from merged_local_cluster_keys if missing.
    for gc in global_clusters:
        gid = gc["global_cluster_id"]
        for local_key in gc.get("merged_local_cluster_keys", []):
            lk = str(local_key or "").strip()
            if not lk:
                continue
            if lk not in local_to_global:
                local_to_global[lk] = {
                    "local_cluster_key": lk,
                    "global_cluster_id": gid,
                    "confidence": None,
                    "mapping_source": "merged_local_cluster_keys",
                }

    missing_local_keys = sorted(expected_local_keys - set(local_to_global.keys()))
    fallback_added = 0

    if missing_local_keys and args.fill_unmapped:
        # Attach all unmapped locals to a dedicated fallback cluster.
        fallback_id = args.fallback_global_cluster_id
        exists = any(gc["global_cluster_id"] == fallback_id for gc in global_clusters)
        if not exists:
            global_clusters.append(
                {
                    "global_cluster_id": fallback_id,
                    "name": "Unmapped Local Cluster",
                    "description": "Fallback bucket for local clusters not explicitly mapped by reduce output.",
                    "representative_persona": "",
                    "merged_local_cluster_keys": [],
                }
            )

        for lk in missing_local_keys:
            local_to_global[lk] = {
                "local_cluster_key": lk,
                "global_cluster_id": fallback_id,
                "confidence": 0.0,
                "mapping_source": "fallback",
            }
            fallback_added += 1

        # Update fallback merged keys in global clusters record.
        for gc in global_clusters:
            if gc["global_cluster_id"] == fallback_id:
                merged = set(gc.get("merged_local_cluster_keys", []))
                merged.update(missing_local_keys)
                gc["merged_local_cluster_keys"] = sorted(merged)
                break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_clusters_sorted = sorted(global_clusters, key=lambda x: x["global_cluster_id"])
    local_to_global_rows = [local_to_global[k] for k in sorted(local_to_global.keys())]

    write_jsonl(out_dir / "global_clusters.jsonl", global_clusters_sorted)
    write_jsonl(out_dir / "local_to_global.jsonl", local_to_global_rows)
    write_jsonl(out_dir / "redundant_pairs.jsonl", [r for r in redundant_pairs if isinstance(r, dict)])
    write_jsonl(out_dir / "reduce_parse_errors.jsonl", errors)

    summary = {
        "batch_output_jsonl": str(args.batch_output_jsonl),
        "requested_custom_id": args.custom_id,
        "used_custom_id": used_custom_id,
        "n_global_clusters": len(global_clusters_sorted),
        "n_local_to_global": len(local_to_global_rows),
        "n_expected_local_clusters": len(expected_local_keys),
        "n_missing_local_keys_before_fallback": len(missing_local_keys),
        "fallback_added": fallback_added,
        "fill_unmapped": args.fill_unmapped,
        "fallback_global_cluster_id": args.fallback_global_cluster_id,
        "files": {
            "global_clusters": str(out_dir / "global_clusters.jsonl"),
            "local_to_global": str(out_dir / "local_to_global.jsonl"),
            "redundant_pairs": str(out_dir / "redundant_pairs.jsonl"),
            "errors": str(out_dir / "reduce_parse_errors.jsonl"),
        },
    }
    write_json(out_dir / "reduce_parse_summary.json", summary)
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse reduce-stage OpenAI batch output into global cluster catalog and local->global mapping."
    )
    parser.add_argument("--batch-output-jsonl", required=True)
    parser.add_argument(
        "--custom-id",
        default="reduce_global_001",
        help="Custom ID of the reduce request to parse.",
    )
    parser.add_argument(
        "--local-cluster-cards-jsonl",
        default="Persona/archetype_generation/out/reduce/reduce_local_cluster_cards.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="Persona/archetype_generation/out/reduce_parsed",
    )
    parser.add_argument(
        "--fill-unmapped",
        action="store_true",
        default=True,
        help="Map unmapped local clusters to fallback global cluster.",
    )
    parser.add_argument(
        "--no-fill-unmapped",
        action="store_true",
        help="Disable fallback mapping for unmapped local clusters.",
    )
    parser.add_argument(
        "--fallback-global-cluster-id",
        default="G_UNMAPPED",
    )
    args = parser.parse_args()
    if args.no_fill_unmapped:
        args.fill_unmapped = False
    return args


def main() -> int:
    args = parse_args()
    parse_reduce_output(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
