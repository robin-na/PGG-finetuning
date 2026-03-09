#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from io_utils import read_jsonl, token_estimate_from_chars, write_json, write_jsonl
from prompt_templates import REDUCE_SYSTEM_PROMPT, build_reduce_user_prompt


def _trim_text(s: Any, max_chars: int) -> str:
    text = str(s or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_reduce_requests(args: argparse.Namespace) -> None:
    clusters = list(read_jsonl(Path(args.map_clusters_jsonl)))
    assignments = list(read_jsonl(Path(args.map_assignments_jsonl)))

    if not clusters:
        raise RuntimeError("No map clusters found. Parse map outputs first.")

    row_meta: Dict[str, Dict[str, Any]] = {}
    if args.map_row_table_jsonl:
        for r in read_jsonl(Path(args.map_row_table_jsonl)):
            row_meta[r["row_id"]] = r

    # Aggregate assignment support stats for local clusters.
    target_weight = defaultdict(float)
    anchor_weight = defaultdict(float)
    anchor_top = defaultdict(list)  # local_key -> list[(prob, row_id)]

    for rec in assignments:
        scope = rec.get("scope")
        probs = rec.get("local_cluster_probs", {})
        row_id = rec.get("row_id")
        if not isinstance(probs, dict):
            continue
        for local_key, p in probs.items():
            try:
                fp = float(p)
            except Exception:
                continue
            if fp <= 0:
                continue
            if scope == "target":
                target_weight[local_key] += fp
            elif scope == "anchor":
                anchor_weight[local_key] += fp
                anchor_top[local_key].append((fp, row_id))

    for k in list(anchor_top.keys()):
        anchor_top[k] = sorted(anchor_top[k], reverse=True)[: args.max_anchor_refs]

    # Tag signature support for interpretability.
    tag_sig_counter: Dict[str, Counter] = defaultdict(Counter)
    for rec in assignments:
        if rec.get("scope") != "target":
            continue
        row_id = rec.get("row_id")
        rm = row_meta.get(row_id, {})
        tag_sig_key = rm.get("tag_signature_key")
        if not tag_sig_key:
            continue
        probs = rec.get("local_cluster_probs", {})
        if not isinstance(probs, dict):
            continue
        for local_key, p in probs.items():
            try:
                fp = float(p)
            except Exception:
                continue
            if fp > 0:
                tag_sig_counter[local_key][tag_sig_key] += fp

    local_cards: List[Dict[str, Any]] = []
    for c in clusters:
        local_key = c.get("local_cluster_key")
        if not local_key:
            continue
        local_cards.append(
            {
                "local_cluster_key": local_key,
                "source_request_custom_id": c.get("request_custom_id"),
                "local_cluster_id": c.get("local_cluster_id"),
                "name": c.get("name"),
                "description": _trim_text(c.get("description"), args.cluster_desc_max_chars),
                "representative_persona": _trim_text(
                    c.get("representative_persona"), args.representative_max_chars
                ),
                "non_redundant_signal": _trim_text(
                    c.get("non_redundant_signal"), args.signal_max_chars
                ),
                "estimated_target_weight": round(target_weight.get(local_key, 0.0), 6),
                "estimated_anchor_weight": round(anchor_weight.get(local_key, 0.0), 6),
                "top_anchor_refs": [
                    {"row_id": rid, "prob": round(prob, 6)}
                    for prob, rid in anchor_top.get(local_key, [])
                ],
                "top_tag_signatures": [
                    {"tag_signature_key": k, "weight": round(v, 6)}
                    for k, v in tag_sig_counter.get(local_key, Counter()).most_common(3)
                ],
            }
        )

    local_cards = sorted(local_cards, key=lambda x: x["local_cluster_key"])

    if args.max_local_clusters > 0 and len(local_cards) > args.max_local_clusters:
        raise RuntimeError(
            f"Local clusters ({len(local_cards)}) exceed --max-local-clusters {args.max_local_clusters}. "
            "Increase the limit or run multi-stage reduce manually."
        )

    reduce_id = args.reduce_id
    user_prompt = build_reduce_user_prompt(reduce_id=reduce_id, local_clusters=local_cards)
    prompt_tokens_est = token_estimate_from_chars(REDUCE_SYSTEM_PROMPT) + token_estimate_from_chars(
        user_prompt
    )

    body = {
        "model": args.model,
        "temperature": args.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if args.max_tokens > 0:
        body["max_tokens"] = args.max_tokens

    request = {
        "custom_id": args.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "reduce_batch_requests.jsonl", [request])
    write_json(
        out_dir / "reduce_request_manifest.json",
        {
            "custom_id": args.custom_id,
            "reduce_id": reduce_id,
            "n_local_clusters": len(local_cards),
            "prompt_tokens_est": prompt_tokens_est,
            "source_files": {
                "map_clusters_jsonl": str(args.map_clusters_jsonl),
                "map_assignments_jsonl": str(args.map_assignments_jsonl),
                "map_row_table_jsonl": str(args.map_row_table_jsonl) if args.map_row_table_jsonl else None,
            },
            "files": {
                "reduce_batch_requests": str(out_dir / "reduce_batch_requests.jsonl"),
            },
        },
    )

    write_jsonl(out_dir / "reduce_local_cluster_cards.jsonl", local_cards)

    summary = {
        "model": args.model,
        "reduce_id": reduce_id,
        "custom_id": args.custom_id,
        "n_local_clusters": len(local_cards),
        "prompt_tokens_est": prompt_tokens_est,
        "output_dir": str(out_dir),
        "files": {
            "batch_requests": str(out_dir / "reduce_batch_requests.jsonl"),
            "manifest": str(out_dir / "reduce_request_manifest.json"),
            "local_cluster_cards": str(out_dir / "reduce_local_cluster_cards.jsonl"),
        },
    }
    write_json(out_dir / "reduce_build_summary.json", summary)
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenAI batch JSONL for reduce-stage global cluster consolidation."
    )
    parser.add_argument(
        "--map-clusters-jsonl",
        default="Persona/archetype_generation/out/map_parsed/map_clusters.jsonl",
    )
    parser.add_argument(
        "--map-assignments-jsonl",
        default="Persona/archetype_generation/out/map_parsed/map_assignments.jsonl",
    )
    parser.add_argument(
        "--map-row-table-jsonl",
        default="Persona/archetype_generation/out/map/map_row_table.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="Persona/archetype_generation/out/reduce",
    )
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--custom-id", default="reduce_global_001")
    parser.add_argument("--reduce-id", default="global_reduce_pass_1")
    parser.add_argument("--cluster-desc-max-chars", type=int, default=1000)
    parser.add_argument("--representative-max-chars", type=int, default=1400)
    parser.add_argument("--signal-max-chars", type=int, default=500)
    parser.add_argument("--max-anchor-refs", type=int, default=5)
    parser.add_argument(
        "--max-local-clusters",
        type=int,
        default=1200,
        help="Safety cap for single-request reduce pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_reduce_requests(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
