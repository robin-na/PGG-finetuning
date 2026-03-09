from __future__ import annotations

import json
from typing import Any, Dict, List


MAP_SYSTEM_PROMPT = (
    "You are a rigorous behavioral scientist. "
    "Cluster persona archetypes using full-person coherence across sections, "
    "produce soft assignments, and aggregate game-level distributions. "
    "Return strict JSON only."
)


def build_map_user_prompt(
    request_id: str,
    shard_id: int,
    part_index: int,
    anchor_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
) -> str:
    payload = {
        "request_meta": {
            "request_id": request_id,
            "shard_id": shard_id,
            "part_index": part_index,
            "n_anchor_rows": len(anchor_rows),
            "n_target_rows": len(target_rows),
        },
        "anchor_rows": anchor_rows,
        "target_rows": target_rows,
    }

    schema = {
        "request_id": "string",
        "clusters": [
            {
                "local_cluster_id": "string (e.g., L01)",
                "name": "string",
                "description": "string",
                "representative_persona": "string",
                "non_redundant_signal": "string",
            }
        ],
        "assignments": {
            "target": [
                {
                    "row_id": "string",
                    "cluster_probs": {"L01": 0.7, "L02": 0.3},
                    "primary_cluster_id": "string",
                    "confidence": 0.0,
                }
            ],
            "anchors": [
                {
                    "row_id": "string",
                    "cluster_probs": {"L01": 0.7, "L02": 0.3},
                    "primary_cluster_id": "string",
                    "confidence": 0.0,
                }
            ],
        },
        "game_distributions": [
            {
                "game_id": "string",
                "cluster_probs": {"L01": 0.6, "L02": 0.4},
                "n_rows": 0,
            }
        ],
    }

    instructions = [
        "Task:",
        "1) Build fine-grained but non-redundant local clusters from ALL rows (anchor_rows + target_rows).",
        "2) Produce soft cluster probabilities for each row in target and anchors separately.",
        "3) Aggregate target rows to game-level distributions.",
        "",
        "Rules:",
        "- Use full persona coherence across all headers/tags in each text. Do not cluster tags independently.",
        "- Do not force uniqueness. Similar players may share a cluster.",
        "- Keep clusters informative; merge near-duplicates.",
        "- It is valid to include low-information or unknown-oriented clusters if needed.",
        "- cluster_probs must be non-negative and sum to 1.0 for each row/game.",
        "- Every target row_id and every anchor row_id must appear exactly once in assignments.",
        "- primary_cluster_id must be the argmax cluster for that row.",
        "",
        "Output:",
        "- Return JSON only, no markdown, no prose wrapper.",
        "- Follow this exact top-level schema shape.",
        json.dumps(schema, ensure_ascii=False),
        "",
        "Input data JSON:",
        json.dumps(payload, ensure_ascii=False),
    ]
    return "\n".join(instructions)


REDUCE_SYSTEM_PROMPT = (
    "You are a rigorous behavioral scientist. "
    "Consolidate local persona clusters into a global non-redundant codebook. "
    "Return strict JSON only."
)


def build_reduce_user_prompt(
    reduce_id: str,
    local_clusters: List[Dict[str, Any]],
) -> str:
    payload = {
        "reduce_id": reduce_id,
        "n_local_clusters": len(local_clusters),
        "local_clusters": local_clusters,
    }

    schema = {
        "reduce_id": "string",
        "global_clusters": [
            {
                "global_cluster_id": "string (e.g., G01)",
                "name": "string",
                "description": "string",
                "representative_persona": "string",
                "merged_local_cluster_keys": ["request_x::L01", "request_y::L03"],
            }
        ],
        "local_to_global": [
            {
                "local_cluster_key": "string",
                "global_cluster_id": "string",
                "confidence": 0.0,
            }
        ],
        "redundant_pairs": [
            {
                "local_cluster_key_a": "string",
                "local_cluster_key_b": "string",
                "reason": "string",
            }
        ],
    }

    instructions = [
        "Task:",
        "Create a global cluster codebook from local clusters.",
        "",
        "Rules:",
        "- Merge only if clusters are semantically and behaviorally redundant.",
        "- Preserve fine-grained distinctions when they are informative.",
        "- Every local_cluster_key must map to exactly one global_cluster_id.",
        "- Confidence in [0,1].",
        "",
        "Output:",
        "- Return JSON only, no markdown, no prose wrapper.",
        "- Follow this exact top-level schema shape.",
        json.dumps(schema, ensure_ascii=False),
        "",
        "Input data JSON:",
        json.dumps(payload, ensure_ascii=False),
    ]
    return "\n".join(instructions)
