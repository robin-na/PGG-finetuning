#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from io_utils import (
    coerce_bool,
    header_signature,
    normalize_key,
    token_estimate_from_chars,
    unknown_ratio_by_section,
    write_json,
    write_jsonl,
)
from prompt_templates import MAP_SYSTEM_PROMPT, build_map_user_prompt


CORE_CONFIG_FIELDS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_MPCR",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]


def _compact_value(v: Any) -> Any:
    b = coerce_bool(v)
    if b is not None:
        return b
    if v is None:
        return None
    if isinstance(v, float):
        if v != v:
            return None
        if v.is_integer():
            return int(v)
        return round(v, 6)
    if isinstance(v, (int, str)):
        return v
    return str(v)


def _config_signature(cfg: Dict[str, Any]) -> str:
    parts = []
    for k in CORE_CONFIG_FIELDS:
        v = cfg.get(k)
        if isinstance(v, bool):
            vv = "T" if v else "F"
        elif v is None:
            vv = "NA"
        else:
            vv = str(v)
        parts.append(f"{k}={vv}")
    return "|".join(parts)


def _unknown_bin(r: float) -> str:
    if r < 0.2:
        return "u_low"
    if r < 0.6:
        return "u_mid"
    return "u_high"


def _length_bins(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = sorted(values)

    def q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        i = (len(arr) - 1) * p
        lo = int(i)
        hi = min(lo + 1, len(arr) - 1)
        w = i - lo
        return arr[lo] * (1 - w) + arr[hi] * w

    return (q(0.33), q(0.66))


def _length_bin(v: float, q1: float, q2: float) -> str:
    if v <= q1:
        return "len_short"
    if v <= q2:
        return "len_mid"
    return "len_long"


def _row_prompt_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "game_id": row["game_id"],
        "player_id": row["player_id"],
        "tag_signature": row["tag_signature"],
        "game_design": row["game_design"],
        "archetype_text": row["text"],
    }


def _load_config_map(path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    if "gameId" not in df.columns:
        raise KeyError(f"Expected gameId in config CSV: {path}")

    cfg_cols = [c for c in df.columns if c.startswith("CONFIG_")]
    cfg_map: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        gid = normalize_key(r.get("gameId"))
        if not gid:
            continue
        cfg = {c: _compact_value(r.get(c)) for c in cfg_cols}
        for c in CORE_CONFIG_FIELDS:
            cfg.setdefault(c, None)
        cfg_map[gid] = cfg
    return cfg_map


def _load_archetypes(path: Path, only_finished: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            finished = obj.get("game_finished")
            if only_finished and finished is not True:
                continue
            game_id = normalize_key(obj.get("experiment"))
            player_id = normalize_key(obj.get("participant"))
            if not game_id or not player_id:
                continue
            text = str(obj.get("text") or "").strip()
            if not text:
                continue
            row_id = f"{game_id}__{player_id}"
            rows.append(
                {
                    "row_id": row_id,
                    "game_id": game_id,
                    "player_id": player_id,
                    "game_finished": finished,
                    "text": text,
                }
            )
    return rows


def _assign_shards(rows: List[Dict[str, Any]], n_shards: int, seed: int) -> None:
    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum_key"]].append(r)

    for skey, items in by_stratum.items():
        rng = random.Random(seed + hash(skey) % 10_000_000)
        rng.shuffle(items)
        for i, row in enumerate(items):
            row["shard_id"] = i % n_shards


def _select_anchors(
    rows: List[Dict[str, Any]],
    anchor_token_budget: int,
    max_anchors: int,
) -> List[str]:
    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum_key"]].append(r)

    for skey in by_stratum:
        by_stratum[skey] = sorted(by_stratum[skey], key=lambda x: (x["prompt_row_tokens"], x["row_id"]))

    chosen: List[str] = []
    chosen_set = set()
    used_tokens = 0

    strata = sorted(by_stratum.keys(), key=lambda k: len(by_stratum[k]))
    depth = 0
    while len(chosen) < max_anchors:
        added = False
        for skey in strata:
            items = by_stratum[skey]
            if depth >= len(items):
                continue
            r = items[depth]
            rid = r["row_id"]
            if rid in chosen_set:
                continue
            t = r["prompt_row_tokens"]
            if used_tokens + t > anchor_token_budget:
                continue
            chosen.append(rid)
            chosen_set.add(rid)
            used_tokens += t
            added = True
            if len(chosen) >= max_anchors:
                break
        if not added:
            break
        depth += 1

    return chosen


def build_requests(args: argparse.Namespace) -> None:
    archetypes = _load_archetypes(Path(args.archetype_jsonl), args.only_finished)
    if args.max_rows > 0:
        archetypes = archetypes[: args.max_rows]

    if not archetypes:
        raise RuntimeError("No archetype rows available after filters.")

    # Ensure unique row_id
    row_ids = [r["row_id"] for r in archetypes]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Duplicate row_id detected in archetype input.")

    cfg_map = _load_config_map(Path(args.config_csv))

    rows: List[Dict[str, Any]] = []
    missing_cfg = 0
    for r in archetypes:
        cfg = cfg_map.get(r["game_id"])
        if cfg is None:
            missing_cfg += 1
            continue

        tag_sig = header_signature(r["text"])
        unknown_ratio = unknown_ratio_by_section(r["text"])
        text_tokens = token_estimate_from_chars(r["text"])
        cfg_sig = _config_signature(cfg)

        row = {
            **r,
            "tag_signature": tag_sig,
            "tag_signature_key": "|".join(tag_sig) if tag_sig else "NO_TAG",
            "unknown_ratio": round(unknown_ratio, 6),
            "unknown_bin": _unknown_bin(unknown_ratio),
            "text_token_est": text_tokens,
            "text_char_len": len(r["text"]),
            "game_design": cfg,
            "config_signature": cfg_sig,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows left after joining CONFIG data.")

    q1, q2 = _length_bins([r["text_token_est"] for r in rows])
    for r in rows:
        r["length_bin"] = _length_bin(r["text_token_est"], q1, q2)
        r["stratum_key"] = (
            f"cfg::{r['config_signature']}||tags::{r['tag_signature_key']}"
            f"||{r['unknown_bin']}||{r['length_bin']}"
        )

    for r in rows:
        prompt_row = _row_prompt_payload(r)
        r["prompt_row_tokens"] = token_estimate_from_chars(
            json.dumps(prompt_row, ensure_ascii=False, separators=(",", ":"))
        )

    _assign_shards(rows, n_shards=args.n_shards, seed=args.seed)
    anchor_ids = _select_anchors(
        rows,
        anchor_token_budget=args.anchor_token_budget,
        max_anchors=args.max_anchors,
    )
    anchor_set = set(anchor_ids)
    for r in rows:
        r["is_anchor"] = r["row_id"] in anchor_set

    by_row = {r["row_id"]: r for r in rows}
    anchor_payloads = [_row_prompt_payload(by_row[rid]) for rid in anchor_ids]

    requests: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    req_count = 0
    for shard_id in range(args.n_shards):
        shard_rows = [r for r in rows if r["shard_id"] == shard_id and not r["is_anchor"]]
        shard_rows = sorted(shard_rows, key=lambda x: (x["game_id"], x["player_id"]))
        if not shard_rows:
            continue

        part_index = 1
        current: List[Dict[str, Any]] = []

        def emit_chunk(chunk: List[Dict[str, Any]], pidx: int) -> None:
            nonlocal req_count
            if not chunk:
                return
            custom_id = f"map_s{shard_id + 1:02d}_p{pidx:03d}"
            request_id = custom_id
            target_payload = [_row_prompt_payload(r) for r in chunk]
            user_prompt = build_map_user_prompt(
                request_id=request_id,
                shard_id=shard_id + 1,
                part_index=pidx,
                anchor_rows=anchor_payloads,
                target_rows=target_payload,
            )
            prompt_tokens_est = token_estimate_from_chars(MAP_SYSTEM_PROMPT) + token_estimate_from_chars(user_prompt)

            body = {
                "model": args.model,
                "temperature": args.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": MAP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if args.max_tokens > 0:
                body["max_tokens"] = args.max_tokens

            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "request_id": request_id,
                    "shard_id": shard_id + 1,
                    "part_index": pidx,
                    "target_row_ids": [r["row_id"] for r in chunk],
                    "anchor_row_ids": anchor_ids,
                    "n_target_rows": len(chunk),
                    "n_anchor_rows": len(anchor_ids),
                    "prompt_tokens_est": prompt_tokens_est,
                }
            )
            req_count += 1

        base_prompt = build_map_user_prompt(
            request_id="_BASE_",
            shard_id=shard_id + 1,
            part_index=0,
            anchor_rows=anchor_payloads,
            target_rows=[],
        )
        base_tokens = token_estimate_from_chars(MAP_SYSTEM_PROMPT) + token_estimate_from_chars(base_prompt)

        if base_tokens > args.target_prompt_tokens:
            raise RuntimeError(
                "Anchor payload exceeds target_prompt_tokens. "
                "Lower --max-anchors or --anchor-token-budget, or increase --target-prompt-tokens."
            )

        current_tokens = base_tokens
        for r in shard_rows:
            rtok = r["prompt_row_tokens"]
            if current and (current_tokens + rtok > args.target_prompt_tokens):
                emit_chunk(current, part_index)
                part_index += 1
                current = []
                current_tokens = base_tokens

            current.append(r)
            current_tokens += rtok

        emit_chunk(current, part_index)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "map_batch_requests.jsonl", requests)
    write_jsonl(out_dir / "map_request_manifest.jsonl", manifest_rows)

    # Persist row table for reproducibility and downstream joining.
    row_table_out = []
    for r in rows:
        row_table_out.append(
            {
                "row_id": r["row_id"],
                "game_id": r["game_id"],
                "player_id": r["player_id"],
                "game_finished": r["game_finished"],
                "text": r["text"],
                "tag_signature": r["tag_signature"],
                "tag_signature_key": r["tag_signature_key"],
                "unknown_ratio": r["unknown_ratio"],
                "unknown_bin": r["unknown_bin"],
                "text_token_est": r["text_token_est"],
                "text_char_len": r["text_char_len"],
                "length_bin": r["length_bin"],
                "config_signature": r["config_signature"],
                "game_design": r["game_design"],
                "stratum_key": r["stratum_key"],
                "shard_id": r["shard_id"] + 1,
                "is_anchor": r["is_anchor"],
                "prompt_row_tokens": r["prompt_row_tokens"],
            }
        )
    write_jsonl(out_dir / "map_row_table.jsonl", row_table_out)

    shard_sizes = defaultdict(int)
    for r in rows:
        shard_sizes[r["shard_id"] + 1] += 1

    summary = {
        "archetype_jsonl": str(args.archetype_jsonl),
        "config_csv": str(args.config_csv),
        "model": args.model,
        "n_rows_input": len(archetypes),
        "n_rows_joined": len(rows),
        "missing_config_rows": missing_cfg,
        "n_shards": args.n_shards,
        "shard_sizes": dict(sorted(shard_sizes.items())),
        "n_anchors": len(anchor_ids),
        "anchor_token_budget": args.anchor_token_budget,
        "target_prompt_tokens": args.target_prompt_tokens,
        "n_requests": len(requests),
        "output_dir": str(out_dir),
        "files": {
            "batch_requests": str(out_dir / "map_batch_requests.jsonl"),
            "request_manifest": str(out_dir / "map_request_manifest.jsonl"),
            "row_table": str(out_dir / "map_row_table.jsonl"),
        },
    }
    write_json(out_dir / "map_build_summary.json", summary)

    print(json_dumps(summary))


def json_dumps(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenAI batch JSONL for LLM-only map-stage clustering+assignment+distribution."
    )
    parser.add_argument(
        "--archetype-jsonl",
        default="Persona/archetype_oracle_gpt51_learn.jsonl",
        help="Input archetype JSONL with experiment/participant/text rows.",
    )
    parser.add_argument(
        "--config-csv",
        default="data/processed_data/df_analysis_learn.csv",
        help="Game-level CONFIG_* table keyed by gameId.",
    )
    parser.add_argument(
        "--output-dir",
        default="Persona/archetype_generation/out/map",
        help="Output directory for batch input and manifests.",
    )
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0, help="Optional chat completion max_tokens.")
    parser.add_argument("--n-shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=110000,
        help="Heuristic prompt token budget per request (chars/4 estimate).",
    )
    parser.add_argument(
        "--anchor-token-budget",
        type=int,
        default=20000,
        help="Heuristic token budget for shared anchors included in every request.",
    )
    parser.add_argument("--max-anchors", type=int, default=120)
    parser.add_argument(
        "--only-finished",
        action="store_true",
        default=True,
        help="Keep only rows where game_finished is true.",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="If set, include unfinished rows (overrides --only-finished).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means all rows.",
    )
    args = parser.parse_args()
    if args.include_unfinished:
        args.only_finished = False
    return args


def main() -> int:
    args = parse_args()
    build_requests(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
