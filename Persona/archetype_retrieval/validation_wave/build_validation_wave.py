#!/usr/bin/env python3
"""
Build validation-wave tag sections and embeddings.

Inputs:
- Persona/archetype_oracle_gpt51_val.jsonl

Outputs (per tag):
- <tag>_sections_input.jsonl
- embeddings_text-embedding-3-large_1536.npy (or configured equivalent)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("Persona/archetype_oracle_gpt51_val.jsonl")
DEFAULT_OUTPUT_DIR = Path("Persona/archetype_retrieval/validation_wave")
DEFAULT_TAGS = [
    "CONTRIBUTION",
    "PUNISHMENT",
    "REWARD",
    "COMMUNICATION",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_END_GAME",
]

SECTION_HEADER_RE = re.compile(r"^\s*<([A-Z_]+)>\s*$")
UNKNOWN_PREFIX_RE = re.compile(r"^unknown(?:[\s\.,:;!?]|$)", flags=re.IGNORECASE)


@dataclass
class ValidationWaveConfig:
    input_jsonl: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    text_column: str = "text"
    tags: List[str] = field(default_factory=lambda: list(DEFAULT_TAGS))
    only_finished: bool = True
    drop_unknown: bool = True
    unknown_mode: str = "startswith"  # startswith | exact

    embedding_backend: str = "openai"  # openai | tfidf
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200
    embedding_max_retries: int = 5
    embedding_retry_seconds: float = 1.0


def to_repo_rel(path_like: Any) -> str:
    p = Path(path_like)
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(p)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return to_repo_rel(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def normalize_tag(tag: str) -> str:
    t = str(tag).strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    return t.strip().upper()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def extract_sections(text: str, tags: Sequence[str]) -> Dict[str, str]:
    tags_set = set(tags)
    buffers: Dict[str, List[str]] = {tag: [] for tag in tags}
    current_tag: Optional[str] = None

    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()
        m = SECTION_HEADER_RE.match(line.strip())
        if m:
            header_tag = m.group(1).upper()
            current_tag = header_tag if header_tag in tags_set else None
            continue
        if current_tag is not None:
            buffers[current_tag].append(line)

    out: Dict[str, str] = {}
    for tag in tags:
        section = "\n".join(buffers[tag]).strip()
        if section:
            out[tag] = section
    return out


def is_unknown_section(section_text: str, mode: str) -> bool:
    norm = normalize_text(section_text)
    if not norm:
        return True
    low = norm.lower()
    if mode == "exact":
        return low in {"unknown", "unknown."}
    if mode == "startswith":
        return bool(UNKNOWN_PREFIX_RE.match(low))
    raise ValueError(f"Unsupported unknown mode: {mode}")


def build_tag_rows(
    records: Sequence[Dict[str, Any]],
    cfg: ValidationWaveConfig,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]], Dict[str, int]]:
    tags = [normalize_tag(t) for t in cfg.tags]
    by_tag: Dict[str, List[Dict[str, Any]]] = {tag: [] for tag in tags}
    stats: Dict[str, Dict[str, int]] = {
        tag: {
            "rows_considered": 0,
            "missing_tag": 0,
            "empty_tag": 0,
            "dropped_unknown": 0,
            "kept": 0,
        }
        for tag in tags
    }
    global_stats = {
        "rows_total": len(records),
        "rows_after_finished_filter": 0,
        "rows_missing_text": 0,
    }

    for rec in records:
        if cfg.only_finished and "game_finished" in rec:
            if parse_bool(rec.get("game_finished")) is not True:
                continue

        global_stats["rows_after_finished_filter"] += 1
        text = rec.get(cfg.text_column)
        if text is None:
            global_stats["rows_missing_text"] += 1
            continue

        sections = extract_sections(str(text), tags)

        game_id = normalize_id(rec.get("gameId"))
        player_id = normalize_id(rec.get("playerId"))
        if not game_id:
            game_id = normalize_id(rec.get("experiment"))
        if not player_id:
            player_id = normalize_id(rec.get("participant"))

        for tag in tags:
            stats[tag]["rows_considered"] += 1
            section_text = sections.get(tag)
            if section_text is None:
                stats[tag]["missing_tag"] += 1
                continue

            section_norm = normalize_text(section_text)
            if not section_norm:
                stats[tag]["empty_tag"] += 1
                continue

            if cfg.drop_unknown and is_unknown_section(section_norm, cfg.unknown_mode):
                stats[tag]["dropped_unknown"] += 1
                continue

            out_row = {
                "gameId": game_id,
                "playerId": player_id,
                # Keep legacy aliases for backward compatibility.
                "experiment": game_id,
                "participant": player_id,
                "game_finished": rec.get("game_finished"),
                "tag": tag,
                "text": section_norm,
                "section_text": section_norm,
            }
            by_tag[tag].append(out_row)
            stats[tag]["kept"] += 1

    return by_tag, stats, global_stats


def build_tag_embeddings(
    texts: Sequence[str],
    cfg: ValidationWaveConfig,
    cache_path: Path,
) -> np.ndarray:
    from cluster.cluster_pipeline import (
        ClusterConfig,
        build_openai_embeddings,
        build_tfidf_embeddings,
    )

    tmp_cfg = ClusterConfig(
        input_jsonl=Path("_unused_input.jsonl"),
        output_jsonl=Path("_unused_output.jsonl"),
        output_dir=cache_path.parent,
        embedding_backend=cfg.embedding_backend,
        embedding_model=cfg.embedding_model,
        embedding_dimensions=cfg.embedding_dimensions,
        embedding_batch_size=cfg.embedding_batch_size,
        embedding_cache_path=cache_path,
        embedding_max_retries=cfg.embedding_max_retries,
        embedding_retry_seconds=cfg.embedding_retry_seconds,
    )

    if cfg.embedding_backend == "openai":
        emb = build_openai_embeddings(texts=texts, cfg=tmp_cfg)
        if not cache_path.exists():
            np.save(cache_path, emb.astype(np.float32))
        return emb.astype(np.float32)

    if cfg.embedding_backend == "tfidf":
        emb = build_tfidf_embeddings(texts=texts, dimensions=cfg.embedding_dimensions)
        np.save(cache_path, emb.astype(np.float32))
        return emb.astype(np.float32)

    raise ValueError(f"Unsupported embedding backend: {cfg.embedding_backend}")


def parse_args() -> ValidationWaveConfig:
    parser = argparse.ArgumentParser(description="Build validation wave tag sections + embeddings")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--keep-unknown", action="store_true")
    parser.add_argument("--unknown-mode", choices=["startswith", "exact"], default="startswith")

    parser.add_argument("--embedding-backend", choices=["openai", "tfidf"], default="openai")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding-dimensions", type=int, default=1536)
    parser.add_argument("--embedding-batch-size", type=int, default=200)
    parser.add_argument("--embedding-max-retries", type=int, default=5)
    parser.add_argument("--embedding-retry-seconds", type=float, default=1.0)

    args = parser.parse_args()
    tags = [normalize_tag(t) for t in args.tags]
    tags = list(dict.fromkeys(tags))

    return ValidationWaveConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        text_column=args.text_column,
        tags=tags,
        only_finished=not args.include_unfinished,
        drop_unknown=not args.keep_unknown,
        unknown_mode=args.unknown_mode,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_retries=args.embedding_max_retries,
        embedding_retry_seconds=args.embedding_retry_seconds,
    )


def run(cfg: ValidationWaveConfig) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(cfg.input_jsonl)
    by_tag, tag_stats, global_stats = build_tag_rows(records, cfg)

    manifest: Dict[str, Any] = {
        "config": to_jsonable(asdict(cfg)),
        "global_stats": global_stats,
        "tag_results": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "errors": [],
    }

    model_tag = cfg.embedding_model.replace("/", "_")
    emb_name = f"embeddings_{model_tag}_{cfg.embedding_dimensions}.npy"

    for tag in cfg.tags:
        tag_dir = cfg.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        section_rows = by_tag[tag]
        section_path = tag_dir / f"{tag.lower()}_sections_input.jsonl"
        write_jsonl(section_path, section_rows)

        tag_result: Dict[str, Any] = {
            "tag": tag,
            "status": "ok",
            "rows_for_embedding": len(section_rows),
            "stats": tag_stats[tag],
            "paths": {
                "section_input_jsonl": to_repo_rel(section_path),
            },
        }

        if not section_rows:
            tag_result["status"] = "skipped_no_rows"
            manifest["tag_results"][tag] = tag_result
            with (tag_dir / "split_summary.json").open("w", encoding="utf-8") as f:
                json.dump(tag_result, f, ensure_ascii=False, indent=2)
                f.write("\n")
            continue

        emb_path = tag_dir / emb_name
        try:
            texts = [str(r.get("text", "")) for r in section_rows]
            emb = build_tag_embeddings(texts=texts, cfg=cfg, cache_path=emb_path)
            if emb.shape[0] != len(section_rows):
                raise ValueError(
                    f"Embedding rows mismatch for {tag}: {emb.shape[0]} vs {len(section_rows)}"
                )
            tag_result["embedding_shape"] = [int(emb.shape[0]), int(emb.shape[1])]
            tag_result["paths"]["embeddings_npy"] = to_repo_rel(emb_path)
        except Exception as exc:
            tag_result["status"] = "embedding_error"
            tag_result["error"] = str(exc)
            manifest["errors"].append(f"{tag}: {exc}")

        manifest["tag_results"][tag] = tag_result
        with (tag_dir / "split_summary.json").open("w", encoding="utf-8") as f:
            json.dump(tag_result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    manifest_path = cfg.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Validation-wave manifest: {to_repo_rel(manifest_path)}")
    for tag in cfg.tags:
        status = manifest["tag_results"].get(tag, {}).get("status")
        rows = manifest["tag_results"].get(tag, {}).get("rows_for_embedding")
        print(f"[{tag}] status={status} rows={rows}")

    if manifest["errors"]:
        print("Completed with errors:")
        for err in manifest["errors"]:
            print(f"- {err}")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
