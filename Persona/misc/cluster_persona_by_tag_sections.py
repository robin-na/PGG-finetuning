#!/usr/bin/env python3
"""
Cluster persona summaries per section tag.

Workflow:
1) Read persona summary JSONL rows.
2) Extract each requested section tag from `text`.
3) Drop rows where the section is missing/empty and optionally "unknown".
4) Run the existing cluster pipeline independently per tag.
5) Write per-tag outputs plus a run manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("Persona/archetype_oracle_gpt51_learn.jsonl")
DEFAULT_OUTPUT_DIR = Path("Persona/misc/tag_section_clusters")
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

SPLIT_NAME_MAP = {
    "CONTRIBUTION": "Contribution Strategy Profiles",
    "PUNISHMENT": "Punishment Behavior Profiles",
    "REWARD": "Reward Behavior Profiles",
    "COMMUNICATION": "Communication Style Profiles",
    "RESPONSE_TO_PUNISHER": "Response-to-Punisher Profiles",
    "RESPONSE_TO_REWARDER": "Response-to-Rewarder Profiles",
    "RESPONSE_TO_OTHERS_OUTCOME": "Outcome-Reaction Profiles",
    "RESPONSE_TO_END_GAME": "End-Game Response Profiles",
}

SECTION_HEADER_RE = re.compile(r"^\s*<([A-Z_]+)>\s*$")
UNKNOWN_PREFIX_RE = re.compile(r"^unknown(?:[\s\.,:;!?]|$)", flags=re.IGNORECASE)


@dataclass
class SectionClusterConfig:
    input_jsonl: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    text_column: str = "text"
    tags: List[str] = field(default_factory=lambda: list(DEFAULT_TAGS))
    only_finished: bool = True
    drop_unknown: bool = True
    unknown_mode: str = "startswith"  # startswith | exact

    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200

    umap_min_dist: float = 0.5
    umap_metric: str = "cosine"
    umap_neighbors_ratio: float = 0.2

    n_clusters: int = 15
    auto_k: bool = False
    k_min: int = 6
    k_max: int = 25
    random_state: int = 42

    summary_backend: str = "openai"
    summary_model: str = "gpt-4o-mini"
    summary_temperature: float = 0.0
    summary_max_examples: int = 20
    summary_max_chars_per_example: int = 280


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return to_repo_rel_str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def to_repo_rel_str(path_like: Any) -> str:
    p = Path(path_like)
    try:
        rel = p.resolve().relative_to(REPO_ROOT.resolve())
        return str(rel)
    except Exception:
        return str(p)


def normalize_tag(tag: str) -> str:
    t = str(tag).strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    return t.strip().upper()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


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
    cfg: SectionClusterConfig,
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
            finished = parse_bool(rec.get("game_finished"))
            if finished is not True:
                continue

        global_stats["rows_after_finished_filter"] += 1
        text = rec.get(cfg.text_column)
        if text is None:
            global_stats["rows_missing_text"] += 1
            continue

        sections = extract_sections(str(text), tags)

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
                "experiment": rec.get("experiment"),
                "participant": rec.get("participant"),
                "game_finished": rec.get("game_finished"),
                "tag": tag,
                "text": section_norm,
                "section_text": section_norm,
            }
            by_tag[tag].append(out_row)
            stats[tag]["kept"] += 1

    return by_tag, stats, global_stats


def make_cluster_config(
    cfg: SectionClusterConfig,
    tag_input_jsonl: Path,
    tag_output_jsonl: Path,
    tag_output_dir: Path,
    n_rows: int,
) -> Any:
    from cluster.cluster_pipeline import ClusterConfig

    model_tag = cfg.embedding_model.replace("/", "_")
    cache_path = None
    if cfg.embedding_backend == "openai":
        cache_path = tag_output_dir / f"embeddings_{model_tag}_{cfg.embedding_dimensions}.npy"

    # Avoid invalid k > n_samples.
    safe_clusters = min(max(2, cfg.n_clusters), max(2, n_rows))

    return ClusterConfig(
        input_jsonl=tag_input_jsonl,
        output_jsonl=tag_output_jsonl,
        output_dir=tag_output_dir,
        text_column="text",
        only_finished=False,
        embedding_backend=cfg.embedding_backend,
        embedding_model=cfg.embedding_model,
        embedding_dimensions=cfg.embedding_dimensions,
        embedding_batch_size=cfg.embedding_batch_size,
        embedding_cache_path=cache_path,
        umap_min_dist=cfg.umap_min_dist,
        umap_metric=cfg.umap_metric,
        umap_neighbors_ratio=cfg.umap_neighbors_ratio,
        n_clusters=safe_clusters,
        auto_k=cfg.auto_k,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        random_state=cfg.random_state,
        summary_backend=cfg.summary_backend,
        summary_model=cfg.summary_model,
        summary_temperature=cfg.summary_temperature,
        summary_max_examples=cfg.summary_max_examples,
        summary_max_chars_per_example=cfg.summary_max_chars_per_example,
    )


def run(cfg: SectionClusterConfig) -> None:
    try:
        from cluster.cluster_pipeline import run as run_cluster_pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import cluster pipeline dependencies. "
            "Install required packages first, e.g. `python3 -m pip install umap-learn scikit-learn pandas numpy`."
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(cfg.input_jsonl)
    by_tag, tag_stats, global_stats = build_tag_rows(records, cfg)

    manifest: Dict[str, Any] = {
        "config": to_jsonable(asdict(cfg)),
        "global_stats": global_stats,
        "tag_results": {},
        "timestamp": datetime.now().isoformat(),
    }

    tags = [normalize_tag(t) for t in cfg.tags]
    for tag in tags:
        split_name = SPLIT_NAME_MAP.get(tag, f"{tag.title()} Profiles")
        tag_dir = cfg.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        section_rows = by_tag[tag]
        tag_input = tag_dir / f"{tag.lower()}_sections_input.jsonl"
        tag_output = tag_dir / f"{tag.lower()}_clustered.jsonl"
        write_jsonl(tag_input, section_rows)

        tag_result: Dict[str, Any] = {
            "tag": tag,
            "split_name": split_name,
            "split_intro": f"Clusters built from <{tag}> section text only.",
            "stats": tag_stats[tag],
            "rows_for_clustering": len(section_rows),
            "paths": {
                "section_input_jsonl": to_repo_rel_str(tag_input),
            },
        }

        if len(section_rows) < 2:
            tag_result["status"] = "skipped_not_enough_rows"
            with (tag_dir / "split_summary.json").open("w", encoding="utf-8") as f:
                json.dump(tag_result, f, ensure_ascii=False, indent=2)
            manifest["tag_results"][tag] = tag_result
            continue

        cluster_cfg = make_cluster_config(
            cfg=cfg,
            tag_input_jsonl=tag_input,
            tag_output_jsonl=tag_output,
            tag_output_dir=tag_dir,
            n_rows=len(section_rows),
        )

        run_cluster_pipeline(cluster_cfg)

        metadata_path = tag_dir / "run_metadata.json"
        catalog_path = tag_dir / "cluster_catalog.json"
        umap_points_path = tag_dir / "umap_points.csv"

        tag_result["status"] = "ok"
        tag_result["paths"].update(
            {
                "clustered_jsonl": to_repo_rel_str(tag_output),
                "cluster_catalog_json": to_repo_rel_str(catalog_path),
                "run_metadata_json": to_repo_rel_str(metadata_path),
                "umap_points_csv": to_repo_rel_str(umap_points_path),
            }
        )

        with (tag_dir / "split_summary.json").open("w", encoding="utf-8") as f:
            json.dump(tag_result, f, ensure_ascii=False, indent=2)

        manifest["tag_results"][tag] = tag_result

    manifest_path = cfg.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> SectionClusterConfig:
    parser = argparse.ArgumentParser(description="Cluster persona summary sections per tag")
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

    parser.add_argument("--umap-min-dist", type=float, default=0.5)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--umap-neighbors-ratio", type=float, default=0.2)

    parser.add_argument("--clusters", type=int, default=15)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--summary-backend", choices=["openai", "keywords"], default="openai")
    parser.add_argument("--summary-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--summary-temperature", type=float, default=0.0)
    parser.add_argument("--summary-max-examples", type=int, default=20)
    parser.add_argument("--summary-max-chars-per-example", type=int, default=280)

    args = parser.parse_args()

    tags = [normalize_tag(t) for t in args.tags]
    tags = list(dict.fromkeys(tags))

    drop_unknown = not args.keep_unknown

    return SectionClusterConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        text_column=args.text_column,
        tags=tags,
        only_finished=not args.include_unfinished,
        drop_unknown=drop_unknown,
        unknown_mode=args.unknown_mode,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_batch_size=args.embedding_batch_size,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_neighbors_ratio=args.umap_neighbors_ratio,
        n_clusters=args.clusters,
        auto_k=args.auto_k,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        summary_backend=args.summary_backend,
        summary_model=args.summary_model,
        summary_temperature=args.summary_temperature,
        summary_max_examples=args.summary_max_examples,
        summary_max_chars_per_example=args.summary_max_chars_per_example,
    )


if __name__ == "__main__":
    run(parse_args())
