#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from cluster.cluster_pipeline import ClusterConfig, run as run_cluster_pipeline
except ModuleNotFoundError:  # pragma: no cover
    from cluster_pipeline import ClusterConfig, run as run_cluster_pipeline


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

TAG_SLUG = {
    "CONTRIBUTION": "contribution",
    "PUNISHMENT": "punishment",
    "REWARD": "reward",
    "COMMUNICATION": "communication",
    "RESPONSE_TO_PUNISHER": "response_to_punisher",
    "RESPONSE_TO_REWARDER": "response_to_rewarder",
    "RESPONSE_TO_OTHERS_OUTCOME": "response_to_others_outcome",
    "RESPONSE_TO_END_GAME": "response_to_end_game",
}

TAG_SPLIT_NAME = {
    "CONTRIBUTION": "Contribution Strategy Profiles",
    "PUNISHMENT": "Punishment Strategy Profiles",
    "REWARD": "Reward Strategy Profiles",
    "COMMUNICATION": "Communication Style Profiles",
    "RESPONSE_TO_PUNISHER": "Response to Punisher Profiles",
    "RESPONSE_TO_REWARDER": "Response to Rewarder Profiles",
    "RESPONSE_TO_OTHERS_OUTCOME": "Response to Others' Outcome Profiles",
    "RESPONSE_TO_END_GAME": "End-Game Response Profiles",
}

TAG_ALIAS = {
    "PAINISHMENT": "PUNISHMENT",
    "PPUNISHMENT": "PUNISHMENT",
    "PUINISHMENT": "PUNISHMENT",
    "PUISHMENT": "PUNISHMENT",
    "PAINPT": "PUNISHMENT",
    "RERWARD": "REWARD",
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_tag_token(token: str) -> Optional[str]:
    t = _normalize_text(token).upper()
    if t in TAG_ALIAS:
        return TAG_ALIAS[t]
    if t in TAG_SLUG:
        return t
    # Small typo recovery fallback.
    best_tag = None
    best_score = -1.0
    for cand in TAG_SLUG:
        score = SequenceMatcher(None, t, cand).ratio()
        if score > best_score:
            best_tag = cand
            best_score = score
    if best_tag is not None and best_score >= 0.82:
        return best_tag
    return None


def _extract_sections(text: str) -> Tuple[Dict[str, str], List[str]]:
    pattern = re.compile(r"<([A-Z_]+)>\s*", flags=0)
    matches = list(pattern.finditer(text))
    if not matches:
        return {}, []

    sections: Dict[str, str] = {}
    unknown_tokens: List[str] = []
    for i, m in enumerate(matches):
        raw_tag = m.group(1)
        normalized_tag = _normalize_tag_token(raw_tag)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = _normalize_text(text[start:end])
        if normalized_tag is None:
            unknown_tokens.append(raw_tag)
            continue
        if not section:
            continue
        if normalized_tag in sections:
            sections[normalized_tag] = _normalize_text(sections[normalized_tag] + " " + section)
        else:
            sections[normalized_tag] = section
    return sections, unknown_tokens


def _build_section_records(
    rows: List[Dict[str, Any]],
    tags: List[str],
    text_column: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]], Dict[str, int]]:
    out: Dict[str, List[Dict[str, Any]]] = {tag: [] for tag in tags}
    stats: Dict[str, Dict[str, int]] = {
        tag: {
            "rows_considered": len(rows),
            "missing_tag": 0,
            "empty_tag": 0,
            "dropped_unknown": 0,
            "kept": 0,
        }
        for tag in tags
    }
    global_unknown_counts: Dict[str, int] = {}

    for row in rows:
        text = _normalize_text(row.get(text_column, ""))
        sections, unknown_tokens = _extract_sections(text)
        for token in unknown_tokens:
            global_unknown_counts[token] = global_unknown_counts.get(token, 0) + 1

        for tag in tags:
            section = sections.get(tag)
            if section is None:
                stats[tag]["missing_tag"] += 1
                continue
            if not section:
                stats[tag]["empty_tag"] += 1
                continue
            out[tag].append(
                {
                    "experiment": row.get("experiment"),
                    "participant": row.get("participant"),
                    "game_finished": row.get("game_finished"),
                    "tag": tag,
                    "text": section,
                    "section_text": section,
                }
            )
            stats[tag]["kept"] += 1

    return out, stats, global_unknown_counts


def _make_paths_for_tag(output_root: Path, tag: str) -> Dict[str, Path]:
    tag_dir = output_root / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    slug = TAG_SLUG[tag]
    return {
        "tag_dir": tag_dir,
        "section_input_jsonl": tag_dir / f"{slug}_sections_input.jsonl",
        "clustered_jsonl": tag_dir / f"{slug}_clustered.jsonl",
        "clustered_polished_jsonl": tag_dir / f"{slug}_clustered_polished.jsonl",
        "cluster_catalog_json": tag_dir / "cluster_catalog.json",
        "cluster_catalog_polished_json": tag_dir / "cluster_catalog_polished.json",
        "cluster_catalog_polish_report_json": tag_dir / "cluster_catalog_polish_report.json",
        "run_metadata_json": tag_dir / "run_metadata.json",
        "umap_points_csv": tag_dir / "umap_points.csv",
        "split_summary_json": tag_dir / "split_summary.json",
    }


def run(args: argparse.Namespace) -> None:
    rows = _load_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"No rows found in: {args.input_jsonl}")

    rows_total = len(rows)
    rows_after_finished = rows
    if not args.include_unfinished:
        rows_after_finished = [r for r in rows if bool(r.get("game_finished", True))]

    rows_missing_text = sum(1 for r in rows_after_finished if not _normalize_text(r.get(args.text_column, "")))
    filtered_rows = [r for r in rows_after_finished if _normalize_text(r.get(args.text_column, ""))]

    tag_list = DEFAULT_TAGS
    if args.only_tags:
        requested = [_normalize_text(t).upper() for t in args.only_tags.split(",") if _normalize_text(t)]
        invalid = [t for t in requested if t not in TAG_SLUG]
        if invalid:
            raise ValueError(f"Invalid tags in --only-tags: {invalid}")
        tag_list = requested

    sections_by_tag, split_stats, unknown_tag_counts = _build_section_records(
        rows=filtered_rows,
        tags=tag_list,
        text_column=args.text_column,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    tag_results: Dict[str, Dict[str, Any]] = {}
    for tag in tag_list:
        paths = _make_paths_for_tag(output_root, tag)
        section_rows = sections_by_tag[tag]
        _write_jsonl(paths["section_input_jsonl"], section_rows)

        split_summary: Dict[str, Any] = {
            "tag": tag,
            "split_name": TAG_SPLIT_NAME.get(tag, tag.title()),
            "split_intro": f"Clusters built from <{tag}> section text only.",
            "stats": split_stats[tag],
            "rows_for_clustering": len(section_rows),
            "paths": {
                "section_input_jsonl": str(paths["section_input_jsonl"]),
                "clustered_jsonl": str(paths["clustered_jsonl"]),
                "cluster_catalog_json": str(paths["cluster_catalog_json"]),
                "run_metadata_json": str(paths["run_metadata_json"]),
                "umap_points_csv": str(paths["umap_points_csv"]),
                "clustered_polished_jsonl": str(paths["clustered_polished_jsonl"]),
                "cluster_catalog_polished_json": str(paths["cluster_catalog_polished_json"]),
                "cluster_catalog_polish_report_json": str(paths["cluster_catalog_polish_report_json"]),
            },
            "status": "ok",
        }

        if args.skip_existing and paths["clustered_jsonl"].exists():
            split_summary["status"] = "skipped_existing"
            paths["split_summary_json"].write_text(
                json.dumps(split_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tag_results[tag] = split_summary
            print(f"[{tag}] skipped existing clustered file: {paths['clustered_jsonl']}")
            continue

        if len(section_rows) < 2:
            split_summary["status"] = "skipped_not_enough_rows"
            paths["split_summary_json"].write_text(
                json.dumps(split_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tag_results[tag] = split_summary
            print(f"[{tag}] skipped: not enough rows ({len(section_rows)})")
            continue

        cache_path: Optional[Path] = None
        if args.embedding_backend == "openai":
            model_tag = args.embedding_model.replace("/", "_")
            cache_path = paths["tag_dir"] / f"embeddings_{model_tag}_{args.embedding_dimensions}.npy"

        cfg = ClusterConfig(
            input_jsonl=paths["section_input_jsonl"],
            output_jsonl=paths["clustered_jsonl"],
            output_dir=paths["tag_dir"],
            text_column="text",
            only_finished=False,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            embedding_batch_size=args.embedding_batch_size,
            embedding_cache_path=cache_path,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            umap_neighbors_ratio=args.umap_neighbors_ratio,
            n_clusters=args.clusters,
            auto_k=args.auto_k,
            k_min=args.k_min,
            k_max=args.k_max,
            random_state=args.random_state,
            cluster_space=args.cluster_space,
            enable_split_merge=not args.disable_split_merge,
            max_split_merge_iters=args.max_split_merge_iters,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            merge_similarity_threshold=args.merge_similarity_threshold,
            target_overlap_rate=args.target_overlap_rate,
            point_overlap_margin=args.point_overlap_margin,
            split_cluster_min_size=args.split_cluster_min_size,
            summary_backend=args.summary_backend,
            summary_model=args.summary_model,
            summary_temperature=args.summary_temperature,
            summary_max_examples=args.summary_max_examples,
            summary_max_chars_per_example=args.summary_max_chars_per_example,
        )

        try:
            print(f"[{tag}] clustering {len(section_rows)} rows...")
            run_cluster_pipeline(cfg)
            split_summary["status"] = "ok"
        except Exception as exc:
            split_summary["status"] = "error"
            split_summary["error"] = str(exc)
            print(f"[{tag}] error: {exc}")

        if split_summary["status"] == "ok" and args.polish_with_llm:
            polish_script = Path(__file__).with_name("polish_cluster_catalog.py")
            cmd = [
                sys.executable,
                str(polish_script),
                "--cluster-catalog",
                str(paths["cluster_catalog_json"]),
                "--clustered-jsonl",
                str(paths["clustered_jsonl"]),
                "--output-catalog",
                str(paths["cluster_catalog_polished_json"]),
                "--write-clustered-jsonl",
                "--output-clustered-jsonl",
                str(paths["clustered_polished_jsonl"]),
                "--overlap-report",
                str(paths["cluster_catalog_polish_report_json"]),
                "--model",
                args.polish_model,
                "--temperature",
                str(args.polish_temperature),
                "--max-examples",
                str(args.polish_max_examples),
                "--max-chars-per-example",
                str(args.polish_max_chars_per_example),
                "--max-retries",
                str(args.polish_max_retries),
                "--retry-seconds",
                str(args.polish_retry_seconds),
                "--max-passes",
                str(args.polish_max_passes),
                "--max-title-similarity",
                str(args.polish_max_title_similarity),
                "--max-intro-similarity",
                str(args.polish_max_intro_similarity),
                "--max-combined-similarity",
                str(args.polish_max_combined_similarity),
            ]
            if args.polish_strict_overlap_check:
                cmd.append("--strict-overlap-check")

            try:
                print(f"[{tag}] running integrated LLM polish...")
                subprocess.run(cmd, check=True)
                split_summary["polish_status"] = "ok"
            except Exception as exc:
                split_summary["polish_status"] = "error"
                split_summary["polish_error"] = str(exc)
                print(f"[{tag}] polish error: {exc}")

        paths["split_summary_json"].write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tag_results[tag] = split_summary

    manifest = {
        "config": {
            "input_jsonl": str(args.input_jsonl),
            "output_dir": str(output_root),
            "text_column": args.text_column,
            "tags": tag_list,
            "only_finished": not args.include_unfinished,
            "drop_unknown": False,
            "unknown_mode": "coerce_to_closest_or_drop",
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model if args.embedding_backend == "openai" else "tfidf",
            "embedding_dimensions": args.embedding_dimensions,
            "embedding_batch_size": args.embedding_batch_size,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": args.umap_metric,
            "umap_neighbors_ratio": args.umap_neighbors_ratio,
            "n_clusters": args.clusters,
            "auto_k": args.auto_k,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "random_state": args.random_state,
            "cluster_space": args.cluster_space,
            "enable_split_merge": not args.disable_split_merge,
            "max_split_merge_iters": args.max_split_merge_iters,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "merge_similarity_threshold": args.merge_similarity_threshold,
            "target_overlap_rate": args.target_overlap_rate,
            "point_overlap_margin": args.point_overlap_margin,
            "split_cluster_min_size": args.split_cluster_min_size,
            "summary_backend": args.summary_backend,
            "summary_model": args.summary_model if args.summary_backend == "openai" else None,
            "summary_temperature": args.summary_temperature,
            "summary_max_examples": args.summary_max_examples,
            "summary_max_chars_per_example": args.summary_max_chars_per_example,
            "polish_with_llm": args.polish_with_llm,
            "polish_model": args.polish_model if args.polish_with_llm else None,
            "polish_temperature": args.polish_temperature,
            "polish_max_examples": args.polish_max_examples,
            "polish_max_chars_per_example": args.polish_max_chars_per_example,
            "polish_max_retries": args.polish_max_retries,
            "polish_retry_seconds": args.polish_retry_seconds,
            "polish_max_passes": args.polish_max_passes,
            "polish_max_title_similarity": args.polish_max_title_similarity,
            "polish_max_intro_similarity": args.polish_max_intro_similarity,
            "polish_max_combined_similarity": args.polish_max_combined_similarity,
            "polish_strict_overlap_check": args.polish_strict_overlap_check,
            "skip_existing": args.skip_existing,
        },
        "global_stats": {
            "rows_total": rows_total,
            "rows_after_finished_filter": len(rows_after_finished),
            "rows_missing_text": rows_missing_text,
            "unknown_tag_tokens": unknown_tag_counts,
        },
        "tag_results": tag_results,
        "timestamp": _now_utc_iso(),
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split persona text by section tags and run clustering per tag."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--only-tags", type=str, default=None, help="Comma-separated subset of tags.")
    parser.add_argument("--skip-existing", action="store_true")

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
    parser.add_argument("--cluster-space", choices=["embedding", "umap"], default="embedding")

    parser.add_argument("--disable-split-merge", action="store_true")
    parser.add_argument("--max-split-merge-iters", type=int, default=20)
    parser.add_argument("--min-clusters", type=int, default=6)
    parser.add_argument("--max-clusters", type=int, default=25)
    parser.add_argument("--merge-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--target-overlap-rate", type=float, default=0.12)
    parser.add_argument("--point-overlap-margin", type=float, default=0.03)
    parser.add_argument("--split-cluster-min-size", type=int, default=60)

    parser.add_argument("--summary-backend", choices=["openai", "keywords"], default="keywords")
    parser.add_argument("--summary-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--summary-temperature", type=float, default=0.0)
    parser.add_argument("--summary-max-examples", type=int, default=20)
    parser.add_argument("--summary-max-chars-per-example", type=int, default=280)

    parser.add_argument("--polish-with-llm", action="store_true")
    parser.add_argument("--polish-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--polish-temperature", type=float, default=0.0)
    parser.add_argument("--polish-max-examples", type=int, default=12)
    parser.add_argument("--polish-max-chars-per-example", type=int, default=280)
    parser.add_argument("--polish-max-retries", type=int, default=5)
    parser.add_argument("--polish-retry-seconds", type=float, default=1.0)
    parser.add_argument("--polish-max-passes", type=int, default=3)
    parser.add_argument("--polish-max-title-similarity", type=float, default=0.88)
    parser.add_argument("--polish-max-intro-similarity", type=float, default=0.92)
    parser.add_argument("--polish-max-combined-similarity", type=float, default=0.84)
    parser.add_argument("--polish-strict-overlap-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
