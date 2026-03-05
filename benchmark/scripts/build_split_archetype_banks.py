#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHETYPE_DIR = REPO_ROOT / "Persona" / "archetype_retrieval"
if str(ARCHETYPE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHETYPE_DIR))

from retrieval_common import (  # noqa: E402
    DEFAULT_TAGS,
    load_jsonl,
    pick_embedding_file,
    pick_sections_input_jsonl,
    write_jsonl,
)


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def row_game_player(row: Dict[str, Any]) -> Tuple[str, str]:
    game_id = normalize_id(row.get("gameId"))
    player_id = normalize_id(row.get("playerId"))
    if not game_id:
        game_id = normalize_id(row.get("experiment"))
    if not player_id:
        player_id = normalize_id(row.get("participant"))
    return game_id, player_id


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def split_rel_path(split_root: Path, split_base_root: Path) -> Path:
    split_root_resolved = split_root.resolve()
    split_base_resolved = split_base_root.resolve()
    benchmark_filtered_root = (split_base_resolved / "data").resolve()
    benchmark_ood_root = (split_base_resolved / "data_ood_splits").resolve()
    benchmark_ood_wave_root = (split_base_resolved / "data_ood_splits_wave_anchored").resolve()

    if split_root_resolved == benchmark_filtered_root:
        return Path("benchmark_filtered")
    try:
        rel_ood = split_root_resolved.relative_to(benchmark_ood_root)
        return Path("benchmark_ood") / rel_ood
    except Exception:
        pass
    try:
        rel_ood_wave = split_root_resolved.relative_to(benchmark_ood_wave_root)
        return Path("benchmark_ood_wave_anchored") / rel_ood_wave
    except Exception:
        pass
    try:
        return split_root_resolved.relative_to(split_base_resolved)
    except Exception:
        return Path(split_root.name)


@dataclass(frozen=True)
class TagSource:
    source_name: str
    tag: str
    sections_path: Path
    embedding_path: Path
    rows: List[Dict[str, Any]]
    embeddings: np.ndarray


def load_tag_source(source_root: Path, tag: str, source_name: str) -> TagSource:
    tag_dir = source_root / tag
    sections_path = pick_sections_input_jsonl(tag_dir)
    embedding_path = pick_embedding_file(tag_dir)

    rows = load_jsonl(sections_path)
    embeddings = np.load(embedding_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if len(rows) != int(embeddings.shape[0]):
        raise ValueError(
            f"{source_name}/{tag}: rows ({len(rows)}) != embeddings ({embeddings.shape[0]})"
        )

    return TagSource(
        source_name=source_name,
        tag=tag,
        sections_path=sections_path,
        embedding_path=embedding_path,
        rows=rows,
        embeddings=embeddings.astype(np.float32, copy=False),
    )


def load_game_player_pairs(player_rounds_csv: Path) -> set[Tuple[str, str]]:
    df = pd.read_csv(player_rounds_csv, usecols=["gameId", "playerId"])
    return set(zip(df["gameId"].astype(str), df["playerId"].astype(str)))


def select_rows_from_source(
    source: TagSource,
    allowed_pairs: set[Tuple[str, str]],
    dedupe_within_source: bool,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, int]]:
    selected_idx: List[int] = []
    seen: set[Tuple[str, str]] = set()
    missing_ids = 0
    duplicate_pairs = 0

    for idx, row in enumerate(source.rows):
        pair = row_game_player(row)
        if not pair[0] or not pair[1]:
            missing_ids += 1
            continue
        if pair not in allowed_pairs:
            continue
        if dedupe_within_source and pair in seen:
            duplicate_pairs += 1
            continue
        selected_idx.append(idx)
        seen.add(pair)

    if selected_idx:
        selected_embeddings = source.embeddings[np.array(selected_idx, dtype=int)]
    else:
        selected_embeddings = np.empty(
            (0, int(source.embeddings.shape[1])), dtype=source.embeddings.dtype
        )
    selected_rows = [source.rows[i] for i in selected_idx]

    stats = {
        "rows_total": int(len(source.rows)),
        "rows_selected": int(len(selected_rows)),
        "missing_id_rows": int(missing_ids),
        "duplicate_pairs_dropped": int(duplicate_pairs),
    }
    return selected_rows, selected_embeddings, stats


def merge_sources_for_pairs(
    sources: Sequence[TagSource],
    allowed_pairs: set[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    merged_rows: List[Dict[str, Any]] = []
    merged_embeddings: List[np.ndarray] = []
    seen_pairs: set[Tuple[str, str]] = set()
    source_stats: Dict[str, Dict[str, int]] = {}
    cross_source_duplicates = 0

    for source in sources:
        rows, embeddings, stats = select_rows_from_source(
            source=source,
            allowed_pairs=allowed_pairs,
            dedupe_within_source=True,
        )
        source_stats[source.source_name] = stats

        for i, row in enumerate(rows):
            pair = row_game_player(row)
            if pair in seen_pairs:
                cross_source_duplicates += 1
                continue
            seen_pairs.add(pair)
            merged_rows.append(row)
            merged_embeddings.append(embeddings[i])

    if merged_embeddings:
        emb = np.stack(merged_embeddings).astype(np.float32, copy=False)
    else:
        emb_dim = int(sources[0].embeddings.shape[1])
        emb = np.empty((0, emb_dim), dtype=np.float32)

    stats = {
        "pairs_requested": int(len(allowed_pairs)),
        "pairs_matched": int(len(seen_pairs)),
        "pair_coverage": float(len(seen_pairs) / len(allowed_pairs))
        if allowed_pairs
        else 0.0,
        "rows_written": int(len(merged_rows)),
        "embedding_rows_written": int(emb.shape[0]),
        "cross_source_duplicates_dropped": int(cross_source_duplicates),
        "source_stats": source_stats,
    }
    return merged_rows, emb, stats


def write_tag_outputs(
    out_tag_dir: Path,
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    sections_filename: str,
    embedding_filename: str,
    overwrite: bool,
) -> None:
    out_tag_dir.mkdir(parents=True, exist_ok=True)
    sections_out = out_tag_dir / sections_filename
    emb_out = out_tag_dir / embedding_filename

    if not overwrite:
        if sections_out.exists() or emb_out.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing files in {out_tag_dir}. "
                "Pass --overwrite to replace."
            )

    write_jsonl(sections_out, rows)
    np.save(emb_out, embeddings.astype(np.float32, copy=False))


def discover_split_roots(splits_root: Path) -> List[Path]:
    roots: List[Path] = []
    for factor_dir in sorted(splits_root.iterdir()):
        if not factor_dir.is_dir():
            continue
        for direction_dir in sorted(factor_dir.iterdir()):
            if not direction_dir.is_dir():
                continue
            train_csv = direction_dir / "raw_data" / "learning_wave" / "player-rounds.csv"
            test_csv = direction_dir / "raw_data" / "validation_wave" / "player-rounds.csv"
            if train_csv.exists() and test_csv.exists():
                roots.append(direction_dir)
    return roots


def build_for_split_root(
    split_root: Path,
    output_root: Path,
    learn_wave_root: Path,
    val_wave_root: Path,
    tags: Sequence[str],
    overwrite: bool,
) -> Dict[str, Any]:
    train_pairs = load_game_player_pairs(
        split_root / "raw_data" / "learning_wave" / "player-rounds.csv"
    )
    test_pairs = load_game_player_pairs(
        split_root / "raw_data" / "validation_wave" / "player-rounds.csv"
    )

    out_root = output_root
    wave_out_train = out_root / "learning_wave"
    wave_out_test = out_root / "validation_wave"

    summary: Dict[str, Any] = {
        "split_root": to_repo_rel(split_root),
        "output_root": to_repo_rel(out_root),
        "learn_source_root": to_repo_rel(learn_wave_root),
        "val_source_root": to_repo_rel(val_wave_root),
        "pairs_requested": {
            "learning_wave": int(len(train_pairs)),
            "validation_wave": int(len(test_pairs)),
        },
        "tags": {},
    }

    for raw_tag in tags:
        tag = str(raw_tag).strip().upper()
        learn_source = load_tag_source(learn_wave_root, tag, "learn_source")
        val_source = load_tag_source(val_wave_root, tag, "val_source")
        sources = [learn_source, val_source]

        train_rows, train_emb, train_stats = merge_sources_for_pairs(
            sources=sources,
            allowed_pairs=train_pairs,
        )
        test_rows, test_emb, test_stats = merge_sources_for_pairs(
            sources=sources,
            allowed_pairs=test_pairs,
        )

        sections_name = learn_source.sections_path.name
        emb_name = learn_source.embedding_path.name
        if val_source.sections_path.name != sections_name:
            sections_name = f"{tag.lower()}_sections_input.jsonl"
        if val_source.embedding_path.name != emb_name:
            emb_name = "embeddings_text-embedding-3-large_1536.npy"

        write_tag_outputs(
            out_tag_dir=wave_out_train / tag,
            rows=train_rows,
            embeddings=train_emb,
            sections_filename=sections_name,
            embedding_filename=emb_name,
            overwrite=overwrite,
        )
        write_tag_outputs(
            out_tag_dir=wave_out_test / tag,
            rows=test_rows,
            embeddings=test_emb,
            sections_filename=sections_name,
            embedding_filename=emb_name,
            overwrite=overwrite,
        )

        summary["tags"][tag] = {
            "learning_wave": train_stats,
            "validation_wave": test_stats,
            "output_files": {
                "learning_sections_jsonl": to_repo_rel(
                    wave_out_train / tag / sections_name
                ),
                "learning_embeddings_npy": to_repo_rel(wave_out_train / tag / emb_name),
                "validation_sections_jsonl": to_repo_rel(
                    wave_out_test / tag / sections_name
                ),
                "validation_embeddings_npy": to_repo_rel(wave_out_test / tag / emb_name),
            },
        }

    manifest_path = out_root / "manifest.json"
    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build split-specific archetype banks (sections + embeddings) for OOD datasets."
        )
    )
    parser.add_argument(
        "--split-root",
        action="append",
        default=[],
        help=(
            "Single split direction root, e.g. "
            "benchmark/data_ood_splits/all_or_nothing/false_to_true "
            "or benchmark/data_ood_splits_wave_anchored/all_or_nothing/false_to_true. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Build for every split direction under --splits-root.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("benchmark/data_ood_splits"),
        help="Parent directory for --all-splits discovery.",
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=Path("benchmark"),
        help=(
            "Base path used to map split roots into --output-root-base when --output-root "
            "is not provided."
        ),
    )
    parser.add_argument(
        "--learning-wave-source",
        type=Path,
        default=Path("Persona/archetype_retrieval/learning_wave"),
        help="Source archetype bank for the original learning wave.",
    )
    parser.add_argument(
        "--validation-wave-source",
        type=Path,
        default=Path("Persona/archetype_retrieval/validation_wave"),
        help="Source archetype bank for the original validation wave.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="archetype_retrieval",
        help="Subdirectory created under each computed output root.",
    )
    parser.add_argument(
        "--output-root-base",
        type=Path,
        default=Path("outputs/benchmark/runs"),
        help=(
            "Base output directory for split artifacts. "
            "Default maps to outputs/benchmark/runs/<split-relative-path>/<output-subdir>."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Explicit output root (single split only), e.g. "
            "outputs/benchmark/runs/benchmark_ood/all_or_nothing/false_to_true/archetype_retrieval."
        ),
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=list(DEFAULT_TAGS),
        help="Tags to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split archetype files if present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    split_roots: List[Path] = [resolve_repo_path(p) for p in args.split_root]
    if args.all_splits:
        split_roots.extend(discover_split_roots(resolve_repo_path(args.splits_root)))

    uniq: List[Path] = []
    seen: set[str] = set()
    for p in split_roots:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    split_roots = uniq

    if not split_roots:
        raise ValueError("No split roots selected. Use --split-root and/or --all-splits.")

    learn_source = resolve_repo_path(args.learning_wave_source)
    val_source = resolve_repo_path(args.validation_wave_source)
    split_base_root = resolve_repo_path(args.split_base_root)
    output_root_base = resolve_repo_path(args.output_root_base)
    tags = [str(t).strip().upper() for t in args.tags if str(t).strip()]

    if args.output_root is not None and len(split_roots) != 1:
        raise ValueError("--output-root can only be used with exactly one --split-root.")

    all_summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "learning_wave_source": to_repo_rel(learn_source),
        "validation_wave_source": to_repo_rel(val_source),
        "split_base_root": to_repo_rel(split_base_root),
        "output_root_base": to_repo_rel(output_root_base),
        "output_subdir": args.output_subdir,
        "tags": tags,
        "splits": {},
    }

    for split_root in split_roots:
        if args.output_root is not None:
            out_root = resolve_repo_path(args.output_root)
        else:
            rel = split_rel_path(split_root, split_base_root)
            out_root = output_root_base / rel / args.output_subdir
        result = build_for_split_root(
            split_root=split_root,
            output_root=out_root,
            learn_wave_root=learn_source,
            val_wave_root=val_source,
            tags=tags,
            overwrite=args.overwrite,
        )
        all_summary["splits"][to_repo_rel(split_root)] = result

    if args.all_splits:
        summary_path = output_root_base / "archetype_build_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
