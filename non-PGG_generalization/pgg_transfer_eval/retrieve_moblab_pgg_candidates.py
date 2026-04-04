#!/usr/bin/env python3
"""Retrieve PGG persona cards for MobLab tasks using the OpenAI retrieval API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from evaluate_batch_results import extract_content, parse_json_object  # type: ignore
from moblab_llm_utils import DEFAULT_OUTPUT_ROOT, load_jsonl, normalize_whitespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "oracle_library"
DEFAULT_QUERY_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-store-id", type=str, required=True)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--library-manifest", type=Path, default=DEFAULT_LIBRARY_DIR / "oracle_library_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QUERY_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-num-results", type=int, default=20)
    parser.add_argument("--rewrite-query", action="store_true", default=True)
    parser.add_argument("--no-rewrite-query", dest="rewrite_query", action="store_false")
    parser.add_argument(
        "--attribute-filter-json",
        type=Path,
        default=None,
        help="Optional JSON file containing the attribute_filter payload for vector_stores.search.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def search_results_to_entries(
    results: Iterable[Any],
    library_by_filename: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in results:
        filename = str(getattr(item, "filename", "") or "")
        if not filename:
            continue
        score = float(getattr(item, "score", 0.0) or 0.0)
        snippets = [
            normalize_whitespace(getattr(chunk, "text", "") or "")
            for chunk in (getattr(item, "content", None) or [])
            if normalize_whitespace(getattr(chunk, "text", "") or "")
        ]
        attributes = getattr(item, "attributes", None) or {}
        current = deduped.get(filename)
        if current is None or score > current["score"]:
            current = {
                "filename": filename,
                "score": score,
                "file_id": getattr(item, "file_id", None),
                "attributes": attributes,
                "snippets": [],
            }
            deduped[filename] = current
        for snippet in snippets:
            if snippet not in current["snippets"]:
                current["snippets"].append(snippet)
        current["snippets"] = current["snippets"][:3]

    ranked = sorted(deduped.values(), key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        library_row = library_by_filename.get(row["filename"])
        if not library_row:
            continue
        row["custom_id"] = library_row["custom_id"]
        row["split"] = library_row["split"]
        row["gameId"] = library_row["gameId"]
        row["playerId"] = library_row["playerId"]
        row["doc_path"] = library_row["doc_path"]
        doc_path = Path(str(library_row["doc_path"]))
        if doc_path.exists():
            row["document_text"] = doc_path.read_text(encoding="utf-8")
        else:
            row["document_text"] = ""
    return ranked


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_id = {str(row["custom_id"]): row for row in load_jsonl(args.manifest_jsonl)}
    library_rows = load_jsonl(args.library_manifest)
    library_by_filename = {str(row["filename"]): row for row in library_rows}
    response_rows = load_jsonl(args.responses_jsonl)
    attribute_filter = None
    if args.attribute_filter_json is not None:
        attribute_filter = json.loads(args.attribute_filter_json.read_text(encoding="utf-8"))

    candidates_path = args.output_dir / "moblab_pgg_candidates.jsonl"
    raw_path = args.output_dir / "moblab_pgg_search_raw.jsonl"
    errors_path = args.output_dir / "moblab_pgg_search_errors.jsonl"
    preview_path = args.output_dir / "moblab_pgg_search_preview.json"
    summary_path = args.output_dir / "moblab_pgg_search_summary.json"

    client = OpenAI()
    total = 0
    retrieved = 0
    errors = 0
    first_preview: Optional[Dict[str, Any]] = None

    with candidates_path.open("w", encoding="utf-8") as cand_f, raw_path.open("w", encoding="utf-8") as raw_f, errors_path.open("w", encoding="utf-8") as err_f:
        for response_row in response_rows:
            custom_id = str(response_row.get("custom_id", ""))
            manifest = manifest_by_id.get(custom_id)
            if manifest is None:
                continue
            total += 1
            content, error, _ = extract_content(response_row)
            if error is not None or not content:
                err_f.write(json.dumps({"custom_id": custom_id, "error": error}, ensure_ascii=False) + "\n")
                errors += 1
                continue
            obj = parse_json_object(content)
            if not isinstance(obj, dict) or not isinstance(obj.get("search_query"), str):
                err_f.write(json.dumps({"custom_id": custom_id, "error": "missing search_query", "content": content}, ensure_ascii=False) + "\n")
                errors += 1
                continue
            query_text = normalize_whitespace(obj["search_query"])
            search_kwargs = {
                "query": query_text,
                "max_num_results": args.max_num_results,
                "rewrite_query": args.rewrite_query,
            }
            if attribute_filter is not None:
                search_kwargs["filters"] = attribute_filter
            page = client.vector_stores.search(args.vector_store_id, **search_kwargs)
            page_rows = getattr(page, "data", None)
            if page_rows is None:
                page_rows = list(page)

            ranked = search_results_to_entries(page_rows, library_by_filename)
            kept = ranked[: args.top_k]
            if kept:
                retrieved += 1
            raw_f.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "instance_id": manifest["instance_id"],
                        "search_query": query_text,
                        "raw_result_count": len(page_rows),
                        "raw_results": ranked,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            row = {
                "custom_id": custom_id,
                "instance_id": manifest["instance_id"],
                "task_type": manifest["task_type"],
                "target_measure": manifest["target_measure"],
                "search_query": query_text,
                "match_cues": obj.get("match_cues"),
                "candidates": kept,
            }
            cand_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if first_preview is None:
                first_preview = row
            if args.limit is not None and total >= args.limit:
                break

    summary = {
        "total_queries": total,
        "retrieved_queries": retrieved,
        "errors": errors,
        "vector_store_id": args.vector_store_id,
        "top_k": args.top_k,
        "rewrite_query": args.rewrite_query,
        "attribute_filter_used": bool(attribute_filter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if first_preview is not None:
        preview_path.write_text(json.dumps(first_preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
