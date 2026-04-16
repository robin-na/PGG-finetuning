#!/usr/bin/env python3
"""Run retrieval-only oracle candidate search from query-writer batch outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_batch_results import extract_content, load_jsonl, parse_json_object  # type: ignore


DEFAULT_LIBRARY_DIR = (
    PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "oracle_library"
)
DEFAULT_QUERY_DIR = (
    PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "joint_social_oracle_query"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-store-id", type=str, required=True)
    parser.add_argument(
        "--responses-jsonl",
        type=Path,
        required=True,
        help="Batch output JSONL from build_joint_social_oracle_query_batch.py",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=DEFAULT_QUERY_DIR / "manifest_joint_social_oracle_query.jsonl",
    )
    parser.add_argument(
        "--library-manifest",
        type=Path,
        default=DEFAULT_LIBRARY_DIR / "oracle_library_manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-num-results", type=int, default=30)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--rewrite-query", action="store_true", default=True)
    parser.add_argument("--no-rewrite-query", dest="rewrite_query", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def extract_query_object(text: str) -> Optional[Dict[str, Any]]:
    obj = parse_json_object(text)
    if isinstance(obj, dict):
        return obj
    return None


def extract_query_text(obj: Dict[str, Any]) -> Optional[str]:
    for key in ("search_query", "query", "retrieval_query"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    return None


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
        if library_row:
            row["custom_id"] = library_row["custom_id"]
            row["split"] = library_row["split"]
            row["gameId"] = library_row["gameId"]
            row["playerId"] = library_row["playerId"]
            row["doc_path"] = library_row["doc_path"]
            row["demographics"] = library_row.get("demographics") or {}
            row["rule_summary_lines"] = library_row.get("rule_summary_lines") or []
    return ranked


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.responses_jsonl.parent / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_jsonl(args.manifest_jsonl)
    response_rows = load_jsonl(args.responses_jsonl)
    library_rows = load_jsonl(args.library_manifest)
    manifest_by_id = {str(row["custom_id"]): row for row in manifest_rows}
    library_by_filename = {str(row["filename"]): row for row in library_rows}

    candidates_path = output_dir / "oracle_candidates.jsonl"
    raw_search_path = output_dir / "oracle_search_raw.jsonl"
    errors_path = output_dir / "oracle_search_errors.jsonl"
    summary_path = output_dir / "oracle_search_summary.json"
    preview_path = output_dir / "oracle_search_preview.json"

    client = OpenAI()
    total = 0
    parsed_queries = 0
    retrieved = 0
    errors = 0
    first_preview: Optional[Dict[str, Any]] = None

    with candidates_path.open("w", encoding="utf-8") as candidates_f, raw_search_path.open(
        "w", encoding="utf-8"
    ) as raw_f, errors_path.open("w", encoding="utf-8") as err_f:
        for response_row in response_rows:
            custom_id = str(response_row.get("custom_id", ""))
            manifest = manifest_by_id.get(custom_id)
            if manifest is None:
                continue
            total += 1
            content, error_info = extract_content(response_row)
            if error_info is not None or content is None:
                err_f.write(
                    json.dumps(
                        {"custom_id": custom_id, "pid": manifest.get("pid"), "error": error_info},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors += 1
                continue

            obj = extract_query_object(content)
            if obj is None:
                err_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "pid": manifest.get("pid"),
                            "error": {"type": "invalid_json", "content": content},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors += 1
                continue

            query_text = extract_query_text(obj)
            if not query_text:
                err_f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "pid": manifest.get("pid"),
                            "error": {"type": "missing_search_query", "content": obj},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors += 1
                continue

            ranking_options: Dict[str, Any] = {"ranker": "auto"}
            if args.score_threshold is not None:
                ranking_options["score_threshold"] = args.score_threshold

            page = client.vector_stores.search(
                args.vector_store_id,
                query=query_text,
                max_num_results=args.max_num_results,
                ranking_options=ranking_options,
                rewrite_query=args.rewrite_query,
            )
            page_rows = getattr(page, "data", None)
            if page_rows is None:
                page_rows = list(page)

            ranked_candidates = search_results_to_entries(page_rows, library_by_filename)
            ranked_candidates = ranked_candidates[: args.top_k]
            parsed_queries += 1
            if ranked_candidates:
                retrieved += 1

            raw_f.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "pid": manifest.get("pid"),
                        "search_query": query_text,
                        "search_query_obj": obj,
                        "raw_result_count": len(page_rows),
                        "raw_results": [
                            {
                                "filename": row["filename"],
                                "score": row["score"],
                                "rank": row["rank"],
                                "snippets": row["snippets"],
                            }
                            for row in search_results_to_entries(page_rows, library_by_filename)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            candidate_row = {
                "custom_id": custom_id,
                "pid": manifest.get("pid"),
                "search_query": query_text,
                "search_query_obj": obj,
                "candidates": ranked_candidates,
            }
            candidates_f.write(json.dumps(candidate_row, ensure_ascii=False) + "\n")
            if first_preview is None:
                first_preview = candidate_row

            if args.limit is not None and parsed_queries >= args.limit:
                break

    if first_preview is not None:
        preview_path.write_text(json.dumps(first_preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "vector_store_id": args.vector_store_id,
        "total_response_rows": len(response_rows),
        "matched_manifest_rows": total,
        "parsed_queries": parsed_queries,
        "retrieval_rows_with_candidates": retrieved,
        "errors": errors,
        "top_k": args.top_k,
        "max_num_results": args.max_num_results,
        "rewrite_query": args.rewrite_query,
        "score_threshold": args.score_threshold,
        "candidates_path": str(candidates_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
