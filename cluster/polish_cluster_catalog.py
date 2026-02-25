#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _parse_summary_json(raw: str) -> Optional[Dict[str, str]]:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None
    title = _normalize_text(obj.get("title", ""))
    intro = _normalize_text(obj.get("intro", ""))
    if not title or not intro:
        return None
    return {"title": title, "intro": intro}


def _compact_text(text: str, max_chars: int) -> str:
    t = _normalize_text(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _norm_for_similarity(text: Any) -> str:
    s = _normalize_text(text).lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _jaccard_tokens(a: str, b: str) -> float:
    sa = set(_norm_for_similarity(a).split())
    sb = set(_norm_for_similarity(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _pair_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    a_title = _norm_for_similarity(a.get("cluster_title", ""))
    b_title = _norm_for_similarity(b.get("cluster_title", ""))
    a_intro = _norm_for_similarity(a.get("cluster_intro", ""))
    b_intro = _norm_for_similarity(b.get("cluster_intro", ""))

    title_seq = float(SequenceMatcher(None, a_title, b_title).ratio())
    intro_seq = float(SequenceMatcher(None, a_intro, b_intro).ratio())
    title_j = _jaccard_tokens(a_title, b_title)
    intro_j = _jaccard_tokens(a_intro, b_intro)
    combined = 0.45 * title_seq + 0.25 * title_j + 0.20 * intro_seq + 0.10 * intro_j
    return {
        "title_seq_similarity": title_seq,
        "intro_seq_similarity": intro_seq,
        "title_jaccard": title_j,
        "intro_jaccard": intro_j,
        "combined_similarity": float(combined),
    }


def _find_overlap_conflicts(
    catalog_rows: List[Dict[str, Any]],
    max_title_similarity: float,
    max_intro_similarity: float,
    max_combined_similarity: float,
) -> List[Dict[str, Any]]:
    rows = sorted(catalog_rows, key=lambda r: int(r["cluster_id"]))
    conflicts: List[Dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = rows[i]
            b = rows[j]
            sim = _pair_similarity(a, b)
            if (
                sim["title_seq_similarity"] >= max_title_similarity
                or sim["intro_seq_similarity"] >= max_intro_similarity
                or sim["combined_similarity"] >= max_combined_similarity
            ):
                conflicts.append(
                    {
                        "cluster_id_a": int(a["cluster_id"]),
                        "cluster_title_a": _normalize_text(a.get("cluster_title", "")),
                        "cluster_id_b": int(b["cluster_id"]),
                        "cluster_title_b": _normalize_text(b.get("cluster_title", "")),
                        **sim,
                    }
                )
    conflicts.sort(
        key=lambda c: (
            c["combined_similarity"],
            c["title_seq_similarity"],
            c["intro_seq_similarity"],
        ),
        reverse=True,
    )
    return conflicts


def _enforce_exact_title_uniqueness(catalog_rows: List[Dict[str, Any]]) -> None:
    """
    Ensure no exact duplicate titles remain. This is a deterministic fallback only.
    """
    title_to_ids: Dict[str, List[int]] = {}
    for row in catalog_rows:
        cid = int(row["cluster_id"])
        title = _normalize_text(row.get("cluster_title", ""))
        title_to_ids.setdefault(title, []).append(cid)

    dup_titles = {t: ids for t, ids in title_to_ids.items() if t and len(ids) > 1}
    if not dup_titles:
        return

    by_id = {int(r["cluster_id"]): r for r in catalog_rows}
    for title, ids in dup_titles.items():
        ids_sorted = sorted(ids)
        for cid in ids_sorted[1:]:
            row = by_id[cid]
            row["cluster_title"] = f"{_normalize_text(row['cluster_title'])} (cluster {cid})"


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "openai package is not available. Install with: python3 -m pip install openai"
        ) from exc
    return OpenAI(api_key=api_key)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_examples_from_clustered(
    clustered_jsonl: Path,
    text_column: str,
    max_examples: int,
    max_chars_per_example: int,
) -> Dict[int, List[str]]:
    rows = _load_jsonl(clustered_jsonl)
    grouped: Dict[int, List[str]] = {}
    seen: Dict[int, set] = {}
    for row in rows:
        cid = int(row["cluster_id"])
        text = row.get(text_column, "")
        compact = _compact_text(text, max_chars_per_example)
        grouped.setdefault(cid, [])
        seen.setdefault(cid, set())
        if compact in seen[cid]:
            continue
        grouped[cid].append(compact)
        seen[cid].add(compact)
        if len(grouped[cid]) > max_examples:
            grouped[cid] = grouped[cid][:max_examples]
    return grouped


def _polish_one_cluster(
    client,
    row: Dict[str, Any],
    peer_titles: List[str],
    conflict_peers: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_retries: int,
    retry_seconds: float,
) -> Dict[str, str]:
    cid = int(row["cluster_id"])
    title = _normalize_text(row.get("cluster_title", ""))
    intro = _normalize_text(row.get("cluster_intro", ""))
    top_terms = row.get("top_terms") or []
    if not isinstance(top_terms, list):
        top_terms = []
    examples = row.get("sample_examples") or []
    if not isinstance(examples, list):
        examples = []

    prompt = (
        "You are polishing cluster labels for a behavioral clustering report.\n"
        "Return a better title and intro for this one cluster.\n\n"
        "Rules:\n"
        "- Title: 4 to 10 words.\n"
        "- Intro: 2 to 4 sentences.\n"
        "- Keep behavior-focused and non-redundant.\n"
        "- Make both title and intro distinct from similar peer clusters.\n"
        "- Do not mention IDs or this prompt.\n\n"
        f"Cluster id: {cid}\n"
        f"Current title: {title}\n"
        f"Current intro: {intro}\n"
        f"Top terms: {', '.join(str(x) for x in top_terms[:12]) if top_terms else 'N/A'}\n"
        f"Peer titles: {', '.join(peer_titles[:25]) if peer_titles else 'N/A'}\n\n"
        "Potentially overlapping peer clusters to differentiate from:\n"
        + ("\n".join(
            f"- peer title: {_normalize_text(p.get('cluster_title',''))}; peer intro: {_compact_text(_normalize_text(p.get('cluster_intro','')), 220)}"
            for p in conflict_peers[:8]
        ) if conflict_peers else "- none")
        + "\n\n"
        "Representative snippets:\n"
        + "\n".join(f"- {_normalize_text(x)}" for x in examples[:12])
        + '\n\nReturn JSON only in this schema: {"title":"...","intro":"..."}'
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_summary_json(raw)
            if parsed:
                return parsed
            raise RuntimeError("Invalid JSON response format.")
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to polish cluster {cid}") from exc
            wait_s = retry_seconds * (2**attempt)
            print(
                f"[polish] retry cluster={cid}, attempt={attempt + 1}, wait={wait_s:.1f}s"
            )
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to polish cluster {cid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polish existing cluster catalog titles/intros with an LLM pass."
    )
    parser.add_argument("--cluster-catalog", type=Path, required=True)
    parser.add_argument("--clustered-jsonl", type=Path, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--output-catalog", type=Path, default=None)
    parser.add_argument("--write-clustered-jsonl", action="store_true")
    parser.add_argument("--output-clustered-jsonl", type=Path, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--max-chars-per-example", type=int, default=280)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=1.0)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--max-title-similarity", type=float, default=0.88)
    parser.add_argument("--max-intro-similarity", type=float, default=0.92)
    parser.add_argument("--max-combined-similarity", type=float, default=0.84)
    parser.add_argument("--overlap-report", type=Path, default=None)
    parser.add_argument("--strict-overlap-check", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(args.cluster_catalog.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("Cluster catalog must be a non-empty JSON list.")

    if args.output_catalog is None:
        args.output_catalog = args.cluster_catalog.with_name(
            f"{args.cluster_catalog.stem}_polished.json"
        )
    if args.overlap_report is None:
        args.overlap_report = args.output_catalog.with_name(
            f"{args.output_catalog.stem}_overlap_report.json"
        )

    extra_examples: Dict[int, List[str]] = {}
    if args.clustered_jsonl is not None:
        extra_examples = _collect_examples_from_clustered(
            clustered_jsonl=args.clustered_jsonl,
            text_column=args.text_column,
            max_examples=args.max_examples,
            max_chars_per_example=args.max_chars_per_example,
        )

    for row in catalog:
        cid = int(row["cluster_id"])
        examples = row.get("sample_examples") or []
        if not isinstance(examples, list):
            examples = []
        combined = [
            _compact_text(str(x), args.max_chars_per_example)
            for x in examples
            if _normalize_text(x)
        ]
        for x in extra_examples.get(cid, []):
            if x not in combined:
                combined.append(x)
        row["sample_examples"] = combined[: args.max_examples]

    client = _get_openai_client()
    current = [dict(r) for r in sorted(catalog, key=lambda r: int(r["cluster_id"]))]
    by_id = {int(r["cluster_id"]): r for r in current}

    pass_summaries: List[Dict[str, Any]] = []
    initial_conflicts = _find_overlap_conflicts(
        catalog_rows=current,
        max_title_similarity=args.max_title_similarity,
        max_intro_similarity=args.max_intro_similarity,
        max_combined_similarity=args.max_combined_similarity,
    )

    for pass_idx in range(1, int(args.max_passes) + 1):
        conflicts_before = _find_overlap_conflicts(
            catalog_rows=current,
            max_title_similarity=args.max_title_similarity,
            max_intro_similarity=args.max_intro_similarity,
            max_combined_similarity=args.max_combined_similarity,
        )
        if pass_idx == 1:
            target_ids = sorted(by_id.keys())
        else:
            ids = set()
            for c in conflicts_before:
                ids.add(int(c["cluster_id_a"]))
                ids.add(int(c["cluster_id_b"]))
            target_ids = sorted(ids)
            if not target_ids:
                pass_summaries.append(
                    {
                        "pass": pass_idx,
                        "target_cluster_ids": [],
                        "n_conflicts_before": len(conflicts_before),
                        "n_conflicts_after": 0,
                        "stopped_early": True,
                    }
                )
                break

        for cid in target_ids:
            row = by_id[cid]
            id_to_title = {
                int(r["cluster_id"]): _normalize_text(r.get("cluster_title", ""))
                for r in current
            }
            peer_titles = [title for other_cid, title in id_to_title.items() if other_cid != cid and title]
            conflict_ids = set()
            for c in conflicts_before:
                if int(c["cluster_id_a"]) == cid:
                    conflict_ids.add(int(c["cluster_id_b"]))
                elif int(c["cluster_id_b"]) == cid:
                    conflict_ids.add(int(c["cluster_id_a"]))
            conflict_peers = [by_id[x] for x in sorted(conflict_ids) if x in by_id]

            polished_text = _polish_one_cluster(
                client=client,
                row=row,
                peer_titles=peer_titles,
                conflict_peers=conflict_peers,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_seconds=args.retry_seconds,
            )
            row["cluster_title"] = polished_text["title"]
            row["cluster_intro"] = polished_text["intro"]
            row["summary_source"] = f"openai-polished-pass-{pass_idx}"
            print(f"[polish] pass={pass_idx} cluster={cid} done")

        _enforce_exact_title_uniqueness(current)
        conflicts_after = _find_overlap_conflicts(
            catalog_rows=current,
            max_title_similarity=args.max_title_similarity,
            max_intro_similarity=args.max_intro_similarity,
            max_combined_similarity=args.max_combined_similarity,
        )
        pass_summaries.append(
            {
                "pass": pass_idx,
                "target_cluster_ids": target_ids,
                "n_conflicts_before": len(conflicts_before),
                "n_conflicts_after": len(conflicts_after),
            }
        )
        if not conflicts_after:
            break

    polished = sorted(current, key=lambda r: int(r["cluster_id"]))
    final_conflicts = _find_overlap_conflicts(
        catalog_rows=polished,
        max_title_similarity=args.max_title_similarity,
        max_intro_similarity=args.max_intro_similarity,
        max_combined_similarity=args.max_combined_similarity,
    )

    args.output_catalog.write_text(
        json.dumps(polished, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved polished catalog: {args.output_catalog}")

    overlap_report = {
        "cluster_catalog": str(args.cluster_catalog),
        "output_catalog": str(args.output_catalog),
        "thresholds": {
            "max_title_similarity": float(args.max_title_similarity),
            "max_intro_similarity": float(args.max_intro_similarity),
            "max_combined_similarity": float(args.max_combined_similarity),
        },
        "passes": pass_summaries,
        "n_conflicts_initial": len(initial_conflicts),
        "n_conflicts_final": len(final_conflicts),
        "final_conflicts": final_conflicts,
    }
    args.overlap_report.write_text(
        json.dumps(overlap_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved overlap report: {args.overlap_report}")

    if args.strict_overlap_check and final_conflicts:
        raise RuntimeError(
            f"Overlap guard failed: {len(final_conflicts)} conflict pairs remain. See {args.overlap_report}"
        )

    if args.write_clustered_jsonl:
        if args.clustered_jsonl is None:
            raise ValueError("--write-clustered-jsonl requires --clustered-jsonl")
        if args.output_clustered_jsonl is None:
            args.output_clustered_jsonl = args.clustered_jsonl.with_name(
                f"{args.clustered_jsonl.stem}_polished.jsonl"
            )
        mapping = {
            int(r["cluster_id"]): {
                "cluster_title": r["cluster_title"],
                "cluster_intro": r["cluster_intro"],
            }
            for r in polished
        }
        rows = _load_jsonl(args.clustered_jsonl)
        out_rows = []
        for row in rows:
            cid = int(row["cluster_id"])
            if cid in mapping:
                row["cluster_title"] = mapping[cid]["cluster_title"]
                row["cluster_intro"] = mapping[cid]["cluster_intro"]
            out_rows.append(row)
        _write_jsonl(args.output_clustered_jsonl, out_rows)
        print(f"Saved polished clustered jsonl: {args.output_clustered_jsonl}")


if __name__ == "__main__":
    main()
