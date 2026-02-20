#!/usr/bin/env python3
"""
Clean clustering pipeline for persona-style JSONL input.

Pipeline:
1) Load JSONL records.
2) Build embeddings (default: OpenAI text-embedding-3-large).
3) Project to 2D with UMAP.
4) Cluster with KMeans.
5) Generate cluster title + introduction (default: OpenAI chat model).
6) Write an enriched JSONL that preserves all original fields and adds cluster fields.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass
class ClusterConfig:
    input_jsonl: Path
    output_jsonl: Path
    output_dir: Path

    text_column: str = "text"
    only_finished: bool = True

    embedding_backend: str = "openai"  # openai | tfidf
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200
    embedding_cache_path: Optional[Path] = None
    embedding_max_retries: int = 5
    embedding_retry_seconds: float = 1.0

    umap_min_dist: float = 0.5
    umap_metric: str = "cosine"
    umap_neighbors_ratio: float = 0.2

    n_clusters: int = 15
    auto_k: bool = False
    k_min: int = 6
    k_max: int = 25
    random_state: int = 42

    summary_backend: str = "openai"  # openai | keywords
    summary_model: str = "gpt-4o-mini"
    summary_temperature: float = 0.0
    summary_max_examples: int = 20
    summary_max_chars_per_example: int = 280
    summary_max_retries: int = 5
    summary_retry_seconds: float = 1.0


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No valid JSON rows found in {path}")
    return pd.DataFrame(rows)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _to_python_scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _to_jsonable_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_python_scalar(v) for k, v in rec.items()}


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


def build_openai_embeddings(texts: Sequence[str], cfg: ClusterConfig) -> np.ndarray:
    cache_path = cfg.embedding_cache_path
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(texts), cfg.embedding_dimensions):
            print(f"[embedding] loaded cache: {cache_path}")
            return cached.astype(np.float32)

    client = _get_openai_client()
    n = len(texts)
    out = np.empty((n, cfg.embedding_dimensions), dtype=np.float32)

    for start in range(0, n, cfg.embedding_batch_size):
        batch = list(texts[start : start + cfg.embedding_batch_size])
        for attempt in range(cfg.embedding_max_retries):
            try:
                resp = client.embeddings.create(
                    input=batch,
                    model=cfg.embedding_model,
                    dimensions=cfg.embedding_dimensions,
                )
                data = sorted(resp.data, key=lambda x: x.index)
                vec = np.asarray([d.embedding for d in data], dtype=np.float32)
                if vec.shape != (len(batch), cfg.embedding_dimensions):
                    raise RuntimeError(
                        f"Unexpected embedding shape: {vec.shape}, expected {(len(batch), cfg.embedding_dimensions)}"
                    )
                out[start : start + len(batch)] = vec
                break
            except Exception as exc:
                if attempt == cfg.embedding_max_retries - 1:
                    raise RuntimeError(f"OpenAI embedding failed at batch {start}") from exc
                wait_s = cfg.embedding_retry_seconds * (2**attempt)
                print(f"[embedding] retry start={start}, attempt={attempt + 1}, wait={wait_s:.1f}s")
                time.sleep(wait_s)
        print(f"[embedding] done {min(start + len(batch), n)}/{n}")

    if cache_path:
        np.save(cache_path, out)
        print(f"[embedding] cache saved: {cache_path}")
    return out


def build_tfidf_embeddings(texts: Sequence[str], dimensions: int) -> np.ndarray:
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=dimensions,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
    )
    return vec.fit_transform(texts).toarray().astype(np.float32)


def build_embeddings(df: pd.DataFrame, cfg: ClusterConfig) -> np.ndarray:
    texts = df["text_norm"].tolist()
    if cfg.embedding_backend == "openai":
        return build_openai_embeddings(texts, cfg)
    if cfg.embedding_backend == "tfidf":
        return build_tfidf_embeddings(texts, cfg.embedding_dimensions)
    raise ValueError(f"Unsupported embedding backend: {cfg.embedding_backend}")


def fit_umap(X: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    n_neighbors = max(2, int(len(X) * cfg.umap_neighbors_ratio))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    x2 = np.asarray(reducer.fit_transform(X), dtype=np.float64)
    if not np.isfinite(x2).all():
        x2 = np.nan_to_num(x2, nan=0.0, posinf=1e6, neginf=-1e6)

    # Normalize 2D coordinates for more stable KMeans distance computations.
    mu = np.mean(x2, axis=0, keepdims=True)
    sigma = np.std(x2, axis=0, keepdims=True) + 1e-12
    x2 = (x2 - mu) / sigma
    if not np.isfinite(x2).all():
        x2 = np.nan_to_num(x2, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(x2, -1e4, 1e4).astype(np.float32, copy=False)


def choose_k(x2: np.ndarray, cfg: ClusterConfig) -> int:
    if not cfg.auto_k:
        return int(cfg.n_clusters)

    n = x2.shape[0]
    low = max(2, cfg.k_min)
    high = min(cfg.k_max, n - 1)
    if low > high:
        return max(2, min(cfg.n_clusters, n - 1))

    best_k = low
    best_score = -1.0
    for k in range(low, high + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=20)
        labels = km.fit_predict(x2)
        score = float(silhouette_score(x2, labels))
        print(f"[auto-k] k={k}, silhouette={score:.6f}")
        if score > best_score:
            best_k = k
            best_score = score
    print(f"[auto-k] selected k={best_k}, silhouette={best_score:.6f}")
    return best_k


def _remap_cluster_ids(labels: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    old_to_new = {int(old): int(new) for new, old in enumerate(order.tolist())}
    new_labels = np.array([old_to_new[int(x)] for x in labels], dtype=int)
    return new_labels, old_to_new


def run_kmeans(x2: np.ndarray, cfg: ClusterConfig) -> Tuple[np.ndarray, np.ndarray]:
    k = choose_k(x2, cfg)
    km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=20)
    labels = km.fit_predict(x2)
    centers = km.cluster_centers_
    labels, mapping = _remap_cluster_ids(labels, centers)

    new_centers = np.zeros_like(centers)
    for old_id, new_id in mapping.items():
        new_centers[new_id] = centers[old_id]
    return labels, new_centers


def _top_terms(texts: Sequence[str], top_k: int = 8) -> List[str]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    x = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())
    scores = np.asarray(x.mean(axis=0)).ravel()
    idx = np.argsort(scores)[-top_k:][::-1]
    return [names[i] for i in idx if scores[i] > 0]


def _keyword_title_from_terms(terms: Sequence[str]) -> str:
    stop = set(ENGLISH_STOP_WORDS) | {"player", "players", "persona", "behavior", "behaviors", "game"}
    cleaned: List[str] = []
    for term in terms:
        parts = [p for p in term.split() if p not in stop]
        if not parts:
            continue
        cleaned.append(" ".join(parts))
        if len(cleaned) >= 3:
            break
    if not cleaned:
        return "Mixed Strategy Profile"
    return " / ".join(x.title() for x in cleaned)


def _compact_text(text: str, max_chars: int) -> str:
    s = normalize_text(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _parse_summary_json(raw: str) -> Optional[Dict[str, str]]:
    raw = raw.strip()
    if not raw:
        return None
    # Accept direct JSON output or JSON embedded in markdown fences.
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
    title = normalize_text(obj.get("title", ""))
    intro = normalize_text(obj.get("intro", ""))
    if not title or not intro:
        return None
    return {"title": title, "intro": intro}


def summarize_one_cluster_openai(
    client,
    cluster_id: int,
    size: int,
    examples: Sequence[str],
    top_terms: Sequence[str],
    cfg: ClusterConfig,
) -> Dict[str, str]:
    prompt = (
        "You are analyzing one cluster of persona descriptions.\n"
        "Write one concise title and one concise introduction.\n\n"
        "Rules:\n"
        "- Title: 4 to 10 words.\n"
        "- Intro: 2 to 4 sentences.\n"
        "- Keep it behavior-focused and generalizable.\n"
        "- Do not mention cluster numbers in the text.\n"
        "- Do not reference this prompt.\n\n"
        f"Cluster id: {cluster_id}\n"
        f"Cluster size: {size}\n"
        f"Top terms: {', '.join(top_terms) if top_terms else 'N/A'}\n\n"
        "Representative persona snippets:\n"
        + "\n".join(f"- {x}" for x in examples)
        + "\n\n"
        'Return JSON only in this schema: {"title":"...","intro":"..."}'
    )

    for attempt in range(cfg.summary_max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.summary_model,
                temperature=cfg.summary_temperature,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_summary_json(raw)
            if parsed:
                return parsed
            raise RuntimeError("Invalid JSON summary format from model")
        except Exception as exc:
            if attempt == cfg.summary_max_retries - 1:
                raise RuntimeError(f"Failed to summarize cluster {cluster_id}") from exc
            wait_s = cfg.summary_retry_seconds * (2**attempt)
            print(
                f"[summary] retry cluster={cluster_id}, attempt={attempt + 1}, wait={wait_s:.1f}s"
            )
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to summarize cluster {cluster_id}")


def summarize_clusters(df: pd.DataFrame, cfg: ClusterConfig) -> List[Dict[str, Any]]:
    cluster_ids = sorted(int(x) for x in df["cluster_id"].unique())
    rows: List[Dict[str, Any]] = []

    client = None
    use_openai = cfg.summary_backend == "openai"
    if use_openai:
        try:
            client = _get_openai_client()
        except Exception as exc:
            print(f"[summary] OpenAI unavailable, fallback to keywords: {exc}")
            use_openai = False

    for cid in cluster_ids:
        sub = df[df["cluster_id"] == cid].copy()
        size = int(len(sub))
        texts = sub["text_norm"].tolist()
        terms = _top_terms(texts, top_k=8) if texts else []

        unique_examples: List[str] = []
        seen = set()
        for text in texts:
            compact = _compact_text(text, cfg.summary_max_chars_per_example)
            if compact in seen:
                continue
            seen.add(compact)
            unique_examples.append(compact)
            if len(unique_examples) >= cfg.summary_max_examples:
                break

        if use_openai:
            try:
                summ = summarize_one_cluster_openai(
                    client=client,
                    cluster_id=cid,
                    size=size,
                    examples=unique_examples,
                    top_terms=terms,
                    cfg=cfg,
                )
                title = summ["title"]
                intro = summ["intro"]
                source = "openai"
            except Exception as exc:
                print(f"[summary] OpenAI fallback to keywords for cluster {cid}: {exc}")
                title = _keyword_title_from_terms(terms)
                intro = (
                    f"This cluster groups personas with related strategy patterns around "
                    f"{', '.join(terms[:5]) if terms else 'mixed behavioral signals'}."
                )
                source = "keywords-fallback"
        else:
            title = _keyword_title_from_terms(terms)
            intro = (
                f"This cluster groups personas with related strategy patterns around "
                f"{', '.join(terms[:5]) if terms else 'mixed behavioral signals'}."
            )
            source = "keywords"

        rows.append(
            {
                "cluster_id": cid,
                "cluster_size": size,
                "cluster_title": title,
                "cluster_intro": intro,
                "summary_source": source,
                "top_terms": terms,
                "sample_examples": unique_examples[: min(5, len(unique_examples))],
            }
        )
        print(f"[summary] cluster={cid}, size={size}, source={source}")

    return rows


def enrich_records(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(catalog, on="cluster_id", how="left")
    return merged


def run(cfg: ClusterConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(cfg.input_jsonl)
    if cfg.text_column not in df.columns:
        raise ValueError(f"Missing text column: {cfg.text_column}")

    if cfg.only_finished and "game_finished" in df.columns:
        df = df[df["game_finished"] == True].copy()
    df = df.reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 rows for clustering.")

    df["text_norm"] = df[cfg.text_column].map(normalize_text)

    X = build_embeddings(df, cfg)
    x2 = fit_umap(X, cfg)
    labels, centers = run_kmeans(x2, cfg)

    df["umap_x"] = x2[:, 0]
    df["umap_y"] = x2[:, 1]
    df["cluster_id"] = labels

    catalog_rows = summarize_clusters(df, cfg)
    catalog = pd.DataFrame(catalog_rows)

    out_df = enrich_records(df, catalog)
    out_df["cluster_size"] = out_df["cluster_size"].fillna(0).astype(int)
    out_df["cluster_id"] = out_df["cluster_id"].astype(int)

    # Keep original fields plus new cluster columns.
    output_records = []
    for rec in out_df.to_dict(orient="records"):
        rec.pop("text_norm", None)
        rec.pop("summary_source", None)
        rec.pop("top_terms", None)
        rec.pop("sample_examples", None)
        output_records.append(_to_jsonable_record(rec))
    write_jsonl(cfg.output_jsonl, output_records)

    cluster_catalog_path = cfg.output_dir / "cluster_catalog.json"
    metadata_path = cfg.output_dir / "run_metadata.json"
    umap_points_csv_path = cfg.output_dir / "umap_points.csv"

    catalog_payload = [_to_jsonable_record(r) for r in catalog_rows]
    cluster_catalog_path.write_text(
        json.dumps(catalog_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_df[
        [cfg.text_column, "umap_x", "umap_y", "cluster_id", "cluster_title", "cluster_intro", "cluster_size"]
    ].to_csv(umap_points_csv_path, index=False)

    metadata = {
        "input_jsonl": str(cfg.input_jsonl),
        "output_jsonl": str(cfg.output_jsonl),
        "rows_used": int(len(df)),
        "text_column": cfg.text_column,
        "only_finished": cfg.only_finished,
        "embedding_backend": cfg.embedding_backend,
        "embedding_model": cfg.embedding_model if cfg.embedding_backend == "openai" else "tfidf",
        "embedding_dimensions": cfg.embedding_dimensions,
        "umap_params": {
            "min_dist": cfg.umap_min_dist,
            "metric": cfg.umap_metric,
            "neighbors_ratio": cfg.umap_neighbors_ratio,
            "random_state": cfg.random_state,
        },
        "clustering_params": {
            "method": "kmeans",
            "n_clusters": int(catalog["cluster_id"].nunique()),
            "auto_k": cfg.auto_k,
            "random_state": cfg.random_state,
            "n_init": 20,
        },
        "summary_backend": cfg.summary_backend,
        "summary_model": cfg.summary_model if cfg.summary_backend == "openai" else None,
        "outputs": {
            "cluster_catalog_json": str(cluster_catalog_path),
            "umap_points_csv": str(umap_points_csv_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> ClusterConfig:
    parser = argparse.ArgumentParser(
        description="Run embedding + UMAP + KMeans clustering and export enriched JSONL."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--include-unfinished", action="store_true")

    parser.add_argument("--embedding-backend", choices=["openai", "tfidf"], default="openai")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding-dimensions", type=int, default=1536)
    parser.add_argument("--embedding-batch-size", type=int, default=200)
    parser.add_argument("--embedding-cache-path", type=Path, default=None)

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
    output_dir = args.output_jsonl.parent
    cache_path = args.embedding_cache_path
    if cache_path is None and args.embedding_backend == "openai":
        model_tag = args.embedding_model.replace("/", "_")
        cache_path = output_dir / f"embeddings_{model_tag}_{args.embedding_dimensions}.npy"

    return ClusterConfig(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        output_dir=output_dir,
        text_column=args.text_column,
        only_finished=not args.include_unfinished,
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
        summary_backend=args.summary_backend,
        summary_model=args.summary_model,
        summary_temperature=args.summary_temperature,
        summary_max_examples=args.summary_max_examples,
        summary_max_chars_per_example=args.summary_max_chars_per_example,
    )


if __name__ == "__main__":
    run(parse_args())
