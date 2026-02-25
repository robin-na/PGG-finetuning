#!/usr/bin/env python3
"""
Clean clustering pipeline for persona-style JSONL input.

Pipeline:
1) Load JSONL records.
2) Build embeddings (default: OpenAI text-embedding-3-large).
3) Project to 2D with UMAP.
4) Cluster with KMeans in embedding space by default, with optional split/merge refinement.
5) Generate cluster title + introduction (default: keyword-based, optional OpenAI chat model).
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
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
try:
    from umap import UMAP
    HAVE_UMAP = True
except Exception:  # pragma: no cover
    HAVE_UMAP = False

    class UMAP:  # type: ignore[override]
        """Minimal UMAP-compatible fallback backed by PCA when umap-learn is unavailable."""

        def __init__(
            self,
            n_components: int = 2,
            n_neighbors: int = 15,
            min_dist: float = 0.1,
            metric: str = "euclidean",
            random_state: int = 42,
            **_: Any,
        ) -> None:
            self.n_components = n_components
            self.random_state = random_state

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            return pca.fit_transform(X)

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
    cluster_space: str = "embedding"  # embedding | umap

    enable_split_merge: bool = True
    max_split_merge_iters: int = 20
    min_clusters: int = 6
    max_clusters: int = 25
    merge_similarity_threshold: float = 0.94
    target_overlap_rate: float = 0.12
    point_overlap_margin: float = 0.03
    split_cluster_min_size: int = 60

    summary_backend: str = "keywords"  # openai | keywords
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
    if not HAVE_UMAP:
        print("[umap] umap-learn unavailable; using PCA fallback for 2D projection.")
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


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)
    mapping = {int(old): int(new) for new, old in enumerate(uniq.tolist())}
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def _compute_centroids(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = int(labels.max()) + 1
    centroids = np.zeros((k, x.shape[1]), dtype=np.float32)
    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            continue
        centroids[cid] = x[mask].mean(axis=0)
    return centroids


def _cluster_overlap_diagnostics(
    x: np.ndarray,
    labels: np.ndarray,
    point_overlap_margin: float,
) -> Dict[str, Any]:
    labels = labels.astype(int)
    k = int(labels.max()) + 1
    n = int(len(labels))
    if k < 2 or n == 0:
        return {
            "overlap_rate": 0.0,
            "cluster_ambiguity": {int(cid): 0.0 for cid in range(k)},
            "max_centroid_similarity": 1.0 if k == 1 else 0.0,
        }

    centroids = _compute_centroids(x, labels)
    x_n = _l2_normalize_rows(x.astype(np.float32, copy=False))
    c_n = _l2_normalize_rows(centroids.astype(np.float32, copy=False))

    sims = np.matmul(x_n, c_n.T)  # [n, k]
    order = np.argsort(sims, axis=1)
    best = order[:, -1]
    second = order[:, -2]
    best_sim = sims[np.arange(n), best]
    second_sim = sims[np.arange(n), second]
    gaps = best_sim - second_sim
    ambiguous = gaps < float(point_overlap_margin)
    overlap_rate = float(np.mean(ambiguous))

    cluster_ambiguity: Dict[int, float] = {}
    for cid in range(k):
        mask = labels == cid
        cluster_ambiguity[cid] = float(np.mean(ambiguous[mask])) if np.any(mask) else 0.0

    c_sim = np.matmul(c_n, c_n.T)
    np.fill_diagonal(c_sim, -1.0)
    max_centroid_similarity = float(np.max(c_sim)) if c_sim.size else 0.0

    return {
        "overlap_rate": overlap_rate,
        "cluster_ambiguity": cluster_ambiguity,
        "max_centroid_similarity": max_centroid_similarity,
    }


def choose_k(x_cluster: np.ndarray, cfg: ClusterConfig) -> int:
    n = x_cluster.shape[0]
    low = max(2, int(cfg.k_min), int(cfg.min_clusters))
    high = min(int(cfg.k_max), int(cfg.max_clusters), n - 1)
    if low > high:
        return max(2, min(int(cfg.n_clusters), n - 1))

    if not cfg.auto_k:
        return int(np.clip(cfg.n_clusters, low, high))

    best_k = low
    best_score = -1.0
    for k in range(low, high + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=20)
        labels = km.fit_predict(x_cluster)
        score = float(silhouette_score(x_cluster, labels))
        print(f"[auto-k] k={k}, silhouette={score:.6f}")
        if score > best_score:
            best_k = k
            best_score = score
    print(f"[auto-k] selected k={best_k}, silhouette={best_score:.6f}")
    return int(best_k)


def _run_kmeans_fixed_k(x_cluster: np.ndarray, k: int, random_state: int) -> np.ndarray:
    km = KMeans(n_clusters=int(k), random_state=random_state, n_init=20)
    labels = km.fit_predict(x_cluster)
    return _reindex_labels(labels)


def _merge_most_similar_pair(
    x_cluster: np.ndarray,
    labels: np.ndarray,
    merge_similarity_threshold: float,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    k = int(labels.max()) + 1
    if k < 2:
        return labels, None

    centroids = _compute_centroids(x_cluster, labels)
    c_n = _l2_normalize_rows(centroids)
    sims = np.matmul(c_n, c_n.T)
    np.fill_diagonal(sims, -1.0)
    i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
    best_sim = float(sims[i, j])
    if best_sim < float(merge_similarity_threshold):
        return labels, None

    keep, drop = (i, j) if i < j else (j, i)
    merged = labels.copy()
    merged[merged == drop] = keep
    merged = _reindex_labels(merged)
    event = {
        "op": "merge",
        "kept_cluster": int(keep),
        "dropped_cluster": int(drop),
        "centroid_cosine_similarity": best_sim,
    }
    return merged, event


def _split_one_cluster(
    x_cluster: np.ndarray,
    labels: np.ndarray,
    cfg: ClusterConfig,
    overlap_diag: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    k = int(labels.max()) + 1
    if k >= int(cfg.max_clusters):
        return labels, None

    centroids = _compute_centroids(x_cluster, labels)
    x_n = _l2_normalize_rows(x_cluster)
    c_n = _l2_normalize_rows(centroids)
    own_sim = np.matmul(x_n, c_n.T)[np.arange(len(labels)), labels]
    cluster_ambiguity = overlap_diag.get("cluster_ambiguity", {})

    best_choice: Optional[Tuple[float, int, int, float, float]] = None
    for cid in range(k):
        mask = labels == cid
        size = int(np.sum(mask))
        if size < int(cfg.split_cluster_min_size):
            continue
        ambiguity = float(cluster_ambiguity.get(cid, 0.0))
        dispersion = float(1.0 - np.mean(own_sim[mask])) if size > 0 else 0.0
        score = ambiguity + 0.5 * dispersion
        candidate = (score, cid, size, ambiguity, dispersion)
        if best_choice is None or candidate[0] > best_choice[0]:
            best_choice = candidate

    if best_choice is None:
        return labels, None

    _, split_cid, split_size, ambiguity, dispersion = best_choice
    mask = labels == split_cid
    sub_x = x_cluster[mask]
    if len(sub_x) < 2:
        return labels, None

    km = KMeans(n_clusters=2, random_state=cfg.random_state, n_init=20)
    sub_labels = km.fit_predict(sub_x)
    if len(np.unique(sub_labels)) < 2:
        return labels, None

    counts = np.bincount(sub_labels, minlength=2)
    keep_local = int(np.argmax(counts))
    new_local = 1 - keep_local

    out = labels.copy()
    idx = np.where(mask)[0]
    out[idx[sub_labels == new_local]] = int(labels.max()) + 1
    out = _reindex_labels(out)
    event = {
        "op": "split",
        "source_cluster": int(split_cid),
        "source_size": int(split_size),
        "source_ambiguity": float(ambiguity),
        "source_dispersion": float(dispersion),
    }
    return out, event


def _remap_cluster_ids_by_umap(labels: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    centroids_2d = _compute_centroids(x2, labels)
    order = np.lexsort((centroids_2d[:, 1], centroids_2d[:, 0]))
    old_to_new = {int(old): int(new) for new, old in enumerate(order.tolist())}
    new_labels = np.array([old_to_new[int(x)] for x in labels], dtype=int)
    return new_labels, old_to_new


def run_kmeans_with_split_merge(
    x_cluster: np.ndarray,
    cfg: ClusterConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    k0 = choose_k(x_cluster, cfg)
    labels = _run_kmeans_fixed_k(x_cluster, k0, cfg.random_state)
    operations: List[Dict[str, Any]] = []

    if cfg.enable_split_merge:
        for iter_idx in range(int(cfg.max_split_merge_iters)):
            changed = False

            while int(labels.max()) + 1 > int(cfg.min_clusters):
                merged_labels, merge_event = _merge_most_similar_pair(
                    x_cluster=x_cluster,
                    labels=labels,
                    merge_similarity_threshold=cfg.merge_similarity_threshold,
                )
                if merge_event is None:
                    break
                new_k = int(np.unique(merged_labels).size)
                labels = _run_kmeans_fixed_k(x_cluster, new_k, cfg.random_state)
                merge_event["iter"] = int(iter_idx + 1)
                operations.append(merge_event)
                changed = True

            overlap_diag = _cluster_overlap_diagnostics(
                x=x_cluster,
                labels=labels,
                point_overlap_margin=cfg.point_overlap_margin,
            )
            overlap_rate = float(overlap_diag["overlap_rate"])

            if (
                overlap_rate > float(cfg.target_overlap_rate)
                and (int(labels.max()) + 1) < int(cfg.max_clusters)
            ):
                split_labels, split_event = _split_one_cluster(
                    x_cluster=x_cluster,
                    labels=labels,
                    cfg=cfg,
                    overlap_diag=overlap_diag,
                )
                if split_event is not None:
                    new_k = int(np.unique(split_labels).size)
                    labels = _run_kmeans_fixed_k(x_cluster, new_k, cfg.random_state)
                    split_event["iter"] = int(iter_idx + 1)
                    operations.append(split_event)
                    changed = True

            if not changed:
                break

    centroids = _compute_centroids(x_cluster, labels)
    final_overlap_diag = _cluster_overlap_diagnostics(
        x=x_cluster,
        labels=labels,
        point_overlap_margin=cfg.point_overlap_margin,
    )
    return labels, centroids, operations, final_overlap_diag


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
    if cfg.cluster_space == "embedding":
        x_cluster = _l2_normalize_rows(X.astype(np.float32, copy=False))
    elif cfg.cluster_space == "umap":
        x_cluster = x2
    else:
        raise ValueError(f"Unsupported cluster space: {cfg.cluster_space}")

    labels, centers, split_merge_events, overlap_diag = run_kmeans_with_split_merge(
        x_cluster=x_cluster,
        cfg=cfg,
    )
    labels, label_mapping = _remap_cluster_ids_by_umap(labels, x2)

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
            "method": "kmeans+split_merge" if cfg.enable_split_merge else "kmeans",
            "cluster_space": cfg.cluster_space,
            "n_clusters": int(catalog["cluster_id"].nunique()),
            "auto_k": cfg.auto_k,
            "random_state": cfg.random_state,
            "n_init": 20,
            "min_clusters": int(cfg.min_clusters),
            "max_clusters": int(cfg.max_clusters),
            "merge_similarity_threshold": float(cfg.merge_similarity_threshold),
            "target_overlap_rate": float(cfg.target_overlap_rate),
            "point_overlap_margin": float(cfg.point_overlap_margin),
            "split_cluster_min_size": int(cfg.split_cluster_min_size),
            "max_split_merge_iters": int(cfg.max_split_merge_iters),
            "split_merge_events": [_to_jsonable_record(e) for e in split_merge_events],
            "label_mapping_pre_to_post_umap_order": {str(k): int(v) for k, v in label_mapping.items()},
            "final_overlap_rate": float(overlap_diag.get("overlap_rate", 0.0)),
            "final_max_centroid_similarity": float(overlap_diag.get("max_centroid_similarity", 0.0)),
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
        description="Run embedding + UMAP + KMeans clustering (with optional split/merge refinement) and export enriched JSONL."
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


if __name__ == "__main__":
    run(parse_args())
