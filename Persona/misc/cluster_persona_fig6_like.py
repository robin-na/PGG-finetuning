#!/usr/bin/env python3
"""
Fig.6-like clustering pipeline for persona summaries.

This script supports two modes:
1) OpenAI-paper style:
   - Embedding: text-embedding-3-large (dimensions=1536, batch=200)
   - UMAP: n_neighbors=int(N/5), min_dist=0.5, metric=cosine
   - Cluster titles: ChatGPT summaries from cluster code lists (SI-style prompt)
2) Local fallback:
   - Embedding: TF-IDF vectors
   - Cluster titles: top-term heuristics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from umap import UMAP

# Load environment variables from .env file
load_dotenv()

DEFAULT_INPUT = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/summary_gpt51_learn.jsonl")
DEFAULT_OUTPUT_DIR = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_fig6_like")

PAPER_STYLE_TITLE_PROMPT = """
Here are {n_clusters} clusters of behavioral codes in the provided list. A behavioral code is a piece
of natural language that describes motivations behind strategic behaviors. Please summarize these
behavioral code clusters in to short titles (e.g., within four words).

The summarized cluster titles should be: (1) understandable, (2) semantically relevant to the
behavioral codes inside the clusters, (3) well covering the whole cluster, (4) distinguishable from
other clusters, (5) generate a generalizable cluster title and avoid including any information
specific to this game scenario.

Before output, please go back and forth review and refine the summarized cluster titles
so that they meet the requirements. Please output the cluster names in lines, followed by the
summarized titles (e.g., "cluster 8AE92B-496: ...").

Clusters:
{cluster_payload}
""".strip()

SINGLE_CLUSTER_TITLE_PROMPT = """
Here is one cluster of behavioral/persona codes. A code is a natural-language description of
motivations behind strategic behavior. Please summarize this cluster into one short title
(within four words), understandable and generalizable.

Output exactly one line:
cluster {cluster_name}: <title>

Cluster:
{cluster_payload}
""".strip()

TITLE_REFINE_PROMPT = """
You are given initial titles for multiple clusters.
Please refine them so the titles are:
1) understandable,
2) semantically relevant,
3) distinguishable from each other,
4) concise (within four words),
5) generalizable.

Output one line per cluster in this exact format:
cluster XX: <title>

Clusters:
{cluster_payload}
""".strip()

EXTENDED_TITLE_PROMPT = """
Here are {n_clusters} clusters of behavioral codes in the provided list. A behavioral code is a piece
of natural language that describes motivations behind strategic behaviors. Please summarize these
behavioral code clusters into extended titles.

CRITICAL REQUIREMENT: Each title MUST be between 8-15 words. Count carefully.

The summarized cluster titles should be:
(1) 8-15 words in length (strictly enforced)
(2) understandable and descriptive
(3) semantically relevant to the behavioral codes inside the clusters
(4) well covering the whole cluster
(5) distinguishable from other clusters
(6) generalizable - avoid information specific to this game scenario

Examples of good extended titles:
- "Conditional cooperators who match group contributions and punish free-riders consistently"
- "Strategic defectors who reduce cooperation when others contribute less to maximize payoff"
- "Unconditional high contributors who maintain full cooperation despite others' free-riding behavior"

Before output, review each title and count the words to ensure 8-15 word requirement is met.
Please output the cluster names in lines, followed by the summarized titles (e.g., "cluster 00-496: ...").

Clusters:
{cluster_payload}
""".strip()

EXTENDED_SINGLE_CLUSTER_TITLE_PROMPT = """
Here is one cluster of behavioral/persona codes. A code is a natural-language description of
motivations behind strategic behavior.

CRITICAL REQUIREMENT: Generate ONE extended title that is EXACTLY 8-15 words. Count carefully.

The title should be:
- 8-15 words in length (strictly enforced)
- understandable and descriptive
- semantically relevant to the codes
- generalizable (avoid game-specific details)

Examples of good 8-15 word titles:
- "Conditional cooperators who match group contributions and punish free-riders consistently" (11 words)
- "Strategic defectors who reduce cooperation when others contribute less" (10 words)

Output exactly one line in this format:
cluster {cluster_name}: <your 8-15 word title here>

Cluster:
{cluster_payload}
""".strip()


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    n_clusters: int = 23
    min_rows_per_cluster_sample: int = 10
    random_state: int = 42
    only_finished: bool = True
    weight_col: Optional[str] = None
    non_negligible_threshold: float = 1e-3

    # Embedding backend.
    embedding_backend: str = "openai"  # openai | tfidf
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200
    embedding_max_retries: int = 5
    embedding_retry_seconds: float = 1.0
    use_embedding_cache: bool = True

    # Cluster title backend.
    title_backend: str = "openai"  # openai | keywords
    title_model: str = "gpt-4o-mini"
    title_temperature: float = 0.0
    title_max_codes_per_cluster: int = 20
    title_max_chars_per_code: int = 220
    title_strategy: str = "auto"  # auto | global | per_cluster
    title_context_token_budget: int = 90000
    title_refine_global: bool = True
    title_max_retries: int = 5
    title_retry_seconds: float = 1.0

    # Extended title configuration
    title_use_extended: bool = False
    title_extended_min_words: int = 8
    title_extended_max_words: int = 15

    # UMAP params aligned with the paper notebook.
    umap_min_dist: float = 0.5
    umap_metric: str = "cosine"

    # Local keyword matrix for fallback titles and diagnostics.
    tfidf_max_features: int = 1536
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    df = pd.DataFrame(rows)
    required = {"experiment", "participant", "game_finished", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _validate_extended_title(title: str, min_words: int = 8, max_words: int = 15) -> bool:
    """
    Check if title meets word count requirements for extended titles.

    Args:
        title: Title string to validate
        min_words: Minimum word count (default 8)
        max_words: Maximum word count (default 15)

    Returns:
        True if title is within word count range, False otherwise
    """
    word_count = len(title.split())
    return min_words <= word_count <= max_words


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it first to run OpenAI-paper mode."
        )
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "openai package is not available. Install it with: python3 -m pip install --user openai"
        ) from exc
    return OpenAI(api_key=api_key)


def build_keyword_matrix(df: pd.DataFrame, cfg: PipelineConfig):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        lowercase=True,
    )
    X = vectorizer.fit_transform(df["text_norm"]).astype(np.float32)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return X, feature_names


def build_openai_embeddings(
    texts: Sequence[str],
    cfg: PipelineConfig,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(texts), cfg.embedding_dimensions):
            print(f"[embedding] loaded cache: {cache_path}")
            return cached.astype(np.float32)

    client = _get_openai_client()
    embeddings = np.empty((len(texts), cfg.embedding_dimensions), dtype=np.float32)
    n_done = 0
    n_total = len(texts)

    for start in range(0, n_total, cfg.embedding_batch_size):
        batch = list(texts[start : start + cfg.embedding_batch_size])
        for attempt in range(cfg.embedding_max_retries):
            try:
                resp = client.embeddings.create(
                    input=batch,
                    model=cfg.embedding_model,
                    dimensions=cfg.embedding_dimensions,
                )
                data = sorted(resp.data, key=lambda x: x.index)
                vecs = np.array([item.embedding for item in data], dtype=np.float32)
                if vecs.shape != (len(batch), cfg.embedding_dimensions):
                    raise RuntimeError(
                        f"Unexpected embedding shape {vecs.shape}, expected {(len(batch), cfg.embedding_dimensions)}"
                    )
                embeddings[start : start + len(batch)] = vecs
                break
            except Exception as exc:
                if attempt == cfg.embedding_max_retries - 1:
                    raise RuntimeError(f"OpenAI embedding failed at batch {start}") from exc
                wait_s = cfg.embedding_retry_seconds * (2 ** attempt)
                print(f"[embedding] retry batch {start}, attempt={attempt + 1}, wait={wait_s:.1f}s")
                time.sleep(wait_s)
        n_done += len(batch)
        print(f"[embedding] done {n_done}/{n_total}")

    if cache_path:
        np.save(cache_path, embeddings)
        print(f"[embedding] cache saved: {cache_path}")
    return embeddings


def make_short_title(top_terms: List[str]) -> str:
    cleaned: List[str] = []
    stop = set(ENGLISH_STOP_WORDS) | {"player", "players", "game", "behavior", "behaviors", "persona"}
    for term in top_terms:
        tokens = [t for t in re.split(r"\s+", term) if t and t not in stop]
        if not tokens:
            continue
        cleaned.append(" ".join(tokens))
        if len(cleaned) >= 3:
            break
    if not cleaned:
        return "Mixed Persona Pattern"
    return " / ".join(t.title() for t in cleaned)


def summarize_clusters_with_keywords(
    df: pd.DataFrame,
    keyword_matrix,
    feature_names: np.ndarray,
    cluster_col: str = "cluster_id",
    top_k_terms: int = 12,
) -> List[dict]:
    summaries: List[dict] = []
    cluster_ids = sorted(df[cluster_col].unique())
    for cid in cluster_ids:
        idx = np.where(df[cluster_col].values == cid)[0]
        sub = keyword_matrix[idx]
        mean_scores = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = np.argsort(mean_scores)[-top_k_terms:][::-1]
        top_terms = [feature_names[i] for i in top_idx if mean_scores[i] > 0]
        summaries.append(
            {
                "cluster_id": int(cid),
                "size": int(len(idx)),
                "title": make_short_title(top_terms),
                "title_source": "keyword-heuristic",
                "top_terms": top_terms,
            }
        )
    return summaries


def _compact_code(text: str, max_chars: int) -> str:
    t = normalize_text(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _cluster_code_examples(df: pd.DataFrame, cid: int, cfg: PipelineConfig) -> Tuple[str, int]:
    sub = df[df["cluster_id"] == cid].copy().sort_values("weight", ascending=False)
    seen = set()
    lines: List[str] = []
    for t in sub["text_norm"].tolist():
        compact = _compact_code(t, cfg.title_max_chars_per_code)
        if compact in seen:
            continue
        seen.add(compact)
        lines.append(f"- {compact}")
        if len(lines) >= cfg.title_max_codes_per_cluster:
            break
    return "\n".join(lines), len(sub)


def _build_cluster_payload(df: pd.DataFrame, cfg: PipelineConfig, cluster_ids: Optional[Sequence[int]] = None) -> str:
    lines: List[str] = []
    ids = sorted(cluster_ids) if cluster_ids is not None else sorted(int(x) for x in df["cluster_id"].unique())
    for cid in ids:
        codes, size = _cluster_code_examples(df, cid, cfg)
        lines.append(f"cluster {int(cid):02d}-{size}:")
        lines.append(codes)
        lines.append("")
    return "\n".join(lines).strip()


def _estimate_tokens(text: str) -> int:
    # rough but stable estimate used only for routing.
    return max(1, len(text) // 4)


def _parse_cluster_titles(raw_text: str, expected_ids: Sequence[int]) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^cluster\s+(\d+)(?:-\d+)?\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            cid = int(m.group(1))
            title = m.group(2).strip().strip('"')
            if title:
                parsed[cid] = title

    # fallback parsing if model uses "XX: title"
    if len(parsed) < len(expected_ids):
        for line in raw_text.splitlines():
            line = line.strip()
            m = re.match(r"^(\d+)\s*[:\-]\s*(.+)$", line)
            if m:
                cid = int(m.group(1))
                title = m.group(2).strip().strip('"')
                if title and cid not in parsed:
                    parsed[cid] = title
    return parsed


def _call_openai_chat(client, prompt: str, cfg: PipelineConfig, stage: str) -> str:
    for attempt in range(cfg.title_max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.title_model,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.title_temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise RuntimeError(f"{stage} failed: prompt exceeds model context window") from exc
            if attempt == cfg.title_max_retries - 1:
                raise RuntimeError(f"{stage} failed after retries") from exc
            wait_s = cfg.title_retry_seconds * (2 ** attempt)
            print(f"[title] retry {stage} attempt={attempt + 1}, wait={wait_s:.1f}s")
            time.sleep(wait_s)
    return ""


def _generate_titles_global(
    client,
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[Dict[int, str], str, str]:
    payload = _build_cluster_payload(df, cfg)

    # Choose prompt based on extended title setting
    if cfg.title_use_extended:
        prompt_template = EXTENDED_TITLE_PROMPT
    else:
        prompt_template = PAPER_STYLE_TITLE_PROMPT

    prompt = prompt_template.format(
        n_clusters=len(df["cluster_id"].unique()),
        cluster_payload=payload,
    )
    raw_text = _call_openai_chat(client, prompt, cfg, stage="global-title")
    expected_ids = sorted(int(x) for x in df["cluster_id"].unique())
    parsed = _parse_cluster_titles(raw_text, expected_ids)

    # Validate extended titles if enabled
    if cfg.title_use_extended:
        for cid, title in parsed.items():
            if not _validate_extended_title(title, cfg.title_extended_min_words, cfg.title_extended_max_words):
                word_count = len(title.split())
                print(f"[title] WARNING: cluster {cid} title has {word_count} words, expected {cfg.title_extended_min_words}-{cfg.title_extended_max_words}")

    return parsed, prompt, raw_text


def _generate_titles_per_cluster(
    client,
    df: pd.DataFrame,
    summaries: List[dict],
    cfg: PipelineConfig,
) -> Tuple[Dict[int, str], str, str]:
    prompt_logs: List[str] = []
    response_logs: List[str] = []
    parsed: Dict[int, str] = {}

    for s in sorted(summaries, key=lambda x: x["cluster_id"]):
        cid = int(s["cluster_id"])
        codes, size = _cluster_code_examples(df, cid, cfg)
        cluster_name = f"{cid:02d}-{size}"

        # Choose prompt based on extended title setting
        if cfg.title_use_extended:
            prompt_template = EXTENDED_SINGLE_CLUSTER_TITLE_PROMPT
        else:
            prompt_template = SINGLE_CLUSTER_TITLE_PROMPT

        prompt = prompt_template.format(cluster_name=cluster_name, cluster_payload=codes)
        raw = _call_openai_chat(client, prompt, cfg, stage=f"per-cluster-title-{cid:02d}")
        got = _parse_cluster_titles(raw, [cid])
        if cid in got:
            parsed[cid] = got[cid]
        else:
            # Fallback: try first non-empty line.
            first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
            if ":" in first:
                first = first.split(":", 1)[1].strip()
            parsed[cid] = first if first else s["title"]

        # Validate extended title if enabled
        if cfg.title_use_extended and cid in parsed:
            if not _validate_extended_title(parsed[cid], cfg.title_extended_min_words, cfg.title_extended_max_words):
                word_count = len(parsed[cid].split())
                print(f"[title] WARNING: cluster {cid} title has {word_count} words, expected {cfg.title_extended_min_words}-{cfg.title_extended_max_words}")

        prompt_logs.append(f"## cluster {cluster_name}\n\n{prompt}")
        response_logs.append(f"## cluster {cluster_name}\n\n{raw}")

    if cfg.title_refine_global and parsed:
        lines = []
        for s in sorted(summaries, key=lambda x: x["cluster_id"]):
            cid = int(s["cluster_id"])
            top_terms = ", ".join(s.get("top_terms", [])[:6])
            lines.append(f"cluster {cid:02d}: title={parsed.get(cid, s['title'])}; top_terms={top_terms}")
        refine_payload = "\n".join(lines)
        refine_prompt = TITLE_REFINE_PROMPT.format(cluster_payload=refine_payload)
        refine_raw = _call_openai_chat(client, refine_prompt, cfg, stage="global-title-refine")
        refined = _parse_cluster_titles(refine_raw, sorted(parsed.keys()))
        if refined:
            parsed.update(refined)
        prompt_logs.append("## refine\n\n" + refine_prompt)
        response_logs.append("## refine\n\n" + refine_raw)

    return parsed, "\n\n".join(prompt_logs), "\n\n".join(response_logs)


def generate_titles_with_openai(
    df: pd.DataFrame,
    summaries: List[dict],
    cfg: PipelineConfig,
) -> Tuple[Dict[int, str], str, str]:
    client = _get_openai_client()

    payload = _build_cluster_payload(df, cfg)
    est_tokens = _estimate_tokens(payload)
    use_global = cfg.title_strategy == "global"
    if cfg.title_strategy == "auto":
        use_global = est_tokens <= cfg.title_context_token_budget

    if use_global:
        print(f"[title] strategy=global, estimated_tokens={est_tokens}")
        try:
            return _generate_titles_global(client, df, cfg)
        except RuntimeError as exc:
            print(f"[title] global failed ({exc}); fallback to per-cluster.")
            return _generate_titles_per_cluster(client, df, summaries, cfg)

    print(f"[title] strategy=per_cluster, estimated_tokens={est_tokens}")
    return _generate_titles_per_cluster(client, df, summaries, cfg)


def apply_titles_to_summaries(
    summaries: List[dict],
    openai_titles: Dict[int, str],
) -> List[dict]:
    out: List[dict] = []
    for s in summaries:
        cid = s["cluster_id"]
        updated = dict(s)
        if cid in openai_titles:
            updated["title"] = openai_titles[cid]
            updated["title_source"] = "openai-chatgpt-summary"
        out.append(updated)
    return out


def _wrap_label(text: str, max_chars_per_line: int = 35) -> str:
    """Wrap a long label into multiple lines."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width=max_chars_per_line))


def plot_clusters(
    df: pd.DataFrame,
    output_png: Path,
    summaries: List[dict],
    non_negligible_threshold: float,
):
    fig, ax = plt.subplots(figsize=(20, 13))
    idx_non = df["weight"] > non_negligible_threshold
    idx_neg = ~idx_non

    if idx_neg.any():
        ax.scatter(
            df.loc[idx_neg, "umap_x"],
            df.loc[idx_neg, "umap_y"],
            s=1,
            marker="+",
            c="grey",
            linewidths=0.2,
            alpha=0.5,
            label="Behavior codes with negligible weights",
        )

    scatter = ax.scatter(
        df.loc[idx_non, "umap_x"],
        df.loc[idx_non, "umap_y"],
        s=np.sqrt(np.clip(df.loc[idx_non, "weight"].to_numpy(), 1e-12, None)) * 300,
        marker="D",
        c=df.loc[idx_non, "cluster_id"],
        cmap="tab20",
        linewidths=0,
        alpha=0.85,
        label="Behavior codes with non-negligible weights",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster ID")

    # Detect whether titles are long (extended mode) and adjust accordingly
    avg_title_len = np.mean([len(s["title"]) for s in summaries]) if summaries else 0
    is_extended = avg_title_len > 30

    # Collect label positions and texts for overlap avoidance
    texts = []
    for s in summaries:
        cid = s["cluster_id"]
        sub = df[df["cluster_id"] == cid]
        if sub.empty:
            continue
        cx = float(sub["umap_x"].mean())
        cy = float(sub["umap_y"].mean())

        if is_extended:
            label = f"{cid:02d}: {_wrap_label(s['title'], 30)}"
            fontsize = 5.5
        else:
            label = f"{cid:02d}: {s['title']}"
            fontsize = 8

        txt = ax.annotate(
            label,
            xy=(cx, cy),
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.3),
        )
        texts.append(txt)

    # Try adjustText for overlap avoidance if available
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    except ImportError:
        pass  # fall back to raw positions

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Persona Clusters (Fig.6-like UMAP projection)")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def dump_cluster_samples(df: pd.DataFrame, summaries: List[dict], output_json: Path, sample_n: int):
    out = []
    for s in summaries:
        cid = s["cluster_id"]
        sub = df[df["cluster_id"] == cid].sort_values("weight", ascending=False)
        sample = sub.head(sample_n)
        out.append(
            {
                "cluster_id": cid,
                "title": s["title"],
                "size": int(len(sub)),
                "samples": [
                    {
                        "experiment": r["experiment"],
                        "participant": r["participant"],
                        "weight": float(r["weight"]),
                        "text": r["text"],
                    }
                    for _, r in sample.iterrows()
                ],
            }
        )
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cfg: PipelineConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_jsonl(cfg.input_path)
    df = df_all.copy()
    if cfg.only_finished:
        df = df[df["game_finished"] == True].copy()
    df = df.reset_index(drop=True)
    df["text_norm"] = df["text"].map(normalize_text)

    if cfg.weight_col:
        if cfg.weight_col not in df.columns:
            raise ValueError(f"weight column not found: {cfg.weight_col}")
        df["weight"] = pd.to_numeric(df[cfg.weight_col], errors="coerce").fillna(0.0)
    else:
        df["weight"] = 1.0

    keyword_matrix, feature_names = build_keyword_matrix(df, cfg)

    if cfg.embedding_backend == "openai":
        model_tag = cfg.embedding_model.replace("/", "_")
        cache_path = cfg.output_dir / f"embeddings_{model_tag}_{cfg.embedding_dimensions}.npy"
        X_embed = build_openai_embeddings(
            texts=df["text_norm"].tolist(),
            cfg=cfg,
            cache_path=cache_path if cfg.use_embedding_cache else None,
        )
        embedding_meta = {
            "method": "openai",
            "model": cfg.embedding_model,
            "dimensions": cfg.embedding_dimensions,
            "batch_size": cfg.embedding_batch_size,
            "cache_path": str(cache_path) if cfg.use_embedding_cache else None,
        }
    elif cfg.embedding_backend == "tfidf":
        X_embed = keyword_matrix.toarray().astype(np.float32)
        embedding_meta = {
            "method": "tfidf",
            "tfidf_params": {
                "max_features": cfg.tfidf_max_features,
                "ngram_range": [cfg.tfidf_ngram_min, cfg.tfidf_ngram_max],
                "min_df": cfg.tfidf_min_df,
                "max_df": cfg.tfidf_max_df,
                "stop_words": "english",
            },
        }
    else:
        raise ValueError(f"Unsupported embedding backend: {cfg.embedding_backend}")

    n_neighbors = max(2, int(len(df) / 5))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    X_umap = np.asarray(reducer.fit_transform(X_embed), dtype=np.float64)
    if not np.isfinite(X_umap).all():
        bad = int(np.size(X_umap) - np.isfinite(X_umap).sum())
        print(f"[umap] detected non-finite values before scaling: {bad}; applying nan_to_num.")
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e6, neginf=-1e6)
    # normalize scale for numerically stable KMeans distance computations
    mu = np.nanmean(X_umap, axis=0, keepdims=True)
    sigma = np.nanstd(X_umap, axis=0, keepdims=True) + 1e-12
    X_umap = (X_umap - mu) / sigma
    if not np.isfinite(X_umap).all():
        bad = int(np.size(X_umap) - np.isfinite(X_umap).sum())
        print(f"[umap] detected non-finite values after scaling: {bad}; applying nan_to_num.")
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e4, neginf=-1e4)
    X_umap = np.clip(X_umap, -1e4, 1e4).astype(np.float32, copy=False)
    df["umap_x"] = X_umap[:, 1]
    df["umap_y"] = X_umap[:, 0]

    km = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init=20)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in matmul", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="overflow encountered in matmul", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in matmul", category=RuntimeWarning)
        df["cluster_id"] = km.fit_predict(X_umap)

    summaries = summarize_clusters_with_keywords(df, keyword_matrix, feature_names)

    title_prompt_text = None
    title_raw_text = None
    if cfg.title_backend == "openai":
        try:
            openai_titles, title_prompt_text, title_raw_text = generate_titles_with_openai(df, summaries, cfg)
            summaries = apply_titles_to_summaries(summaries, openai_titles)
        except Exception as exc:
            print(f"[title] OpenAI generation failed; keep keyword titles. reason={exc}")
    elif cfg.title_backend != "keywords":
        raise ValueError(f"Unsupported title backend: {cfg.title_backend}")

    title_map = {s["cluster_id"]: s["title"] for s in summaries}
    df["cluster_title"] = df["cluster_id"].map(title_map)
    df["non_negligible"] = df["weight"] > cfg.non_negligible_threshold

    points_csv = cfg.output_dir / "persona_umap_points.csv"
    cluster_csv = cfg.output_dir / "persona_clusters.csv"
    summary_json = cfg.output_dir / "cluster_summaries.json"
    summary_txt = cfg.output_dir / "cluster_summaries.txt"
    samples_json = cfg.output_dir / "cluster_samples.json"
    plot_png = cfg.output_dir / "persona_umap_clusters.png"
    meta_json = cfg.output_dir / "run_metadata.json"
    title_prompt_txt = cfg.output_dir / "cluster_title_prompt.txt"
    title_response_txt = cfg.output_dir / "cluster_title_response.txt"

    df[
        [
            "experiment",
            "participant",
            "game_finished",
            "weight",
            "non_negligible",
            "text",
            "umap_x",
            "umap_y",
            "cluster_id",
            "cluster_title",
        ]
    ].to_csv(points_csv, index=False)
    df[["participant", "cluster_id", "cluster_title", "weight", "non_negligible"]].to_csv(cluster_csv, index=False)

    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_txt.write_text(
        "\n".join(
            f"cluster {s['cluster_id']:02d} ({s['size']}): {s['title']} | "
            f"title_source={s['title_source']} | top_terms={', '.join(s['top_terms'][:8])}"
            for s in summaries
        ),
        encoding="utf-8",
    )

    if title_prompt_text:
        title_prompt_txt.write_text(title_prompt_text, encoding="utf-8")
    if title_raw_text:
        title_response_txt.write_text(title_raw_text, encoding="utf-8")

    dump_cluster_samples(df, summaries, samples_json, sample_n=cfg.min_rows_per_cluster_sample)
    plot_clusters(df, plot_png, summaries, non_negligible_threshold=cfg.non_negligible_threshold)

    metadata = {
        "input_path": str(cfg.input_path),
        "rows_total": int(len(df_all)),
        "rows_used": int(len(df)),
        "only_finished": cfg.only_finished,
        "embedding": embedding_meta,
        "title_backend": cfg.title_backend,
        "title_model": cfg.title_model if cfg.title_backend == "openai" else None,
        "title_max_codes_per_cluster": cfg.title_max_codes_per_cluster,
        "title_max_chars_per_code": cfg.title_max_chars_per_code,
        "title_strategy": cfg.title_strategy,
        "title_context_token_budget": cfg.title_context_token_budget,
        "title_refine_global": cfg.title_refine_global,
        "weight_col": cfg.weight_col,
        "non_negligible_threshold": cfg.non_negligible_threshold,
        "non_negligible_count": int(df["non_negligible"].sum()),
        "umap_params": {
            "n_components": 2,
            "n_neighbors": n_neighbors,
            "min_dist": cfg.umap_min_dist,
            "metric": cfg.umap_metric,
            "random_state": cfg.random_state,
        },
        "clustering_params": {
            "method": "KMeans",
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "n_init": 20,
        },
        "outputs": {
            "persona_umap_points_csv": str(points_csv),
            "persona_clusters_csv": str(cluster_csv),
            "cluster_summaries_json": str(summary_json),
            "cluster_summaries_txt": str(summary_txt),
            "cluster_samples_json": str(samples_json),
            "persona_umap_clusters_png": str(plot_png),
            "cluster_title_prompt_txt": str(title_prompt_txt) if title_prompt_text else None,
            "cluster_title_response_txt": str(title_response_txt) if title_raw_text else None,
        },
    }
    meta_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Fig.6-like clustering for persona JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", type=int, default=23)
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--embedding-backend", choices=["openai", "tfidf"], default="openai")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding-dimensions", type=int, default=1536)
    parser.add_argument("--embedding-batch-size", type=int, default=200)
    parser.add_argument("--no-embedding-cache", action="store_true")

    parser.add_argument("--title-backend", choices=["openai", "keywords"], default="openai")
    parser.add_argument("--title-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--title-max-codes-per-cluster", type=int, default=20)
    parser.add_argument("--title-max-chars-per-code", type=int, default=220)
    parser.add_argument("--title-strategy", choices=["auto", "global", "per_cluster"], default="auto")
    parser.add_argument("--title-context-token-budget", type=int, default=90000)
    parser.add_argument("--no-title-refine-global", action="store_true")
    parser.add_argument("--title-temperature", type=float, default=0.0)
    parser.add_argument("--title-use-extended", action="store_true",
        help="Generate extended titles (8-15 words) instead of short titles (2-4 words)")

    parser.add_argument("--weight-col", type=str, default=None)
    parser.add_argument("--weight-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        n_clusters=args.clusters,
        only_finished=not args.include_unfinished,
        random_state=args.random_state,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_batch_size=args.embedding_batch_size,
        use_embedding_cache=not args.no_embedding_cache,
        title_backend=args.title_backend,
        title_model=args.title_model,
        title_max_codes_per_cluster=args.title_max_codes_per_cluster,
        title_max_chars_per_code=args.title_max_chars_per_code,
        title_strategy=args.title_strategy,
        title_context_token_budget=args.title_context_token_budget,
        title_refine_global=not args.no_title_refine_global,
        title_temperature=args.title_temperature,
        title_use_extended=args.title_use_extended,
        weight_col=args.weight_col,
        non_negligible_threshold=args.weight_threshold,
    )


if __name__ == "__main__":
    run(parse_args())
