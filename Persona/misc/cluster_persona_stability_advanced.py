#!/usr/bin/env python3
"""
Advanced persona cluster stability analysis.

Adds three analyses on top of the previous stability workflow:
1) Shared-space comparison: fit one global embedding->UMAP space and compare splits there.
2) Delexicalized comparison: remove mechanism words/tags and rerun stability.
3) Bootstrap CIs: estimate uncertainty (alignment + silhouette) with resampling.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from umap import UMAP

from cluster_persona_fig6_like import (
    build_openai_embeddings,
    load_jsonl,
    normalize_text,
)


DEFAULT_INPUT = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/summary_gpt51_learn.jsonl")
DEFAULT_EXISTING_DIR = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_analysis")
DEFAULT_OUTPUT_DIR = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced")
DEFAULT_ORIGINAL_EMBED_CACHE = Path(
    "/Users/kehangzh/Desktop/PGG-finetuning/Persona/test_extended_titles/embeddings_text-embedding-3-large_1536.npy"
)


@dataclass
class AdvancedConfig:
    input_path: Path = DEFAULT_INPUT
    existing_dir: Path = DEFAULT_EXISTING_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    split_tags: List[str] = field(default_factory=lambda: ["PUNISHMENT", "REWARD", "COMMUNICATION"])
    base_n_clusters: int = 15
    adaptive_k: bool = True
    min_samples_per_cluster: int = 15
    random_state: int = 42

    # Step 1: original shared-space
    original_embedding_backend: str = "cached_openai"  # cached_openai | openai | tfidf
    original_embedding_cache_path: Optional[Path] = DEFAULT_ORIGINAL_EMBED_CACHE
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200

    # Step 2: delexicalized
    delex_embedding_backend: str = "tfidf"  # openai | tfidf
    delex_embedding_cache_path: Optional[Path] = None
    delex_mode: str = "word_mask"  # word_mask | drop_sections | word_mask_and_drop_sections
    mechanism_tags: List[str] = field(default_factory=lambda: ["PUNISHMENT", "REWARD", "COMMUNICATION"])
    mechanism_word_pattern: str = (
        r"\b(punishment|punish|punished|punishing|reward|rewards|rewarded|rewarding|"
        r"communication|communicate|chat|chats|message|messages|sanction|sanctions)\b"
    )

    # Shared UMAP params
    umap_min_dist: float = 0.5
    umap_metric: str = "cosine"

    # TF-IDF params
    tfidf_max_features: int = 1536
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95

    # Bootstrap
    bootstrap_iters: int = 100


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def filter_by_tag_presence(df: pd.DataFrame, tag: str, present: bool = True) -> pd.DataFrame:
    pattern = f"<{tag}>"
    mask = df["text"].str.contains(pattern, regex=False, na=False)
    return df[mask if present else ~mask].copy()


def compute_adaptive_k(n_samples: int, base_k: int, min_samples_per_cluster: int) -> int:
    base_samples_per_cluster = 3912 / base_k
    adaptive_k = max(2, int(n_samples / base_samples_per_cluster))
    max_k = max(2, n_samples // min_samples_per_cluster)
    return min(adaptive_k, max_k)


def delexicalize_text(text: str, cfg: AdvancedConfig) -> str:
    s = str(text)
    mode = cfg.delex_mode
    if mode not in {"word_mask", "drop_sections", "word_mask_and_drop_sections"}:
        raise ValueError(f"Unsupported delex_mode: {mode}")

    if mode in {"drop_sections", "word_mask_and_drop_sections"}:
        for tag in cfg.mechanism_tags:
            # Case 1: paired section tags (if present).
            s = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>",
                " ",
                s,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Case 2: open section without explicit closing tag;
            # remove from tag start to next section tag like <CONTRIBUTION>/<REWARD>/...
            s = re.sub(
                rf"<{tag}[^>]*>.*?(?=<[A-Z_]+[^>]*>|$)",
                " ",
                s,
                flags=re.IGNORECASE | re.DOTALL,
            )

    # Remove mechanism tags explicitly (for residual tag tokens).
    for tag in cfg.mechanism_tags:
        s = re.sub(rf"</?{tag}[^>]*>", " ", s, flags=re.IGNORECASE)

    if mode in {"word_mask", "word_mask_and_drop_sections"}:
        # Remove mechanism words.
        s = re.sub(cfg.mechanism_word_pattern, " ", s, flags=re.IGNORECASE)
    return normalize_text(s)


def build_tfidf_embeddings(texts: Sequence[str], cfg: AdvancedConfig) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        lowercase=True,
    )
    return vectorizer.fit_transform(texts).toarray().astype(np.float32)


def get_embeddings(
    texts: Sequence[str],
    cfg: AdvancedConfig,
    backend: str,
    cache_path: Optional[Path],
) -> np.ndarray:
    if backend == "cached_openai":
        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(f"cached_openai requires existing cache_path, got: {cache_path}")
        X = np.load(cache_path)
        if X.shape[0] != len(texts):
            raise ValueError(f"cached embedding rows mismatch: {X.shape[0]} vs {len(texts)}")
        return X.astype(np.float32)

    if backend == "openai":
        class TempConfig:
            def __init__(self, acfg: AdvancedConfig):
                self.embedding_model = acfg.embedding_model
                self.embedding_dimensions = acfg.embedding_dimensions
                self.embedding_batch_size = acfg.embedding_batch_size
                self.embedding_max_retries = 5
                self.embedding_retry_seconds = 1.0

        return build_openai_embeddings(texts=texts, cfg=TempConfig(cfg), cache_path=cache_path)

    if backend == "tfidf":
        return build_tfidf_embeddings(texts, cfg)

    raise ValueError(f"Unsupported backend: {backend}")


def fit_global_umap(X_embed: np.ndarray, cfg: AdvancedConfig) -> np.ndarray:
    n_neighbors = max(2, int(X_embed.shape[0] / 5))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    X_umap = np.asarray(reducer.fit_transform(X_embed), dtype=np.float64)
    if not np.isfinite(X_umap).all():
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e6, neginf=-1e6)
    mu = np.nanmean(X_umap, axis=0, keepdims=True)
    sigma = np.nanstd(X_umap, axis=0, keepdims=True) + 1e-12
    X_umap = (X_umap - mu) / sigma
    if not np.isfinite(X_umap).all():
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(X_umap, -1e4, 1e4).astype(np.float32, copy=False)


def cluster_and_metrics(X2: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        labels = km.fit_predict(X2)

    centroids = np.zeros((k, 2), dtype=np.float32)
    for cid in range(k):
        m = labels == cid
        if np.any(m):
            centroids[cid, :] = X2[m].mean(axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        metrics = {
            "silhouette_score": float(silhouette_score(X2, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(X2, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(X2, labels)),
        }
    return labels, centroids, metrics


def align_clusters(centroids_a: np.ndarray, centroids_b: np.ndarray) -> Tuple[Dict[int, int], float, np.ndarray]:
    cost = cdist(centroids_a, centroids_b, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    score = float(cost[row_ind, col_ind].mean())
    return mapping, score, cost


def analyze_shared_space(
    df_all: pd.DataFrame,
    X_shared: np.ndarray,
    cfg: AdvancedConfig,
    analysis_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work = df_all.copy()
    work["umap_x"] = X_shared[:, 0]
    work["umap_y"] = X_shared[:, 1]

    results: Dict[str, Any] = {}
    for tag in cfg.split_tags:
        df_with = filter_by_tag_presence(work, tag, present=True).reset_index(drop=True)
        df_without = filter_by_tag_presence(work, tag, present=False).reset_index(drop=True)

        if cfg.adaptive_k:
            k_with = compute_adaptive_k(len(df_with), cfg.base_n_clusters, cfg.min_samples_per_cluster)
            k_without = compute_adaptive_k(len(df_without), cfg.base_n_clusters, cfg.min_samples_per_cluster)
        else:
            k_with = cfg.base_n_clusters
            k_without = cfg.base_n_clusters

        X_with = df_with[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
        X_without = df_without[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)

        labels_with, centroids_with, qm_with = cluster_and_metrics(X_with, k_with, cfg.random_state)
        labels_without, centroids_without, qm_without = cluster_and_metrics(X_without, k_without, cfg.random_state)
        mapping, score, cost = align_clusters(centroids_with, centroids_without)

        df_with["cluster_id"] = labels_with
        df_without["cluster_id"] = labels_without

        tag_dir = output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        df_with.to_csv(tag_dir / "with_points.csv", index=False)
        df_without.to_csv(tag_dir / "without_points.csv", index=False)

        res = {
            "analysis": analysis_name,
            "tag": tag,
            "k_with": int(k_with),
            "k_without": int(k_without),
            "n_with": int(len(df_with)),
            "n_without": int(len(df_without)),
            "alignment_map": {str(k): int(v) for k, v in mapping.items()},
            "alignment_score": float(score),
            "quality_metrics_with": qm_with,
            "quality_metrics_without": qm_without,
        }
        (tag_dir / "alignment.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        results[tag] = {
            **res,
            "_df_with": df_with,
            "_df_without": df_without,
            "_cost": cost,
        }

        plot_shared_side_by_side(
            df_with=df_with,
            df_without=df_without,
            title=f"{analysis_name}: {tag} (alignment={score:.3f})",
            output_path=tag_dir / "comparison_side_by_side.png",
        )

    summary_rows = []
    for tag in cfg.split_tags:
        r = results[tag]
        summary_rows.append(
            {
                "tag": tag,
                "alignment_score": r["alignment_score"],
                "k_with": r["k_with"],
                "k_without": r["k_without"],
                "n_with": r["n_with"],
                "n_without": r["n_without"],
                "sil_with": r["quality_metrics_with"]["silhouette_score"],
                "sil_without": r["quality_metrics_without"]["silhouette_score"],
                "db_with": r["quality_metrics_with"]["davies_bouldin_score"],
                "db_without": r["quality_metrics_without"]["davies_bouldin_score"],
                "ch_with": r["quality_metrics_with"]["calinski_harabasz_score"],
                "ch_without": r["quality_metrics_without"]["calinski_harabasz_score"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    plot_summary_grid(summary_df, output_dir / "summary_dashboard.png", title=f"{analysis_name} Summary")

    for tag in list(results.keys()):
        # remove heavy in-memory fields from manifest copy
        for k in ["_df_with", "_df_without", "_cost"]:
            results[tag].pop(k, None)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "analysis": analysis_name,
                "config": {
                    "base_n_clusters": cfg.base_n_clusters,
                    "adaptive_k": cfg.adaptive_k,
                    "min_samples_per_cluster": cfg.min_samples_per_cluster,
                    "random_state": cfg.random_state,
                },
                "results_summary": results,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return results


def bootstrap_ci(
    df_with: pd.DataFrame,
    df_without: pd.DataFrame,
    k_with: int,
    k_without: int,
    n_iters: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X_with = df_with[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
    X_without = df_without[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
    n_with = X_with.shape[0]
    n_without = X_without.shape[0]

    align_scores: List[float] = []
    sil_with_scores: List[float] = []
    sil_without_scores: List[float] = []

    for i in range(n_iters):
        idx_w = rng.integers(0, n_with, size=n_with)
        idx_wo = rng.integers(0, n_without, size=n_without)
        sw = X_with[idx_w]
        swo = X_without[idx_wo]

        labels_w, cent_w, qm_w = cluster_and_metrics(sw, k_with, seed + i + 1)
        labels_wo, cent_wo, qm_wo = cluster_and_metrics(swo, k_without, seed + i + 1001)
        _, align, _ = align_clusters(cent_w, cent_wo)

        align_scores.append(float(align))
        sil_with_scores.append(float(qm_w["silhouette_score"]))
        sil_without_scores.append(float(qm_wo["silhouette_score"]))

    def ci(arr: List[float]) -> Dict[str, float]:
        x = np.array(arr, dtype=float)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "ci95_low": float(np.quantile(x, 0.025)),
            "ci95_high": float(np.quantile(x, 0.975)),
        }

    return {
        "alignment": ci(align_scores),
        "silhouette_with": ci(sil_with_scores),
        "silhouette_without": ci(sil_without_scores),
        "_samples": {
            "alignment": align_scores,
            "silhouette_with": sil_with_scores,
            "silhouette_without": sil_without_scores,
        },
    }


def plot_shared_side_by_side(
    df_with: pd.DataFrame,
    df_without: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    for ax, df, name in [
        (axes[0], df_with, "WITH"),
        (axes[1], df_without, "WITHOUT"),
    ]:
        k = int(df["cluster_id"].nunique())
        cmap = plt.get_cmap("tab20", k)
        for cid in sorted(df["cluster_id"].unique()):
            m = df["cluster_id"] == cid
            ax.scatter(
                df.loc[m, "umap_x"],
                df.loc[m, "umap_y"],
                s=20,
                alpha=0.65,
                c=[cmap(int(cid))],
                linewidths=0,
            )
            cx = df.loc[m, "umap_x"].mean()
            cy = df.loc[m, "umap_y"].mean()
            ax.text(cx, cy, str(int(cid)), fontsize=9, ha="center", va="center")
        ax.set_title(f"{name} (n={len(df)}, k={k})", fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_grid(summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    tags = summary_df["tag"].tolist()
    x = np.arange(len(tags))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    axes[0].bar(x - width / 2, summary_df["sil_with"], width, label="WITH", alpha=0.85)
    axes[0].bar(x + width / 2, summary_df["sil_without"], width, label="WITHOUT", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags)
    axes[0].set_title("Silhouette (higher better)")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].bar(tags, summary_df["alignment_score"], alpha=0.85, color="seagreen")
    axes[1].set_title("Alignment Score (lower better)")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x - width / 2, summary_df["n_with"], width, label="WITH", alpha=0.85)
    axes[2].bar(x + width / 2, summary_df["n_without"], width, label="WITHOUT", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags)
    axes[2].set_title("Split Sizes")
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend()

    axes[3].bar(x - width / 2, summary_df["k_with"], width, label="WITH", alpha=0.85)
    axes[3].bar(x + width / 2, summary_df["k_without"], width, label="WITHOUT", alpha=0.85)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(tags)
    axes[3].set_title("Cluster Counts")
    axes[3].grid(alpha=0.25, axis="y")
    axes[3].legend()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_existing_summary(existing_manifest: Dict[str, Any], output_path: Path) -> None:
    rows = []
    for tag, v in existing_manifest.get("results_summary", {}).items():
        rows.append(
            {
                "tag": tag,
                "alignment_score": v["alignment_score"],
                "k_with": v["k_with"],
                "k_without": v["k_without"],
                "n_with": v["n_with"],
                "n_without": v["n_without"],
                "sil_with": v["quality_metrics_with"]["silhouette_score"],
                "sil_without": v["quality_metrics_without"]["silhouette_score"],
                "db_with": v["quality_metrics_with"]["davies_bouldin_score"],
                "db_without": v["quality_metrics_without"]["davies_bouldin_score"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    plot_summary_grid(df, output_path, title="Existing Stability Results (Enhanced View)")


def plot_step12_compare(df1: pd.DataFrame, df2: pd.DataFrame, output_path: Path) -> None:
    # df1: shared-original, df2: shared-delex
    m = df1.merge(df2, on="tag", suffixes=("_orig", "_delex"))
    tags = m["tag"].tolist()
    x = np.arange(len(tags))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(x - width / 2, m["alignment_score_orig"], width, label="original", alpha=0.85)
    axes[0].bar(x + width / 2, m["alignment_score_delex"], width, label="delex", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags)
    axes[0].set_title("Alignment Score (lower better)")
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x - width / 2, m["sil_with_orig"], width, label="WITH original", alpha=0.85)
    axes[1].bar(x + width / 2, m["sil_with_delex"], width, label="WITH delex", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tags)
    axes[1].set_title("Silhouette WITH split")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x - width / 2, m["sil_without_orig"], width, label="WITHOUT original", alpha=0.85)
    axes[2].bar(x + width / 2, m["sil_without_delex"], width, label="WITHOUT delex", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags)
    axes[2].set_title("Silhouette WITHOUT split")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("Step1 vs Step2 Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    m.to_csv(output_path.with_suffix(".csv"), index=False)


def run_bootstrap_for_analysis(
    analysis_results: Dict[str, Any],
    base_dir: Path,
    cfg: AdvancedConfig,
    name: str,
) -> pd.DataFrame:
    rows = []
    plot_rows = []
    for tag in cfg.split_tags:
        with_df = pd.read_csv(base_dir / tag / "with_points.csv")
        without_df = pd.read_csv(base_dir / tag / "without_points.csv")
        k_with = int(analysis_results[tag]["k_with"])
        k_without = int(analysis_results[tag]["k_without"])

        res = bootstrap_ci(
            df_with=with_df,
            df_without=without_df,
            k_with=k_with,
            k_without=k_without,
            n_iters=cfg.bootstrap_iters,
            seed=cfg.random_state + 17,
        )
        rows.append(
            {
                "analysis": name,
                "tag": tag,
                "alignment_mean": res["alignment"]["mean"],
                "alignment_ci95_low": res["alignment"]["ci95_low"],
                "alignment_ci95_high": res["alignment"]["ci95_high"],
                "sil_with_mean": res["silhouette_with"]["mean"],
                "sil_with_ci95_low": res["silhouette_with"]["ci95_low"],
                "sil_with_ci95_high": res["silhouette_with"]["ci95_high"],
                "sil_without_mean": res["silhouette_without"]["mean"],
                "sil_without_ci95_low": res["silhouette_without"]["ci95_low"],
                "sil_without_ci95_high": res["silhouette_without"]["ci95_high"],
            }
        )

        for v in res["_samples"]["alignment"]:
            plot_rows.append({"analysis": name, "tag": tag, "metric": "alignment", "value": v})
        for v in res["_samples"]["silhouette_with"]:
            plot_rows.append({"analysis": name, "tag": tag, "metric": "sil_with", "value": v})
        for v in res["_samples"]["silhouette_without"]:
            plot_rows.append({"analysis": name, "tag": tag, "metric": "sil_without", "value": v})

    df = pd.DataFrame(rows)
    sample_df = pd.DataFrame(plot_rows)
    df.to_csv(base_dir / f"bootstrap_{name}_summary.csv", index=False)
    sample_df.to_csv(base_dir / f"bootstrap_{name}_samples.csv", index=False)
    plot_bootstrap_ci(df, base_dir / f"bootstrap_{name}_ci.png", title=f"Bootstrap CI ({name})")
    return df


def plot_bootstrap_ci(df: pd.DataFrame, output_path: Path, title: str) -> None:
    tags = df["tag"].tolist()
    x = np.arange(len(tags))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def err(mid, low, high):
        return np.vstack([mid - low, high - mid])

    axes[0].errorbar(
        x,
        df["alignment_mean"],
        yerr=err(df["alignment_mean"], df["alignment_ci95_low"], df["alignment_ci95_high"]),
        fmt="o",
        capsize=5,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags)
    axes[0].set_title("Alignment (mean +/- 95% CI)")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        x,
        df["sil_with_mean"],
        yerr=err(df["sil_with_mean"], df["sil_with_ci95_low"], df["sil_with_ci95_high"]),
        fmt="o",
        capsize=5,
        color="tab:blue",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tags)
    axes[1].set_title("Silhouette WITH (95% CI)")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        x,
        df["sil_without_mean"],
        yerr=err(df["sil_without_mean"], df["sil_without_ci95_low"], df["sil_without_ci95_high"]),
        fmt="o",
        capsize=5,
        color="tab:orange",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags)
    axes[2].set_title("Silhouette WITHOUT (95% CI)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def run(cfg: AdvancedConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all = load_jsonl(cfg.input_path)
    df_all = df_all[df_all["game_finished"] == True].copy().reset_index(drop=True)
    df_all["text_norm"] = df_all["text"].map(normalize_text)
    df_all["text_delex"] = df_all["text"].map(lambda t: delexicalize_text(t, cfg))

    # Step0: existing visualization
    existing_manifest_path = cfg.existing_dir / "manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = json.loads(existing_manifest_path.read_text())
        plot_existing_summary(existing_manifest, cfg.output_dir / "step0_existing_enhanced.png")

    # Step1: shared global space on original text
    X_orig = get_embeddings(
        texts=df_all["text_norm"].tolist(),
        cfg=cfg,
        backend=cfg.original_embedding_backend,
        cache_path=cfg.original_embedding_cache_path,
    )
    X_orig_umap = fit_global_umap(X_orig, cfg)
    np.save(cfg.output_dir / "step1_global_umap_original.npy", X_orig_umap)
    step1_dir = cfg.output_dir / "step1_shared_space_original"
    step1_results = analyze_shared_space(
        df_all=df_all,
        X_shared=X_orig_umap,
        cfg=cfg,
        analysis_name="Step1 Shared-Space (Original)",
        output_dir=step1_dir,
    )

    # Step2: delexicalized rerun
    X_delex = get_embeddings(
        texts=df_all["text_delex"].tolist(),
        cfg=cfg,
        backend=cfg.delex_embedding_backend,
        cache_path=cfg.delex_embedding_cache_path,
    )
    X_delex_umap = fit_global_umap(X_delex, cfg)
    np.save(cfg.output_dir / "step2_global_umap_delex.npy", X_delex_umap)
    step2_dir = cfg.output_dir / "step2_shared_space_delex"
    step2_results = analyze_shared_space(
        df_all=df_all,
        X_shared=X_delex_umap,
        cfg=cfg,
        analysis_name="Step2 Shared-Space (Delexicalized)",
        output_dir=step2_dir,
    )

    # Step1 vs Step2 visualization
    s1 = pd.read_csv(step1_dir / "summary_metrics.csv")
    s2 = pd.read_csv(step2_dir / "summary_metrics.csv")
    plot_step12_compare(s1, s2, cfg.output_dir / "step12_compare.png")

    # Step3: bootstrap CI on both analyses
    bs1 = run_bootstrap_for_analysis(step1_results, step1_dir, cfg, name="step1_original")
    bs2 = run_bootstrap_for_analysis(step2_results, step2_dir, cfg, name="step2_delex")
    plot_step12_compare(
        bs1.rename(
            columns={
                "alignment_mean": "alignment_score",
                "sil_with_mean": "sil_with",
                "sil_without_mean": "sil_without",
            }
        )[["tag", "alignment_score", "sil_with", "sil_without"]],
        bs2.rename(
            columns={
                "alignment_mean": "alignment_score",
                "sil_with_mean": "sil_with",
                "sil_without_mean": "sil_without",
            }
        )[["tag", "alignment_score", "sil_with", "sil_without"]],
        cfg.output_dir / "step3_bootstrap_mean_compare.png",
    )

    manifest = {
        "config": _to_jsonable(asdict(cfg)),
        "rows_used": int(len(df_all)),
        "outputs": {
            "step0_existing_enhanced_png": str(cfg.output_dir / "step0_existing_enhanced.png"),
            "step1_dir": str(step1_dir),
            "step2_dir": str(step2_dir),
            "step12_compare_png": str(cfg.output_dir / "step12_compare.png"),
            "bootstrap_step1_csv": str(step1_dir / "bootstrap_step1_original_summary.csv"),
            "bootstrap_step2_csv": str(step2_dir / "bootstrap_step2_delex_summary.csv"),
            "bootstrap_step1_png": str(step1_dir / "bootstrap_step1_original_ci.png"),
            "bootstrap_step2_png": str(step2_dir / "bootstrap_step2_delex_ci.png"),
            "bootstrap_compare_png": str(cfg.output_dir / "step3_bootstrap_mean_compare.png"),
        },
        "timestamp": datetime.now().isoformat(),
    }
    (cfg.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> AdvancedConfig:
    parser = argparse.ArgumentParser(description="Advanced persona stability analysis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--existing-dir", type=Path, default=DEFAULT_EXISTING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-tags", nargs="+", default=["PUNISHMENT", "REWARD", "COMMUNICATION"])
    parser.add_argument("--base-clusters", type=int, default=15)
    parser.add_argument("--no-adaptive-k", action="store_true")
    parser.add_argument("--min-samples-per-cluster", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--original-embedding-backend",
        choices=["cached_openai", "openai", "tfidf"],
        default="cached_openai",
    )
    parser.add_argument("--original-embedding-cache-path", type=Path, default=DEFAULT_ORIGINAL_EMBED_CACHE)

    parser.add_argument("--delex-embedding-backend", choices=["openai", "tfidf"], default="tfidf")
    parser.add_argument("--delex-embedding-cache-path", type=Path, default=None)
    parser.add_argument(
        "--delex-mode",
        choices=["word_mask", "drop_sections", "word_mask_and_drop_sections"],
        default="word_mask",
    )

    parser.add_argument("--bootstrap-iters", type=int, default=100)
    args = parser.parse_args()

    return AdvancedConfig(
        input_path=args.input,
        existing_dir=args.existing_dir,
        output_dir=args.output_dir,
        split_tags=args.split_tags,
        base_n_clusters=args.base_clusters,
        adaptive_k=not args.no_adaptive_k,
        min_samples_per_cluster=args.min_samples_per_cluster,
        random_state=args.random_state,
        original_embedding_backend=args.original_embedding_backend,
        original_embedding_cache_path=args.original_embedding_cache_path,
        delex_embedding_backend=args.delex_embedding_backend,
        delex_embedding_cache_path=args.delex_embedding_cache_path,
        delex_mode=args.delex_mode,
        bootstrap_iters=args.bootstrap_iters,
    )


if __name__ == "__main__":
    run(parse_args())
