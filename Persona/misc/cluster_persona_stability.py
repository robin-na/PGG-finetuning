#!/usr/bin/env python3
"""
Persona clustering stability analysis.

This script evaluates the stability of persona clusters across different game variants
by splitting personas based on tag presence (e.g., WITH/WITHOUT PUNISHMENT) and comparing
clustering results between splits.

The core insight: All personas come from the same subject pool playing PGG, but different
game configurations produce different behavioral tags. Stability analysis reveals whether
similar behavioral patterns emerge regardless of game mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from umap import UMAP

# Import from existing clustering script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cluster_persona_fig6_like import (
    load_jsonl,
    normalize_text,
    build_openai_embeddings,
    build_keyword_matrix,
    summarize_clusters_with_keywords,
    generate_titles_with_openai,
    apply_titles_to_summaries,
    _get_openai_client,
)

load_dotenv()

DEFAULT_INPUT = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/summary_gpt51_learn.jsonl")
DEFAULT_OUTPUT_DIR = Path("/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_analysis")


@dataclass
class StabilityConfig:
    """Configuration for stability analysis pipeline."""
    input_path: Path
    output_dir: Path

    # Clustering parameters (inherited from base)
    embedding_backend: str = "openai"  # openai | tfidf
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 200
    use_embedding_cache: bool = True
    embedding_cache_path: Optional[Path] = None

    # Stability-specific parameters
    split_tags: List[str] = field(default_factory=lambda: ["PUNISHMENT", "REWARD", "COMMUNICATION"])
    base_n_clusters: int = 23  # Base k value for full dataset
    adaptive_k: bool = True     # Adjust k based on split size
    min_samples_per_cluster: int = 15  # Minimum samples required per cluster
    random_state: int = 42

    # Analysis parameters
    alignment_method: str = "hungarian"
    compute_silhouette: bool = True

    # Title generation
    title_backend: str = "openai"  # openai | keywords
    title_model: str = "gpt-4o-mini"
    title_use_extended: bool = True  # Use extended titles (8-15 words)
    title_temperature: float = 0.0
    title_max_codes_per_cluster: int = 20
    title_max_chars_per_code: int = 220
    title_strategy: str = "auto"
    title_context_token_budget: int = 90000
    title_refine_global: bool = True
    title_max_retries: int = 5
    title_retry_seconds: float = 1.0

    # UMAP parameters
    umap_min_dist: float = 0.5
    umap_metric: str = "cosine"

    # TF-IDF fallback parameters
    tfidf_max_features: int = 1536
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95


def filter_by_tag_presence(df: pd.DataFrame, tag: str, present: bool = True) -> pd.DataFrame:
    """
    Filter personas based on whether a specific tag appears in their text.

    Args:
        df: Full persona dataframe with columns: experiment, participant, game_finished, text
        tag: Tag name (e.g., "PUNISHMENT", "REWARD", "COMMUNICATION")
        present: If True, return personas WITH tag; if False, WITHOUT tag

    Returns:
        Filtered dataframe
    """
    pattern = f"<{tag}>"
    mask = df["text"].str.contains(pattern, regex=False, na=False)
    return df[mask if present else ~mask].copy()


def compute_adaptive_k(n_samples: int, base_k: int, min_samples_per_cluster: int) -> int:
    """
    Compute cluster count based on data size.

    Strategy: Maintain approximately same samples-per-cluster ratio as base dataset.

    Args:
        n_samples: Number of samples in current split
        base_k: Base number of clusters (e.g., 23 for full dataset)
        min_samples_per_cluster: Minimum samples required per cluster

    Returns:
        Adjusted k value
    """
    # Estimate samples per cluster from base (assuming ~3912 personas total)
    base_samples_per_cluster = 3912 / base_k  # ~170 samples/cluster

    # Calculate proportional k
    adaptive_k = max(2, int(n_samples / base_samples_per_cluster))

    # Ensure minimum samples per cluster
    max_k = n_samples // min_samples_per_cluster
    adaptive_k = min(adaptive_k, max_k)

    return adaptive_k


def compute_cluster_quality(X_embed: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute intrinsic clustering quality metrics.

    Args:
        X_embed: Embedding matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)

    Returns:
        Dictionary with quality metrics
    """
    return {
        "silhouette_score": float(silhouette_score(X_embed, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_embed, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_embed, labels)),
    }


def align_clusters_hungarian(
    split_a_centroids: np.ndarray,
    split_b_centroids: np.ndarray,
) -> Tuple[Dict[int, int], float, np.ndarray]:
    """
    Align clusters between two splits using Hungarian algorithm on centroid distances.

    Args:
        split_a_centroids: (k_a, 2) array of UMAP centroids for split A
        split_b_centroids: (k_b, 2) array of UMAP centroids for split B

    Returns:
        alignment_map: Dict mapping split_a_cluster_id -> split_b_cluster_id
        alignment_score: Average centroid distance of aligned pairs (lower is better)
        cost_matrix: Full pairwise centroid distance matrix
    """
    # Compute pairwise centroid distances
    cost_matrix = cdist(split_a_centroids, split_b_centroids, metric='euclidean')

    # Solve optimal assignment (minimizes total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build alignment map
    alignment_map = {int(row_ind[i]): int(col_ind[i]) for i in range(len(row_ind))}

    # Compute alignment score (average distance of aligned pairs)
    aligned_distances = cost_matrix[row_ind, col_ind]
    alignment_score = float(aligned_distances.mean())

    return alignment_map, alignment_score, cost_matrix


def run_clustering_on_split(
    df: pd.DataFrame,
    cfg: StabilityConfig,
    output_subdir: str,
    n_clusters: int,
    df_all_with_embeddings: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Run full clustering pipeline on a data split.

    Args:
        df: Split dataframe (WITH or WITHOUT certain tag)
        cfg: Stability configuration
        output_subdir: Subdirectory name for outputs (e.g., "PUNISHMENT_WITH")
        n_clusters: Number of clusters for this split
        df_all_with_embeddings: Tuple of (full_df, full_embeddings) for reusing cached embeddings

    Returns:
        Dictionary with:
            - df: dataframe with cluster assignments
            - summaries: cluster summaries list
            - embeddings: embedding matrix for this split
            - cluster_centroids: (n_clusters, 2) UMAP centroids
            - quality_metrics: dict with silhouette/calinski/davies-bouldin scores
    """
    output_dir = cfg.output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare split data
    df = df.reset_index(drop=True)
    df["text_norm"] = df["text"].map(normalize_text)

    # Get embeddings for this split
    if df_all_with_embeddings is not None:
        # Reuse cached embeddings by matching on experiment+participant
        df_all, full_embeddings = df_all_with_embeddings

        # Create lookup for indices
        df_all_lookup = {
            (str(row["experiment"]), str(row["participant"])): idx
            for idx, row in df_all.iterrows()
        }

        # Extract embeddings for current split
        indices = []
        for _, row in df.iterrows():
            key = (str(row["experiment"]), str(row["participant"]))
            if key in df_all_lookup:
                indices.append(df_all_lookup[key])

        X_embed = full_embeddings[indices].astype(np.float32)
    elif cfg.embedding_backend == "openai":
        # Build embeddings from scratch (fallback)
        print(f"    Building OpenAI embeddings for {len(df)} samples...")
        cache_path = output_dir / f"embeddings_{cfg.embedding_model.replace('/', '_')}_{cfg.embedding_dimensions}.npy"

        # Create temporary config object
        class TempConfig:
            def __init__(self, scfg):
                self.embedding_model = scfg.embedding_model
                self.embedding_dimensions = scfg.embedding_dimensions
                self.embedding_batch_size = scfg.embedding_batch_size
                self.embedding_max_retries = scfg.title_max_retries
                self.embedding_retry_seconds = scfg.title_retry_seconds

        X_embed = build_openai_embeddings(
            texts=df["text_norm"].tolist(),
            cfg=TempConfig(cfg),
            cache_path=cache_path if cfg.use_embedding_cache else None,
        )
    else:  # tfidf
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=cfg.tfidf_max_features,
            ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
            min_df=cfg.tfidf_min_df,
            max_df=cfg.tfidf_max_df,
            lowercase=True,
        )
        X_embed = vectorizer.fit_transform(df["text_norm"]).toarray().astype(np.float32)

    # UMAP reduction
    n_neighbors = max(2, int(len(df) / 5))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    X_umap = np.asarray(reducer.fit_transform(X_embed), dtype=np.float64)

    # Handle non-finite values
    if not np.isfinite(X_umap).all():
        bad = int(np.size(X_umap) - np.isfinite(X_umap).sum())
        print(f"    [warn] detected {bad} non-finite values in UMAP; applying nan_to_num")
        X_umap = np.nan_to_num(X_umap, nan=0.0, posinf=1e6, neginf=-1e6)

    # Normalize for stable KMeans
    mu = np.nanmean(X_umap, axis=0, keepdims=True)
    sigma = np.nanstd(X_umap, axis=0, keepdims=True) + 1e-12
    X_umap = (X_umap - mu) / sigma
    X_umap = np.clip(X_umap, -1e4, 1e4).astype(np.float32)

    df["umap_x"] = X_umap[:, 0]
    df["umap_y"] = X_umap[:, 1]

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init=20)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        df["cluster_id"] = km.fit_predict(X_umap)

    # Compute cluster centroids in UMAP space
    cluster_centroids = np.zeros((n_clusters, 2), dtype=np.float32)
    for cid in range(n_clusters):
        mask = df["cluster_id"] == cid
        if mask.any():
            cluster_centroids[cid, 0] = df.loc[mask, "umap_x"].mean()
            cluster_centroids[cid, 1] = df.loc[mask, "umap_y"].mean()

    # Build keyword matrix for fallback titles
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        lowercase=True,
    )
    keyword_matrix = vectorizer.fit_transform(df["text_norm"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Generate initial titles with keywords
    summaries = summarize_clusters_with_keywords(df, keyword_matrix, feature_names)

    # Generate titles with OpenAI if configured
    if cfg.title_backend == "openai":
        try:
            # Create temporary config compatible with title generation
            class TitleConfig:
                def __init__(self, scfg):
                    self.title_model = scfg.title_model
                    self.title_temperature = scfg.title_temperature
                    self.title_max_codes_per_cluster = scfg.title_max_codes_per_cluster
                    self.title_max_chars_per_code = scfg.title_max_chars_per_code
                    self.title_strategy = scfg.title_strategy
                    self.title_context_token_budget = scfg.title_context_token_budget
                    self.title_refine_global = scfg.title_refine_global
                    self.title_max_retries = scfg.title_max_retries
                    self.title_retry_seconds = scfg.title_retry_seconds
                    self.title_use_extended = scfg.title_use_extended
                    self.title_extended_min_words = 8
                    self.title_extended_max_words = 15

            openai_titles, _, _ = generate_titles_with_openai(df, summaries, TitleConfig(cfg))
            summaries = apply_titles_to_summaries(summaries, openai_titles)
        except Exception as exc:
            print(f"    [warn] OpenAI title generation failed: {exc}; using keyword titles")

    # Add titles to dataframe
    title_map = {s["cluster_id"]: s["title"] for s in summaries}
    df["cluster_title"] = df["cluster_id"].map(title_map)

    # Compute quality metrics
    quality_metrics = compute_cluster_quality(X_umap, df["cluster_id"].values)

    # Save outputs
    df[["experiment", "participant", "game_finished", "text", "umap_x", "umap_y", "cluster_id", "cluster_title"]].to_csv(
        output_dir / "persona_umap_points.csv", index=False
    )

    with open(output_dir / "cluster_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    with open(output_dir / "quality_metrics.json", "w") as f:
        json.dump(quality_metrics, f, indent=2)

    return {
        "df": df,
        "summaries": summaries,
        "embeddings": X_embed,
        "umap_embeddings": X_umap,
        "cluster_centroids": cluster_centroids,
        "quality_metrics": quality_metrics,
    }


def plot_split_comparison(
    split_a_df: pd.DataFrame,
    split_b_df: pd.DataFrame,
    split_a_summaries: List[dict],
    split_b_summaries: List[dict],
    alignment_map: Dict[int, int],
    cost_matrix: np.ndarray,
    output_path: Path,
    tag_name: str,
):
    """
    Create side-by-side UMAP plot comparing two splits.

    Args:
        split_a_df: DataFrame for split A (WITH tag)
        split_b_df: DataFrame for split B (WITHOUT tag)
        split_a_summaries: Cluster summaries for split A
        split_b_summaries: Cluster summaries for split B
        alignment_map: Dict mapping cluster_a_id -> cluster_b_id
        cost_matrix: Pairwise centroid distance matrix
        output_path: Path to save figure
        tag_name: Name of tag being analyzed
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Create color map for aligned pairs
    n_clusters_a = len(split_a_summaries)
    n_clusters_b = len(split_b_summaries)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters_a, n_clusters_b)))

    # Plot split A (WITH tag)
    for cid_a in range(n_clusters_a):
        mask = split_a_df["cluster_id"] == cid_a
        if mask.any():
            ax1.scatter(
                split_a_df.loc[mask, "umap_x"],
                split_a_df.loc[mask, "umap_y"],
                c=[colors[cid_a]],
                s=50,
                alpha=0.6,
                label=f"Cluster {cid_a}"
            )

            # Add cluster label
            cx = split_a_df.loc[mask, "umap_x"].mean()
            cy = split_a_df.loc[mask, "umap_y"].mean()
            title = split_a_summaries[cid_a]["title"]
            # Truncate long titles for display
            if len(title) > 60:
                title = title[:57] + "..."
            ax1.text(
                cx, cy, f"{cid_a}",
                fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=colors[cid_a], linewidth=2, pad=3)
            )

    ax1.set_xlabel("UMAP-1", fontsize=12)
    ax1.set_ylabel("UMAP-2", fontsize=12)
    ax1.set_title(f"WITH {tag_name} (n={len(split_a_df)}, k={n_clusters_a})", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot split B (WITHOUT tag)
    for cid_b in range(n_clusters_b):
        mask = split_b_df["cluster_id"] == cid_b
        if mask.any():
            # Find aligned cluster from A (if any)
            aligned_from_a = [ka for ka, vb in alignment_map.items() if vb == cid_b]
            color_idx = aligned_from_a[0] if aligned_from_a else cid_b

            ax2.scatter(
                split_b_df.loc[mask, "umap_x"],
                split_b_df.loc[mask, "umap_y"],
                c=[colors[color_idx]],
                s=50,
                alpha=0.6,
                label=f"Cluster {cid_b}"
            )

            # Add cluster label
            cx = split_b_df.loc[mask, "umap_x"].mean()
            cy = split_b_df.loc[mask, "umap_y"].mean()
            title = split_b_summaries[cid_b]["title"]
            if len(title) > 60:
                title = title[:57] + "..."
            ax2.text(
                cx, cy, f"{cid_b}",
                fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=colors[color_idx], linewidth=2, pad=3)
            )

    ax2.set_xlabel("UMAP-1", fontsize=12)
    ax2.set_ylabel("UMAP-2", fontsize=12)
    ax2.set_title(f"WITHOUT {tag_name} (n={len(split_b_df)}, k={n_clusters_b})", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add alignment info
    avg_distance = cost_matrix[list(alignment_map.keys()), list(alignment_map.values())].mean()
    fig.suptitle(
        f"Stability Analysis: {tag_name} Split Comparison\\nAlignment Score (avg centroid distance): {avg_distance:.3f}",
        fontsize=16, fontweight="bold", y=0.98
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved comparison plot: {output_path}")


def plot_stability_summary(
    results_by_tag: Dict[str, Dict],
    output_path: Path,
):
    """
    Create bar chart summary of stability metrics across all tags.

    Args:
        results_by_tag: Dictionary with tag names as keys and result dicts as values
        output_path: Path to save figure
    """
    if not results_by_tag:
        print("  No results to summarize")
        return

    tags = list(results_by_tag.keys())
    n_tags = len(tags)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Metric 1: Silhouette Scores
    with_scores = [results_by_tag[tag]["quality_metrics_with"]["silhouette_score"] for tag in tags]
    without_scores = [results_by_tag[tag]["quality_metrics_without"]["silhouette_score"] for tag in tags]

    x = np.arange(n_tags)
    width = 0.35

    axes[0].bar(x - width/2, with_scores, width, label="WITH tag", alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, without_scores, width, label="WITHOUT tag", alpha=0.8, color='coral')
    axes[0].set_ylabel("Silhouette Score", fontsize=12)
    axes[0].set_title("Clustering Quality: Silhouette Score (higher is better)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Metric 2: Alignment Scores
    alignment_scores = [results_by_tag[tag]["alignment_score"] for tag in tags]
    axes[1].bar(tags, alignment_scores, alpha=0.8, color='mediumseagreen')
    axes[1].set_ylabel("Alignment Score (centroid distance)", fontsize=12)
    axes[1].set_title("Cluster Alignment: Avg Centroid Distance (lower is better)", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Metric 3: Split Sizes
    n_with = [results_by_tag[tag]["n_with"] for tag in tags]
    n_without = [results_by_tag[tag]["n_without"] for tag in tags]

    axes[2].bar(x - width/2, n_with, width, label="WITH tag", alpha=0.8, color='steelblue')
    axes[2].bar(x + width/2, n_without, width, label="WITHOUT tag", alpha=0.8, color='coral')
    axes[2].set_ylabel("Number of Personas", fontsize=12)
    axes[2].set_title("Split Sizes", fontsize=13, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    # Metric 4: Cluster Counts
    k_with = [results_by_tag[tag]["k_with"] for tag in tags]
    k_without = [results_by_tag[tag]["k_without"] for tag in tags]

    axes[3].bar(x - width/2, k_with, width, label="WITH tag", alpha=0.8, color='steelblue')
    axes[3].bar(x + width/2, k_without, width, label="WITHOUT tag", alpha=0.8, color='coral')
    axes[3].set_ylabel("Number of Clusters", fontsize=12)
    axes[3].set_title("Cluster Counts (k)", fontsize=13, fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(tags)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3, axis='y')

    fig.suptitle("Stability Analysis Summary Across Tags", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary plot: {output_path}")


def run_stability_analysis(cfg: StabilityConfig):
    """
    Main pipeline for stability analysis.
    """
    print(f"{'='*80}")
    print(f"Persona Clustering Stability Analysis")
    print(f"{'='*80}\n")

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    print("[1/6] Loading persona dataset...")
    df_all = load_jsonl(cfg.input_path)
    df_all = df_all[df_all["game_finished"] == True].copy().reset_index(drop=True)
    df_all["text_norm"] = df_all["text"].map(normalize_text)
    print(f"  Loaded {len(df_all)} completed game personas")

    # Load or build embeddings once for entire dataset
    cached_embeddings = None
    if cfg.embedding_backend == "openai":
        print(f"\n[2/6] Building embeddings for full dataset...")
        if cfg.embedding_cache_path and cfg.embedding_cache_path.exists():
            print(f"  Loading cached embeddings from {cfg.embedding_cache_path}")
            cached_embeddings = np.load(cfg.embedding_cache_path)
        else:
            cache_path = cfg.output_dir / f"embeddings_{cfg.embedding_model.replace('/', '_')}_{cfg.embedding_dimensions}.npy"
            # Create temporary PipelineConfig-like object for compatibility
            class TempConfig:
                def __init__(self, scfg):
                    self.embedding_model = scfg.embedding_model
                    self.embedding_dimensions = scfg.embedding_dimensions
                    self.embedding_batch_size = scfg.embedding_batch_size
                    self.embedding_max_retries = scfg.title_max_retries
                    self.embedding_retry_seconds = scfg.title_retry_seconds

            cached_embeddings = build_openai_embeddings(
                texts=df_all["text_norm"].tolist(),
                cfg=TempConfig(cfg),
                cache_path=cache_path if cfg.use_embedding_cache else None,
            )

    results_by_tag = {}

    # Analyze each tag
    for tag_idx, tag in enumerate(cfg.split_tags, 1):
        print(f"\n{'='*80}")
        print(f"[{tag_idx+2}/6] Analyzing stability for tag: {tag}")
        print(f"{'='*80}")

        # Split by tag presence
        df_with = filter_by_tag_presence(df_all, tag, present=True)
        df_without = filter_by_tag_presence(df_all, tag, present=False)

        print(f"\n  Split sizes:")
        print(f"    WITH {tag}: {len(df_with)} personas ({100*len(df_with)/len(df_all):.1f}%)")
        print(f"    WITHOUT {tag}: {len(df_without)} personas ({100*len(df_without)/len(df_all):.1f}%)")

        # Compute adaptive k for each split
        if cfg.adaptive_k:
            k_with = compute_adaptive_k(len(df_with), cfg.base_n_clusters, cfg.min_samples_per_cluster)
            k_without = compute_adaptive_k(len(df_without), cfg.base_n_clusters, cfg.min_samples_per_cluster)
            print(f"\n  Adaptive cluster counts:")
            print(f"    k_with = {k_with} ({len(df_with)/k_with:.1f} samples/cluster)")
            print(f"    k_without = {k_without} ({len(df_without)/k_without:.1f} samples/cluster)")
        else:
            k_with = k_without = cfg.base_n_clusters
            print(f"\n  Using fixed k = {cfg.base_n_clusters} for both splits")

        # Verify sufficient data
        if len(df_with) < k_with * cfg.min_samples_per_cluster:
            print(f"\n  WARNING: Insufficient data for WITH split ({len(df_with)} < {k_with * cfg.min_samples_per_cluster})")
            print(f"  Skipping {tag} analysis.")
            continue

        if len(df_without) < k_without * cfg.min_samples_per_cluster:
            print(f"\n  WARNING: Insufficient data for WITHOUT split ({len(df_without)} < {k_without * cfg.min_samples_per_cluster})")
            print(f"  Skipping {tag} analysis.")
            continue

        # Run clustering on each split
        print(f"\n  Running clustering on WITH split...")
        results_with = run_clustering_on_split(
            df_with, cfg,
            output_subdir=f"{tag}_WITH",
            n_clusters=k_with,
            df_all_with_embeddings=(df_all, cached_embeddings) if cached_embeddings is not None else None,
        )

        print(f"  Running clustering on WITHOUT split...")
        results_without = run_clustering_on_split(
            df_without, cfg,
            output_subdir=f"{tag}_WITHOUT",
            n_clusters=k_without,
            df_all_with_embeddings=(df_all, cached_embeddings) if cached_embeddings is not None else None,
        )

        # Align clusters
        print(f"  Computing cluster alignment...")
        alignment_map, alignment_score, cost_matrix = align_clusters_hungarian(
            results_with["cluster_centroids"],
            results_without["cluster_centroids"]
        )

        # Save alignment results
        alignment_output = {
            "tag": tag,
            "k_with": k_with,
            "k_without": k_without,
            "n_with": len(df_with),
            "n_without": len(df_without),
            "alignment_map": {str(k): int(v) for k, v in alignment_map.items()},
            "alignment_score": alignment_score,
            "quality_metrics_with": results_with["quality_metrics"],
            "quality_metrics_without": results_without["quality_metrics"],
        }

        comparison_dir = cfg.output_dir / f"{tag}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        with open(comparison_dir / "alignment.json", "w") as f:
            json.dump(alignment_output, f, indent=2)

        # Generate visualizations
        print(f"  Generating visualizations...")
        try:
            plot_split_comparison(
                results_with["df"], results_without["df"],
                results_with["summaries"], results_without["summaries"],
                alignment_map, cost_matrix,
                comparison_dir / "comparison_side_by_side.png",
                tag
            )
        except Exception as exc:
            print(f"    [warn] Visualization failed: {exc}")

        results_by_tag[tag] = alignment_output

        print(f"\n  {tag} analysis complete!")

    # Generate cross-tag summary
    if results_by_tag:
        print(f"\n{'='*80}")
        print(f"[{len(cfg.split_tags)+3}/6] Generating cross-tag summary...")
        print(f"{'='*80}")

        try:
            plot_stability_summary(
                results_by_tag,
                cfg.output_dir / "stability_summary.png"
            )
        except Exception as exc:
            print(f"  [warn] Summary visualization failed: {exc}")

        # Save manifest
        manifest = {
            "config": {
                "input_path": str(cfg.input_path),
                "split_tags": cfg.split_tags,
                "base_n_clusters": cfg.base_n_clusters,
                "adaptive_k": cfg.adaptive_k,
                "min_samples_per_cluster": cfg.min_samples_per_cluster,
                "embedding_backend": cfg.embedding_backend,
                "title_backend": cfg.title_backend,
                "title_use_extended": cfg.title_use_extended,
                "random_state": cfg.random_state,
            },
            "results_summary": results_by_tag,
            "timestamp": datetime.now().isoformat(),
        }

        with open(cfg.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Saved manifest: {cfg.output_dir / 'manifest.json'}")

    print(f"\n{'='*80}")
    print(f"Stability analysis complete!")
    print(f"Results saved to: {cfg.output_dir}")
    print(f"{'='*80}")


def parse_args() -> StabilityConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Persona clustering stability analysis across game variants"
    )

    parser.add_argument("--input", type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL file with persona summaries")
    parser.add_argument("--output-dir", type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results")

    parser.add_argument("--split-tags", nargs="+",
        default=["PUNISHMENT", "REWARD", "COMMUNICATION"],
        help="Tags to analyze for stability")
    parser.add_argument("--base-clusters", type=int, default=23,
        help="Base number of clusters for full dataset")
    parser.add_argument("--no-adaptive-k", action="store_true",
        help="Use fixed k instead of adaptive k based on split size")
    parser.add_argument("--min-samples-per-cluster", type=int, default=15,
        help="Minimum samples required per cluster")

    parser.add_argument("--embedding-backend", choices=["openai", "tfidf"], default="openai",
        help="Embedding method")
    parser.add_argument("--embedding-cache-path", type=Path, default=None,
        help="Path to existing embedding cache to reuse")

    parser.add_argument("--title-backend", choices=["openai", "keywords"], default="openai",
        help="Title generation method")
    parser.add_argument("--title-use-extended", action="store_true", default=True,
        help="Generate extended titles (8-15 words)")

    parser.add_argument("--random-state", type=int, default=42,
        help="Random seed for reproducibility")

    args = parser.parse_args()

    return StabilityConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        split_tags=args.split_tags,
        base_n_clusters=args.base_clusters,
        adaptive_k=not args.no_adaptive_k,
        min_samples_per_cluster=args.min_samples_per_cluster,
        embedding_backend=args.embedding_backend,
        use_embedding_cache=args.embedding_cache_path is not None,
        embedding_cache_path=args.embedding_cache_path,
        title_backend=args.title_backend,
        title_use_extended=args.title_use_extended,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    run_stability_analysis(parse_args())
