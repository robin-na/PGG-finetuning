# Cluster Pipeline (Embedding + KMeans + Optional Split/Merge)

This folder contains a clean, standalone pipeline to cluster persona-style JSONL data.

## What this pipeline does

Given one input JSONL file (for example `summary_gpt51_learn.jsonl`), it will:

1. Read all rows and normalize the text field.
2. Build embeddings (default: OpenAI `text-embedding-3-large`).
3. Reduce embeddings to 2D with UMAP (for visualization/export).
4. Run KMeans clustering (default in full embedding space, not UMAP 2D).
5. Optionally refine clusters via iterative split/merge until overlap decreases.
6. Generate a **cluster title** and **cluster introduction** for each cluster (default: keyword backend).
7. Export a new JSONL file that keeps all original fields and appends cluster fields.

## Input requirements

- File format: JSONL (`.jsonl`)
- Required: a text column (default column name: `text`)
- Optional: `game_finished` column  
  If present, the script keeps only `game_finished == true` by default.

## Main output

The primary output is the file you specify with `--output-jsonl`.

Each output row preserves all original fields and adds:

- `umap_x`
- `umap_y`
- `cluster_id`
- `cluster_title`
- `cluster_intro`
- `cluster_size`

## Additional output files

The script also writes:

- `cluster_catalog.json`: one record per cluster (title, intro, size, terms, examples)
- `run_metadata.json`: run configuration and output paths
- `umap_points.csv`: text + UMAP + cluster columns for quick inspection
- `embeddings_<model>_<dim>.npy`: embedding cache (when using OpenAI embeddings)
- `split_summary.json` (from `run_tag_section_clustering.py`): per-tag status and paths
- `manifest.json` (from `run_tag_section_clustering.py`): global run summary across tags
- `cluster_catalog_polish_report.json` (when polishing): overlap diagnostics and unresolved conflict pairs

## Quick start

Run from the repository root:

```bash
python3 cluster/cluster_pipeline.py \
  --input-jsonl Persona/summary_gpt51_learn.jsonl \
  --output-jsonl cluster/output/summary_gpt51_learn_clustered.jsonl \
  --cluster-space embedding \
  --summary-backend keywords
```

## One-command tag-section run (learn or val)

This wrapper reads persona JSONL with inline section tags (e.g., `<CONTRIBUTION> ...`),
splits per tag, and runs clustering for all tags.

Learn (from `Persona/summary_gpt51_learn.jsonl`):

```bash
python3 cluster/run_tag_section_clustering.py \
  --input-jsonl Persona/summary_gpt51_learn.jsonl \
  --output-root Persona/misc/tag_section_clusters_openai_learn \
  --cluster-space embedding \
  --summary-backend keywords
```

Validation (from `Persona/summary_gpt51_val.jsonl`):

```bash
python3 cluster/run_tag_section_clustering.py \
  --input-jsonl Persona/summary_gpt51_val.jsonl \
  --output-root Persona/misc/tag_section_clusters_openai_val \
  --cluster-space embedding \
  --summary-backend keywords
```

Validation with integrated LLM polishing + overlap guard:

```bash
python3 cluster/run_tag_section_clustering.py \
  --input-jsonl Persona/summary_gpt51_val.jsonl \
  --output-root Persona/misc/tag_section_clusters_openai_val \
  --cluster-space embedding \
  --summary-backend keywords \
  --polish-with-llm \
  --polish-max-passes 3 \
  --polish-strict-overlap-check
```

## OpenAI requirements

Set your API key first:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Install dependencies if needed:

```bash
python3 -m pip install openai numpy pandas scikit-learn umap-learn
```

## Useful options

- `--text-column text_column_name`: choose the text field to embed.
- `--include-unfinished`: keep rows where `game_finished` is false.
- `--cluster-space embedding|umap`: default `embedding`.
- `--clusters 15`: fixed number of clusters (default: 15).
- `--auto-k --k-min 6 --k-max 25`: choose k automatically by silhouette score.
- Split/merge controls:
  - `--disable-split-merge`
  - `--max-split-merge-iters 20`
  - `--min-clusters 6 --max-clusters 25`
  - `--merge-similarity-threshold 0.94`
  - `--target-overlap-rate 0.12`
  - `--point-overlap-margin 0.03`
  - `--split-cluster-min-size 60`
- `--embedding-backend openai|tfidf`: default is `openai`.
- `--summary-backend openai|keywords`: default is `keywords`.
- `--summary-model gpt-4o-mini`: model for cluster title/introduction.
- `--embedding-cache-path /path/to/cache.npy`: custom embedding cache location.

## Optional: polish labels with LLM after clustering

```bash
python3 cluster/polish_cluster_catalog.py \
  --cluster-catalog Persona/misc/tag_section_clusters_openai/CONTRIBUTION/cluster_catalog.json \
  --clustered-jsonl Persona/misc/tag_section_clusters_openai/CONTRIBUTION/contribution_clustered.jsonl \
  --output-catalog Persona/misc/tag_section_clusters_openai/CONTRIBUTION/cluster_catalog_polished.json \
  --write-clustered-jsonl \
  --output-clustered-jsonl Persona/misc/tag_section_clusters_openai/CONTRIBUTION/contribution_clustered_polished.jsonl
```

Useful overlap-guard options for polish:
- `--max-passes 3`
- `--max-title-similarity 0.88`
- `--max-intro-similarity 0.92`
- `--max-combined-similarity 0.84`
- `--strict-overlap-check`

## Notes

- Default configuration is aligned with overlap reduction: embeddings + KMeans + split/merge + keyword labels.
- Cluster IDs are remapped to stable ordering by centroid coordinates for cleaner outputs.
- Cluster summaries are generated per cluster; LLM polishing can be integrated in wrapper (`--polish-with-llm`) or run separately.
- Embeddings are cached locally by default under each tag folder as `embeddings_<model>_<dim>.npy`.
