# Cluster Pipeline (LLM Embedding + UMAP + KMeans)

This folder contains a clean, standalone pipeline to cluster persona-style JSONL data.

## What this pipeline does

Given one input JSONL file (for example `summary_gpt51_learn.jsonl`), it will:

1. Read all rows and normalize the text field.
2. Build embeddings (default: OpenAI `text-embedding-3-large`).
3. Reduce embeddings to 2D with UMAP.
4. Run KMeans clustering in the 2D space.
5. Generate a **cluster title** and **cluster introduction** for each cluster (default: OpenAI chat model).
6. Export a new JSONL file that keeps all original fields and appends cluster fields.

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

## Quick start

Run from the repository root:

```bash
python3 ./PGG-finetuning/Cluster/cluster_pipeline.py \
  --input-jsonl ./PGG-finetuning/Persona/summary_gpt51_learn.jsonl \
  --output-jsonl ,/PGG-finetuning/Cluster/output/summary_gpt51_learn_clustered.jsonl
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
- `--clusters 15`: fixed number of clusters (default: 15).
- `--auto-k --k-min 6 --k-max 25`: choose k automatically by silhouette score.
- `--embedding-backend openai|tfidf`: default is `openai`.
- `--summary-backend openai|keywords`: default is `openai`.
- `--summary-model gpt-4o-mini`: model for cluster title/introduction.
- `--embedding-cache-path /path/to/cache.npy`: custom embedding cache location.

## Example: local dry run without OpenAI

```bash
python3 ./PGG-finetuning/Cluster/cluster_pipeline.py \
  --input-jsonl ./PGG-finetuning/Persona/summary_gpt51_learn.jsonl \
  --output-jsonl ./PGG-finetuning/Cluster/output/dry_run_clustered.jsonl \
  --embedding-backend tfidf \
  --summary-backend keywords
```

## Notes

- Default configuration is aligned with your current workflow: OpenAI embeddings + UMAP + KMeans.
- Cluster IDs are remapped to stable ordering by centroid coordinates for cleaner outputs.
- Cluster summaries are generated per cluster (not one giant prompt) to avoid context-length failures.

