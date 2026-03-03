# Benchmark Reports

This directory stores analysis-facing outputs derived from benchmark runs.

## Structure

- `plots/embeddings/`
  - Embedding-retrieval analysis outputs:
    - Cross-split similarity/regret plots and summary tables.
    - Consolidated figure + index/tables for per-split detailed plots.
  - Produced by:
    - `benchmark/scripts/plot_benchmark_validation_performance.py`
    - `benchmark/scripts/plot_benchmark_validation_detailed.py`
- `micro_behavior/`
  - Micro-behavior evaluation analysis outputs (metrics tables + plots + manifests).
  - Produced by: `Micro_behavior_eval/analysis/run_analysis.py`

## Relationship to `outputs/benchmark`

- `outputs/benchmark` stores raw/generated run artifacts.
- `reports/benchmark` stores analysis/report products built from those runs.
