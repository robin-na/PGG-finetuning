# Benchmark Scripts

These scripts run benchmark pipelines and analyses without writing into `benchmark/data*`.
They write only to `outputs/benchmark` and `reports/benchmark` (not `outputs/default` / `reports/default`).

## Scripts

- `build_split_archetype_banks.py`
  - Builds split-specific archetype banks from the global archetype banks under `Persona/archetype_retrieval/`.
  - Default output root: `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/`
  - Split path mapping:
    - `benchmark/data` -> `benchmark_filtered`
    - `benchmark/data_ood_splits/<factor>/<direction>` -> `benchmark_ood/<factor>/<direction>`
- `build_union_archetype_summary_pool.py`
  - Creates merged learn+val archetype summary pool.
  - Default output: `outputs/benchmark/cache/archetype/summary_gpt51_learn_val_union_finished.jsonl`
- `run_split_pipeline.py`
  - Runs micro eval and/or archetype train/validate for one split root.
  - Supports `--stages index` to only refresh split pointers without running new jobs.
  - Reads data from `benchmark/data*`, writes generated outputs to `outputs/benchmark/runs/...`
  - Archetype flags:
    - `--archetype-mode none`
    - `--archetype-mode random_summary --archetype-summary-pool <pool_jsonl>`
    - `--archetype-mode matched_summary --archetype-summary-pool <oracle_or_retrieved_jsonl>`
  - Micro output layout:
    - `outputs/benchmark/runs/<split-relative-path>/micro_behavior_eval/<variant>/<run_id>/...`
    - variants: `no_archetype`, `random_archetype`, `oracle_archetype`, `retrieved_archetype`
  - Writes split run index:
    - `outputs/benchmark/runs/<split-relative-path>/run_index.json`
    - includes latest archetype run, latest micro run per variant, and retrieved summary pointers when present.
- `analyze_micro_variant_runs.py`
  - Picks latest run under each named variant folder and runs `Micro_behavior_eval/analysis/run_analysis.py`.
  - Avoids "latest four overall" heuristics by selecting one latest run per explicit variant folder.
- `migrate_micro_runs_layout.py`
  - Migrates old flat micro run folders into variant-based layout.
  - Default destination: `outputs/default/runs/source_default/micro_behavior_eval/`
- `plot_benchmark_validation_performance.py`
  - Consolidated cross-split plots (retrieved vs oracle and regret).
  - Default outputs under `reports/benchmark/plots/embeddings/`
- `plot_benchmark_validation_detailed.py`
  - Split-by-split detailed plotting (random baseline + error bars) plus consolidated figure.
  - Default outputs under `reports/benchmark/plots/embeddings/`

## Typical Flow

1. Build datasets in `benchmark/` if needed.
2. Run split pipeline with `run_split_pipeline.py`.
3. Run micro comparison via `analyze_micro_variant_runs.py` (for micro behavior).
4. Build embedding reports with one of the plotting scripts.

Example micro comparison:

```bash
python benchmark/scripts/analyze_micro_variant_runs.py \
  --split-root benchmark/data
```
