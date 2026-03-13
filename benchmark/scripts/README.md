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
    - `benchmark/data_ood_splits_wave_anchored/<factor>/<direction>` -> `benchmark_ood_wave_anchored/<factor>/<direction>`
- `build_union_archetype_summary_pool.py`
  - Creates merged learn+val archetype summary pool.
  - Default output: `outputs/benchmark/cache/archetype/archetype_oracle_gpt51_learn_val_union_finished.jsonl`
- `run_split_pipeline.py`
  - Runs micro eval, macro simulation eval, and/or archetype train/validate for one split root.
  - `archetype-synthetic` stage precomputes split-specific synthetic archetype summaries from the latest trained run.
  - Supports `--stages index` to only refresh split pointers without running new jobs.
  - Supports `--game-ids-csv <manifest.csv>` (with `--game-ids-column`) to enforce the same game set in micro and macro runs.
  - Reads data from `benchmark/data*`, writes generated outputs to `outputs/benchmark/runs/...`
  - Safety guard: rejects output roots under `benchmark/` so run artifacts cannot be written into dataset folders.
  - Archetype flags:
    - `--archetype-mode none`
    - `--archetype-mode random_summary` (default pool: `Persona/archetype_oracle_gpt51_learn.jsonl`)
    - `--archetype-mode random_summary --archetype-summary-pool <pool_jsonl>` (override)
    - `--archetype-mode matched_summary --archetype-summary-pool <oracle_or_retrieved_jsonl>`
    - `--archetype-mode config_bank_archetype --archetype-summary-pool Persona/archetype_oracle_gpt51_learn.jsonl`
  - Micro output layout:
    - `outputs/benchmark/runs/<split-relative-path>/micro_behavior_eval/<variant>/<run_id>/...`
    - variants: `no_archetype`, `random_archetype`, `oracle_archetype`, `retrieved_archetype`, `config_bank_archetype`
  - Macro output layout:
    - `outputs/benchmark/runs/<split-relative-path>/macro_simulation_eval/<variant>/<run_id>/...`
    - variants: `no_archetype`, `random_archetype`, `oracle_archetype`, `retrieved_archetype`, `config_bank_archetype`
  - Writes split run index:
    - `outputs/benchmark/runs/<split-relative-path>/run_index.json`
    - includes latest archetype run, latest micro run per variant, latest macro run per variant, and retrieved summary pointers when present.
  - Synthetic archetype output (default):
    - `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/validation_wave/synthetic_archetype_ridge_val.jsonl`
- `analyze_micro_variant_runs.py`
  - Picks latest run under each named variant folder and runs `Micro_behavior_eval/analysis/run_analysis.py`.
  - Avoids "latest four overall" heuristics by selecting one latest run per explicit variant folder.
- `analyze_macro_variant_runs.py`
  - Picks latest run under each named variant folder under `macro_simulation_eval/`.
  - Acts as a benchmark helper and calls `Macro_simulation_eval/analysis/run_analysis.py`.
  - Writes outputs to `reports/benchmark/macro_simulation_eval/<analysis_run_id>/`.
- `precompute_synthetic_archetypes.py`
  - Precomputes synthetic archetype JSONL pools per split (used by retrieved archetype mode).
  - Can optionally auto-train retrieval models when `latest_run.txt` is missing.
  - Supports passthrough `--train-arg` / `--validate-arg` and a fast auto-train preset via `--fast-default-train` (`--models mean linear ridge --allow-tag-errors`).
- `build_paired_eval_manifest.py`
  - Builds strict-filtered paired manifests with one game for `CONFIG_punishmentExists=False` and one for `True` per `CONFIG_configId`.
  - Writes both a manifest CSV and a comma-separated game-id text file for direct `--game_ids` usage.
- `shard_game_manifest.py`
  - Splits a manifest CSV into stable shards for multi-GPU runs.
  - Useful when launching one benchmark job per GPU against disjoint game subsets.
- `sample_game_manifest.py`
  - Selects a stable subset of games from a manifest CSV.
  - Useful for quick calibration runs or autosweeps before launching a full benchmark.
- `merge_macro_shard_runs.py`
  - Merges multiple shard-local macro run directories into one combined macro run directory.
  - Useful after data-parallel benchmark runs where each GPU handled a disjoint manifest shard.
- `migrate_micro_runs_layout.py`
  - Migrates old flat micro run folders into variant-based layout.
  - Default destination: `outputs/default/runs/source_default/micro_behavior_eval/`
- `plot_benchmark_validation_performance.py`
  - Consolidated cross-split plots (retrieved vs oracle and regret).
  - By default discovers both `benchmark/data_ood_splits` and `benchmark/data_ood_splits_wave_anchored` (plus `benchmark/data` with `--include-default`).
  - Default outputs under `reports/benchmark/plots/embeddings/`
- `plot_benchmark_validation_detailed.py`
  - Split-by-split detailed plotting (random baseline + error bars) plus consolidated figure.
  - By default discovers both `benchmark/data_ood_splits` and `benchmark/data_ood_splits_wave_anchored` (plus `benchmark/data` with `--include-default`).
  - Default outputs under `reports/benchmark/plots/embeddings/`

## Typical Flow

1. Build datasets in `benchmark/` if needed.
2. Run split pipeline with `run_split_pipeline.py`.
3. Run micro comparison via `analyze_micro_variant_runs.py` (for micro behavior).
4. Run macro comparison via `analyze_macro_variant_runs.py` (for normalized-efficiency + directional CONFIG effects).
5. Build embedding reports with one of the plotting scripts.

Example micro comparison:

```bash
python benchmark/scripts/analyze_micro_variant_runs.py \
  --split-root benchmark/data
```

Example macro comparison:

```bash
python benchmark/scripts/analyze_macro_variant_runs.py \
  --split-root benchmark/data \
  --allow-missing
```
