# Benchmark Folder

This folder stores benchmark datasets and the scripts that define benchmark workflows.
Generated experiment artifacts are stored outside this folder.

## What Lives Here

- `data/`: filtered default benchmark dataset (condition-2 filter)
- `data_ood_splits/`: one-factor OOD split datasets
- `data_ood_splits_wave_anchored/`: one-factor OOD splits with original wave roles preserved
- `build_filtered_dataset.py`: builds `benchmark/data`
- `build_ood_splits.py`: builds `benchmark/data_ood_splits`
- `build_ood_splits_wave_anchored.py`: builds `benchmark/data_ood_splits_wave_anchored`
- `scripts/`: benchmark pipeline + plotting scripts

## Output Policy

- Raw/generated experiment outputs: `outputs/benchmark/`
- Plots/tables/reports: `reports/benchmark/`
- Original non-benchmark (`data/`) outputs should go to:
  - `outputs/default/`
  - `reports/default/`

Keep `benchmark/data*` immutable once built.

## Dataset Build Commands

Build filtered base dataset:
```bash
python benchmark/build_filtered_dataset.py
```

Build one-factor OOD splits:
```bash
python benchmark/build_ood_splits.py
```

Build wave-anchored one-factor OOD splits:
```bash
python benchmark/build_ood_splits_wave_anchored.py
```

## Pipeline Scripts (`benchmark/scripts`)

- `build_split_archetype_banks.py`
- `build_union_archetype_summary_pool.py`
- `run_split_pipeline.py`
- `precompute_synthetic_archetypes.py`
- `build_paired_eval_manifest.py`
- `analyze_micro_variant_runs.py`
- `analyze_macro_variant_runs.py`
- `migrate_micro_runs_layout.py`
- `plot_benchmark_validation_performance.py`
- `plot_benchmark_validation_detailed.py`

### Split archetype banks

```bash
python benchmark/scripts/build_split_archetype_banks.py \
  --split-root benchmark/data_ood_splits/all_or_nothing/false_to_true
```

Default output:
- `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/{learning_wave,validation_wave}`
where `<split-relative-path>` maps as:
- `benchmark/data` -> `benchmark_filtered`
- `benchmark/data_ood_splits/<factor>/<direction>` -> `benchmark_ood/<factor>/<direction>`
- `benchmark/data_ood_splits_wave_anchored/<factor>/<direction>` -> `benchmark_ood_wave_anchored/<factor>/<direction>`

### Run split pipeline

```bash
python benchmark/scripts/run_split_pipeline.py \
  --split-root benchmark/data_ood_splits/all_or_nothing/false_to_true \
  --stages all
```

Index-only refresh (no model/simulation run):

```bash
python benchmark/scripts/run_split_pipeline.py \
  --split-root benchmark/data_ood_splits/all_or_nothing/false_to_true \
  --stages index
```

Default outputs:
- `outputs/benchmark/runs/<split-relative-path>/micro_behavior_eval/<variant>/<run_id>`
- `outputs/benchmark/runs/<split-relative-path>/macro_simulation_eval/<variant>/<run_id>`
- `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/model_runs`
- `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/validation_wave/synthetic_archetype_<model>_val.jsonl`
- `outputs/benchmark/runs/<split-relative-path>/run_index.json`
- `outputs/benchmark/cache/archetype/archetype_oracle_gpt51_learn_val_union_finished.jsonl` (only when explicitly requested)

Micro eval wiring uses:
- `Micro_behavior_eval/run_micro_behavior_eval.py --data_root <split-root> --wave validation_wave`
Macro simulation wiring uses:
- `Macro_simulation_eval/run_macro_simulation_eval.py --data_root <split-root> --wave validation_wave`
- Archetype options:
  - `--archetype-mode none`
  - `--archetype-mode random_summary` (default pool: `Persona/archetype_oracle_gpt51_learn.jsonl`)
  - `--archetype-mode random_summary --archetype-summary-pool <pool_jsonl>` (override)
  - `--archetype-mode matched_summary --archetype-summary-pool Persona/archetype_oracle_gpt51_val.jsonl` (oracle)
  - `--archetype-mode matched_summary --archetype-summary-pool <retrieved_jsonl>` (retrieved)

`run_index.json` tracks the latest pointers for that split:
- latest archetype run (`archetype_retrieval.latest_run_dir`)
- latest micro run per variant (`micro_behavior_eval.variants.<variant>.latest_run_dir`)
- latest macro run per variant (`macro_simulation_eval.variants.<variant>.latest_run_dir`)
- retrieved archetype summary JSONL/trace (when available)

Precompute synthetic archetypes (for retrieved archetype mode):
```bash
python benchmark/scripts/run_split_pipeline.py \
  --split-root benchmark/data_ood_splits/all_or_nothing/false_to_true \
  --stages archetype-synthetic,index \
  --synthetic-model ridge
```

Batch-precompute synthetic archetypes across splits:
```bash
python benchmark/scripts/precompute_synthetic_archetypes.py \
  --include-default
```

Batch-precompute for wave-anchored OOD splits only (auto-train missing runs, faster model set):
```bash
python benchmark/scripts/precompute_synthetic_archetypes.py \
  --ood-root benchmark/data_ood_splits_wave_anchored \
  --auto-train-if-missing \
  --fast-default-train
```

Build one-pair-per-config manifest (strict filtered) for matched micro/macro eval:
```bash
python benchmark/scripts/build_paired_eval_manifest.py \
  --data-root benchmark/data \
  --wave validation_wave \
  --seed 0
```
This writes:
- manifest CSV (`gameId`, `CONFIG_configId`, `CONFIG_punishmentExists`)
- comma-separated game ID list for direct `--game_ids` use

Use that manifest in pipeline runs:
```bash
python benchmark/scripts/run_split_pipeline.py \
  --split-root benchmark/data_ood_splits/player_count/high_to_low \
  --stages micro,macro \
  --game-ids-csv reports/benchmark/manifests/<manifest>.csv \
  --game-ids-column gameId
```

### Micro Comparison (Named Variants)

```bash
python benchmark/scripts/analyze_micro_variant_runs.py \
  --split-root benchmark/data_ood_splits/all_or_nothing/false_to_true
```

This selects the latest run in:
- `no_archetype/`
- `random_archetype/`
- `oracle_archetype/`
- `retrieved_archetype/`

and runs `Micro_behavior_eval/analysis/run_analysis.py` automatically.

### Macro Comparison (Named Variants)

```bash
python benchmark/scripts/analyze_macro_variant_runs.py \
  --split-root benchmark/data \
  --allow-missing
```

Outputs include:
- game-level simulated vs human normalized efficiency
- aggregate MAE/RMSE/correlation for normalized efficiency
- directional CONFIG-effect comparison (human vs simulated sign agreement)

Note:
- `Macro_simulation_eval/analysis/` is the primary macro analysis package.
- `Analysis_robin/` is retained as reference code.
- benchmark workflows should use `benchmark/scripts/analyze_micro_variant_runs.py` and `benchmark/scripts/analyze_macro_variant_runs.py` as the primary entry points.

### Migrate Old Micro Run Layout

```bash
python benchmark/scripts/migrate_micro_runs_layout.py \
  --source-root Micro_behavior_eval/output \
  --destination-root outputs/default/runs/source_default/micro_behavior_eval
```

### Detailed validation plots (random baseline + error bars)

```bash
python benchmark/scripts/plot_benchmark_validation_detailed.py \
  --include-default \
  --run-per-split \
  --top-k 5 \
  --weight-by-rows
```

Default report outputs:
- `reports/benchmark/plots/embeddings/per_split_plot_index.csv`
- `reports/benchmark/plots/embeddings/consolidated_split_model_metrics_with_random.csv`
- `reports/benchmark/plots/embeddings/across_splits_regret_and_hit_with_random.png`

The per-split figure files are written under each split run's `validation_eval/figures`.
