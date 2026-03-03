# Benchmark Folder

This folder stores benchmark datasets and the scripts that define benchmark workflows.
Generated experiment artifacts are stored outside this folder.

## What Lives Here

- `data/`: filtered default benchmark dataset (condition-2 filter)
- `data_ood_splits/`: one-factor OOD split datasets
- `build_filtered_dataset.py`: builds `benchmark/data`
- `build_ood_splits.py`: builds `benchmark/data_ood_splits`
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

## Pipeline Scripts (`benchmark/scripts`)

- `build_split_archetype_banks.py`
- `build_union_archetype_summary_pool.py`
- `run_split_pipeline.py`
- `analyze_micro_variant_runs.py`
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
- `outputs/benchmark/runs/<split-relative-path>/archetype_retrieval/model_runs`
- `outputs/benchmark/runs/<split-relative-path>/run_index.json`
- `outputs/benchmark/cache/archetype/summary_gpt51_learn_val_union_finished.jsonl`

Micro eval wiring uses:
- `Micro_behavior_eval/run_micro_behavior_eval.py --data_root <split-root> --wave validation_wave`
- Archetype options:
  - `--archetype-mode none`
  - `--archetype-mode random_summary --archetype-summary-pool <pool_jsonl>`
  - `--archetype-mode matched_summary --archetype-summary-pool Persona/summary_gpt51_val.jsonl` (oracle)
  - `--archetype-mode matched_summary --archetype-summary-pool <retrieved_jsonl>` (retrieved)

`run_index.json` tracks the latest pointers for that split:
- latest archetype run (`archetype_retrieval.latest_run_dir`)
- latest micro run per variant (`micro_behavior_eval.variants.<variant>.latest_run_dir`)
- retrieved archetype summary JSONL/trace (when available)

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
