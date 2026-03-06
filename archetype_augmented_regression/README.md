# Archetype Augmented Regression

Code for predicting macro outcomes with CONFIG features augmented by archetype-cluster distributions.

## Main scripts

- `eval_style_augmented_regression.py`
  - Runs one learn->validation evaluation.
  - Supports:
    - style source: `oracle`, `synthetic`, `both`
    - granularity: `game`, `config_treatment`, `both`
  - Writes row-level OOS `r2_oos_train_mean` deltas and per-granularity noise ceiling blocks.
- `run_style_augmented_ood_wave_anchored.py`
  - Batch-runs all splits under `benchmark/data_ood_splits_wave_anchored/*/*`.
  - Writes timestamped run folders under:
    - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/`
- `plot_ood_results.py`
  - Builds method/noise-ceiling comparison plots from an OOD run root.
  - Exports figures and plot-ready CSVs under:
    - `reports/archetype_augmented_regression/figures/<run_id>/<model>/`
- `plot_config_treatment_non_ood.py`
  - Single-run (`learn -> val`) CONFIG-treatment plot with:
    - `R2` (test-mean), `R2` (train-mean), `RMSE`
    - sampling noise ceiling from `noise_ceiling_sampling_config_treatment`
- `plot_config_treatment_ood.py`
  - OOD split plot at CONFIG-treatment level with:
    - split means + 95% CI,
    - split heatmaps of deltas vs CONFIG baseline,
    - gap-to-ceiling distributions.
- `bootstrap_inference.py`
  - Paired bootstrap utilities for:
    - method-level CIs,
    - pairwise delta CIs and significance flags.
- `config_treatment_eval.py`
  - Shared evaluator to reconstruct CONFIG-treatment predictions for:
    - CONFIG-only baseline,
    - oracle-style augmented,
    - synthetic-style augmented.

## Module map

- `eval_pipeline.py`
  - End-to-end single-split pipeline orchestration.
  - CLI args + fit/eval loop + artifact writing.
- `ood_batch_pipeline.py`
  - Batch orchestration over OOD split folders.
  - Per-run aggregate tables + manifest writing.
- `style_features.py`
  - Text-cluster distribution construction.
  - Synthetic CONFIG->style mapper (OOF ridge).
- `modeling.py`
  - Shared preprocessing and metric helpers.
- `noise_ceiling.py`
  - CONFIG-level variability ceiling helpers.
  - Computes `oracle_test_config_mean` and `train_to_test_config_mean` metrics.
- `io_utils.py`
  - JSONL loading, target resolution, CONFIG coercion, grouped aggregation, JSON-safe serialization.

## Typical usage

Single split:

```bash
python archetype_augmented_regression/eval_style_augmented_regression.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```

OOD batch:

```bash
python archetype_augmented_regression/run_style_augmented_ood_wave_anchored.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```

Plot a completed OOD run (ridge):

```bash
python archetype_augmented_regression/plot_ood_results.py \
  --run-root outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id> \
  --model ridge
```

Single-run CONFIG-treatment plot (ridge):

```bash
python archetype_augmented_regression/plot_config_treatment_non_ood.py \
  --results-csv reports/archetype_augmented_regression/results_itt_efficiency_both_granularity.csv \
  --summary-json reports/archetype_augmented_regression/summary_itt_efficiency_both_granularity.json \
  --model ridge
```

OOD CONFIG-treatment plot (ridge):

```bash
python archetype_augmented_regression/plot_config_treatment_ood.py \
  --run-root outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id> \
  --model ridge
```
