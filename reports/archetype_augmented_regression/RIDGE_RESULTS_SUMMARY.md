# Ridge Summary: CONFIG + Archetype Distribution Features

## Scope

This document summarizes ridge-only experiments for predicting `itt_relative_efficiency` on validation-wave data using:

- CONFIG-only features (baseline),
- CONFIG + oracle archetype distribution features,
- CONFIG + synthetic archetype distribution features.

The summary reflects runs completed on **March 5, 2026**.

## Objective

Evaluate whether archetype-distribution features improve out-of-sample prediction beyond CONFIG-only models, and compare:

- **Oracle style features**: observed per-game cluster shares from validation persona text (upper bound).
- **Synthetic style features**: per-game cluster shares predicted from CONFIG alone (deployable setting).

## Data

- Main split:
  - Train: `benchmark/data/processed_data/df_analysis_learn.csv`
  - Test: `benchmark/data/processed_data/df_analysis_val.csv`
- Persona text:
  - Learn: `Persona/archetype_oracle_gpt51_learn.jsonl`
  - Validation: `Persona/archetype_oracle_gpt51_val.jsonl`
- OOD sweep:
  - `benchmark/data_ood_splits_wave_anchored/*/*`
  - 18 splits total, each preserving learning-wave as train and validation-wave as test.

## Methodology

### 1. Feature set and target

- Target: `itt_relative_efficiency`
- Baseline predictors: overlapping `CONFIG_*` columns from learn/val analysis tables, excluding:
  - `CONFIG_configId`
  - `CONFIG_treatmentName`
- Preprocessing:
  - Numeric: median imputation + standardization
  - Categorical: most-frequent imputation + one-hot encoding

### 2. Archetype clustering

- Input text: persona `text` rows keyed by `experiment` (game ID).
- Representation: TF-IDF (`max_features=5000`, `min_df=2`, `ngram_range=(1,2)`).
- Clustering: KMeans (`random_state=42`, `n_init=20`) with `k in {4,6,8,10,12,16}`.
- Game-level style vector: normalized cluster share vector per game.

### 3. Style feature variants

- Oracle variant:
  - Use observed game-level cluster-share vectors for both learn and validation games.
- Synthetic variant:
  - Train ridge mapper: `CONFIG -> cluster-share vector` on learn games.
  - Generate **OOF predictions** (5-fold KFold) for learn games to avoid leakage into final target model fit.
  - Generate predictions for validation games using mapper fit on all learn games.
  - Clip negatives at zero and renormalize each predicted vector to sum to 1.

### 4. Final ridge predictor

- For each `k` and style source, train ridge (`alpha=1.0`, `random_state=42`) on:
  - `CONFIG` only (baseline), and
  - `CONFIG + style shares` (augmented).

### 5. Evaluation granularities

- `game`:
  - One row per game.
- `config_treatment`:
  - Aggregate by `CONFIG_treatmentName` before fitting/evaluation.
  - Numeric columns averaged; categorical columns use first value.
  - Target averaged within group.

### 6. Metrics

- `R2`, `RMSE`, `MAE` on validation side.
- Improvement is measured against CONFIG-only baseline:
  - `delta_r2 = augmented_r2 - baseline_r2` (higher is better),
  - `delta_rmse = augmented_rmse - baseline_rmse` (lower is better),
  - `delta_mae = augmented_mae - baseline_mae` (lower is better).
- `R2` denominator is validation-side SST:
  \[
  \sum_i (y_i - \bar y_{val})^2
  \]

### 7. Model selection note

For each style source / model / granularity / split, the reported “best” row picks `k` by:

1. minimum `delta_rmse`,
2. tie-break by minimum `delta_mae`.

This is convenient for exploration but optimistic if interpreted as fixed, preregistered model selection.

## Results

## A. Main split (`benchmark/data`) — Ridge only

### Game granularity

- Baseline: `R2=0.0133`, `RMSE=0.2603`, `MAE=0.1931`
- Oracle best (`k=8`):
  - `R2=0.1058`, `RMSE=0.2478`, `MAE=0.1781`
  - `delta_r2=+0.0925`, `delta_rmse=-0.0125`, `delta_mae=-0.0150`
- Synthetic best (`k=10`):
  - `R2=0.0377`, `RMSE=0.2571`, `MAE=0.1937`
  - `delta_r2=+0.0244`, `delta_rmse=-0.0032`, `delta_mae=+0.0006`

### `CONFIG_treatmentName` granularity

- Baseline: `R2=-0.0318`, `RMSE=0.1294`, `MAE=0.0968`
- Oracle best (`k=16`):
  - `R2=0.0841`, `RMSE=0.1219`, `MAE=0.0943`
  - `delta_r2=+0.1158`, `delta_rmse=-0.0075`, `delta_mae=-0.0025`
- Synthetic best (`k=16`):
  - `R2=0.1358`, `RMSE=0.1184`, `MAE=0.0895`
  - `delta_r2=+0.1676`, `delta_rmse=-0.0110`, `delta_mae=-0.0073`

Interpretation: on this split, synthetic is weaker than oracle at game-level, but stronger at `CONFIG_treatmentName`-averaged level.

## B. Wave-anchored OOD sweep (18 splits) — Ridge only

Win-rate definition below is **vs CONFIG-only baseline**, not synthetic-vs-oracle.

### Oracle augmentation

- `game`:
  - `R2+RMSE` win rate: `16/18 = 88.9%`
  - `all3` (R2+RMSE+MAE) win rate: `15/18 = 83.3%`
  - Mean deltas: `delta_r2=+0.1419`, `delta_rmse=-0.0171`, `delta_mae=-0.0175`
- `config_treatment`:
  - `R2+RMSE` win rate: `13/18 = 72.2%`
  - `all3` win rate: `12/18 = 66.7%`
  - Mean deltas: `delta_r2=+0.2627`, `delta_rmse=-0.0136`, `delta_mae=-0.0115`

### Synthetic augmentation

- `game`:
  - `R2+RMSE` win rate: `14/18 = 77.8%`
  - `all3` win rate: `10/18 = 55.6%`
  - Mean deltas: `delta_r2=+0.0009`, `delta_rmse=-0.0006`, `delta_mae=+0.0003`
- `config_treatment`:
  - `R2+RMSE` win rate: `13/18 = 72.2%`
  - `all3` win rate: `12/18 = 66.7%`
  - Mean deltas: `delta_r2=+0.0538`, `delta_rmse=-0.0034`, `delta_mae=-0.0030`

Interpretation:

- Synthetic is frequently helpful vs CONFIG-only.
- Oracle is generally stronger, especially at game-level.
- Synthetic’s gains are more competitive at config-averaged granularity.

### Synthetic ridge at game granularity: OOD losses (4 splits)

- `num_rounds/high_to_low`
- `num_rounds/low_to_high`
- `reward_exists/false_to_true`
- `show_n_rounds/false_to_true`

## Output artifacts

- Main split summary:
  - `reports/archetype_augmented_regression/summary_itt_relative_efficiency_both_granularity.json`
  - `reports/archetype_augmented_regression/results_itt_relative_efficiency_both_granularity.csv`
- OOD aggregated outputs:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/all_splits_best_rows.csv`
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/comparison_synthetic_vs_oracle.csv`
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/comparison_counts.csv`
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/config_only_improvement_counts.csv`
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/config_only_improvement_split_level.csv`

## Repro command

Run both style sources and both granularities on one split:

```bash
python archetype_augmented_regression/eval_style_augmented_regression.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```

Run on one OOD split by overriding analysis CSVs:

```bash
python archetype_augmented_regression/eval_style_augmented_regression.py \
  --learn-analysis-csv benchmark/data_ood_splits_wave_anchored/mpcr/high_to_low/processed_data/df_analysis_learn.csv \
  --val-analysis-csv benchmark/data_ood_splits_wave_anchored/mpcr/high_to_low/processed_data/df_analysis_val.csv \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```
