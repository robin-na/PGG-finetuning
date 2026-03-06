# Archetype-Augmented Regression: Runbook

This folder stores outputs for CONFIG + archetype-distribution regression experiments.

## Canonical Code

- Single split evaluator:
  - `archetype_augmented_regression/eval_style_augmented_regression.py`
- Wave-anchored OOD batch runner:
  - `archetype_augmented_regression/run_style_augmented_ood_wave_anchored.py`

## Recommended Organization

- Keep all new OOD runs under:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/`
- Keep one pointer file:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/latest_run.txt`
- Keep narrative docs in this root:
  - `RIDGE_RESULTS_SUMMARY.md`
  - `ITT_EFFICIENCY_OOS_NOISE_REPORT.md`
  - `OOD_WAVE_ANCHORED_ITT_EFFICIENCY_OOS_NOISE.md`
  - `OOD_RIDGE_PLOT_INDEX.md`
  - `NOISE_CEILING_DEFINITION_AND_PLOTS.md`
  - `BOOTSTRAP_INFERENCE_NOTATION.md`
- Legacy snapshot (pre-run-structured outputs):
  - `reports/archetype_augmented_regression/legacy/ood_wave_anchored_snapshot/`

## OOD Run Output Layout

For each run (`runs/<run_id>/`):

- Per split:
  - `<factor>/<direction>/results.csv`
  - `<factor>/<direction>/summary.json`
- Aggregates:
  - `all_splits_best_rows.csv`
  - `comparison_synthetic_vs_oracle.csv`
  - `comparison_counts.csv`
  - `config_only_improvement_split_level.csv`
  - `config_only_improvement_counts.csv`
  - `noise_ceiling_by_split.csv`
  - `noise_ceiling_summary.csv`
  - `run_manifest.json`

## Figures

- OOD run figures are written to:
  - `reports/archetype_augmented_regression/figures/<run_id>/<model>/`
- For `run_full_oos_noise_itt_efficiency` ridge:
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/`
- CONFIG-treatment OOD figures:
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/`
- CONFIG-treatment bootstrap figures/tables:
  - `reports/archetype_augmented_regression/figures/non_ood_single/ridge/`
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment_bootstrap/`
- Non-OOD single-run figures:
  - `reports/archetype_augmented_regression/figures/non_ood_single/ridge/`

## Naming Conventions

- `results_*.csv`: row-level model outputs from a single evaluator run.
- `summary_*.json`: compact best-row summary from a single evaluator run.
- `comparison_*`: synthetic-vs-oracle comparisons.
- `config_only_improvement_*`: augmentation-vs-baseline comparisons.
- `noise_ceiling_*`: CONFIG-level variability ceilings by split and aggregated over splits.
  - Example single-split export: `noise_ceiling_itt_efficiency_both_granularity.csv`.

## Metric Notes

- `r2`: standard test-set R² with test-mean denominator.
- `r2_oos_train_mean`: OOS R² with training-mean denominator.
- Noise ceiling block (in each `summary.json`):
  - `oracle_test_config_mean`: upper bound from test CONFIG-group means.
  - `train_to_test_config_mean`: train CONFIG means mapped to test (fallback to train mean for unseen CONFIGs).

## Repro Commands

Single split:

```bash
python archetype_augmented_regression/eval_style_augmented_regression.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```

Batch OOD (all wave-anchored splits):

```bash
python archetype_augmented_regression/run_style_augmented_ood_wave_anchored.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both
```

Smoke test (first 2 splits only):

```bash
python archetype_augmented_regression/run_style_augmented_ood_wave_anchored.py \
  --target itt_relative_efficiency \
  --style-source both \
  --eval-granularity both \
  --max-splits 2
```

## Suggested Workflow

1. Run one full OOD batch.
2. Review `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/run_manifest.json` for failures.
3. Use `config_only_improvement_counts.csv` for baseline-improvement win rates.
4. Use `comparison_counts.csv` only for synthetic-vs-oracle contrast.
5. Update `RIDGE_RESULTS_SUMMARY.md` with key new findings.
