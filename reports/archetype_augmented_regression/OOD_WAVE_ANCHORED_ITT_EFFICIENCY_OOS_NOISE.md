# OOD Wave-Anchored (ITT Efficiency): OOS R2 + Noise Ceiling

Full run id: `run_full_oos_noise_itt_efficiency`

Run root:

- `outputs/archetype_augmented_regression/ood_wave_anchored/runs/run_full_oos_noise_itt_efficiency/`

## Main output tables

- `all_splits_best_rows.csv`
- `config_only_improvement_counts.csv`
- `config_only_improvement_split_level.csv`
- `comparison_counts.csv`
- `comparison_synthetic_vs_oracle.csv`
- `noise_ceiling_by_split.csv`
- `noise_ceiling_summary.csv`
- `run_manifest.json`

## Ridge: CONFIG+style vs CONFIG-only (18 splits)

From `config_only_improvement_counts.csv`:

- Oracle, game: `r2_win_rate=0.944`, `r2_oos_train_mean_win_rate=0.944`, `mean_delta_r2=+0.167`, `mean_delta_r2_oos_train_mean=+0.165`, `mean_delta_rmse=-0.0116`
- Oracle, config_treatment: `r2_win_rate=0.889`, `r2_oos_train_mean_win_rate=0.889`, `mean_delta_r2=+0.160`, `mean_delta_r2_oos_train_mean=+0.154`, `mean_delta_rmse=-0.0067`
- Synthetic, game: `r2_win_rate=0.722`, `r2_oos_train_mean_win_rate=0.722`, `mean_delta_r2=+0.0276`, `mean_delta_r2_oos_train_mean=+0.0273`, `mean_delta_rmse=-0.0017`
- Synthetic, config_treatment: `r2_win_rate=0.667`, `r2_oos_train_mean_win_rate=0.667`, `mean_delta_r2=+0.0662`, `mean_delta_r2_oos_train_mean=+0.0615`, `mean_delta_rmse=-0.0026`

## Noise ceiling summary (18 splits)

From `noise_ceiling_summary.csv`:

- Game:
  - `mean_unseen_test_share=1.0`
  - `mean_oracle_r2=0.2565`
  - `mean_oracle_r2_oos_train_mean=0.2712`
  - `mean_train_to_test_r2=-0.0212`
  - `mean_train_to_test_r2_oos_train_mean=0.0`
- Config_treatment:
  - `mean_unseen_test_share=1.0`
  - `mean_oracle_r2=1.0`
  - `mean_oracle_r2_oos_train_mean=1.0`
  - `mean_train_to_test_r2=-0.1242`
  - `mean_train_to_test_r2_oos_train_mean=0.0`

Interpretation: across these OOD splits, effective CONFIG overlap is absent (`mean_unseen_test_share=1.0`), so train-config mean transfer collapses to the train-mean baseline for OOS R2.
