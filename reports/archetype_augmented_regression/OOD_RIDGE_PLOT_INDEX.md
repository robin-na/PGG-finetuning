# OOD Ridge Plot Index

Run: `run_full_oos_noise_itt_efficiency`

Figure directory:

- `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/`

Figures:

- `ridge_ood_method_comparison_with_noise_ceiling.png`
  - Mean R2 and RMSE across splits for CONFIG-only, synthetic, oracle, and noise ceiling.
- `ridge_ood_gap_to_noise_ceiling.png`
  - Distribution of per-split gaps to noise ceiling (R2 and RMSE).
- `ridge_ood_split_delta_heatmaps.png`
  - Split-level heatmaps of oracle/synthetic improvement over CONFIG-only.

CONFIG-treatment-only figures:

- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_mean_with_ci.png`
- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_gap_to_ceiling.png`
- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_split_heatmaps.png`

CONFIG-treatment bootstrap (granular per split):

- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment_bootstrap/ridge_ood_config_treatment_split_method_errorbars.png`
- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment_bootstrap/ridge_ood_config_treatment_split_pairwise_delta_errorbars.png`
- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment_bootstrap/ridge_ood_config_treatment_split_bootstrap_method_ci.csv`
- `figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment_bootstrap/ridge_ood_config_treatment_split_bootstrap_pairwise_delta.csv`

Plot tables:

- `ridge_methods_with_ceiling_by_split.csv`
- `ridge_delta_vs_config_by_split.csv`
- `ridge_gap_to_noise_ceiling_by_split.csv`
