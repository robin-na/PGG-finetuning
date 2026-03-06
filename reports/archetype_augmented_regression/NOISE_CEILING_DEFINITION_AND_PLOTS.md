# Noise Ceiling Definition and Plot Outputs

Definition aligned to `noise_ceiling (1).pdf`:

- For each CONFIG-treatment group `x` in validation:
  - outcomes: `y_{x,1..n_x}`
  - sample variance: `s_x^2`
  - mean-estimation noise term: `s_x^2 / n_x`
- Sampling-noise floor:
  - `MSE_floor = mean_x(s_x^2 / n_x)`
  - `RMSE_floor = sqrt(MSE_floor)`
- Implied R2 ceilings at CONFIG-treatment level:
  - test-mean denominator: `1 - MSE_floor / mean((ybar_x - mean(ybar))^2)`
  - train-mean denominator: `1 - MSE_floor / mean((ybar_x - mean_train)^2)`

Where this is computed:

- `archetype_augmented_regression/noise_ceiling.py`
  - `compute_sampling_noise_ceiling(...)`

Where it is stored:

- Single run summary:
  - `reports/archetype_augmented_regression/summary_itt_efficiency_both_granularity.json`
  - key: `noise_ceiling_sampling_config_treatment`
- OOD run:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/run_full_oos_noise_itt_efficiency/noise_ceiling_sampling_config_treatment_by_split.csv`
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/run_full_oos_noise_itt_efficiency/noise_ceiling_sampling_config_treatment_summary.csv`

Plot outputs:

- Non-OOD single split:
  - `reports/archetype_augmented_regression/figures/non_ood_single/ridge/ridge_non_ood_config_treatment_method_comparison.png`
- OOD CONFIG-treatment:
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_mean_with_ci.png`
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_split_heatmaps.png`
  - `reports/archetype_augmented_regression/figures/run_full_oos_noise_itt_efficiency/ridge/config_treatment/ridge_ood_config_treatment_gap_to_ceiling.png`
