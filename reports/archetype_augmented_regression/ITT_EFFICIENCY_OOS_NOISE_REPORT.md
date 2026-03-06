# ITT Efficiency: OOS R2 + Noise Ceiling Update

This note summarizes the refreshed single-split run with:

- target: `itt_efficiency`
- style source: `both` (`oracle` and `synthetic`)
- granularity: `both` (`game` and `config_treatment`)
- model focus: ridge

Run artifacts:

- `reports/archetype_augmented_regression/results_itt_efficiency_both_granularity.csv`
- `reports/archetype_augmented_regression/summary_itt_efficiency_both_granularity.json`

## OOS Metric Definition

- Standard R2 (`r2`) uses test-mean denominator.
- OOS R2 (`r2_oos_train_mean`) uses training-mean denominator:
  - `1 - sum((y - y_hat)^2) / sum((y - mean(y_train))^2)`

## Ridge Results (Best k per mode)

### Game granularity

- Baseline (CONFIG only): `r2=0.0305`, `r2_oos_train_mean=0.0308`
- Oracle best (`k=6`): `r2=0.1920` (`+0.1614`), `r2_oos_train_mean=0.1922` (`+0.1614`)
- Synthetic best (`k=16`): `r2=0.0483` (`+0.0178`), `r2_oos_train_mean=0.0487` (`+0.0178`)

### CONFIG_treatment granularity

- Baseline (CONFIG only): `r2=0.1837`, `r2_oos_train_mean=0.2000`
- Oracle best (`k=6`): `r2=0.2069` (`+0.0231`), `r2_oos_train_mean=0.2226` (`+0.0227`)
- Synthetic best (`k=8`): `r2=0.2742` (`+0.0905`), `r2_oos_train_mean=0.2886` (`+0.0887`)

## Noise Ceiling (from `summary_itt_efficiency_both_granularity.json`)

### Game

- `unseen_test_share=1.0`
- `oracle_test_config_mean`: `r2=0.2498`, `r2_oos_train_mean=0.2500`
- `train_to_test_config_mean`: `r2=-0.00034`, `r2_oos_train_mean=0.0`

### CONFIG_treatment

- `unseen_test_share=1.0`
- `oracle_test_config_mean`: `r2=1.0`, `r2_oos_train_mean=1.0`
- `train_to_test_config_mean`: `r2=-0.0203`, `r2_oos_train_mean=0.0`

Interpretation: the CONFIG overlap between learn and validation is effectively zero (`unseen_test_share=1.0`), so train-config mean mapping has no predictive leverage and falls back to train mean.
