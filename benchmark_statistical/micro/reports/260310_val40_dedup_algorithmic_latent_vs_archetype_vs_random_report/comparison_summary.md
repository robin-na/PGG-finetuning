# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| algorithmic_latent_family | 260310_val40_dedup_algorithmic_latent_family | 4666/4666 |
| archetype_cluster | 260310_val40_dedup_archetype_cluster | 4609/4609 |
| random_baseline | 260309_val40_dedup_random_baseline | 4609/4609 |

## Plot Guide

- `compare_contrib_mae.png`: Mean contribution MAE across runs with 95% CI (`↓ better`).
- `compare_target_f1.png`: Mean typed-target F1 across runs with 95% CI (`↑ better`).
- `compare_action_exact_match.png`: Mean exact action-match rate across runs with 95% CI (`↑ better`).
- `compare_target_hit_any.png`: Mean target-hit-any rate across runs with 95% CI (`↑ better`).
- If enabled, `prev-round baseline` is shown as a dotted horizontal reference line (with faint 95% CI band) in bar plots.
- `compare_by_round_contrib_mae.png`: Round-wise contribution MAE with 95% CI bands (`↓ better`).
- `compare_by_round_target_f1.png`: Round-wise target F1 with 95% CI bands (`↑ better`).
- `compare_contrib_mae_by_all_or_nothing.png`: Contribution MAE split by `CONFIG_allOrNothing` (`↓ better`).
- `compare_contrib_binary_by_all_or_nothing_true.png`: Binary contribution metrics for all-or-nothing games (`↑ better`).

## Standard Error and CI

- For each metric within a run: `SE = s / sqrt(n)`.
- `s` is sample standard deviation of row-level metric values; `n` is row count.
- 95% CI is `mean ± 1.96 * SE`.

## Overall Means

| Run | contrib_mae | target_f1 | action_exact_match | target_hit_any |
|---|---:|---:|---:|---:|
| algorithmic_latent_family | 6.2285 | 0.4280 | 0.3832 | 0.4788 |
| archetype_cluster | 7.8772 | 0.6958 | 0.6897 | 0.7019 |
| random_baseline | 9.3628 | 0.7341 | 0.7325 | 0.7360 |

## Contribution By Game Regime

| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |
|---|---|---:|---:|---:|---:|---:|
| continuous | algorithmic_latent_family | 2544 | 6.5417 | 0.3271 | NaN | NaN |
| continuous | archetype_cluster | 2544 | 7.4002 | 0.3700 | NaN | NaN |
| continuous | random_baseline | 2544 | 8.8494 | 0.4425 | NaN | NaN |
| all-or-nothing | algorithmic_latent_family | 2122 | 5.8530 | 0.2926 | 0.7074 | 0.7982 |
| all-or-nothing | archetype_cluster | 2065 | 8.4649 | 0.4232 | 0.5768 | 0.7063 |
| all-or-nothing | random_baseline | 2065 | 9.9952 | 0.4998 | 0.5002 | 0.6161 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | algorithmic_latent_family vs archetype_cluster | 6.2285 | 7.8772 | -1.6487 | 0.0000 | yes |
| contrib_mae (↓ better) | algorithmic_latent_family vs random_baseline | 6.2285 | 9.3628 | -3.1343 | 0.0000 | yes |
| contrib_mae (↓ better) | archetype_cluster vs random_baseline | 7.8772 | 9.3628 | -1.4856 | 0.0000 | yes |
| target_f1 (↑ better) | algorithmic_latent_family vs archetype_cluster | 0.4280 | 0.6958 | -0.2678 | 0.0000 | yes |
| target_f1 (↑ better) | algorithmic_latent_family vs random_baseline | 0.4280 | 0.7341 | -0.3060 | 0.0000 | yes |
| target_f1 (↑ better) | archetype_cluster vs random_baseline | 0.6958 | 0.7341 | -0.0382 | 0.0000 | yes |
| action_exact_match (↑ better) | algorithmic_latent_family vs archetype_cluster | 0.3832 | 0.6897 | -0.3065 | 0.0000 | yes |
| action_exact_match (↑ better) | algorithmic_latent_family vs random_baseline | 0.3832 | 0.7325 | -0.3493 | 0.0000 | yes |
| action_exact_match (↑ better) | archetype_cluster vs random_baseline | 0.6897 | 0.7325 | -0.0427 | 0.0000 | yes |
| target_hit_any (↑ better) | algorithmic_latent_family vs archetype_cluster | 0.4788 | 0.7019 | -0.2231 | 0.0000 | yes |
| target_hit_any (↑ better) | algorithmic_latent_family vs random_baseline | 0.4788 | 0.7360 | -0.2572 | 0.0000 | yes |
| target_hit_any (↑ better) | archetype_cluster vs random_baseline | 0.7019 | 0.7360 | -0.0341 | 0.0003 | yes |

