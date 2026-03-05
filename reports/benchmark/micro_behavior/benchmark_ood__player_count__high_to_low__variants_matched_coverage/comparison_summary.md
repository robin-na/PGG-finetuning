# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| no archetype | no_archetype/2603022352 | 668/668 |
| random archetype | random_archetype/9980072_r0_random | 668/668 |
| oracle archetype | oracle_archetype/9977694_r0_oracle | 668/668 |
| retrieved archetype | retrieved_archetype/9976921_r1_retrieved | 668/668 |

## Plot Guide

- `compare_contrib_mae.png`: Mean contribution MAE across runs with 95% CI (`↓ better`).
- `compare_target_f1.png`: Mean typed-target F1 across runs with 95% CI (`↑ better`).
- `compare_action_exact_match.png`: Mean exact action-match rate across runs with 95% CI (`↑ better`).
- `compare_target_hit_any.png`: Mean target-hit-any rate across runs with 95% CI (`↑ better`).
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
| no archetype | 4.6213 | 0.5008 | 0.4296 | 0.5299 |
| random archetype | 4.9731 | 0.5914 | 0.5254 | 0.6213 |
| oracle archetype | 4.4117 | 0.7553 | 0.6841 | 0.7799 |
| retrieved archetype | 4.7859 | 0.6571 | 0.5958 | 0.6751 |

## Contribution By Game Regime

| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |
|---|---|---:|---:|---:|---:|---:|
| continuous | no archetype | 436 | 4.6491 | 0.2325 | NaN | NaN |
| continuous | oracle archetype | 436 | 4.4656 | 0.2233 | NaN | NaN |
| continuous | random archetype | 436 | 5.4633 | 0.2732 | NaN | NaN |
| continuous | retrieved archetype | 436 | 5.3601 | 0.2680 | NaN | NaN |
| all-or-nothing | no archetype | 232 | 4.5690 | 0.2284 | 0.7716 | 0.8571 |
| all-or-nothing | oracle archetype | 232 | 4.3103 | 0.2155 | 0.7845 | 0.8596 |
| all-or-nothing | random archetype | 232 | 4.0517 | 0.2026 | 0.7974 | 0.8760 |
| all-or-nothing | retrieved archetype | 232 | 3.7069 | 0.1853 | 0.8147 | 0.8883 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | no archetype vs random archetype | 4.6213 | 4.9731 | -0.3518 | 0.4100 | no |
| contrib_mae (↓ better) | no archetype vs oracle archetype | 4.6213 | 4.4117 | 0.2096 | 0.6277 | no |
| contrib_mae (↓ better) | no archetype vs retrieved archetype | 4.6213 | 4.7859 | -0.1647 | 0.6623 | no |
| contrib_mae (↓ better) | random archetype vs oracle archetype | 4.9731 | 4.4117 | 0.5614 | 0.1776 | no |
| contrib_mae (↓ better) | random archetype vs retrieved archetype | 4.9731 | 4.7859 | 0.1871 | 0.6530 | no |
| contrib_mae (↓ better) | oracle archetype vs retrieved archetype | 4.4117 | 4.7859 | -0.3743 | 0.3813 | no |
| target_f1 (↑ better) | no archetype vs random archetype | 0.5008 | 0.5914 | -0.0906 | 0.0010 | yes |
| target_f1 (↑ better) | no archetype vs oracle archetype | 0.5008 | 0.7553 | -0.2545 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs retrieved archetype | 0.5008 | 0.6571 | -0.1563 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs oracle archetype | 0.5914 | 0.7553 | -0.1639 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs retrieved archetype | 0.5914 | 0.6571 | -0.0657 | 0.0148 | yes |
| target_f1 (↑ better) | oracle archetype vs retrieved archetype | 0.7553 | 0.6571 | 0.0983 | 0.0001 | yes |
| action_exact_match (↑ better) | no archetype vs random archetype | 0.4296 | 0.5254 | -0.0958 | 0.0009 | yes |
| action_exact_match (↑ better) | no archetype vs oracle archetype | 0.4296 | 0.6841 | -0.2545 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs retrieved archetype | 0.4296 | 0.5958 | -0.1662 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs oracle archetype | 0.5254 | 0.6841 | -0.1587 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs retrieved archetype | 0.5254 | 0.5958 | -0.0704 | 0.0142 | yes |
| action_exact_match (↑ better) | oracle archetype vs retrieved archetype | 0.6841 | 0.5958 | 0.0883 | 0.0012 | yes |
| target_hit_any (↑ better) | no archetype vs random archetype | 0.5299 | 0.6213 | -0.0913 | 0.0012 | yes |
| target_hit_any (↑ better) | no archetype vs oracle archetype | 0.5299 | 0.7799 | -0.2500 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs retrieved archetype | 0.5299 | 0.6751 | -0.1452 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs oracle archetype | 0.6213 | 0.7799 | -0.1587 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs retrieved archetype | 0.6213 | 0.6751 | -0.0539 | 0.0520 | no |
| target_hit_any (↑ better) | oracle archetype vs retrieved archetype | 0.7799 | 0.6751 | 0.1048 | 0.0000 | yes |

