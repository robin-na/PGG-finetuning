# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| exact_sequence_structured | 260310_val40_dedup_exact_sequence_structured_archetype | 4666/4666 |
| history_archetype | 260310_val40_dedup_history_archetype | 4609/4609 |
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
| exact_sequence_structured | 7.8333 | 0.8060 | 0.7990 | 0.8080 |
| history_archetype | 4.9594 | 0.6923 | 0.6646 | 0.7140 |
| archetype_cluster | 7.8772 | 0.6958 | 0.6897 | 0.7019 |
| random_baseline | 9.3628 | 0.7341 | 0.7325 | 0.7360 |

## Contribution By Game Regime

| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |
|---|---|---:|---:|---:|---:|---:|
| continuous | archetype_cluster | 2544 | 7.4002 | 0.3700 | NaN | NaN |
| continuous | exact_sequence_structured | 2544 | 11.0810 | 0.5540 | NaN | NaN |
| continuous | history_archetype | 2544 | 4.3388 | 0.2169 | NaN | NaN |
| continuous | random_baseline | 2544 | 8.8494 | 0.4425 | NaN | NaN |
| all-or-nothing | archetype_cluster | 2065 | 8.4649 | 0.4232 | 0.5768 | 0.7063 |
| all-or-nothing | exact_sequence_structured | 2122 | 3.9397 | 0.1970 | 0.8030 | 0.8879 |
| all-or-nothing | history_archetype | 2065 | 5.7240 | 0.2862 | 0.7138 | 0.8143 |
| all-or-nothing | random_baseline | 2065 | 9.9952 | 0.4998 | 0.5002 | 0.6161 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | exact_sequence_structured vs history_archetype | 7.8333 | 4.9594 | 2.8738 | 0.0000 | yes |
| contrib_mae (↓ better) | exact_sequence_structured vs archetype_cluster | 7.8333 | 7.8772 | -0.0439 | 0.8094 | no |
| contrib_mae (↓ better) | exact_sequence_structured vs random_baseline | 7.8333 | 9.3628 | -1.5295 | 0.0000 | yes |
| contrib_mae (↓ better) | history_archetype vs archetype_cluster | 4.9594 | 7.8772 | -2.9178 | 0.0000 | yes |
| contrib_mae (↓ better) | history_archetype vs random_baseline | 4.9594 | 9.3628 | -4.4033 | 0.0000 | yes |
| contrib_mae (↓ better) | archetype_cluster vs random_baseline | 7.8772 | 9.3628 | -1.4856 | 0.0000 | yes |
| target_f1 (↑ better) | exact_sequence_structured vs history_archetype | 0.8060 | 0.6923 | 0.1138 | 0.0000 | yes |
| target_f1 (↑ better) | exact_sequence_structured vs archetype_cluster | 0.8060 | 0.6958 | 0.1102 | 0.0000 | yes |
| target_f1 (↑ better) | exact_sequence_structured vs random_baseline | 0.8060 | 0.7341 | 0.0720 | 0.0000 | yes |
| target_f1 (↑ better) | history_archetype vs archetype_cluster | 0.6923 | 0.6958 | -0.0036 | 0.7373 | no |
| target_f1 (↑ better) | history_archetype vs random_baseline | 0.6923 | 0.7341 | -0.0418 | 0.0000 | yes |
| target_f1 (↑ better) | archetype_cluster vs random_baseline | 0.6958 | 0.7341 | -0.0382 | 0.0001 | yes |
| action_exact_match (↑ better) | exact_sequence_structured vs history_archetype | 0.7990 | 0.6646 | 0.1344 | 0.0000 | yes |
| action_exact_match (↑ better) | exact_sequence_structured vs archetype_cluster | 0.7990 | 0.6897 | 0.1092 | 0.0000 | yes |
| action_exact_match (↑ better) | exact_sequence_structured vs random_baseline | 0.7990 | 0.7325 | 0.0665 | 0.0000 | yes |
| action_exact_match (↑ better) | history_archetype vs archetype_cluster | 0.6646 | 0.6897 | -0.0252 | 0.0117 | yes |
| action_exact_match (↑ better) | history_archetype vs random_baseline | 0.6646 | 0.7325 | -0.0679 | 0.0000 | yes |
| action_exact_match (↑ better) | archetype_cluster vs random_baseline | 0.6897 | 0.7325 | -0.0427 | 0.0000 | yes |
| target_hit_any (↑ better) | exact_sequence_structured vs history_archetype | 0.8080 | 0.7140 | 0.0939 | 0.0000 | yes |
| target_hit_any (↑ better) | exact_sequence_structured vs archetype_cluster | 0.8080 | 0.7019 | 0.1061 | 0.0000 | yes |
| target_hit_any (↑ better) | exact_sequence_structured vs random_baseline | 0.8080 | 0.7360 | 0.0720 | 0.0000 | yes |
| target_hit_any (↑ better) | history_archetype vs archetype_cluster | 0.7140 | 0.7019 | 0.0122 | 0.2177 | no |
| target_hit_any (↑ better) | history_archetype vs random_baseline | 0.7140 | 0.7360 | -0.0219 | 0.0211 | yes |
| target_hit_any (↑ better) | archetype_cluster vs random_baseline | 0.7019 | 0.7360 | -0.0341 | 0.0003 | yes |

