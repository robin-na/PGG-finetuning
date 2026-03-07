# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| no archetype | no_archetype/2602262019 | 2810/2810 |
| oracle archetype | oracle_archetype/2602270203 | 4873/4873 |
| retrieved archetype | retrieved_archetype/2602270212 | 5158/5158 |
| random archetype | random_archetype/2602270902 | 5134/5134 |
| prev-round baseline | __baseline_prev_round__ | 2810/2810 |

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
| no archetype | 4.0683 | 0.4832 | 0.4203 | 0.5192 |
| oracle archetype | 3.4599 | 0.6681 | 0.5922 | 0.7092 |
| retrieved archetype | 3.7846 | 0.6582 | 0.5977 | 0.6915 |
| random archetype | 4.4375 | 0.5578 | 0.4940 | 0.6026 |
| prev-round baseline | 4.4790 | 0.8057 | 0.7637 | 0.8224 |

## Contribution By Game Regime

| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |
|---|---|---:|---:|---:|---:|---:|
| continuous | oracle archetype | 2715 | 2.7109 | 0.1355 | NaN | NaN |
| continuous | random archetype | 2727 | 3.7851 | 0.1893 | NaN | NaN |
| continuous | retrieved archetype | 2727 | 3.2274 | 0.1614 | NaN | NaN |
| all-or-nothing | oracle archetype | 2158 | 4.4022 | 0.2201 | 0.7799 | 0.8436 |
| all-or-nothing | random archetype | 2407 | 5.1766 | 0.2588 | 0.7412 | 0.8219 |
| all-or-nothing | retrieved archetype | 2431 | 4.4097 | 0.2205 | 0.7795 | 0.8581 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | no archetype vs oracle archetype | 4.0683 | 3.4599 | 0.6084 | 0.0002 | yes |
| contrib_mae (↓ better) | no archetype vs retrieved archetype | 4.0683 | 3.7846 | 0.2837 | 0.0874 | no |
| contrib_mae (↓ better) | no archetype vs random archetype | 4.0683 | 4.4375 | -0.3691 | 0.0322 | yes |
| contrib_mae (↓ better) | no archetype vs prev-round baseline | 4.0683 | 4.4790 | -0.4107 | 0.0371 | yes |
| contrib_mae (↓ better) | oracle archetype vs retrieved archetype | 3.4599 | 3.7846 | -0.3247 | 0.0183 | yes |
| contrib_mae (↓ better) | oracle archetype vs random archetype | 3.4599 | 4.4375 | -0.9776 | 0.0000 | yes |
| contrib_mae (↓ better) | oracle archetype vs prev-round baseline | 3.4599 | 4.4790 | -1.0191 | 0.0000 | yes |
| contrib_mae (↓ better) | retrieved archetype vs random archetype | 3.7846 | 4.4375 | -0.6529 | 0.0000 | yes |
| contrib_mae (↓ better) | retrieved archetype vs prev-round baseline | 3.7846 | 4.4790 | -0.6944 | 0.0000 | yes |
| contrib_mae (↓ better) | random archetype vs prev-round baseline | 4.4375 | 4.4790 | -0.0415 | 0.8106 | no |
| target_f1 (↑ better) | no archetype vs oracle archetype | 0.4832 | 0.6681 | -0.1849 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs retrieved archetype | 0.4832 | 0.6582 | -0.1751 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs random archetype | 0.4832 | 0.5578 | -0.0747 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs prev-round baseline | 0.4832 | 0.8057 | -0.3226 | 0.0000 | yes |
| target_f1 (↑ better) | oracle archetype vs retrieved archetype | 0.6681 | 0.6582 | 0.0098 | 0.2903 | no |
| target_f1 (↑ better) | oracle archetype vs random archetype | 0.6681 | 0.5578 | 0.1102 | 0.0000 | yes |
| target_f1 (↑ better) | oracle archetype vs prev-round baseline | 0.6681 | 0.8057 | -0.1376 | 0.0000 | yes |
| target_f1 (↑ better) | retrieved archetype vs random archetype | 0.6582 | 0.5578 | 0.1004 | 0.0000 | yes |
| target_f1 (↑ better) | retrieved archetype vs prev-round baseline | 0.6582 | 0.8057 | -0.1475 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs prev-round baseline | 0.5578 | 0.8057 | -0.2479 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs oracle archetype | 0.4203 | 0.5922 | -0.1720 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs retrieved archetype | 0.4203 | 0.5977 | -0.1774 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs random archetype | 0.4203 | 0.4940 | -0.0737 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs prev-round baseline | 0.4203 | 0.7637 | -0.3434 | 0.0000 | yes |
| action_exact_match (↑ better) | oracle archetype vs retrieved archetype | 0.5922 | 0.5977 | -0.0055 | 0.5919 | no |
| action_exact_match (↑ better) | oracle archetype vs random archetype | 0.5922 | 0.4940 | 0.0983 | 0.0000 | yes |
| action_exact_match (↑ better) | oracle archetype vs prev-round baseline | 0.5922 | 0.7637 | -0.1715 | 0.0000 | yes |
| action_exact_match (↑ better) | retrieved archetype vs random archetype | 0.5977 | 0.4940 | 0.1038 | 0.0000 | yes |
| action_exact_match (↑ better) | retrieved archetype vs prev-round baseline | 0.5977 | 0.7637 | -0.1660 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs prev-round baseline | 0.4940 | 0.7637 | -0.2697 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs oracle archetype | 0.5192 | 0.7092 | -0.1900 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs retrieved archetype | 0.5192 | 0.6915 | -0.1723 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs random archetype | 0.5192 | 0.6026 | -0.0834 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs prev-round baseline | 0.5192 | 0.8224 | -0.3032 | 0.0000 | yes |
| target_hit_any (↑ better) | oracle archetype vs retrieved archetype | 0.7092 | 0.6915 | 0.0177 | 0.0594 | no |
| target_hit_any (↑ better) | oracle archetype vs random archetype | 0.7092 | 0.6026 | 0.1066 | 0.0000 | yes |
| target_hit_any (↑ better) | oracle archetype vs prev-round baseline | 0.7092 | 0.8224 | -0.1132 | 0.0000 | yes |
| target_hit_any (↑ better) | retrieved archetype vs random archetype | 0.6915 | 0.6026 | 0.0889 | 0.0000 | yes |
| target_hit_any (↑ better) | retrieved archetype vs prev-round baseline | 0.6915 | 0.8224 | -0.1309 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs prev-round baseline | 0.6026 | 0.8224 | -0.2198 | 0.0000 | yes |

