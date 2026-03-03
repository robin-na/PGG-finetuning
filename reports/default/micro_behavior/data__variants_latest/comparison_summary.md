# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| no archetype | no_archetype/2602262019 | 2810/2810 |
| random archetype | random_archetype/2602270902 | 5134/5134 |
| oracle archetype | oracle_archetype/2602270203 | 4873/4873 |
| retrieved archetype | retrieved_archetype/2602270212 | 5158/5158 |

## Plot Guide

- `compare_contrib_mae.png`: Mean contribution MAE across runs with 95% CI (`↓ better`).
- `compare_target_f1.png`: Mean typed-target F1 across runs with 95% CI (`↑ better`).
- `compare_action_exact_match.png`: Mean exact action-match rate across runs with 95% CI (`↑ better`).
- `compare_target_hit_any.png`: Mean target-hit-any rate across runs with 95% CI (`↑ better`).
- `compare_by_round_contrib_mae.png`: Round-wise contribution MAE with 95% CI bands (`↓ better`).
- `compare_by_round_target_f1.png`: Round-wise target F1 with 95% CI bands (`↑ better`).

## Standard Error and CI

- For each metric within a run: `SE = s / sqrt(n)`.
- `s` is sample standard deviation of row-level metric values; `n` is row count.
- 95% CI is `mean ± 1.96 * SE`.

## Overall Means

| Run | contrib_mae | target_f1 | action_exact_match | target_hit_any |
|---|---:|---:|---:|---:|
| no archetype | 4.0683 | 0.4832 | 0.4203 | 0.5192 |
| random archetype | 4.4375 | 0.5578 | 0.4940 | 0.6026 |
| oracle archetype | 3.4599 | 0.6681 | 0.5922 | 0.7092 |
| retrieved archetype | 3.7846 | 0.6582 | 0.5977 | 0.6915 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | no archetype vs random archetype | 4.0683 | 4.4375 | -0.3691 | 0.0328 | yes |
| contrib_mae (↓ better) | no archetype vs oracle archetype | 4.0683 | 3.4599 | 0.6084 | 0.0002 | yes |
| contrib_mae (↓ better) | no archetype vs retrieved archetype | 4.0683 | 3.7846 | 0.2837 | 0.0881 | no |
| contrib_mae (↓ better) | random archetype vs oracle archetype | 4.4375 | 3.4599 | 0.9776 | 0.0000 | yes |
| contrib_mae (↓ better) | random archetype vs retrieved archetype | 4.4375 | 3.7846 | 0.6529 | 0.0000 | yes |
| contrib_mae (↓ better) | oracle archetype vs retrieved archetype | 3.4599 | 3.7846 | -0.3247 | 0.0190 | yes |
| target_f1 (↑ better) | no archetype vs random archetype | 0.4832 | 0.5578 | -0.0747 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs oracle archetype | 0.4832 | 0.6681 | -0.1849 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs retrieved archetype | 0.4832 | 0.6582 | -0.1751 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs oracle archetype | 0.5578 | 0.6681 | -0.1102 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs retrieved archetype | 0.5578 | 0.6582 | -0.1004 | 0.0000 | yes |
| target_f1 (↑ better) | oracle archetype vs retrieved archetype | 0.6681 | 0.6582 | 0.0098 | 0.2878 | no |
| action_exact_match (↑ better) | no archetype vs random archetype | 0.4203 | 0.4940 | -0.0737 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs oracle archetype | 0.4203 | 0.5922 | -0.1720 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs retrieved archetype | 0.4203 | 0.5977 | -0.1774 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs oracle archetype | 0.4940 | 0.5922 | -0.0983 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs retrieved archetype | 0.4940 | 0.5977 | -0.1038 | 0.0000 | yes |
| action_exact_match (↑ better) | oracle archetype vs retrieved archetype | 0.5922 | 0.5977 | -0.0055 | 0.5771 | no |
| target_hit_any (↑ better) | no archetype vs random archetype | 0.5192 | 0.6026 | -0.0834 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs oracle archetype | 0.5192 | 0.7092 | -0.1900 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs retrieved archetype | 0.5192 | 0.6915 | -0.1723 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs oracle archetype | 0.6026 | 0.7092 | -0.1066 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs retrieved archetype | 0.6026 | 0.6915 | -0.0889 | 0.0000 | yes |
| target_hit_any (↑ better) | oracle archetype vs retrieved archetype | 0.7092 | 0.6915 | 0.0177 | 0.0611 | no |

