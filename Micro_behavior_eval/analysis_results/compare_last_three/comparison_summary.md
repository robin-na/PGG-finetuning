# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| no persona | 2602262019 | 2810/2810 |
| oracle persona | 2602270203 | 4373/4373 |
| retrieved persona | 2602270212 | 4590/4590 |

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
| no persona | 4.0683 | 0.4832 | 0.4203 | 0.5192 |
| oracle persona | 3.3762 | 0.6779 | 0.6037 | 0.7183 |
| retrieved persona | 3.8693 | 0.6783 | 0.6227 | 0.7100 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | no persona vs oracle persona | 4.0683 | 3.3762 | 0.6922 | 0.0000 | yes |
| contrib_mae (↓ better) | no persona vs retrieved persona | 4.0683 | 3.8693 | 0.1990 | 0.2788 | no |
| contrib_mae (↓ better) | oracle persona vs retrieved persona | 3.3762 | 3.8693 | -0.4931 | 0.0007 | yes |
| target_f1 (↑ better) | no persona vs oracle persona | 0.4832 | 0.6779 | -0.1947 | 0.0000 | yes |
| target_f1 (↑ better) | no persona vs retrieved persona | 0.4832 | 0.6783 | -0.1951 | 0.0000 | yes |
| target_f1 (↑ better) | oracle persona vs retrieved persona | 0.6779 | 0.6783 | -0.0004 | 0.9698 | no |
| action_exact_match (↑ better) | no persona vs oracle persona | 0.4203 | 0.6037 | -0.1834 | 0.0000 | yes |
| action_exact_match (↑ better) | no persona vs retrieved persona | 0.4203 | 0.6227 | -0.2024 | 0.0000 | yes |
| action_exact_match (↑ better) | oracle persona vs retrieved persona | 0.6037 | 0.6227 | -0.0190 | 0.0874 | no |
| target_hit_any (↑ better) | no persona vs oracle persona | 0.5192 | 0.7183 | -0.1991 | 0.0000 | yes |
| target_hit_any (↑ better) | no persona vs retrieved persona | 0.5192 | 0.7100 | -0.1908 | 0.0000 | yes |
| target_hit_any (↑ better) | oracle persona vs retrieved persona | 0.7183 | 0.7100 | 0.0082 | 0.4228 | no |

