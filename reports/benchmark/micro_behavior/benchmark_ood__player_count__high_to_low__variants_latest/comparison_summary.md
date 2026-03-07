# Comparison Summary

## Runs

| Label | Run ID | Rows (post/pre) |
|---|---|---:|
| no archetype | no_archetype/2603022352 | 1205/1205 |
| random archetype | random_archetype/9980072_r0_random | 1205/1205 |
| oracle archetype | oracle_archetype/9977694_r0_oracle | 1205/1205 |
| retrieved archetype | retrieved_archetype/9976921_r1_retrieved | 1205/1205 |
| prev-round baseline | __baseline_prev_round__ | 1205/1205 |

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
| no archetype | 3.9676 | 0.4399 | 0.3884 | 0.4606 |
| random archetype | 4.1427 | 0.5824 | 0.5344 | 0.6025 |
| oracle archetype | 3.8515 | 0.5814 | 0.5295 | 0.5992 |
| retrieved archetype | 4.0631 | 0.5262 | 0.4805 | 0.5402 |
| prev-round baseline | 4.1353 | 0.8677 | 0.8332 | 0.8788 |

## Contribution By Game Regime

| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |
|---|---|---:|---:|---:|---:|---:|
| continuous | no archetype | 853 | 3.8230 | 0.1911 | NaN | NaN |
| continuous | oracle archetype | 853 | 3.7292 | 0.1865 | NaN | NaN |
| continuous | prev-round baseline | 853 | 3.8722 | 0.1936 | NaN | NaN |
| continuous | random archetype | 853 | 4.3048 | 0.2152 | NaN | NaN |
| continuous | retrieved archetype | 853 | 4.1923 | 0.2096 | NaN | NaN |
| all-or-nothing | no archetype | 352 | 4.3182 | 0.2159 | 0.7841 | 0.8555 |
| all-or-nothing | oracle archetype | 352 | 4.1477 | 0.2074 | 0.7926 | 0.8571 |
| all-or-nothing | prev-round baseline | 352 | 4.7727 | 0.2386 | 0.7614 | 0.8359 |
| all-or-nothing | random archetype | 352 | 3.7500 | 0.1875 | 0.8125 | 0.8764 |
| all-or-nothing | retrieved archetype | 352 | 3.7500 | 0.1875 | 0.8125 | 0.8778 |

## Pairwise Significance (BH-adjusted)

| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |
|---|---|---:|---:|---:|---:|---|
| contrib_mae (↓ better) | no archetype vs random archetype | 3.9676 | 4.1427 | -0.1751 | 0.6514 | no |
| contrib_mae (↓ better) | no archetype vs oracle archetype | 3.9676 | 3.8515 | 0.1162 | 0.7907 | no |
| contrib_mae (↓ better) | no archetype vs retrieved archetype | 3.9676 | 4.0631 | -0.0954 | 0.8389 | no |
| contrib_mae (↓ better) | no archetype vs prev-round baseline | 3.9676 | 4.1353 | -0.1676 | 0.6583 | no |
| contrib_mae (↓ better) | random archetype vs oracle archetype | 4.1427 | 3.8515 | 0.2913 | 0.3784 | no |
| contrib_mae (↓ better) | random archetype vs retrieved archetype | 4.1427 | 4.0631 | 0.0797 | 0.8707 | no |
| contrib_mae (↓ better) | random archetype vs prev-round baseline | 4.1427 | 4.1353 | 0.0075 | 0.9778 | no |
| contrib_mae (↓ better) | oracle archetype vs retrieved archetype | 3.8515 | 4.0631 | -0.2116 | 0.5500 | no |
| contrib_mae (↓ better) | oracle archetype vs prev-round baseline | 3.8515 | 4.1353 | -0.2838 | 0.3881 | no |
| contrib_mae (↓ better) | retrieved archetype vs prev-round baseline | 4.0631 | 4.1353 | -0.0722 | 0.8720 | no |
| target_f1 (↑ better) | no archetype vs random archetype | 0.4399 | 0.5824 | -0.1425 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs oracle archetype | 0.4399 | 0.5814 | -0.1414 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs retrieved archetype | 0.4399 | 0.5262 | -0.0863 | 0.0000 | yes |
| target_f1 (↑ better) | no archetype vs prev-round baseline | 0.4399 | 0.8677 | -0.4278 | 0.0000 | yes |
| target_f1 (↑ better) | random archetype vs oracle archetype | 0.5824 | 0.5814 | 0.0011 | 0.9778 | no |
| target_f1 (↑ better) | random archetype vs retrieved archetype | 0.5824 | 0.5262 | 0.0562 | 0.0076 | yes |
| target_f1 (↑ better) | random archetype vs prev-round baseline | 0.5824 | 0.8677 | -0.2853 | 0.0000 | yes |
| target_f1 (↑ better) | oracle archetype vs retrieved archetype | 0.5814 | 0.5262 | 0.0551 | 0.0087 | yes |
| target_f1 (↑ better) | oracle archetype vs prev-round baseline | 0.5814 | 0.8677 | -0.2864 | 0.0000 | yes |
| target_f1 (↑ better) | retrieved archetype vs prev-round baseline | 0.5262 | 0.8677 | -0.3415 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs random archetype | 0.3884 | 0.5344 | -0.1461 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs oracle archetype | 0.3884 | 0.5295 | -0.1411 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs retrieved archetype | 0.3884 | 0.4805 | -0.0921 | 0.0000 | yes |
| action_exact_match (↑ better) | no archetype vs prev-round baseline | 0.3884 | 0.8332 | -0.4448 | 0.0000 | yes |
| action_exact_match (↑ better) | random archetype vs oracle archetype | 0.5344 | 0.5295 | 0.0050 | 0.8720 | no |
| action_exact_match (↑ better) | random archetype vs retrieved archetype | 0.5344 | 0.4805 | 0.0539 | 0.0123 | yes |
| action_exact_match (↑ better) | random archetype vs prev-round baseline | 0.5344 | 0.8332 | -0.2988 | 0.0000 | yes |
| action_exact_match (↑ better) | oracle archetype vs retrieved archetype | 0.5295 | 0.4805 | 0.0490 | 0.0239 | yes |
| action_exact_match (↑ better) | oracle archetype vs prev-round baseline | 0.5295 | 0.8332 | -0.3037 | 0.0000 | yes |
| action_exact_match (↑ better) | retrieved archetype vs prev-round baseline | 0.4805 | 0.8332 | -0.3527 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs random archetype | 0.4606 | 0.6025 | -0.1419 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs oracle archetype | 0.4606 | 0.5992 | -0.1386 | 0.0000 | yes |
| target_hit_any (↑ better) | no archetype vs retrieved archetype | 0.4606 | 0.5402 | -0.0797 | 0.0002 | yes |
| target_hit_any (↑ better) | no archetype vs prev-round baseline | 0.4606 | 0.8788 | -0.4183 | 0.0000 | yes |
| target_hit_any (↑ better) | random archetype vs oracle archetype | 0.6025 | 0.5992 | 0.0033 | 0.9136 | no |
| target_hit_any (↑ better) | random archetype vs retrieved archetype | 0.6025 | 0.5402 | 0.0622 | 0.0036 | yes |
| target_hit_any (↑ better) | random archetype vs prev-round baseline | 0.6025 | 0.8788 | -0.2763 | 0.0000 | yes |
| target_hit_any (↑ better) | oracle archetype vs retrieved archetype | 0.5992 | 0.5402 | 0.0589 | 0.0060 | yes |
| target_hit_any (↑ better) | oracle archetype vs prev-round baseline | 0.5992 | 0.8788 | -0.2797 | 0.0000 | yes |
| target_hit_any (↑ better) | retrieved archetype vs prev-round baseline | 0.5402 | 0.8788 | -0.3386 | 0.0000 | yes |

