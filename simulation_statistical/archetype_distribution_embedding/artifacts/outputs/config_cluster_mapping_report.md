# CONFIG to cluster-mix interpretation

This report summarizes how validation-wave `CONFIG` changes map to the six archetype clusters. It uses treatment-level averages, so these are descriptive treatment associations, not isolated causal coefficients. The English cluster labels below are analyst shorthand derived from the exemplar summaries in `clustering_report.md`; the model itself still uses `cluster_1` to `cluster_6`.

## Cluster meanings

- `cluster_1`: `strong_mildly_conditional_cooperator`. Strong cooperator with a mild conditional component. Validation mean mass `0.183`.
- `cluster_2`: `opportunistic_free_rider`. Strategic or opportunistic free-rider with intermittent cooperation. Validation mean mass `0.147`.
- `cluster_3`: `fragile_high_cooperator`. Initially enthusiastic high cooperator who adapts downward when others look unreliable. Validation mean mass `0.180`.
- `cluster_4`: `near_unconditional_full_cooperator`. Near-unconditional full cooperator. Validation mean mass `0.225`.
- `cluster_5`: `unknown_or_sparse_info`. Unknown or unidentifiable summary cluster driven by sparse information. Validation mean mass `0.074`.
- `cluster_6`: `moderate_payoff_aware_conditional_cooperator`. Moderately cooperative, payoff-aware conditional cooperator. Validation mean mass `0.191`.

## Strongest treatment-level shifts in actual validation mixtures

### `CONFIG_chat` (1 - 0)

- decreases `moderate_payoff_aware_conditional_cooperator` by `-0.462`
- decreases `strong_mildly_conditional_cooperator` by `-0.406`
- increases `fragile_high_cooperator` by `+0.299`
- increases `near_unconditional_full_cooperator` by `+0.380`

### `CONFIG_punishmentExists` (1 - 0)

- decreases `moderate_payoff_aware_conditional_cooperator` by `-0.324`
- decreases `opportunistic_free_rider` by `-0.071`
- increases `near_unconditional_full_cooperator` by `+0.092`
- increases `strong_mildly_conditional_cooperator` by `+0.337`

### `CONFIG_rewardExists` (1 - 0)

- decreases `opportunistic_free_rider` by `-0.048`
- decreases `fragile_high_cooperator` by `-0.042`
- increases `strong_mildly_conditional_cooperator` by `+0.035`
- increases `near_unconditional_full_cooperator` by `+0.037`

### `CONFIG_allOrNothing` (1 - 0)

- decreases `moderate_payoff_aware_conditional_cooperator` by `-0.117`
- decreases `strong_mildly_conditional_cooperator` by `-0.115`
- increases `opportunistic_free_rider` by `+0.093`
- increases `near_unconditional_full_cooperator` by `+0.120`

### `CONFIG_showOtherSummaries` (1 - 0)

- decreases `fragile_high_cooperator` by `-0.121`
- decreases `near_unconditional_full_cooperator` by `-0.052`
- increases `moderate_payoff_aware_conditional_cooperator` by `+0.044`
- increases `opportunistic_free_rider` by `+0.096`

### `CONFIG_showPunishmentId` (1 - 0)

- decreases `opportunistic_free_rider` by `-0.078`
- decreases `moderate_payoff_aware_conditional_cooperator` by `-0.040`
- increases `fragile_high_cooperator` by `+0.071`
- increases `near_unconditional_full_cooperator` by `+0.078`

### `CONFIG_showNRounds` (1 - 0)

- decreases `fragile_high_cooperator` by `-0.077`
- decreases `unknown_or_sparse_info` by `-0.014`
- increases `moderate_payoff_aware_conditional_cooperator` by `+0.021`
- increases `opportunistic_free_rider` by `+0.038`

### `CONFIG_playerCount` (> 6.5 minus <= 6.5)

- decreases `opportunistic_free_rider` by `-0.115`
- increases `moderate_payoff_aware_conditional_cooperator` by `+0.035`
- increases `unknown_or_sparse_info` by `+0.037`

### `CONFIG_numRounds` (> 14.5 minus <= 14.5)

- decreases `opportunistic_free_rider` by `-0.067`
- decreases `strong_mildly_conditional_cooperator` by `-0.062`
- increases `near_unconditional_full_cooperator` by `+0.054`
- increases `fragile_high_cooperator` by `+0.119`

### `CONFIG_MPCR` (> 0.485 minus <= 0.485)

- decreases `near_unconditional_full_cooperator` by `-0.108`
- decreases `unknown_or_sparse_info` by `-0.005`
- increases `strong_mildly_conditional_cooperator` by `+0.032`
- increases `moderate_payoff_aware_conditional_cooperator` by `+0.046`

## Does the learned CONFIG->cluster model capture the same directions?

Broadly yes for the largest effects, but with much smaller amplitude. The predicted treatment mixtures shrink the actual shifts toward the mean rather than inventing different directions.

### `CONFIG_chat` (1 - 0)

- predicted decrease in `moderate_payoff_aware_conditional_cooperator` by `-0.212`
- predicted decrease in `strong_mildly_conditional_cooperator` by `-0.209`
- predicted increase in `fragile_high_cooperator` by `+0.175`
- predicted increase in `near_unconditional_full_cooperator` by `+0.262`

### `CONFIG_punishmentExists` (1 - 0)

- predicted decrease in `moderate_payoff_aware_conditional_cooperator` by `-0.150`
- predicted decrease in `fragile_high_cooperator` by `-0.036`
- predicted increase in `near_unconditional_full_cooperator` by `+0.040`
- predicted increase in `strong_mildly_conditional_cooperator` by `+0.174`

### `CONFIG_allOrNothing` (1 - 0)

- predicted decrease in `strong_mildly_conditional_cooperator` by `-0.092`
- predicted decrease in `moderate_payoff_aware_conditional_cooperator` by `-0.059`
- predicted increase in `fragile_high_cooperator` by `+0.049`
- predicted increase in `near_unconditional_full_cooperator` by `+0.065`

### `CONFIG_showOtherSummaries` (1 - 0)

- predicted decrease in `near_unconditional_full_cooperator` by `-0.048`
- predicted decrease in `unknown_or_sparse_info` by `-0.014`
- predicted increase in `fragile_high_cooperator` by `+0.021`
- predicted increase in `opportunistic_free_rider` by `+0.036`

### `CONFIG_numRounds` (> 14.5 minus <= 14.5)

- predicted decrease in `opportunistic_free_rider` by `-0.085`
- predicted decrease in `moderate_payoff_aware_conditional_cooperator` by `-0.003`
- predicted increase in `unknown_or_sparse_info` by `+0.014`
- predicted increase in `fragile_high_cooperator` by `+0.057`

### `CONFIG_MPCR` (> 0.485 minus <= 0.485)

- predicted decrease in `near_unconditional_full_cooperator` by `-0.064`
- predicted decrease in `opportunistic_free_rider` by `-0.046`
- predicted increase in `moderate_payoff_aware_conditional_cooperator` by `+0.042`
- predicted increase in `strong_mildly_conditional_cooperator` by `+0.053`

## Highest-loading validation treatments by cluster

### `cluster_1` / `strong_mildly_conditional_cooperator`

- `VALIDATION_19_T` with treatment-average mass `0.930`
- `VALIDATION_2_T` with treatment-average mass `0.930`
- `VALIDATION_6_T` with treatment-average mass `0.910`

### `cluster_2` / `opportunistic_free_rider`

- `VALIDATION_18_C` with treatment-average mass `0.620`
- `VALIDATION_10_C` with treatment-average mass `0.511`
- `VALIDATION_0_C` with treatment-average mass `0.412`

### `cluster_3` / `fragile_high_cooperator`

- `VALIDATION_7_C` with treatment-average mass `0.903`
- `VALIDATION_7_T` with treatment-average mass `0.465`
- `VALIDATION_14_C` with treatment-average mass `0.450`

### `cluster_4` / `near_unconditional_full_cooperator`

- `VALIDATION_13_T` with treatment-average mass `0.660`
- `VALIDATION_12_T` with treatment-average mass `0.536`
- `VALIDATION_0_T` with treatment-average mass `0.529`

### `cluster_5` / `unknown_or_sparse_info`

- `VALIDATION_15_T` with treatment-average mass `0.120`
- `VALIDATION_6_T` with treatment-average mass `0.077`
- `VALIDATION_11_T` with treatment-average mass `0.076`

### `cluster_6` / `moderate_payoff_aware_conditional_cooperator`

- `VALIDATION_2_C` with treatment-average mass `0.979`
- `VALIDATION_1_C` with treatment-average mass `0.960`
- `VALIDATION_19_C` with treatment-average mass `0.950`

## Main takeaway

- Chat and punishment are the two strongest observed composition shifters.
- Chat moves mass away from the quieter conditional-cooperation clusters and toward stronger cooperative clusters, while also increasing one more polarized/free-riding cluster.
- Punishment shifts mass away from moderate conditional cooperation and toward the strong-cooperation clusters.
- All-or-nothing treatments look more polarized, increasing both the strongest cooperator cluster and the opportunistic free-rider cluster.
- Larger `K` only helped in oracle direct prediction. In the deployable setting, `K=6` remains best because the `CONFIG -> cluster mixture` step becomes harder as `K` grows.