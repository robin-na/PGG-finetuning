# Cluster Label Proposals for K=8 and K=10

These are analyst shorthand labels for the `K=8` and `K=10` GMM refits. They are based on the exemplar summaries in [clustering_report_k8.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/clustering_report_k8.md) and [clustering_report_k10.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/clustering_report_k10.md). They are not model-native labels.

## K=8

- `cluster_1`: `opportunistic_extreme_toggler`
  - Mixed-strategy conditional cooperator that flips between full cooperation and full free-riding.
- `cluster_2`: `rigid_unconditional_full_cooperator`
  - Stable, repeated-game full cooperator with almost no downward adjustment.
- `cluster_3`: `unknown_or_sparse_info`
  - Unidentifiable summaries with too little observable behavior.
- `cluster_4`: `pure_free_rider`
  - Near-zero or zero contribution with almost no cooperative response.
- `cluster_5`: `cooperation_first_moderate_conditional_cooperator`
  - Generally prosocial, moderate-to-high contributor with conditional adjustment.
- `cluster_6`: `high_conditional_cooperator`
  - High contributor anchored on cooperation, but willing to move down when the environment deteriorates.
- `cluster_7`: `idealistic_full_cooperator`
  - Principled full contributor, often visible in one-shot or norm-driven settings.
- `cluster_8`: `payoff_sensitive_mixed_conditional_cooperator`
  - Moderately cooperative, flexible, and somewhat opportunistic; uses a wider range of contribution levels.

## K=10

- `cluster_1`: `self_protective_high_cooperator`
  - Near-max contributor with opportunistic or self-protective dips.
- `cluster_2`: `strong_but_conditional_full_leaning_cooperator`
  - Mostly full contributor, but more reactive to punishment or local instability.
- `cluster_3`: `strict_norm_enforcing_full_cooperator`
  - Rigid full cooperator with explicit moral or norm-based commitment.
- `cluster_4`: `patterned_opportunistic_cooperator`
  - Alternates between full cooperation and free-riding in a deliberate, payoff-aware pattern.
- `cluster_5`: `strict_free_rider`
  - Stable zero-contributor / uncompromising defector.
- `cluster_6`: `unknown_or_sparse_info`
  - Unidentifiable summaries with too little observable behavior.
- `cluster_7`: `idealistic_full_cooperator`
  - Clean principled full contributor, especially in one-shot or low-incentive environments.
- `cluster_8`: `midrange_conditional_cooperator`
  - Moderate contributor with strategic shading and verbal pro-cooperation in some cases.
- `cluster_9`: `norm_following_high_cooperator_with_quiet_deviations`
  - Strong cooperator that usually stays near full contribution but is not perfectly rigid.
- `cluster_10`: `tolerant_unconditional_full_cooperator`
  - Very stable full cooperator with high tolerance for being exploited.

## Interpretation

- `K=8` creates one clearly distinct new anti-cooperation cluster relative to `K=6`: a pure free-rider cluster.
- `K=8` also splits the old cooperative mass into:
  - rigid full cooperators
  - idealistic full cooperators
  - high conditional cooperators
  - more payoff-sensitive mixed cooperators
- `K=10` mostly keeps splitting the cooperative side into narrower flavors rather than discovering many brand-new behavioral regimes.
- That is consistent with the direct-prediction sweep:
  - higher `K` contains extra oracle signal
  - but the extra granularity is hard for the `CONFIG -> cluster mixture` model to predict reliably
