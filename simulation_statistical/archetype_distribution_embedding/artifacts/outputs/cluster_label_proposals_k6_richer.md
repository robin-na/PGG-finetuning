# Richer Cluster Labels for K=6

This note supersedes the earlier contribution-only shorthand for the current `K=6` clustering. It uses the post-hoc behavioral profile in [cluster_posthoc_profile.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_posthoc_profile.md), so the labels reflect contribution, punishment, reward, and communication rather than contribution alone.

## Proposed labels

- `cluster_1`: `punitive_norm_enforcing_cooperator`
  - High contributor with the strongest punishment signature in the current six-cluster solution.
  - Contribution is clearly prosocial, but the distinctive feature is active norm enforcement through punishment rather than reward.

- `cluster_2`: `opportunistic_free_rider_with_low_social_engagement`
  - Lowest-contribution cluster among the identified behavioral types.
  - Characterized by frequent zero contribution, opportunistic cooperation, and generally weak positive sanctioning.
  - Communication is present more often than in some other clusters, but the content reads more disengaged or minimally participatory than coordinating.

- `cluster_3`: `communicative_prosocial_coordinator_rewarder`
  - Very high contributor, highly communicative, and the clearest reward-heavy cooperative cluster.
  - Distinctive features are chat/coordination language plus strong positive reinforcement of cooperation.

- `cluster_4`: `principled_unconditional_full_cooperator`
  - Cleanest unconditional full-cooperation cluster.
  - Extremely high and stable contribution with little evidence of strategic adjustment.
  - The post-hoc text profile is mostly non-punitive and non-rewarding; this is the pure “I contribute because that is the right rule” cluster.

- `cluster_5`: `unknown_or_sparse_information`
  - Sparse or low-confidence summaries that do not support a stable behavioral interpretation.
  - Should not be treated as a substantive behavioral type.

- `cluster_6`: `reward_oriented_nonpunitive_conditional_cooperator`
  - Moderately high contributor with clear reward use and essentially no punishment.
  - More payoff-aware and conditional than `cluster_4`, but much less punitive than `cluster_1`.
  - This is the clearest “reward instead of punish” cluster in the current solution.

## Why this is better than the old shorthand

- The old labels mostly tracked contribution level because the cluster reports previewed the beginning of each summary, which usually starts with `<CONTRIBUTION>`.
- The actual embeddings were fit on the full cleaned archetype text, including `COMMUNICATION`, `PUNISHMENT`, `REWARD`, and response tags.
- The post-hoc profile shows that sanctioning and communication are real dimensions of the cluster space, especially for:
  - `cluster_1` versus `cluster_6`
  - `cluster_3` versus `cluster_4`

## Suggested usage

- For macro treatment interpretation, use these richer labels instead of the older contribution-only names.
- For modeling, still keep the canonical IDs `cluster_1` to `cluster_6` in code and files.
- If we later retrain the clustering, these names should be treated as provisional and revalidated against the new post-hoc profile.
