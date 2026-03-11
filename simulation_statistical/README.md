# Simulation Statistical

`simulation_statistical` is the numerical simulation package for PGG behavior. It shares the benchmark data layout with the LLM evaluation code, but the decision policy is a seeded statistical simulator instead of a prompted model.

Data and outputs live under `benchmark_statistical/`.

Retrospective report for the current experiment cycle:

- [EXPERIMENT_REPORT_2026-03-10.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/EXPERIMENT_REPORT_2026-03-10.md)
- [config_cluster_mapping_report.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/config_cluster_mapping_report.md)
- [cluster_label_proposals_k6_richer.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_label_proposals_k6_richer.md)
- [cluster_label_proposals_k8_k10.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_label_proposals_k8_k10.md)
- [cluster_posthoc_profile.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_posthoc_profile.md)
- [raw_behavior_cluster_report.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/raw_behavior_cluster/artifacts/outputs/raw_behavior_cluster_report.md)
- [wave_anchored_ood_report.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/artifacts/outputs/wave_anchored_ood_report.md)

## Strategy Set

Current supported `--strategy` values:

- `random_baseline`
- `archetype_cluster`
- `archetype_cluster_oracle_treatment`
- `archetype_cluster_plus`
- `archetype_cluster_plus_oracle_treatment`
- `history_archetype`
- `exact_sequence_archetype`
- `exact_sequence_oracle_treatment`
- `exact_sequence_history_only`
- `gpu_sequence_archetype`

The current recommendation for macro rollout is:

- strongest learned macro engine so far: `archetype_cluster`
- best next interpretable extension: `archetype_cluster_plus`
- not recommended for macro rollout without further redesign: `exact_sequence_*`

For direct benchmark-level macro prediction in `simulation_statistical/macro/analysis`:

- `linear_config` and `ridge_config` use raw `CONFIG` only
- `linear_cluster_pred` and `ridge_cluster_pred` use the predicted archetype-cluster distribution only
- `linear_cluster_pred_plus_config` and `ridge_cluster_pred_plus_config` use predicted cluster distribution plus raw `CONFIG`

Current evidence is that direct prediction from the predicted cluster distribution is stronger than direct prediction from raw `CONFIG` alone for normalized efficiency.

## Model Families

### `random_baseline`

No training.

Per player-round:

1. contribution:
   - `0` or `CONFIG_endowment` when `CONFIG_allOrNothing=true`
   - integer from `0..CONFIG_endowment` otherwise
2. punishment:
   - if enabled, one random target with probability `0.10`
   - units fixed at `1`
3. reward:
   - if enabled, one random target with probability `0.10`
   - units fixed at `1`

### `archetype_cluster`

This is the simple cluster-conditioned engine in [trained_policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/trained_policy.py).

Training steps:

1. learning-wave LLM summaries are embedded and clustered in `simulation_statistical/archetype_distribution_embedding`
2. each learning-wave player gets a hard cluster assignment from the learned cluster weights
3. contribution priors are stored by:
   - cluster
   - round phase
   - all-or-nothing flag
4. sanction/reward behavior is stored by:
   - cluster
   - round phase
   - contribution-rank orientation of targets

Simulation steps:

1. predict a treatment-level cluster distribution from `CONFIG` with the Dirichlet env model
2. sample one hard cluster per player at game start
3. sample contribution from the cluster-conditioned empirical pool
4. sample punish/reward from the cluster-conditioned empirical action model

`archetype_cluster_oracle_treatment` uses the same behavior model, but replaces the env-model cluster distribution with a validation-treatment oracle average.

### `archetype_cluster_plus`

This is the new light-history extension in [cluster_plus_policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/cluster_plus_policy.py).

Design goal:

- keep the stable treatment-level archetype engine
- add only a small amount of structured visible history
- avoid the macro action collapse seen in the exact-sequence family

Training data:

- learning-wave player-round rows from `benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv`
- learning-wave cluster assignments from `simulation_statistical/archetype_distribution_embedding/artifacts/outputs/player_cluster_weights_learn.parquet`
- learning-wave game-level `CONFIG` rows from `benchmark_statistical/data/processed_data/df_analysis_learn.csv`

Contribution model:

- empirical contribution samples indexed by:
  - sampled cluster
  - visible round phase
  - all-or-nothing flag
  - last-round own contribution bin
  - last-round peer-mean contribution bin
  - whether the player was punished last round
  - whether the player was rewarded last round
- backoff chain falls from the most specific history key to cluster-only and then global pools

Action model:

- separate punish and reward empirical models
- stored quantities:
  - probability of acting at all
  - total units
  - number of targets
  - average contribution-rank orientation of targets
- indexed by:
  - sampled cluster
  - visible round phase
  - last-round history signature
  - current-round own contribution bin
  - current-round peer-mean contribution bin
- same target cannot be both punished and rewarded
- final actions are pruned to respect the player’s within-round budget

Visible history used by `archetype_cluster_plus`:

- last-round own contribution
- last-round peer mean contribution
- whether the focal player was punished last round
- whether the focal player was rewarded last round
- current-round own and peer-mean contribution bins for sanctioning
- if `CONFIG_showNRounds=true`, phase is `early/mid/late`
- otherwise phase uses absolute round buckets to avoid leaking the known horizon

`archetype_cluster_plus_oracle_treatment` uses the same behavior model, but swaps in validation-treatment oracle cluster distributions.

### `history_archetype`

This is the first history-conditioned model in [history_conditioned_policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/history_conditioned_policy.py).

Training:

- learning-wave player-round rows
- hard cluster assignment as an input feature
- structured summary-history features
- gradient boosting heads for contribution and sanction/reward components

History representation:

- compressed aggregates, not exact ordered state
- examples:
  - own previous contribution
  - peer mean and variance
  - visible punishment/reward summaries
  - payoff summaries

### `exact_sequence_*`

These models live in [structured_sequence_policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/structured_sequence_policy.py).

Training:

- learning-wave player-round rows
- exact ordered round history kept as structured state, not free text
- contribution head:
  - binary classifier for all-or-nothing games
  - 5-bin classifier for continuous games over `{0, 5, 10, 15, 20}`
- action head:
  - per focal-target categorical model over `none/punish/reward`
  - units fixed to `1`

Variants:

- `exact_sequence_archetype`: includes sampled cluster from the env model
- `exact_sequence_oracle_treatment`: oracle treatment cluster distribution
- `exact_sequence_history_only`: no cluster at all

These models improved one-step micro behavior but have shown unstable macro rollouts, especially action collapse.

### `gpu_sequence_archetype`

PyTorch/GPU rewrite attempt in [gpu_sequence_policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/gpu_sequence_policy.py).

- same general structured-history setup as `exact_sequence_*`
- MLP heads instead of sklearn SGD
- useful only on a real GPU/tensorized path

## Layout

- `simulation_statistical/micro`
  - micro-style prediction vs actual rows
  - outputs default to `benchmark_statistical/micro/runs`
  - reports default to `benchmark_statistical/micro/reports`
- `simulation_statistical/macro`
  - full-game numerical rollouts
  - outputs default to `benchmark_statistical/macro/runs`
  - reports default to `benchmark_statistical/macro/reports`
  - compare reports also write:
    - `direct_regression_baseline_summaries.csv`
    - `noise_ceiling_summary.csv`
    - `noise_ceiling_by_benchmark.csv`
- `simulation_statistical/archetype_distribution_embedding`
  - LLM-summary embedding, clustering, and env-model pipeline
- `simulation_statistical/algorithmic_latent`
  - planned redesign for algorithm-based latent strategies with LLM-proposed priors and data-calibrated inference
- `simulation_statistical/ood`
  - true wave-anchored one-factor OOD direct benchmarks built from `benchmark_statistical/data`
- shared utilities:
  - [common.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/common.py)
  - [policy.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/policy.py)

## Training Commands

Cluster-only behavior model:

```bash
python simulation_statistical/archetype_distribution_embedding/run_pipeline.py all-post-embed
```

Cluster-plus behavior model:

```bash
python simulation_statistical/train_cluster_plus_policy.py
```

History-conditioned gradient boosting model:

```bash
python simulation_statistical/train_history_conditioned_policy.py
```

Structured exact-sequence model:

```bash
python simulation_statistical/train_exact_sequence_policy.py
```

Structured exact-sequence no-cluster ablation:

```bash
python simulation_statistical/train_exact_sequence_policy.py --no_cluster
```

GPU sequence model:

```bash
python simulation_statistical/train_gpu_sequence_model.py
```

True OOD direct-benchmark sweep:

```bash
python simulation_statistical/ood/run_direct_ood_benchmarks.py
```

## Run

Micro rollout:

```bash
python simulation_statistical/micro/run_micro_simulation.py --strategy archetype_cluster_plus
```

Macro rollout:

```bash
python simulation_statistical/macro/run_macro_simulation.py --strategy archetype_cluster_plus
```

The same `--strategy` argument accepts any of the strategy names listed above.

## Notes

- Statistical policies do not emit chat text; `data.chat_message` stays blank.
- Standard macro validation uses treatment-level contexts from `df_analysis_val_averaged.csv`.
- For fairer stochastic evaluation, macro runs can also be generated on all raw validation games and aggregated later by `CONFIG_treatmentName`.
- The next-generation redesign is documented separately in [algorithmic_latent/README.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/README.md) so it does not get mixed into the current cluster-based baseline family.
