# Algorithmic Latent Redesign

This folder is a separate redesign track for `simulation_statistical`.

The goal is not to extend the current cluster-based simulator further. The goal is to replace opaque cluster IDs with interpretable algorithmic latent strategies that:

- have direct meaning to the simulator
- can be fit from observed human traces
- can be shifted by `CONFIG` in a structured way
- are amenable to strict OOD tests

This redesign is motivated by the limits of the current pipeline:

- `archetype_cluster` is strong on treatment-level macro correlation, but its latent state is not interpretable enough
- richer history-conditioned simulators improved micro imitation, but they degraded macro rollout
- text-cluster latents are useful as a prior, but they are not themselves the behavioral object we want to study

## Core Idea

Represent each player with:

- an `algorithm family`
- a small numeric parameter vector within that family

Examples:

- `conditional_cooperator`
- `endgame_defector`
- `retaliatory_punisher`
- `reward_oriented_cooperator`

The simulator then rolls out behavior from those interpretable decision rules instead of from cluster-conditioned empirical pools or black-box history models.

The LLM is not asked to directly predict game outcomes. Instead, it is used to:

- propose candidate behavioral families and state variables
- label or summarize human traces into candidate algorithm families
- produce priors over how `CONFIG` should shift family mixtures or parameter signs

Human data remains the final authority:

- parameters are fit from observed traces
- family assignments are inferred from data
- LLM priors are calibrated, shrunk, or ignored when they do not help held-out performance

## Folder Layout

- [DSL_SPEC.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/DSL_SPEC.md)
  - the restricted strategy language and state variables
- [FAMILY_LIBRARY.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/FAMILY_LIBRARY.md)
  - first-pass family set and parameters
- [TRAINING_PLAN.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/TRAINING_PLAN.md)
  - inference, fitting, and evaluation plan
- [OOD_EVAL_PLAN.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/OOD_EVAL_PLAN.md)
  - strict out-of-distribution benchmark plan
- [prompts/README.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/prompts/README.md)
  - how LLM prompting should be used in this branch
- [simulator/README.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/simulator/README.md)
  - planned runtime structure
- [inference/README.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/README.md)
  - planned inference and calibration modules

## Design Principles

1. Interpretable latent state
   - latent variables should correspond to recognizable behavioral rules, not anonymous clusters
2. Visibility-respecting state
   - the simulator can only condition on information the player actually sees
3. Probabilistic, not brittle
   - avoid a single exact hard-coded program per person; infer a posterior over families and parameters
4. Data-calibrated LLM usage
   - the LLM proposes priors and hypotheses; the data decides what survives
5. OOD-first evaluation
   - the design should be judged on held-out environment families, not only random validation splits

## First Build Target

The first practical version should be:

- a small family library
- structured visible state only
- contribution and per-target sanction/reward probabilities from parameterized rules
- soft player-family assignments inferred from learning-wave traces
- `CONFIG -> family mixture` model

This is intentionally narrower than a free-form program synthesis system. The point is to get a disciplined, fit-able, interpretable baseline before considering a richer algorithm space.

## First Implemented Artifact

Stage 1 is now implemented as a reusable state-table builder:

- [inference/build_state_table.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/build_state_table.py)

It produces two visibility-aware parquet tables:

- `learning_wave_contribution_stage.parquet`
- `learning_wave_action_stage.parquet`

and the same for `validation_wave` when requested.

Run:

```bash
python simulation_statistical/algorithmic_latent/inference/build_state_table.py --wave learning_wave
```

Default output root:

- `simulation_statistical/algorithmic_latent/artifacts/state_tables`

The contribution table is one row per player-round contribution decision.

Contribution-stage conditioning is:

- `CONFIG`
- visible history from rounds `1..T-1` in structured form
- previous-round summaries
- cumulative history summaries such as own mean contribution, peer mean contribution, cumulative punish/reward received, and visible peer summary aggregates

It does not see round-`T` peer contributions.

The action table is one row per focal-target sanction/reward decision edge in rounds where punishment or reward exists.

Action-stage conditioning is:

- the same `CONFIG + visible history from rounds 1..T-1`
- current-round contributions for the focal player and all peers
- target-relative current-round contribution features

It does not see other players' round-`T` punish/reward choices before the focal player acts. Punisher/rewarder identity only enters later rounds if the corresponding visibility flag reveals it after the round resolves.

Stage 2 fitting entrypoint:

- [inference/fit_family_policies.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_family_policies.py)

Run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_family_policies.py
```

That trains one contribution head and one action head for each family in the initial family library, using the learning-wave state tables as input.

Stage 3 posterior inference entrypoint:

- [inference/infer_player_posteriors.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/infer_player_posteriors.py)

Run:

```bash
python simulation_statistical/algorithmic_latent/inference/infer_player_posteriors.py
```

That scores each learning-wave player trajectory under every family and outputs soft player-family posteriors plus treatment-level family mixtures.

Stage 4 env-mixture entrypoint:

- [inference/fit_env_family_mixture.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_env_family_mixture.py)

Run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_env_family_mixture.py
```

That fits a `CONFIG -> family mixture` environment model from the player posterior tables and writes both learning/validation predictions plus evaluation summaries.

Optional rollout-calibration entrypoint:

- [inference/fit_action_rate_calibration.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_action_rate_calibration.py)

Run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_action_rate_calibration.py
```

That fits a post-hoc punish/reward rate shrinkage artifact for the first simulator runtime.

Stage 5 first simulator runtime:

- [simulator/runtime.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/simulator/runtime.py)

Run:

```bash
python simulation_statistical/micro/run_micro_simulation.py --strategy algorithmic_latent_family
python simulation_statistical/macro/run_macro_simulation.py --strategy algorithmic_latent_family
```

The current runtime samples one family per player from the fitted env-mixture model, uses family-specific contribution heads for contributions, and family-specific per-target action heads for punishment/reward with fixed unit `1`.
