# Inference Plan

This folder is reserved for inference and calibration code for the algorithmic-latent simulator.

Planned components:

- `build_state_table.py`
  - create contribution-stage and action-stage structured rows
- `fit_family_policies.py`
  - fit family-level contribution and sanction/reward heads
- `infer_player_posteriors.py`
  - infer soft family assignments from player trajectories
- `fit_env_family_mixture.py`
  - learn `CONFIG -> family mixture`
- `fit_action_rate_calibration.py`
  - calibrate row-level punish/reward rates for rollout
- `calibrate_llm_priors.py`
  - learn how strongly to trust LLM-proposed priors

First implementation target:

- regularized GLM or multinomial-logit family policies
- likelihood-based player-family posteriors
- simplex-preserving env mixture model

This should stay statistically transparent before introducing richer neural inference.

## Implemented Stage 1

Current implemented entrypoint:

- [build_state_table.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/build_state_table.py)

It builds:

- contribution-stage rows:
  - one row per player-round
  - visible history over rounds `1..T-1`
  - cumulative structured summaries, not just last-round state
  - actual observed contribution label
- action-stage rows:
  - one row per focal-target pair in sanction/reward-enabled games
  - current-round contribution context for all peers
  - visible retaliation/reciprocity indicators only when IDs are shown
  - observed action labels `none / punish / reward / both`

Timing assumptions:

- contribution is chosen before any round-`T` contribution or sanction identity is revealed
- sanction/reward is chosen after round-`T` contributions are visible
- current-round sanction/reward identities are not visible before the focal player acts

Example:

```bash
python simulation_statistical/algorithmic_latent/inference/build_state_table.py --wave learning_wave
python simulation_statistical/algorithmic_latent/inference/build_state_table.py --wave validation_wave
```

## Implemented Stage 2

Current family-policy trainer:

- [fit_family_policies.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_family_policies.py)

It fits:

- one contribution classifier per family
- one action classifier per family

using:

- family-specific feature subsets from [family_definitions.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/family_definitions.py)
- learning-wave contribution/action state tables
- regularized probabilistic linear models

Default run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_family_policies.py
```

Default outputs:

- `simulation_statistical/algorithmic_latent/artifacts/models/family_policy_bundle.pkl`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/family_policy_train_summary.csv`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/family_policy_train_summary.json`

## Implemented Stage 3

Current posterior-inference entrypoint:

- [infer_player_posteriors.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/infer_player_posteriors.py)

It:

- scores each player trajectory under every family policy
- balances contribution-stage and action-stage likelihoods so action-edge counts do not swamp contribution evidence
- outputs soft player-family posteriors
- aggregates those posteriors into treatment-level family mixtures

Default run:

```bash
python simulation_statistical/algorithmic_latent/inference/infer_player_posteriors.py
```

Default outputs:

- `simulation_statistical/algorithmic_latent/artifacts/outputs/learning_wave_player_family_posteriors.parquet`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/learning_wave_treatment_family_mixture.csv`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/learning_wave_player_family_posteriors_summary.json`

## Implemented Stage 4

Current env-mixture trainer:

- [fit_env_family_mixture.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_env_family_mixture.py)

It:

- aggregates player posteriors into game-level family mixtures
- joins those mixtures to game-level `CONFIG`
- fits a Dirichlet regression from `CONFIG -> family mixture`
- evaluates on both game-level and treatment-level validation mixtures

This stage uses a Dirichlet regressor rather than hard-label multinomial classification so the predicted output stays on the family-mixture simplex.

Default run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_env_family_mixture.py
```

Default outputs:

- `simulation_statistical/algorithmic_latent/artifacts/models/env_family_mixture_model.pkl`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/learning_wave_env_family_predictions.csv`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/validation_wave_env_family_predictions.csv`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/env_family_mixture_eval_summary.csv`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/env_family_mixture_eval_summary.json`

## Implemented Action-Rate Calibration

Current rollout-calibration entrypoint:

- [fit_action_rate_calibration.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/inference/fit_action_rate_calibration.py)

It:

- scores the learning-wave action-edge table under the family policy bundle
- mixes those edge probabilities using the inferred player-family posteriors
- fits global punish/reward shrinkage factors so player-round any-punish / any-reward rates match the learning data

Default run:

```bash
python simulation_statistical/algorithmic_latent/inference/fit_action_rate_calibration.py
```

Default outputs:

- `simulation_statistical/algorithmic_latent/artifacts/outputs/action_rate_calibration.json`
- `simulation_statistical/algorithmic_latent/artifacts/outputs/action_rate_calibration_summary.csv`
