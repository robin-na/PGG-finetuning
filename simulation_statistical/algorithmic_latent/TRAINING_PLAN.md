# Training Plan

This document specifies how the algorithmic-latent redesign should be fit in a data-driven way.

## High-Level Model

For player `i` in game `g` and round `t`:

- `f_i` = latent algorithm family
- `theta_i` = latent parameter vector within family
- `s_i,t` = visible structured state

The model factorization is:

```text
p(f_i | CONFIG_g)
p(theta_i | f_i, CONFIG_g)
p(contribution_i,t | s_i,t, f_i, theta_i)
p(actions_i,t | s_i,t, contribution_t, f_i, theta_i)
```

The first implementation should simplify this to:

- shared family parameters instead of fully player-specific `theta_i`
- soft family assignments for players
- optional treatment-level offsets

## Data Inputs

Primary inputs:

- learning-wave player-round rows
- game-level `CONFIG`
- player-level identifiers and avatars
- validation-wave data only for held-out evaluation

Optional later inputs:

- chat summaries
- LLM-generated strategy summaries
- treatment-level literature priors

## Training Stages

### Stage 1: Build Structured State

Create one contribution-stage row and one action-stage row per decision.

Each row should include:

- visible `CONFIG`
- visible history features
- current-round target features for sanctioning
- observed human action label

This stage should be deterministic and reusable across models.

### Stage 2: Fit Family-Level Policies

For each family:

- contribution head:
  - multinomial over legal contribution bins
- punish head:
  - Bernoulli or multinomial over visible targets
- reward head:
  - Bernoulli or multinomial over visible targets

First fitting option:

- regularized generalized linear models

Why:

- interpretable coefficients
- easy likelihood computation
- stable baselines before neural variants

### Stage 3: Infer Player Family Posteriors

For each player:

- compute likelihood of their learning-wave trajectory under each family
- combine with family prior
- normalize to get posterior family weights

```text
p(f_i | trajectory_i) proportional to p(trajectory_i | f_i) * p(f_i)
```

This gives:

- soft family assignment, not a brittle hard label
- a human-interpretable latent state

### Stage 4: Learn `CONFIG -> Family Mixture`

Train a treatment/env model that predicts family mixture from `CONFIG`.

Current first implementation:

- Dirichlet regression on game-level soft family mixtures

Later options:

- Dirichlet-multinomial regression
- hierarchical Bayesian mixture model

Important:

- `CONFIG` should be allowed to affect both family composition and family parameters later
- do not assume design effects act only through mixture shift

### Stage 5: Rollout Calibration

After fitting one-step policies, validate the full simulator in closed-loop rollout.

If needed:

- add post-hoc calibration on contribution bin probabilities
- add post-hoc calibration on punish/reward rates
- calibrate by family and treatment, not only globally

## Where the LLM Enters

The LLM should not produce the final fitted parameters.

Allowed LLM roles:

- propose family candidates
- suggest which state variables should matter for a family
- provide priors over coefficient signs or relative strength
- summarize player histories into candidate family priors
- predict how new `CONFIG`s should shift family prevalence or parameter signs

All of those outputs should be used as priors or features, not as the final simulator state.

## Calibration Strategy

Use a shrinkage framework:

```text
parameter = data_estimate + lambda * llm_prior_signal
```

or

```text
posterior proportional to likelihood * prior_from_llm
```

Where `lambda` or prior strength is fit on held-out environments.

If the LLM prior does not help held-out performance, it should shrink toward zero automatically.

## First Implementation Scope

Keep the first build narrow:

- no free-form code generation
- no giant family library
- no player-specific neural hidden states
- no chat yet

Build only:

- shared family coefficients
- soft player-family posteriors
- `CONFIG -> family mixture`
- structured rollout simulator

That is enough to test whether algorithmic latent state is a better abstraction than anonymous clusters.
