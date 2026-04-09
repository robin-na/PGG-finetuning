# Two-Stage Trust / Punishment / Helping Analysis Overview

## Purpose

This note describes a proposed evaluation layer for from-scratch forecasting on the two-stage trustworthiness dataset.

The analysis should be role-specific but parallel in structure to the PGG benchmark:

1. per-config mean alignment
2. macro alignment across design cells
3. micro response-vector alignment
4. human-vs-human reference ceilings

## Human Reference Set

Recommended human comparison set:

- `E1`, `E2`, `E4`, `E5` files for the check-based experiments
- `E3a` and `E3b` for the decision-time experiment

The main design-cell label should be:

- `experiment_id x role x condition`

This is the cleanest analogue to `CONFIG_treatmentName` in PGG.

## Player A Evaluation

Primary Player A summaries:

- check rate or fast-rate
- prosocial-action rate
- mean return percentage
- joint pattern frequencies such as unchecked-and-helped or checked-and-did-not-punish

Macro scores:

- RMSE of config-level means
- MAE of config-level means

Micro scores:

- Wasserstein distance over return percentages
- Wasserstein or total-variation style comparisons over the joint decision patterns

## Player B Evaluation

Player B uses a conditional strategy method.

Primary summaries:

- mean send amount in each scenario cue
- premium for uncalculating versus calculating prosociality
- premium for prosocial action versus no action

Macro scores:

- RMSE of scenario-specific config means
- MAE of scenario-specific config means

Micro scores:

- Wasserstein distance over conditional-send vectors
- distance over derived trust-premium summaries

## Macro Pointwise Alignment

The macro view should aggregate over all `experiment x role x condition` cells.

Because the cue structure differs across experiments, this layer should compare only like-with-like:

- help-cost cells against help-cost cells
- punish-cost cells against punish-cost cells
- time-cue cells against time-cue cells
- help-impact cells against help-impact cells
- punish-impact cells against punish-impact cells

The key question is whether the model recovers the directional response surface across observable versus hidden conditions and across cue types.

## Human Noise Ceiling

This dataset does support a meaningful PGG-style human-vs-human reference.

Recommended procedure:

1. stratify by `experiment_id x role x condition`
2. bootstrap independent pseudo-model and pseudo-human samples within that exact cell
3. score them with the same mean-alignment and vector-alignment metrics used for model runs

This is conceptually cleaner than the longitudinal trust dataset because the experimental design has explicit repeated cells with hundreds of observations each.

## Important Caveats

### 1. One unified parser is not enough

Player A and Player B have different response schemas, and `E3` differs from the checking experiments.

### 2. Decision-time should be bucketed, not treated as raw latency

The theoretical signal is fast versus slow. Predicting exact seconds would add noise without improving interpretability.

### 3. This is not a rollout task

The benchmark is closer to structured stage-conditioned forecasting than to sequential transcript generation.
