# Multi-Game Battery with LLM Delegation Analysis Overview

## Purpose

This note describes a proposed evaluation layer for from-scratch forecasting on the multi-game delegation dataset.

The main analysis layers should be:

1. per-config decision-mean alignment
2. macro alignment across treatment-scenario cells
3. within-cell distribution alignment
4. human-vs-human reference ceilings with subject-level clustering

## Human Reference Set

Recommended human comparison set:

- the subject-level experiment-session table
- each row is one `SubjectID x TreatmentCode` session with nested scenario outputs

Use two levels of grouping:

- top-level design for sampling and clustered uncertainty: `TreatmentCode`
- nested evaluation cell after exploding parsed outputs: `TreatmentCode x Scenario x Case`

So there are:

- `6` repeated designs at the experiment-session level
- `16` nested evaluation cells after expansion

## Per-Run Comparison

Primary decision targets:

- `UGProposer`
- `Responder`
- `Sender`
- `Receiver`
- `PD`
- `SH`
- `C`

Optional secondary targets:

- delegation indicators where defined

Per-run outputs should compare generated and human means within each nested config cell for each game decision, while preserving subject clustering at the session level.

## Macro Pointwise Alignment

The macro view should ask whether the model gets the nested config-to-config response surface right across the `16` cells.

Primary macro score families:

- RMSE of config means for each game variable
- MAE of config means for each game variable
- pooled categorical divergence for the coordination-game choice distribution

This is the strongest non-PGG analogue to the PGG treatment-mean benchmark because the same scenario definitions repeat many times.

## Micro Distribution Alignment

Macro config means can hide unrealistic within-cell heterogeneity.

A micro layer should therefore compare generated and human within-cell distributions for:

- UG offers
- responder thresholds
- trust-game return amounts
- cooperation rates in PD and SH
- coordination-game choice frequencies
- delegation rates, when included

For numeric decisions, the core score should be 1-Wasserstein distance. For categorical choices such as the coordination game, the evaluator should use a distributional distance over the five-option choice vector.

## Human Noise Ceiling

This dataset supports a strong PGG-style human-vs-human ceiling, but it needs one change relative to PGG:

- resampling should happen at the `SubjectID` level, not at the scenario-row level

Recommended procedure:

1. bootstrap subject-session rows with replacement within each treatment arm
2. expand those sampled rows back into their nested scenario outputs
3. compare pseudo-model and pseudo-human summaries within each `TreatmentCode x Scenario x Case` cell

This preserves the repeated-measures structure and avoids overstating the effective sample size.

## Important Caveats

### 1. Nested scenarios are not independent

The same subject contributes multiple scenario outputs inside one session, so any bootstrap or standard-error calculation must respect clustering at the subject/session level.

### 2. Delegation fields are treatment-dependent

Some delegation variables are meaningful only in voluntary-delegation settings, so the evaluation should keep "core game decisions" and "delegation decisions" as separate score families.

### 3. The coordination game is categorical

It should not be scored as if it were an interval-scale numeric target unless that coding choice is made explicit.
