# Minority Game + BRET Analysis Overview

## Purpose

This note describes a proposed evaluation layer for from-scratch forecasting on the minority-game plus BRET dataset.

The main analysis layers should be:

1. per-treatment sequence alignment
2. config-level mean alignment
3. within-treatment distribution alignment
4. model-vs-human noise-ceiling comparison

## Human Reference Set

Recommended human comparison set:

- finished participants only
- one row per participant from the main wide file
- optional demographic join from the Prolific export

The main config key is:

- `participant.in_deception`

That gives two repeated design cells, which is enough for a PGG-style human-vs-human ceiling even though it is much smaller than the PGG treatment grid.

## Per-Run Comparison

For each generated participant and each human participant, the evaluator should derive:

- round-by-round `A` or `B` choices
- mean `A` rate
- switch round from `A` to `B`
- total potential payoff
- distance from the optimal `AAAAABBBBB*` family

If BRET is included, it should be scored separately with:

- mean boxes collected
- SD of boxes collected
- Wasserstein distance over the boxes-collected distribution

## Macro Pointwise Alignment

The cleanest macro view is:

- compare generated and human `A` rates for each round within each deception cell
- aggregate over the two treatment cells

Primary macro scores:

- RMSE of config-round means
- MAE of config-round means
- average payoff error by deception cell

Interpretation:

- lower is better
- this directly tests whether the model recovers the human transition from early `A` behavior toward later `B` behavior

## Micro Distribution Alignment

A model can get round means right while collapsing heterogeneity in switching behavior.

A micro layer should therefore compare generated and human distributions for:

- total number of `A` choices
- switch round
- total potential payoff
- longest initial `A` streak
- boxes collected in BRET, if included

The core score family should be 1-Wasserstein distance within deception cell.

## Human Noise Ceiling

This dataset supports a close analogue of the PGG noise ceiling.

Recommended procedure:

1. split human comparisons within each `participant.in_deception` cell
2. bootstrap one pseudo-model sample and one pseudo-human sample from that same cell
3. score them with the same metrics used for model evaluation

This is a clean ceiling because:

- the environment is repeated many times under the same rules
- the config definition is explicit
- each participant contributes one independent sequence

The only limitation relative to PGG is the small number of config cells.

## Important Caveats

### 1. The benchmark is cleaner if it uses the true transition rule

The original experiment displayed simulated group feedback, but those displayed counts are not the cleanest observable state for a from-scratch benchmark.

### 2. BRET should probably remain secondary at first

The bonus game is the main repeated social task. BRET is valuable, but it is a separate risk measure rather than the core sequential game.

### 3. Deception effects are small

That is substantively interesting, but it also means the main variation may come more from sequence dynamics than from between-condition mean shifts.
