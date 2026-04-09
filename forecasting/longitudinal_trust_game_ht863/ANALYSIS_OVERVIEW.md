# Longitudinal Trust Game Analysis Overview

## Purpose

This note describes a proposed evaluation layer for from-scratch forecasting on the longitudinal trust-game dataset.

The main analysis layers should be:

1. cell-level mean alignment
2. trajectory-level alignment across days
3. participant-level distribution alignment
4. human-vs-human reference ceilings

## Human Reference Set

Recommended human comparison set:

- participants who completed all `10` sessions
- raw trial-level files as the canonical source of ratings
- `matrix_mean.csv` as a convenience summary rather than the only source of truth

Because the same participant contributes repeated measurements, all bootstrap and uncertainty calculations should resample whole participants, not individual trial rows.

## Per-Run Comparison

The most natural base comparison is between generated and human means for each:

- `Day x cooperation_probability x stake` cell
- `Day` mean

Suggested output families:

- cell mean alignment tables
- day-trajectory alignment tables
- participant-summary distribution tables

Useful participant summaries include:

- overall mean willingness to play
- day `1` to day `10` slope
- sensitivity to cooperation probability
- sensitivity to stake
- within-person response variance

## Macro Pointwise Alignment

The cleanest macro analogue to the PGG treatment-level plots is to treat each `Day x trial-type` combination as a config cell.

That gives `10 x 16 = 160` repeated cells.

Primary macro scores:

- RMSE of generated vs human cell means
- MAE of generated vs human day means
- correlation of generated vs human cell-level response surfaces

Interpretation:

- lower RMSE and MAE are better
- higher correlation means the model recovered the human response surface over stake and partner reliability

## Micro Distribution Alignment

Macro cell means can look reasonable even if the model collapses participant heterogeneity.

A micro layer should therefore compare generated and human distributions for:

- participant mean rating
- participant day slope
- participant cooperation-probability sensitivity
- participant stake sensitivity
- participant within-session variability

The natural score family is within-cell or pooled 1-Wasserstein distance over these summaries.

## Human Noise Ceiling

This dataset only partially matches the PGG noise-ceiling setup.

There is not a rich grid of repeated experimental treatments in the same way as PGG. Instead, the closest analogue is:

- resample participants with replacement
- compare human subset A vs human subset B on the same `Day x cooperation_probability x stake` cells

That gives a valid empirical human-vs-human reference, but it is weaker conceptually than the PGG ceiling because:

- everyone faces the same repeated design
- the main structure is within-person over time rather than across many distinct treatment cells

Recommended ceiling views:

1. cellwise human-vs-human RMSE over the `160` cells
2. participant-summary Wasserstein distances after resampling whole schedules

## Important Caveats

### 1. This is not a transcript benchmark

The output is a repeated rating panel, not a multi-player sequence of observed actions.

### 2. Demographic overlap with Twin is limited

If gender is not recovered cleanly into the final benchmark table, the corrected Twin mode may only be matchable on age.

### 3. Prompt length can become large

A full `160`-rating output is still manageable, but it is much more structured than the one-shot games in the other non-PGG datasets.
