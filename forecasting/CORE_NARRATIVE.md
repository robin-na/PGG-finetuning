# Core Narrative

This file is the canonical internal source of truth for what the `forecasting/` benchmark is trying to do.

Use this document for:

- the stable project framing
- the main benchmark conditions
- the current headline findings
- the shared interpretation across datasets

Do not use the paper draft alone as the working source of truth. The paper will lag the codebase. If the benchmark changes, update this file first and then update dataset-specific docs and the draft.

Related files:

- [README.md](README.md)
- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)
- [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md)
- [DECISION_LOG.md](DECISION_LOG.md)

## Project Goal

The goal is to test whether LLMs can simulate human behavior in strategic and economic experiments by combining:

- a description of the experimental environment
- sampled participant context
- no in-domain behavioral prefix from the target experiment

The core use case is transportability:

- we often have one dataset with rich participant profiles
- and another dataset with the institution or game we want to predict
- but not both in the same study

The working question is whether LLMs can use the first kind of data to improve predictions in the second kind of setting.

## Core Hypothesis

Providing sampled participant profiles from the Twin dataset helps LLMs make better from-scratch predictions of human behavior.

More specifically:

1. `baseline` gives the model only the task, rules, and environment.
2. `demographic_only_row_resampled_seed_0` adds only lightweight demographic context sampled from the target population.
3. `twin_sampled_seed_0` adds richer Twin-derived participant profiles, sampled and then corrected to match the target population over shared demographics.
4. `twin_sampled_unadjusted_seed_0` adds the same kind of Twin-derived profiles without demographic correction.

The current benchmark is therefore not just “does profile augmentation help?” but:

- does demographics alone help?
- does richer Twin-derived context help beyond demographics?
- does demographic correction of Twin sampling matter?

## Current Headline Pattern

The repeated pattern so far is:

- Twin-sampled augmentation helps meaningfully relative to baseline.
- Demographic-only augmentation adds little or almost nothing relative to baseline.
- The gap between corrected and unadjusted Twin sampling is often small relative to the gap between either Twin condition and the non-Twin conditions.

That means the practical value seems to come mainly from richer behavioral or dispositional grounding, not from demographics alone.

## Beyond PGG

The original benchmark emphasis was PGG forecasting, but the project now tests the same idea across multiple games.

Current datasets in scope:

- PGG forecasting
- Minority game with BRET
- Longitudinal trust game
- Two-stage trust / punishment / helping
- Multi-game battery with LLM delegation

The purpose of expanding beyond PGG is to test whether the benefit of Twin-derived profile sampling is specific to one cooperative-game setting or reflects a broader transportability property.

## Benchmark Conditions

The four core conditions should be treated as the default comparison set across datasets:

1. `baseline`
2. `demographic_only_row_resampled_seed_0`
3. `twin_sampled_seed_0`
4. `twin_sampled_unadjusted_seed_0`

Unless there is a dataset-specific reason not to, new forecasting benchmarks should expose these same four conditions so comparisons remain consistent.

## Critical Unit Definition

The most important shared rule is:

- one LLM request should predict one full experimental unit

That means:

- if the experiment is a multi-player game, one request should predict the whole game
- if the experiment is a subject-level session with multiple nested scenario blocks, one request should predict the whole session
- if a design is repeated many times with different recruited participants, those repeated rows are separate experimental units under the same design

Equally important:

- sample rows, not content within rows

This is the direct analogue of how the PGG benchmark is treated:

- a design is repeated multiple times
- those repeated experiments differ because different people are recruited into the same design
- simulation and evaluation should preserve the integrity of each experimental unit

So in all datasets:

- top-level sampling happens at the experiment row or session row
- nested content inside that row stays attached to the row
- evaluation can explode nested structures after parsing, but not before sampling

## What Counts As A Design

The word `design` should always mean the repeated experimental environment, not the exploded scoring cell.

Examples:

- in PGG, the design is the treatment configuration, and multiple human games are run under that same design
- in the multi-game delegation benchmark, the design is the treatment arm (`TRP`, `TRU`, `TDP`, `TDU`, `ODP`, `ODU`), while `Scenario x Case` is nested inside the session for scoring
- in the two-stage benchmark, the design is the repeated condition cell such as `experiment x role x visibility`

This distinction is critical because the human noise ceiling and the model sampling logic depend on it.

## Evaluation Philosophy

The current evaluation philosophy is:

- prioritize distributional alignment over point prediction
- use human-vs-human noise ceilings as the main empirical reference
- preserve the real sampling unit when bootstrapping uncertainty

In practice this means:

- Wasserstein-style or other distributional distances should be primary
- row-level exact match or MAE can still be reported, but as secondary diagnostics
- when participants contribute multiple nested outputs inside a session, bootstrap the session row, not the exploded nested rows

## Relationship To The Paper Draft

The paper draft in `/Users/robinna/Downloads/llm_transportability.pdf` captures the original transportability framing well:

- borrowed populations
- separate profile and institutional datasets
- no in-domain calibration on the target game
- Twin-style profile augmentation as an integration layer

What has changed since that draft:

1. the benchmark comparison now consistently uses four conditions rather than only baseline vs Twin
2. the current empirical takeaway is that demographics alone provide little value, while Twin-style behavioral grounding does the real work
3. the project now includes multiple non-PGG games to test broader transportability

The draft should therefore be treated as an external writeup of the core idea, while this file tracks the current benchmark definition and live results framing.

## How To Maintain This File

Update this file when:

- the project-level framing changes
- a benchmark-wide finding becomes stable enough to state as a core result
- the default interpretation of the four conditions changes
- the definition of the prediction unit or design unit changes

Do not use this file as a dated changelog. Put dated implementation or benchmark decisions in [DECISION_LOG.md](DECISION_LOG.md).
