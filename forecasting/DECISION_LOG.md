# Decision Log

This file records dated benchmark decisions that affect prompt construction, sampling, parsing, or evaluation.

Use this file for:

- changes to benchmark units
- changes to default conditions
- evaluation-method changes
- important interpretation decisions

Use [CORE_NARRATIVE.md](CORE_NARRATIVE.md) for the stable project story.

## 2026-04-09

### Four-Condition Benchmark Is The Default

The default benchmark comparison is now:

1. `baseline`
2. `demographic_only_row_resampled_seed_0`
3. `twin_sampled_seed_0`
4. `twin_sampled_unadjusted_seed_0`

This replaces the earlier effective emphasis on only baseline vs corrected Twin.

### Demographics-Are-Minimal Interpretation

The current cross-benchmark interpretation is:

- Twin-sampled augmentation helps
- demographic-only augmentation adds little relative to baseline
- demographic correction of Twin sampling may matter less than the existence of richer Twin-derived behavioral grounding itself

This is a working empirical summary and should be updated if later benchmarks contradict it.

### Distributional Metrics Are Primary

The preferred evaluation target is distributional alignment, not point prediction.

Operationally:

- Wasserstein-style and related distributional distances are primary where appropriate
- row-level exact match, MAE, and similar metrics remain secondary diagnostics
- human-vs-human noise ceilings are the main empirical reference standard

### Sampling Rule: Sample Rows, Not Within-Row Content

For all forecasting benchmarks:

- sample whole experimental rows or session rows
- do not independently sample or reshuffle nested content inside those rows

Rationale:

- preserves the integrity of the experimental unit
- matches the PGG logic where repeated experiments under the same design differ by who was recruited
- avoids breaking within-session dependence structure

### Multi-Game Delegation Benchmark Unit

For `forecasting/multi_game_llm_fvk2c/`:

- one row / one LLM request = one subject-level session under one treatment arm
- top-level design = `TreatmentCode`
- nested evaluation cells after parsing = `TreatmentCode x Scenario x Case`

Important implication:

- the older scenario-row description is not the correct top-level unit for sampling or request construction

### Multi-Game Downsampling Rule

The current multi-game batch-input default is:

- `50` subject-level session rows per treatment arm
- `6` treatment arms
- `300` requests total per run

This sampling happens at the session-row level and preserves all nested scenarios inside each sampled row.

### Two-Stage Benchmark Downsampling Rule

The current two-stage batch-input default is:

- `50` records per treatment cell
- `20` treatment cells
- `1,000` requests total per run

### Two-Stage Benchmark Output Prioritization

For `forecasting/two_stage_trust_punishment_y2hgu/`:

- treatment-level distribution-distance outputs are primary
- row-level exact-match or absolute-error outputs are secondary

### Multi-Game Noise Ceiling Rule

For `forecasting/multi_game_llm_fvk2c/`:

- bootstrap subject-session rows within treatment arm
- expand nested scenarios only after resampling
- do not bootstrap exploded scenario rows independently

This preserves the correct repeated-measures structure.

## Maintenance Rule

Append new entries when a benchmark-wide decision changes.

Do not rewrite older entries unless they were incorrect; instead add a new dated entry that supersedes the old one.
