# Forecasting Registry

This directory contains machine-readable lookup tables for the forecasting pipeline.

Files:

- `experiment_registry.csv` / `experiment_registry.json`
- `analysis_registry.csv` / `analysis_registry.json`

## Experiment Registry

One row per run manifest found in `forecasting/metadata/`.

Important fields:

- `run_name`: concrete run identifier used throughout `batch_input/`, `metadata/`, and `results/`
- `is_core_run`: whether this is one of the main four-mode comparison runs
- `raw_variant_name`: the variant string stored in the original manifest
- `canonical_variant`: normalized variant label used for cross-run comparison
- `variant_family`: one of `baseline`, `demographic_only`, `twin_corrected`, `twin_unadjusted`, or `other`
- `profile_source`: where the player-background cards come from
- `persona_assignment_file`: seat-level game assignment source, if the run uses augmentation
- `persona_cards_file`: prompt-facing player card source, if the run uses augmentation
- `vs_human_num_generated_games`: number of generated games that were successfully parsed/evaluated in the per-run treatment comparison
- `vs_human_generated_game_gap`: `total_request_count - vs_human_num_generated_games`
- `notes`: normalized warnings or interpretation notes for the run

## Analysis Registry

One row per analysis family.

Important fields:

- `analysis_id`: stable identifier for the analysis family
- `script`: script that builds the outputs
- `output_location`: where the outputs live
- `analysis_level`: macro, micro, or exploratory
- `unit_of_comparison`: what is being compared
- `metrics_or_outputs`: main reported metrics
- `noise_ceiling_method`: how the human reference ceiling is computed, if any
- `current_scope`: what the currently committed output actually covers
- `risk_or_note`: current ambiguity, limitation, or caveat that should stay visible
