# Forecasting Refactor Plan

Historical note:

- Path examples that mention top-level `forecasting/build_batch_inputs.py` or top-level `forecasting/results/` describe the pre-refactor layout.
- The active PGG pipeline now lives under `forecasting/pgg/`.

## Goal

Separate the pipeline into clearer layers so that:

1. Twin profile construction is a shared upstream artifact pipeline.
2. Profile assignment is a shared reusable module, not duplicated across PGG and non-PGG.
3. Dataset-to-benchmark normalization is isolated from prompt construction.
4. Prompt construction is isolated from batch writing.
5. Evaluation stays benchmark-specific where it needs to diverge.

The target is not a full rewrite. The target is to extract shared code first, keep current run names and artifact paths stable, and avoid breaking existing results.

## Current Pipeline

The current end-to-end flow is:

1. Build deterministic Twin profiles from raw Twin responses.
2. Render those profiles into prompt-facing cards.
3. Sample synthetic profiles for the target benchmark:
   - baseline: none
   - demographic-only: sampled from target-study demographics
   - Twin corrected: sampled from Twin after matching target-study demographic distribution
   - Twin unadjusted: sampled from Twin without demographic correction
4. Build benchmark records and prompts.
5. Write batch inputs and sidecar metadata.
6. Parse batch outputs.
7. Evaluate against human data and noise ceilings.

### Where The Current Logic Lives

#### Shared Twin artifact construction

- [`non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py)
- [`non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py)

#### PGG-specific profile assignment and prompt assembly

- [`forecasting/pgg/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/pgg/build_batch_inputs.py)
- [`forecasting/pgg/profile_sampling/sample_twin_personas_for_pgg_validation.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/pgg/profile_sampling/sample_twin_personas_for_pgg_validation.py)
- [`forecasting/pgg/profile_sampling/sample_pgg_demographic_only_profiles_for_validation.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/pgg/profile_sampling/sample_pgg_demographic_only_profiles_for_validation.py)

#### Non-PGG shared builder

- [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py)

This file currently owns:

- dataset ingestion
- benchmark row normalization
- demographic harmonization
- Twin loading
- demographic-only sampling
- Twin-corrected and Twin-uncorrected sampling
- profile-block rendering
- prompt construction
- batch writing
- metadata writing
- token estimation

#### Benchmark-specific wrappers

- [`forecasting/minority_game_bret_njzas/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/minority_game_bret_njzas/build_batch_inputs.py)
- [`forecasting/longitudinal_trust_game_ht863/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/longitudinal_trust_game_ht863/build_batch_inputs.py)
- [`forecasting/two_stage_trust_punishment_y2hgu/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/two_stage_trust_punishment_y2hgu/build_batch_inputs.py)
- [`forecasting/multi_game_llm_fvk2c/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/multi_game_llm_fvk2c/build_batch_inputs.py)

#### Evaluation

- PGG eval mostly lives at top-level `forecasting/`
- non-PGG eval mostly lives in each benchmark folder

## Main Problems

### 1. Shared profile-assignment logic is split across two systems

Shared Twin artifact construction now lives canonically under `non-PGG_generalization/twin_profiles`, while PGG-specific profile assignment lives under `forecasting/pgg/profile_sampling` and non-PGG profile assignment lives in `forecasting/non_pgg_batch_builder.py`.

That creates:

- duplicated demographic matching logic
- duplicated Twin-card loading logic
- different conventions for assignments and profile blocks

### 2. `non_pgg_batch_builder.py` has too many responsibilities

This file acts as:

- dataset adapter
- sampling engine
- prompt builder
- run writer
- metadata writer

That makes it hard to:

- change one part without touching others
- test benchmark construction independently from prompt generation
- reuse profile logic across additional datasets

### 3. Benchmark code and benchmark artifacts are mixed

Each benchmark folder contains:

- docs
- build script
- batch inputs
- batch outputs
- metadata
- results
- plots
- eval scripts

That is workable for a small number of benchmarks, but the code boundaries are now harder to see.

### 4. The old `task_grounding` name became too broad

The canonical folder is now `non-PGG_generalization/twin_profiles/`. The older `task_grounding/` path is retained only as a compatibility symlink for previously generated artifacts. Conceptually, this layer is shared Twin profile infrastructure, not task-specific grounding.

## Recommended Target Architecture

Keep the benchmark artifact folders. Refactor shared code into explicit layers.

```text
forecasting/
  common/
    profiles/
      twin_artifacts.py
      sampling.py
      render_blocks.py
      shared_notes.py
    runs/
      batch_writer.py
      token_estimation.py
      manifests.py
    schemas.py
  datasets/
    pgg.py
    minority_game_bret_njzas.py
    longitudinal_trust_game_ht863.py
    two_stage_trust_punishment_y2hgu.py
    multi_game_llm_fvk2c.py
  prompts/
    pgg.py
    minority_game_bret_njzas.py
    longitudinal_trust_game_ht863.py
    two_stage_trust_punishment_y2hgu.py
    multi_game_llm_fvk2c.py
  runners/
    build_batch_inputs.py
    parse_outputs.py
    evaluate_outputs.py
```

The benchmark folders then remain mainly as artifact and benchmark-specific analysis locations:

- `forecasting/`
- `forecasting/minority_game_bret_njzas/`
- `forecasting/longitudinal_trust_game_ht863/`
- `forecasting/two_stage_trust_punishment_y2hgu/`
- `forecasting/multi_game_llm_fvk2c/`

## Proposed Responsibility Split

### Layer 1: Shared Profile Artifacts

This layer is upstream and deterministic.

It should own:

- loading Twin extended profiles
- loading Twin rendered cards
- validating Twin card/profile schemas
- building shared prompt-note files

Recommended module:

- `forecasting/common/profiles/twin_artifacts.py`

Initial source material:

- [`non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py)
- [`non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py)

Important note:

- keep the actual build scripts where they are for now
- only extract reusable loaders/helpers into `forecasting/common/profiles`

### Layer 2: Shared Profile Sampling

This layer should own:

- demographic-only sampling
- Twin-corrected sampling
- Twin-uncorrected sampling
- assignment manifests
- reuse-count tracking
- demographic matching fallback logic

Recommended module:

- `forecasting/common/profiles/sampling.py`

Code to extract from:

- [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py)
  - `_sample_demographic_profiles`
  - `_sample_twin_profiles`
  - `_build_candidate_maps`
  - `_choose_pid`
  - `_load_twin_personas`
  - `_load_twin_cards`
- [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py)
  - Twin assignment loading and card loading helpers

This is the highest-priority extraction because it removes the most duplication.

### Layer 3: Dataset Adapters

Each benchmark should expose one canonical adapter that returns a shared bundle-like object:

- `records`
- `units`
- `demographic_source`
- `dataset_key`
- `display_name`
- `matching_fields`

Recommended modules:

- `forecasting/datasets/pgg.py`
- `forecasting/datasets/minority_game_bret_njzas.py`
- `forecasting/datasets/longitudinal_trust_game_ht863.py`
- `forecasting/datasets/two_stage_trust_punishment_y2hgu.py`
- `forecasting/datasets/multi_game_llm_fvk2c.py`

Code to move:

- PGG record-selection and metadata-loading logic from [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py)
- non-PGG `_build_*_bundle` functions from [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py)

The purpose is to make dataset normalization testable independently from prompt construction.

### Layer 4: Prompt Builders

Each benchmark should expose one prompt builder with the same interface:

- input: canonical record row + optional profile block
- output: `system_prompt`, `user_prompt`

Recommended modules:

- `forecasting/prompts/pgg.py`
- `forecasting/prompts/minority_game_bret_njzas.py`
- `forecasting/prompts/longitudinal_trust_game_ht863.py`
- `forecasting/prompts/two_stage_trust_punishment_y2hgu.py`
- `forecasting/prompts/multi_game_llm_fvk2c.py`

Code to move:

- PGG prompt assembly from [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py)
- non-PGG `_build_prompt_*` functions from [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py)

### Layer 5: Shared Run Writer

This layer should own:

- request JSONL writing
- token estimation
- gold-target writing
- request manifest writing
- metadata manifest writing

Recommended modules:

- `forecasting/common/runs/batch_writer.py`
- `forecasting/common/runs/token_estimation.py`
- `forecasting/common/runs/manifests.py`

Code to move:

- `_estimate_input_tokens`
- `_batch_entry`
- `_write_jsonl`
- `_build_run`

from [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py), and analogous pieces from the PGG builder.

## Concrete End State For Your Five Pipeline Parts

Your five-part decomposition should become:

### 1. Twin raw data -> deterministic profile

Canonical home:

- `non-PGG_generalization/twin_profiles/`

Recommended role:

- source artifact generation only

Do not mix benchmark-specific sampling into this layer.

### 2. Random assignment of profiles to target benchmark

Canonical home:

- `forecasting/common/profiles/sampling.py`

This should become the only place that knows:

- demographic-only row resampling
- corrected Twin sampling
- uncorrected Twin sampling

### 3. Experimental setting -> batch input records

Canonical home:

- `forecasting/datasets/*.py`
- `forecasting/common/runs/batch_writer.py`

This layer should not know how profile cards are built internally.

### 4. Connect profile to prediction prompt

Canonical home:

- `forecasting/common/profiles/render_blocks.py`
- `forecasting/prompts/*.py`

Separation:

- `render_blocks.py` turns sampled profile objects into prompt-ready text blocks
- `prompts/*.py` decides where that block is inserted in the benchmark prompt

### 5. Analysis / evaluation

Canonical home:

- keep benchmark-specific eval in each benchmark folder

Reason:

- evaluation metrics legitimately diverge across benchmarks
- forcing one shared evaluator would make the code less clear

## Migration Plan

### Phase 1: Extract shared profile code

Create:

- `forecasting/common/profiles/twin_artifacts.py`
- `forecasting/common/profiles/sampling.py`
- `forecasting/common/profiles/render_blocks.py`
- `forecasting/common/profiles/shared_notes.py`

Move or wrap:

- Twin loading
- profile-card loading
- demographic-only sampling
- Twin sampling
- shared note generation
- profile block rendering

Keep existing builders working by importing the new helpers.

Success criterion:

- no change to run names
- no change to output file paths
- no change to generated prompts for existing seeds

### Phase 2: Extract dataset adapters

Create:

- `forecasting/datasets/*.py`

Move:

- PGG benchmark row selection
- non-PGG `_build_*_bundle` functions

Success criterion:

- both PGG and non-PGG builders consume the same kind of canonical dataset bundle

### Phase 3: Extract prompt builders

Create:

- `forecasting/prompts/*.py`

Move:

- prompt builder functions from both top-level PGG and non-PGG builders

Success criterion:

- prompt construction becomes testable independently from sampling and benchmark row generation

### Phase 4: Extract shared run writer

Create:

- `forecasting/common/runs/batch_writer.py`

Move:

- request JSONL writing
- token estimation
- metadata writing

Success criterion:

- one run-writing path for PGG and non-PGG

### Phase 5: Thin the entry points

At the end:

- top-level PGG `build_batch_inputs.py` becomes a thin orchestration script
- `non_pgg_batch_builder.py` disappears or becomes an orchestration script
- per-benchmark wrappers remain thin

## What Not To Refactor Yet

Do not change these until the shared pipeline extraction is stable:

1. benchmark artifact directory layout
2. result file naming
3. run naming conventions
4. evaluation folder locations
5. existing plotting scripts

Those are downstream consumers. Breaking them early creates unnecessary churn.

## Minimal First Refactor

If time is limited, do only this first:

1. extract shared profile loading and sampling
2. extract profile-block rendering
3. leave dataset adapters and prompt builders where they are

That gets most of the clarity benefit with low migration risk.

## Recommended File Moves

### Move into `forecasting/common/profiles`

From [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py):

- `_load_twin_cards`
- `_load_twin_personas`
- `_build_candidate_maps`
- `_choose_pid`
- `_sample_demographic_profiles`
- `_sample_twin_profiles`
- `_common_shared_note_lines`
- `_dataset_specific_caveats`
- `_write_shared_notes_file`
- `_render_demographic_profile_block`
- `_render_twin_profile_block`

From [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py):

- `_load_twin_game_assignments`
- `_load_twin_profile_cards`
- Twin-assignment path resolution helpers

### Move into `forecasting/datasets`

From [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py):

- `_build_minority_bundle`
- `_build_longitudinal_bundle`
- `_build_two_stage_bundle`
- `_build_multi_game_bundle`
- shared demographic harmonizers used only for dataset normalization

From [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py):

- game loading and selected-game bundle creation helpers

### Move into `forecasting/prompts`

From [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py):

- `_build_prompt_minority`
- `_build_prompt_longitudinal`
- `_build_prompt_two_stage`
- `_build_prompt_multi_game`

From [`forecasting/build_batch_inputs.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/build_batch_inputs.py):

- PGG prompt templates and transcript prompt assembly

### Move into `forecasting/common/runs`

From [`forecasting/non_pgg_batch_builder.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/non_pgg_batch_builder.py):

- `_estimate_input_tokens`
- `_batch_entry`
- `_write_jsonl`
- `_build_run`

## Suggested Order Of Work

1. extract `common/profiles`
2. make both PGG and non-PGG builders import it
3. extract `datasets`
4. extract `prompts`
5. extract `common/runs`
6. rename or retire `non_pgg_batch_builder.py`

## Practical End State

At the end, the repo should answer these questions cleanly:

- Where are Twin-derived artifacts built?
  - `non-PGG_generalization/twin_profiles`

- Where is profile sampling defined?
  - `forecasting/common/profiles`

- Where is a dataset converted into benchmark rows?
  - `forecasting/datasets`

- Where is prompt wording defined?
  - `forecasting/prompts`

- Where are run JSONLs and metadata written?
  - `forecasting/common/runs`

- Where are benchmark-specific evaluations and plots?
  - benchmark folders under `forecasting/`

That is the architecture I would move toward.
