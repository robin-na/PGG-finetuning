# Twin-to-PGG Validation Persona Sampling

## Purpose

This sampler builds seat-level persona assignments for the PGG validation wave. It supports two modes:

- `adjusted_joint_demographics`: match the **aggregate** validation-wave PGG demographic distribution
- `unadjusted_random`: draw a random baseline with no demographic correction

It does **not** attempt to map source personas to individual PGG participants in either mode.

The intended use is:

- preserve the actual validation-wave game/config structure
- preserve the configured number of seats per game
- support both corrected and uncorrected persona sampling before using these personas in PGG simulations

## Sampling Modes

### `adjusted_joint_demographics`

This is the corrected sampler. It targets the joint distribution over:

- `age_bracket`
- `education_harmonized`
- `sex_male_female`

### `unadjusted_random`

This is the baseline sampler. It:

- uses the same valid-start games
- uses the same configured seat counts
- allows reuse across games
- disallows within-game reuse unless forced
- samples uniformly from the full source persona pool

It does **not** allocate demographic quotas and does **not** constrain sampling by age, education, or male/female.

## Target Distribution

The corrected sampler targets the joint distribution over:

- `age_bracket`
- `education_harmonized`
- `sex_male_female`

These dimensions are matched jointly, not marginally. The unadjusted baseline does not use target dimensions.

Implementation:

- PGG source file: `PGG-finetuning/data/raw_data/validation_wave/player-inputs.csv`
- valid games file: `PGG-finetuning/data/processed_data/df_analysis_val.csv`
- Twin persona pool: `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_extended_profiles/twin_extended_profiles.jsonl`
- prompt-facing card pool: `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`

## Harmonization

### Age

PGG numeric age is bucketed as:

- `< 30` -> `18-29`
- `30-49`
- `50-64`
- `65+`

Twin already stores age in matching bracket form.

### Education

Completed-degree harmonization:

- PGG `high-school` -> `high school`
- PGG `bachelor` -> `college/postsecondary`
- PGG `other` -> `college/postsecondary`
- PGG `master` -> `postgraduate`

Twin harmonization:

- `Less than high school` -> `high school`
- `High school graduate` -> `high school`
- `Some college, no degree` -> `high school`
- `Associate's degree` -> `college/postsecondary`
- `College graduate/some postgrad` -> `college/postsecondary`
- `Postgraduate` -> `postgraduate`

### Sex / Gender

Twin only provides `sex assigned at birth` with `Male/Female`.

PGG stores open-text `data.gender`, which is normalized using the same parser as the demographic preprocessing pipeline:

- `man` -> `male`
- `woman` -> `female`
- `non_binary` and `unknown` are not representable in Twin

Because of that mismatch, the sex-targeted quota estimation uses only the subset of valid-game PGG rows that have:

- complete age
- complete education
- normalized `male/female`

Excluded rows are explicitly counted in the outputs.

## Sampling Unit

The matching unit is the **population distribution**, not the individual PGG participant.

What the sampler does:

1. identify valid-start validation games
2. expand each game to its configured seat count
3. estimate the observed PGG joint distribution over `age x education x male/female`
4. convert that distribution to integer seat quotas using largest-remainder rounding
5. assign target demographic cells to seats
6. sample Twin personas inside each exact target cell

What it does not do:

- it does not use a PGG participant's own age/education/sex to pick that seat's Twin persona
- it does not impute missing PGG demographics per participant for seat matching

## Replacement Policy

Sampling is **with replacement across games** and **without replacement within a game** unless forced.

Concretely:

- a Twin persona may appear in multiple different games
- the same Twin persona is not reused inside the same game unless the candidate pool is exhausted

In the current corrected `seed_0` run:

- `within_game_reuse_count = 0`
- global replacement is active

Global replacement is necessary because the validation-wave seat count exceeds the Twin persona pool size.

## Candidate Selection

Within a target cell, candidate Twin personas are sampled with inverse-reuse weighting:

- weight = `1 / (1 + reuse_count)`

This reduces repeated use of the same Twin persona while still allowing reuse across games.

Fallback order:

1. exact `age x education x sex`
2. exact `age x education`
3. exact `age x sex`
4. exact `education x sex`
5. age only
6. education only
7. sex only
8. full pool

In the current corrected `seed_0` run, every seat matched at the strictest level:

- `exact_joint3_no_within_game = 3583`

In the current unadjusted `seed_0` run:

- `full_pool_random_no_within_game = 3583`

## Card Mode

The sampler currently points to the `pgg_prompt_min` profile cards:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`

This mode is meant for token-efficient prompt augmentation:

- universal methodological caveats and cue definitions are written once in `shared_prompt_notes.md`
- per-player cards mostly keep values, anchors, and player-specific limits only

The fuller `pgg_prompt` mode also exists when more inline explanation is useful. The sampler does not currently point to `compact`, `standard`, `audit`, or the fuller `pgg_prompt`.

## Path Format

Assignment artifacts are written with repo-rooted paths, not local machine-specific absolute paths.

Format:

- `PGG-finetuning/...`

This applies to:

- `summary.json`
- `manifest.json`
- `seat_assignments.jsonl`

and to the embedded Twin profile/card references inside each seat assignment row.

## Validation-Wave Seat Count Note

One valid-start game is internally inconsistent:

- gameId: `pJFEFMc5YWW7XyLuN`
- `CONFIG_playerCount = 19`
- listed `playerIds = 18`

Per explicit user instruction, the sampler uses the configured count and creates `19` seats for that game. The extra seat has no roster-linked `pgg_roster_playerId`.

## Current Seed-0 Outputs

### Corrected sampler

Base directory:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_to_pgg_validation_persona_sampling/seed_0`

Main files:

- `seat_assignments.csv`
- `seat_assignments.jsonl`
- `game_assignments.jsonl`
- `summary.json`
- `manifest.json`
- `sampling_notes.md`
- `observed_pgg_distribution_targetable_rows.csv`
- `target_distribution_allocated.csv`
- `assigned_twin_distribution.csv`
- `distribution_checks.csv`
- `twin_usage_summary.csv`

Demographic validation files:

- `demographic_validation/sampled_twin_vs_pgg_demographics.csv`
- `demographic_validation/validation_metrics.csv`
- `demographic_validation/validation_meta.json`

### Unadjusted baseline

Base directory:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_to_pgg_validation_persona_sampling_unadjusted/seed_0`

Main files:

- `seat_assignments.csv`
- `seat_assignments.jsonl`
- `game_assignments.jsonl`
- `summary.json`
- `manifest.json`
- `sampling_notes.md`
- `observed_pgg_distribution_targetable_rows.csv`
- `assigned_twin_distribution.csv`
- `distribution_checks.csv`
- `twin_usage_summary.csv`

## Prompt Assembly

For augmentation, split the prompt into:

### 1. Global prompt block

Use once per game:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_extended_profile_cards/pgg_prompt_min/shared_prompt_notes.md`

This file contains:

- the general interpretation note
- shared methodological caveats
- the cue glossary and cue definitions

This content is intentionally not repeated inside each player card.

### 2. Player-specific blocks

Use once per sampled player:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`

The seat-level manifest points to the corresponding player by `twin_pid`.

For each sampled seat, the player-specific block should come from:

- the matching `twin_pid`
- the game seat assignment in `seat_assignments.jsonl` or `game_assignments.jsonl`

In `pgg_prompt_min`, the player-specific card keeps:

- headline
- summary
- short background
- behavioral signature
- observed anchors
- cue values
- player-specific limits only when present

In `pgg_prompt_min`, the player-specific card does **not** repeat:

- global caveats
- cue definitions
- per-cue construction notes
- generic transfer-scope notes

### 3. Game/config block

Add the PGG game context separately:

- game rules
- config features
- round structure
- sanction/reward/chat availability
- any simulation instructions

### Recommended prompt order

1. game/config context
2. shared prompt notes
3. player cards for the sampled seats
4. simulation task / prediction instruction

## Current Seed-0 Summary

### Corrected sampler

From `summary.json`:

- valid games: `417`
- configured seats: `3583`
- actual listed roster seats: `3582`
- observed valid PGG player-input rows: `3334`
- observed complete age+education rows: `3326`
- observed targetable age+education+male/female rows: `3260`
- excluded complete rows for sex targeting: `66`
  - `non_binary = 56`
  - `unknown = 10`
- unique Twin personas used: `1576`
- max reuse count: `14`
- within-game reuse: `0`

Distribution checks:

- target-to-assigned TVD for age: `0.0`
- target-to-assigned TVD for education: `0.0`
- target-to-assigned TVD for sex: `0.0`
- target-to-assigned TVD for full `age x education x sex`: `0.0`

Observed-to-target TVDs are nonzero but tiny because `3260` observed targetable PGG rows are being scaled to `3583` configured seats with integer rounding.

### Unadjusted baseline

From `summary.json`:

- valid games: `417`
- configured seats: `3583`
- actual listed roster seats: `3582`
- observed valid PGG player-input rows: `3334`
- observed targetable age+education+male/female rows: `3260`
- unique personas used: `1696`
- max reuse count: `8`
- within-game reuse: `0`

Observed-to-assigned TVDs:

- age: `0.28129870`
- education: `0.07175294`
- sex: `0.00269704`
- full `age x education x sex`: `0.28129870`

## Regeneration

Corrected mode:

```bash
python PGG-finetuning/non-PGG_generalization/task_grounding/sample_twin_personas_for_pgg_validation.py
```

This writes the default corrected run to:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_to_pgg_validation_persona_sampling/seed_0`

Unadjusted baseline:

```bash
python PGG-finetuning/non-PGG_generalization/task_grounding/sample_twin_personas_for_pgg_validation.py \
  --sampling-mode unadjusted_random
```

This writes to:

- `PGG-finetuning/non-PGG_generalization/task_grounding/output/twin_to_pgg_validation_persona_sampling_unadjusted/seed_0`
