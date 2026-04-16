# PGG Validation Demographic-Only Profiles

## Purpose

This pipeline builds demographic-only player cards for the valid-start validation-wave PGG games.

It is meant for the experiment:

- prompt the model with demographic information only
- do not use Twin-derived profiles
- do not use behavioral-task evidence
- do not preserve the real player-to-game linkage from the validation wave

The goal is to test whether demographic prompting alone improves prediction.

## Source Frame

Game structure:

- `PGG-finetuning/data/processed_data/df_analysis_val.csv`

Merged PGG-side demographics:

- `PGG-finetuning/demographics/merged_demographcs_prolific.csv`

Only rows whose `PGGEXIT_gameId` belongs to a valid-start validation game are used.

Current `seed_0` source coverage:

- valid-start validation games: `417`
- merged demographic rows used: `3334`
- configured validation seats generated: `3583`

Learning-wave rows are not used anywhere in this pipeline.

## Included Fields

The cards can include:

- `age`
- `sex_or_gender`
- `education`
- `ethnicity`
- `country_of_birth`
- `country_of_residence`
- `nationality`
- `employment_status`

Field sources:

- age: `PROLIFIC_Age`, with fallback to `PGGEXIT_data.age`
- sex/gender: `PROLIFIC_Sex`, with fallback to normalized `PGGEXIT_data.gender`
- education: `PGGEXIT_data.education`
- ethnicity: `PROLIFIC_Ethnicity simplified`
- country of birth: `PROLIFIC_Country of birth`
- country of residence: `PROLIFIC_Country of residence`
- nationality: `PROLIFIC_Nationality`
- employment: `PROLIFIC_Employment status`

If a field is unavailable in a sampled profile, it is omitted from the rendered card.

## Two Sampling Modes

### 1. Independent Marginals

Output directory:

- `PGG-finetuning/forecasting/pgg/profile_sampling/output/pgg_validation_demographic_only_sampling/seed_0`

This mode assumes we only know the validation-wave field distributions, not individual-level profiles.

It:

1. estimates a separate marginal distribution for each field
2. scales each field distribution to the configured validation seat total
3. samples each field independently
4. combines the independently sampled fields into one card per seat

This preserves fieldwise distributions but does not preserve cross-field correlations.

### 2. Row-Resampled Validation Profiles

Output directory:

- `PGG-finetuning/forecasting/pgg/profile_sampling/output/pgg_validation_demographic_only_sampling_row_resampled/seed_0`

This mode resamples whole validation-wave demographic rows as intact profiles.

It:

1. draws source rows from the valid-start validation demographic frame
2. preserves cross-field correlations within those rows
3. assigns the sampled rows to synthetic seats in the validation game structure
4. does not link the sampled rows back to the actual players who played each game

Reuse across games is allowed. Within-game reuse is avoided unless forced by exhaustion.

## Seat Assignment Logic

Both modes respect the validation-wave game structure:

- same valid-start game set
- same configured number of seats per game

One valid-start game is internally inconsistent:

- `gameId = pJFEFMc5YWW7XyLuN`
- `CONFIG_playerCount = 19`
- listed `playerIds = 18`

Per prior user instruction, both demographic-only pipelines use the configured count and therefore create `19` seats for that game.

## Player-Specific Cards

Each sampled seat points to:

- `demographic_profile_cards.jsonl`

Each card contains:

- a one-line demographic summary
- the available demographic fields only

There is no behavioral content in these cards.

## Main Artifacts

Each seed directory contains:

- `demographic_profile_cards.jsonl`
- `demographic_profile_cards.csv`
- `preview_demographic_profile_cards.json`
- `preview_demographic_profile_cards.md`
- `seat_assignments.csv`
- `seat_assignments.jsonl`
- `game_assignments.jsonl`
- `field_distribution_comparison.csv`
- `summary.json`
- `manifest.json`

The row-resampled version also writes:

- `source_row_selection.csv`
- `source_row_usage_summary.csv`

## Prompt Assembly

For demographic-only augmentation:

1. load the seat-to-profile mapping from `seat_assignments.jsonl` or `game_assignments.jsonl`
2. pull the corresponding player card from `demographic_profile_cards.jsonl`
3. append the game rules and simulation instruction

## Regeneration

Independent marginals:

```bash
python PGG-finetuning/forecasting/pgg/profile_sampling/sample_pgg_demographic_only_profiles_for_validation.py \
  --sampling-mode independent_field_marginals \
  --seed 0
```

Row-resampled:

```bash
python PGG-finetuning/forecasting/pgg/profile_sampling/sample_pgg_demographic_only_profiles_for_validation.py \
  --sampling-mode row_resampled_validation_rows \
  --seed 0
```
