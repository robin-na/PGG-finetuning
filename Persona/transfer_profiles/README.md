# PGG Transfer Profiles

This folder holds the transfer-oriented profile extraction pipeline for the PGG data.

## Goal

Generate one structured profile per PGG participant that is useful for downstream retrieval into the Twin economic-game benchmark.

The generated profile is meant to:

- describe general economic / social-preference tendencies, not only PGG-specific surface behavior
- stay grounded in the exact repeated public-goods-game rules where the behavior was observed
- separate observed PGG behavior from broader inferred latent traits
- produce both structured fields for retrieval and a short persona card for prompt augmentation

## Current scope

The current request builder is:

- `Persona/misc/build_transfer_profile_requests.py`

It now targets **both learning and validation waves**.

Direct raw-data profiles are built from:

- `data/raw_data/learning_wave/player-rounds.csv`
- `data/raw_data/validation_wave/player-rounds.csv`
- `data/processed_data/df_analysis_learn.csv`
- `data/processed_data/df_analysis_val.csv`

The participant transcript evidence comes from:

- `Persona/transcripts_learn.jsonl`
- `Persona/transcripts_val.jsonl`

The builder first constructs a deterministic raw profile bank, then filters to participants who appear to have played through the full game. The current filter is strict:

- if `data.contribution` is missing in any observed round for a participant, that participant is excluded from the LLM extraction batch

In the current snapshot:

- raw profiles built: `8,041`
- complete profiles after the missing-contribution filter: `7,349`
- batch requests emitted: `7,315`
- complete profiles skipped because no participant transcript was available: `34`

## Prompt inputs

Each extraction request is built from four evidence sources:

1. Exact PGG rule context
   - derived directly from the game `CONFIG_*` values

2. Deterministic behavioral summary from raw game data
   - built directly from `player-rounds.csv` and `df_analysis_*.csv`
   - includes contribution summaries, mechanism availability, conditionality, punishment/reward modules, completion flags, and event responses

3. Compact participant transcript evidence
   - selected round snippets rather than the full transcript
   - prioritizes first/last rounds, extreme contribution rounds, sanction/reward events, and chat evidence

Demographics are no longer included in the LLM prompt by default.

## Output shape

The batch requests ask the model to return JSON only, with:

- `observed_in_pgg`
- `latent_traits`
- `transfer_hypotheses`
- `uncertainties`
- `evidence`
- `persona_card`

The output intentionally does **not** ask the model to generate:

- participant identifiers
- game identifiers
- config identifiers
- a separate `game_context` field

Those are handled in the manifest instead.

## Build requests

From repo root:

```bash
python Persona/misc/build_transfer_profile_requests.py
```

Useful options:

```bash
python Persona/misc/build_transfer_profile_requests.py --limit 5
python Persona/misc/build_transfer_profile_requests.py --model gpt-5.1
python Persona/misc/build_transfer_profile_requests.py --max-round-snippets 6
```

## Outputs

By default, artifacts are written under:

- `Persona/transfer_profiles/output/all_waves/`

Files:

- `raw_profiles_all.jsonl`
- `requests_transfer_profiles_<model>.jsonl`
- `manifest_transfer_profiles.jsonl`
- `preview_transfer_profiles_<model>.json`
- `token_estimate_transfer_profiles_<model>.json`

The manifest carries the external join keys (`gameId`, `playerId`) so the model output itself can stay ID-free.
