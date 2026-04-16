# Twin Profiles and Transfer Notes

## Digital-Twin-Simulation Setup

- Same participant appears across waves 1 through 4.
- `wave1_3_persona_text` is the prompt source: it is a text rendering of that participant's answers from waves 1-3.
- `wave4_Q_wave4_A` is the held-out target answered by that same participant in wave 4.
- `wave4_Q_wave1_3_A` is the matched earlier-wave answer block for the same target questions and same participant, giving a human consistency comparator.
- The standard wave-4 benchmark targets 64 question IDs and is dominated by heuristics/biases and pricing blocks, not trust/ultimatum/dictator.

## Recommended PGG-Transfer Benchmark

- Primary targets: trust, ultimatum, dictator.
- Secondary targets: mental accounting, time preference, risk preference.
- Use Twin demographics, personality, cognitive tests, and non-target behavioral tasks as profile inputs.
- Exclude target-family items from the Twin profile during prediction.

## First-Pass Leakage Exclusions

- Trust target: exclude `QID117-122` and trust thought-text `QID271-272`.
- Ultimatum target: exclude `QID224-230`.
- Dictator target: exclude `QID231` and dictator thought-text `QID275`.
- Secondary target families: exclude same-family items when those families are targets.

## Why Not Reuse Their Benchmark As-Is

- Their benchmark is well-suited for Twin-internal persona completion.
- It is not the cleanest primary benchmark for PGG transfer because the main wave-4 target set is mostly outside the canonical social-preference game space.

## Validation-Wave Twin Sampling

- `sample_twin_personas_for_pgg_validation.py` builds a seat-level Twin persona assignment for the PGG validation wave.
- It targets the aggregate PGG demographic distribution over `age x education x male/female`, not individual PGG participants.
- It currently points to the `pgg_prompt_min` profile cards.
- Sampling uses replacement across games but disallows reuse within a game unless forced.
- Detailed method and output notes are in `TWIN_TO_PGG_VALIDATION_PERSONA_SAMPLING.md`.

## Validation-Wave Demographic-Only Sampling

- `sample_pgg_demographic_only_profiles_for_validation.py` builds synthetic demographic-only cards for the same valid-start validation games.
- It uses only PGG-side demographic fields from the merged validation demographic file.
- It uses no Twin data, no behavioral-task information, and no actual player-to-game linkage.
- It supports both independent fieldwise sampling and row-resampled whole-profile sampling.
- Detailed method and output notes are in `PGG_VALIDATION_DEMOGRAPHIC_ONLY_PROFILES.md`.

## Files

- `twin_question_inventory.csv`
- `wave4_target_inventory.csv`
- `wave4_target_inventory_full_text.csv`
- `TWIN_EXTENDED_PROFILE_SPEC.md`
- `twin_extended_profile_schema.json`
- `twin_extended_profile_card_schema.json`
- `twin_extended_profile_mapping.csv`
- `build_twin_extended_profile_mapping.py`
- `build_twin_extended_profiles.py`
- `render_twin_extended_profile_cards.py`
- `analyze_twin_profile_card_distributions.py`
- `sample_twin_personas_for_pgg_validation.py`
- `sample_pgg_demographic_only_profiles_for_validation.py`
- `TWIN_TO_PGG_VALIDATION_PERSONA_SAMPLING.md`
- `PGG_VALIDATION_DEMOGRAPHIC_ONLY_PROFILES.md`
- `output/twin_extended_profiles/twin_extended_profiles.jsonl`
- `output/twin_extended_profile_cards/compact/twin_extended_profile_cards.jsonl`
- `output/twin_extended_profile_cards/standard/twin_extended_profile_cards.jsonl`
- `output/twin_extended_profile_cards/audit/twin_extended_profile_cards.jsonl`
- `output/twin_extended_profile_cards/pgg_prompt/twin_extended_profile_cards.jsonl`
- `output/twin_extended_profile_cards/pgg_prompt/shared_prompt_notes.md`
- `output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`
- `output/twin_extended_profile_cards/pgg_prompt_min/shared_prompt_notes.md`
- `output/twin_to_pgg_validation_persona_sampling/seed_0/seat_assignments.jsonl`
- `output/twin_to_pgg_validation_persona_sampling/seed_0/summary.json`
- `output/pgg_validation_demographic_only_sampling/seed_0/demographic_profile_cards.jsonl`
- `output/pgg_validation_demographic_only_sampling/seed_0/seat_assignments.jsonl`
- `output/pgg_validation_demographic_only_sampling/seed_0/summary.json`
- `output/pgg_validation_demographic_only_sampling_row_resampled/seed_0/demographic_profile_cards.jsonl`
- `output/pgg_validation_demographic_only_sampling_row_resampled/seed_0/seat_assignments.jsonl`
- `output/pgg_validation_demographic_only_sampling_row_resampled/seed_0/summary.json`
