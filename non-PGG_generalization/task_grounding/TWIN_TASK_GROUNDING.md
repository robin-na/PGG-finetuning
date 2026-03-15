# Twin Task Grounding

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

## Files

- `twin_question_inventory.csv`
- `wave4_target_inventory.csv`
- `wave4_target_inventory_full_text.csv`
