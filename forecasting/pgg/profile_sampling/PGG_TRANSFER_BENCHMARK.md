# PGG Transfer Benchmark

## Benchmark Goal

- Primary benchmark: predict held-out Trust, Ultimatum, or Dictator responses using the rest of the Twin profile.
- Main condition: keep all non-target economic families in the prompt/profile.
- Ablations: intermediate (only non-social economic families) and strict (no economic families).
- Evaluation target: choice questions only in the primary scoreboard. Target-family thought-text items stay excluded from inputs and are not part of the main metric.

## Why This Is Not The Public Wave-4 Benchmark

- The released Twin `wave_split` benchmark block does not contain the economic-preference battery.
- This benchmark therefore uses a held-out-family design over the `wave1_3_persona_json` profile instead of the public wave-4 target block.
- If private wave-4 economic-game targets become available later, the same condition definitions can be reused unchanged.

## Conditions

### Main benchmark

- `main`: Use the full Twin profile plus all non-target economic families. Exclude the target family entirely.

### Intermediate ablation

- `intermediate`: Use the full Twin profile plus only non-social economic families (mental accounting, time preference, risk preference).

### Strict ablation

- `strict`: Use the full Twin profile with no economic-preference items at all.

## Target Families

### Trust

- Choice targets: `QID117, QID118, QID119, QID120, QID121, QID122`
- Exclude target-family thought text from inputs: `QID271, QID272`
- Availability in cached Twin data: 2058/2058 to 2058/2058 participants across target choice items.

| Condition | Allowed Families | Allowed Input Questions | Excluded Input Questions |
| --- | --- | ---: | ---: |
| Main benchmark | demographics, personality, cognitive_tests, ultimatum, dictator, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 149 | 9 |
| Intermediate ablation | demographics, personality, cognitive_tests, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 140 | 18 |
| Strict ablation | demographics, personality, cognitive_tests | 123 | 35 |

### Ultimatum

- Choice targets: `QID224, QID225, QID226, QID227, QID228, QID229, QID230`
- Availability in cached Twin data: 2058/2058 to 2058/2058 participants across target choice items.

| Condition | Allowed Families | Allowed Input Questions | Excluded Input Questions |
| --- | --- | ---: | ---: |
| Main benchmark | demographics, personality, cognitive_tests, trust, dictator, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 150 | 8 |
| Intermediate ablation | demographics, personality, cognitive_tests, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 140 | 18 |
| Strict ablation | demographics, personality, cognitive_tests | 123 | 35 |

### Dictator

- Choice targets: `QID231`
- Exclude target-family thought text from inputs: `QID275`
- Availability in cached Twin data: 2058/2058 to 2058/2058 participants across target choice items.

| Condition | Allowed Families | Allowed Input Questions | Excluded Input Questions |
| --- | --- | ---: | ---: |
| Main benchmark | demographics, personality, cognitive_tests, trust, ultimatum, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 155 | 3 |
| Intermediate ablation | demographics, personality, cognitive_tests, mental_accounting, time_preference, risk_preference_gain, risk_preference_loss | 140 | 18 |
| Strict ablation | demographics, personality, cognitive_tests | 123 | 35 |

## Recommended Scoreboard

- Primary scoreboard: `main` condition for Trust, Ultimatum, and Dictator.
- Secondary scoreboard: `intermediate` and `strict` ablations for the same three target families.
- Recommended baselines per cell: Twin-only profile, Twin-only + non-target economic behavior, PGG-augmented retrieval, random-PGG retrieval.
- Keep the benchmark keying on `(block_name, question_id)` internally to avoid QID collisions across Twin blocks.

## Files

- `pgg_transfer_benchmark_spec.json`
- `pgg_transfer_benchmark_cells.csv`
