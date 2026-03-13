# Archetype Sampling

Shared archetype-assignment runtime for the LLM evaluation pipelines in:

- `Micro_behavior_eval/`
- `Macro_simulation_eval/`

The goal is to keep persona/archetype assignment logic in `Persona/`, not inside the statistical benchmark folders.

## Modes

- `matched_summary`
  - Use the finished archetype summary that matches the target `gameId` and `playerId`.
  - This is the oracle path for validation-wave runs.

- `random_summary`
  - Uniformly sample finished archetype summaries from the provided pool.

- `config_bank_archetype`
  - Train a config-only regression from learn-wave `CONFIG_*` values to the reduced persona embedding space.
  - Score every finished learn-wave archetype summary in the bank.
  - Convert scores to softmax weights with temperature `tau`.
  - Sample `N = CONFIG_playerCount` summaries without replacement.

`config_bank_archetype` is the retrieval mode for heterogeneity-preserving, config-conditioned sampling.

## Data Sources

The config-bank sampler depends on:

- learn-wave archetype text:
  - `Persona/archetype_oracle_gpt51_learn.jsonl`
- learn-wave persona/config table:
  - `simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet`
- learn-wave reduced embedding matrix:
  - `simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_matrix_learn.parquet`

Only rows with `game_finished == true` are used for training and sampling.

In the current learn-wave bank this leaves `3691` usable personas and excludes unfinished/unknown cases.

## Runtime Outputs

Both micro and macro eval runs now write:

- normal row output CSV
- transcript JSONL
- debug JSONL files
- `archetype_assignments.jsonl`

`archetype_assignments.jsonl` contains one row per target game/player assignment with:

- target game/treatment/player slot
- assignment mode
- source oracle `experiment` and `participant`
- full source archetype text
- soft-bank score/weight/rank when applicable
- precomputed assignment metadata when a fixed manifest is used

This file is the easiest place to inspect which oracle persona was actually injected into a run.

## CLI

Micro:

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --archetype config_bank_archetype \
  --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl
```

Macro:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --archetype config_bank_archetype \
  --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl
```

Precompute fixed assignments first, then reuse them in runs:

```bash
python Persona/archetype_sampling/generate_game_assignments.py \
  --seeds 0,1,2,3,4,5,6,7,8,9

python Micro_behavior_eval/run_micro_behavior_eval.py \
  --archetype config_bank_archetype \
  --archetype_assignments_in_path Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed0.jsonl

python Macro_simulation_eval/run_macro_simulation_eval.py \
  --archetype config_bank_archetype \
  --archetype_assignments_in_path Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed0.jsonl
```

Notes:

- If `--archetype config_bank_archetype` is selected and the pool is left at the validation default, the eval entrypoints rewrite it to `Persona/archetype_oracle_gpt51_learn.jsonl`.
- Sampling is without replacement.
- No diversity penalty is applied.
- Multiple players can still land in the same behavioral region; the only thing disallowed is reusing the exact same bank persona inside one game.
- `generate_game_assignments.py` writes one fixed assignment JSONL per seed so repeated simulations can reuse exactly the same retrieved personas.

## Validation-Treatment Reports

Use the standalone report builder to inspect the 40 validation treatment configs without running the LLM simulators:

```bash
python Persona/archetype_sampling/build_validation_reports.py
```

Default outputs:

- `Persona/archetype_sampling/outputs/validation_config_bank/validation_treatment_single_sample_seed0.jsonl`
  - one sampled persona roster per treatment
- `Persona/archetype_sampling/outputs/validation_config_bank/validation_treatment_bank_distribution.csv`
  - exact soft-bank weights over all learn-wave personas for each treatment
- `Persona/archetype_sampling/outputs/validation_config_bank/validation_treatment_resample_summary.csv`
  - repeated-draw summary statistics per treatment
- `Persona/archetype_sampling/outputs/validation_config_bank/validation_treatment_top_personas.jsonl`
  - top repeatedly selected personas per treatment with full text
- `Persona/archetype_sampling/outputs/validation_config_bank/manifest.json`

Fixed game-level assignment outputs:

- `Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed<seed>.jsonl`
- `Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignment_summary.csv`
- `Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/manifest.json`

## Implementation Notes

- `runtime.py`
  - shared pool loader
  - supported mode registry
  - config-conditioned bank sampler
  - precomputed assignment manifest loader
  - assignment manifest generation

- `build_validation_reports.py`
  - treatment-level offline reporting over the 40 validation configs

- `generate_game_assignments.py`
  - fixed-seed game-level assignment manifest generation for repeatable micro/macro runs

The micro and macro evaluators should treat this package as the single source of truth for archetype assignment behavior.
