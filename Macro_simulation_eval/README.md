# Macro Simulation Eval

`Macro_simulation_eval` runs full-game PGG simulation at `gameId` granularity using:
- game config from `processed_data/df_analysis_{val|learn}.csv`
- player roster from `raw_data/<wave>/player-rounds.csv`
- avatar lookup from `raw_data/<wave>/players.csv`
- demographics from `demographics/demographics_numeric_{val|learn}.csv`

## Default I/O Layout

- eval outputs root: `outputs/default/runs/source_default/macro_simulation_eval`
- per-run folder: `<output_root>/<run_id_or_timestamp>/`
- key files:
  - `macro_simulation_eval.csv`
  - `history_transcripts.jsonl`
  - `archetype_assignments.jsonl`
  - `macro_simulation_debug.jsonl` (unless debug is off)
  - `config.json`

## Archetype Modes

- no archetype: omit `--archetype`
- random archetype: `--archetype random_summary --archetype_summary_pool <pool_jsonl>`
- oracle archetype: `--archetype matched_summary --archetype_summary_pool Persona/archetype_oracle_gpt51_val.jsonl`
- retrieved archetype: `--archetype matched_summary --archetype_summary_pool <retrieved_jsonl>`
- config-bank retrieved archetype: `--archetype config_bank_archetype --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl`

`config_bank_archetype`:

- trains a config-only embedding regressor on finished learn-wave oracle personas
- scores all learn-wave bank personas for the target config
- samples `N = CONFIG_playerCount` personas without replacement
- records the exact sampled persona text and source IDs in `archetype_assignments.jsonl`
- uses `--archetype_soft_bank_temperature` to control the softness of the bank weights
- if `--archetype_assignments_in_path <manifest.jsonl>` is set, uses those precomputed assignments instead of resampling

## Action Prompt Mode

- default: `--action_prompt_mode binary_targets`
- asks only for punish/reward target lists; each selected target receives exactly one unit
- simulated punish/reward outputs therefore default to unit-binary dictionaries (`avatar -> 1`)
- legacy opt-in: `--action_prompt_mode legacy_units`

## Continuation Gate

- default: on via `--action_continuation_gate True`
- repeated same-dyad actions from the immediately previous simulated round are probabilistically thinned
- defaults: `--punish_continuation_keep_prob 0.5`, `--reward_continuation_keep_prob 0.35`

## Demographics In Prompt

- default: `--include_demographics False`
- opt-in: `--include_demographics True`

## Examples

Default validation data:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider openai \
  --openai_model gpt-5-mini
```

vLLM:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider vllm \
  --base_model google/gemma-3-27b-it \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --max_parallel_games 4 \
  --vllm_max_concurrency 8
```

Benchmark split data:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider openai \
  --openai_model gpt-5-mini \
  --data_root benchmark/data_ood_splits/all_or_nothing/false_to_true \
  --wave validation_wave \
  --archetype matched_summary \
  --archetype_summary_pool outputs/benchmark/cache/archetype/archetype_oracle_gpt51_learn_val_union_finished.jsonl \
  --output_root outputs/benchmark/runs/benchmark_ood/all_or_nothing/false_to_true/macro_simulation_eval/oracle_archetype

python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider openai \
  --openai_model gpt-5-mini \
  --archetype config_bank_archetype \
  --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl \
  --archetype_soft_bank_temperature 0.07

python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider openai \
  --openai_model gpt-5-mini \
  --archetype config_bank_archetype \
  --archetype_assignments_in_path Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed0.jsonl
```

Resume an interrupted run in place:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --output_root outputs/default/runs/source_default/macro_simulation_eval \
  --resume_from_run oracle_archetype/2602270212
```

Resume behavior:

- existing output files are preserved and reopened in append mode
- fully completed games are skipped automatically
- interrupted games keep their completed rounds; only rows from the first unfinished round onward are trimmed before continuing
- the resumed prompts/transcript state are rebuilt from the existing CSV rows, so continuation starts from the prior simulated history
- run arguments are restored from the saved `config.json` before continuing, except for redacted API keys and live logging flags
- `config.json` keeps the original `created_at_utc` and appends resume metadata under `resume` / `resume_history`

Concurrency notes:

- `--max_parallel_games` now overlaps multiple games for remote providers (`openai` / `vllm`).
- Local HF inference still runs sequentially across games; use `vllm` if you want high request concurrency on a single model server.

The main simulation CSV now also includes:

- `archetype_source_gameId`
- `archetype_source_playerId`
- `archetype_source_rank`
- `archetype_source_score`
- `archetype_source_weight`

## Analysis

Use:

```bash
python Macro_simulation_eval/analysis/run_analysis.py --help
```

Primary analysis docs:
- `Macro_simulation_eval/analysis/README.md`
