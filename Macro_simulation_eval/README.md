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
  - `macro_simulation_debug.jsonl` (unless debug is off)
  - `config.json`

## Archetype Modes

- no archetype: omit `--archetype`
- random archetype: `--archetype random_summary --archetype_summary_pool <pool_jsonl>`
- oracle archetype: `--archetype matched_summary --archetype_summary_pool Persona/archetype_oracle_gpt51_val.jsonl`
- retrieved archetype: `--archetype matched_summary --archetype_summary_pool <retrieved_jsonl>`

## Examples

Default validation data:

```bash
python Macro_simulation_eval/run_macro_simulation_eval.py \
  --provider openai \
  --openai_model gpt-5-mini
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
```

## Analysis

Use:

```bash
python Macro_simulation_eval/analysis/run_analysis.py --help
```

Primary analysis docs:
- `Macro_simulation_eval/analysis/README.md`
