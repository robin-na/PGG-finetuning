# Micro_behavior_eval

Round-level micro behavior evaluator for PGG data.

It predicts each player's behavior in round `T` from game history through `T-1`, then writes eval-ready CSV rows with predicted vs actual round-`T` behavior.

## Defaults

- `--data_root`: unset (optional convenience root)
- `--wave`: `validation_wave`
- `--output_root`: `outputs/default/runs/source_default/micro_behavior_eval`
- `--rounds_csv`: `data/raw_data/validation_wave/player-rounds.csv`
- `--analysis_csv`: `data/processed_data/df_analysis_val.csv`
- `--demographics_csv`: `demographics/demographics_numeric_val.csv`
- `--players_csv`: `data/raw_data/validation_wave/players.csv`
- `--games_csv`: `data/raw_data/validation_wave/games.csv`

If `--data_root` is set, default CSV inputs are auto-resolved as:
- `raw_data/<wave>/player-rounds.csv`
- `raw_data/<wave>/players.csv`
- `raw_data/<wave>/games.csv`
- `processed_data/df_analysis_val.csv` or `df_analysis_learn.csv` (based on `--wave`)
- `demographics/demographics_numeric_val.csv` or `demographics_numeric_learn.csv` (based on `--wave`)

## Prompt/Context behavior

- History format follows transcript-style game context (`<ROUND>`, `<CONTRIB>`, `<PEERS_CONTRIBUTIONS>`, etc.).
- Demographics are inserted right after `You are playing an online public goods game (PGG).`
- If chat is enabled, the system text uses: `You may send messages to the group chat.`
- Elicitation format mirrors `Simulation_robin`:
  - JSON-only outputs
  - contribution / actions stages only (no round-`T` chat prediction)
  - optional reasoning via `--include_reasoning`
  - same provider/model controls (`local` or `openai`)
- Optional archetype conditioning:
  - `--archetype matched_summary` or `--archetype random_summary`
  - default summary pool: `Persona/archetype_oracle_gpt51_val.jsonl`
  - archetype text is prepended as `# YOUR ARCHETYPE` with `<SUMMARY STARTS> ... <SUMMARY ENDS>`

## Run

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py
```

Example (OpenAI):

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --provider openai \
  --openai_model gpt-4o-mini \
  --openai_api_key_env OPENAI_API_KEY
```

Subset example:

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --game_ids gucxYCpGb5Z3d39ya,HyMH2wFdBsGCcERZ7 \
  --start_round 2 \
  --max_games 2

python Micro_behavior_eval/run_micro_behavior_eval.py \
  --archetype matched_summary \
  --archetype_summary_pool Persona/archetype_oracle_gpt51_val.jsonl
```

Benchmark split example via `--data_root`:

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --data_root benchmark/data_ood_splits/all_or_nothing/false_to_true \
  --wave validation_wave \
  --archetype matched_summary \
  --archetype_summary_pool outputs/benchmark/cache/archetype/archetype_oracle_gpt51_learn_val_union_finished.jsonl \
  --output_root outputs/benchmark/runs/benchmark_ood/all_or_nothing/false_to_true/micro_behavior_eval/oracle_archetype
```

Four-way archetype comparison setup:

- `no archetype`: omit `--archetype`
- `random archetype`: `--archetype random_summary --archetype_summary_pool <pool_jsonl>`
- `oracle archetype`: `--archetype matched_summary --archetype_summary_pool Persona/archetype_oracle_gpt51_val.jsonl`
- `retrieved archetype`: `--archetype matched_summary --archetype_summary_pool <retrieved_jsonl>`

## Output

Each run writes a timestamped folder under `outputs/default/runs/source_default/micro_behavior_eval/<run_id_or_timestamp>/` with:

- `micro_behavior_eval.csv`: prediction/eval rows
- `history_transcripts.jsonl`: transcript-form histories used by the pipeline
- `micro_behavior_debug.jsonl`: prompt/output debug records (unless `debug_level=off`)
- `config.json`: run arguments/parameters + resolved selection/output paths

Write behavior:

- `micro_behavior_eval.csv` is appended and flushed during each game (after each evaluated round).
- `history_transcripts.jsonl` and debug JSONL files are appended and flushed after each game.

The CSV includes:

- game/player/round identifiers
- predicted contribution, punishment/reward dictionaries
- actual round-`T` behavior from data
- parsing flags and simple contribution error (`contribution_abs_error`)

## Analysis

Run analysis on a produced run:

```bash
python Micro_behavior_eval/analysis/run_analysis.py \
  --run_id <run_id>
```

For benchmark outputs, point to the benchmark eval root:

```bash
python Micro_behavior_eval/analysis/run_analysis.py \
  --eval_root outputs/benchmark/runs/benchmark_ood/all_or_nothing/false_to_true/micro_behavior_eval \
  --run_id oracle_archetype/<run_id>
```

Default analysis output root:
- `reports/default/micro_behavior/<analysis_run_id>/`

For original `data/` runs, use:
- `--eval_root outputs/default/runs/source_default/micro_behavior_eval`
- `--analysis_root reports/default/micro_behavior`
