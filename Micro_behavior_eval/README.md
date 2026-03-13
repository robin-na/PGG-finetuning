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
  - same provider/model controls (`local`, `openai`, or `vllm`)
- Action prompt mode:
  - default: `--action_prompt_mode binary_targets`
  - asks only for punish/reward target lists; each selected target receives exactly one unit
  - replayed human punish/reward history is also recoded to one unit per directed edge for prompt consistency
  - legacy opt-in: `--action_prompt_mode legacy_units`
- Continuation gate:
  - default: on via `--action_continuation_gate True`
  - repeated same-dyad actions from the immediately previous round are probabilistically thinned
  - defaults: `--punish_continuation_keep_prob 0.5`, `--reward_continuation_keep_prob 0.35`
- Demographics in prompt:
  - default: `--include_demographics False`
  - opt-in: `--include_demographics True`
- Optional archetype conditioning:
  - `--archetype matched_summary`, `--archetype random_summary`, or `--archetype config_bank_archetype`
  - default summary pool:
    - `matched_summary` / `random_summary`: `Persona/archetype_oracle_gpt51_val.jsonl`
    - `config_bank_archetype`: `Persona/archetype_oracle_gpt51_learn.jsonl`
  - `config_bank_archetype` trains a config-only embedding regressor on finished learn-wave oracle personas and samples without replacement from the induced bank distribution
  - temperature is controlled by `--archetype_soft_bank_temperature` (default `0.07`)
  - if `--archetype_assignments_in_path <manifest.jsonl>` is set, the run uses those fixed assignments instead of resampling
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

Example (vLLM):

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --provider vllm \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --max_parallel_games 4 \
  --vllm_max_concurrency 8
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

python Micro_behavior_eval/run_micro_behavior_eval.py \
  --archetype config_bank_archetype \
  --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl \
  --archetype_soft_bank_temperature 0.07

python Persona/archetype_sampling/generate_game_assignments.py \
  --seeds 0,1,2,3,4,5,6,7,8,9

python Micro_behavior_eval/run_micro_behavior_eval.py \
  --archetype config_bank_archetype \
  --archetype_assignments_in_path Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed0.jsonl
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

Resume an interrupted run in place:

```bash
python Micro_behavior_eval/run_micro_behavior_eval.py \
  --output_root outputs/default/runs/source_default/micro_behavior_eval \
  --resume_from_run retrieved_archetype/2602270212
```

Resume behavior:

- existing output files are preserved and reopened in append mode
- completed games are skipped automatically
- interrupted games resume from the first unfinished evaluated round
- run arguments are restored from the saved `config.json` before continuing, except for redacted API keys and live logging flags
- `config.json` keeps the original `created_at_utc` and appends resume metadata under `resume` / `resume_history`

Four-way archetype comparison setup:

- `no archetype`: omit `--archetype`
- `random archetype`: `--archetype random_summary --archetype_summary_pool <pool_jsonl>`
- `oracle archetype`: `--archetype matched_summary --archetype_summary_pool Persona/archetype_oracle_gpt51_val.jsonl`
- `retrieved archetype`: `--archetype matched_summary --archetype_summary_pool <retrieved_jsonl>`
- `config-bank retrieved archetype`: `--archetype config_bank_archetype --archetype_summary_pool Persona/archetype_oracle_gpt51_learn.jsonl`

## Output

Each run writes a timestamped folder under `outputs/default/runs/source_default/micro_behavior_eval/<run_id_or_timestamp>/` with:

- `micro_behavior_eval.csv`: prediction/eval rows
- `history_transcripts.jsonl`: transcript-form histories used by the pipeline
- `archetype_assignments.jsonl`: one row per target game/player with the exact injected oracle persona text and source IDs
- `micro_behavior_debug.jsonl`: prompt/output debug records (unless `debug_level=off`)
- `config.json`: run arguments/parameters + resolved selection/output paths

Write behavior:

- `micro_behavior_eval.csv` is appended and flushed during each game (after each evaluated round).
- `history_transcripts.jsonl` and debug JSONL files are appended and flushed after each game.
- `--max_parallel_games` now overlaps multiple games for remote providers (`openai` / `vllm`). Local HF generation still runs sequentially.
- if `--resume_from_run <run_id_or_dir>` is set, existing records are preserved and the run resumes in place instead of truncating outputs

The CSV includes:

- game/player/round identifiers
- assigned archetype source metadata (`archetype_source_gameId`, `archetype_source_playerId`, score/rank/weight for soft-bank mode)
- predicted contribution, punishment/reward dictionaries
  - under default `binary_targets` mode, punish/reward dictionaries are unit-binary (`avatar -> 1`)
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
