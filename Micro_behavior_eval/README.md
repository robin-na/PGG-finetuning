# Micro_behavior_eval

Round-level micro behavior evaluator for PGG data.

It predicts each player's behavior in round `T` from game history through `T-1`, then writes eval-ready CSV rows with predicted vs actual round-`T` behavior.

## Defaults

- `--rounds_csv`: `data/raw_data/validation_wave/player-rounds.csv`
- `--analysis_csv`: `data/processed_data/df_analysis_val.csv`
- `--demographics_csv`: `demographics/demographics_numeric_val.csv`
- `--players_csv`: `data/raw_data/validation_wave/players.csv`
- `--games_csv`: `data/raw_data/validation_wave/games.csv`

## Prompt/Context behavior

- History format follows transcript-style game context (`<ROUND>`, `<CONTRIB>`, `<PEERS_CONTRIBUTIONS>`, etc.).
- Demographics are inserted right after `You are playing an online public goods game (PGG).`
- If chat is enabled, the system text uses: `You may send messages to the group chat.`
- Elicitation format mirrors `Simulation_robin`:
  - JSON-only outputs
  - contribution / actions stages only (no round-`T` chat prediction)
  - optional reasoning via `--include_reasoning`
  - same provider/model controls (`local` or `openai`)
- Optional persona conditioning:
  - `--persona matched_summary` or `--persona random_summary`
  - default summary pool: `Persona/summary_gpt51_val.jsonl`
  - persona text is prepended as `# YOUR PERSONA` with `<SUMMARY STARTS> ... <SUMMARY ENDS>`

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
  --persona matched_summary \
  --persona_summary_pool Persona/summary_gpt51_val.jsonl
```

## Output

Each run writes a timestamped folder under `Micro_behavior_eval/output/<run_id_or_timestamp>/` with:

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
