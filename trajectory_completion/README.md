# Trajectory Completion

Within-game baselines for forecasting the rest of a public goods game from its first `k` rounds.

## Scope

- Data source: learning-wave tables under `data/raw_data/learning_wave/`
- Game filter: only games with no missing `data.contribution` rows, so no player leaves mid-game
- Current default evaluation slice: games with observed horizon `> 10` rounds
- Current default prefix lengths: `k = 1, 3, 5, 8`

## Baselines

- `persistence`: repeat the previous round's contribution and sanction/reward actions
- `ewma`: recency-weighted contribution forecast plus heuristic sanction/reward targeting based on the previous round
- `within_game_ar`: per-player ridge autoregression fit only on the observed prefix, with `ewma` fallback when `k` is too small

## Run

From the repo root:

```bash
python -m trajectory_completion.evaluate
```

Outputs are written by default to:

`trajectory_completion/results/learning_wave_complete_gt10_k1358/`

The evaluator writes:

- `actor_level_predictions.csv`
- `round_level_predictions.csv`
- `game_summary.csv`
- `overall_summary.csv`
- `manifest.json`

## Plot

Render the current summary figure:

```bash
python -m trajectory_completion.plot_results
```

Default output:

`trajectory_completion/results/learning_wave_complete_gt10_k1358/trajectory_completion_summary.png`

The plot uses game-level means with standard-error error bars from
`game_summary.csv`.

## Prompting

Prompt layout and JSON output shape:

- [PROMPTING.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/trajectory_completion/PROMPTING.md)

Build OpenAI Batch inputs for observer-view trajectory completion:

```bash
python -m trajectory_completion.build_openai_batch_inputs \
  --split val \
  --k-values 1,3,5,8
```

Default output:

`trajectory_completion/batch_inputs/validation_wave_complete_gt10_k1358_observer/`

## Compact Observer Batches

Build the compact observer-format batch inputs:

```bash
python -m trajectory_completion.build_compact_observer_batch_inputs \
  --split val \
  --selection-mode one_per_treatment \
  --require-valid-starting-players \
  --k-values 0
```

Notes:

- `k=0` is a full rollout task from round 1, not a continuation-from-prefix task.
- The prompt asks for `### GAME EXPLANATION` at the beginning and `### ROUND N EXPLANATION` at the start of each predicted round.
- For punishment-only or reward-only games, action `unit` values are positive.
- Only mixed punishment+reward games use signed units: positive for reward and negative for punishment.
- Use `--repeats-per-game N` to duplicate each selected prompt `N` times.
- Use `--repeat-count-mode match_valid_start_treatment_counts` to duplicate each selected treatment prompt to the number of validation-wave games with `valid_number_of_starting_players == True` for that treatment.
- If you request a single `k` value, the builder writes only the per-`k` files such as `request_batch_k0.jsonl` and `gold_continuations_k0.jsonl`.
- The aggregate `request_batch_all_k.jsonl` and `gold_continuations_all_k.jsonl` are only written for multi-`k` runs.

## Parse Compact Outputs

Parse a completed OpenAI Batch output back into structured round predictions:

```bash
python -m trajectory_completion.parse_compact_observer_outputs \
  --input-jsonl trajectory_completion/batch_inputs/.../output_batch_k0.jsonl \
  --request-manifest-csv trajectory_completion/batch_inputs/.../request_manifest.csv \
  --output-jsonl trajectory_completion/batch_inputs/.../parsed_output_batch_k0.jsonl
```

Notes:

- The parser is intentionally permissive about extra bookkeeping text around the transcript.
- For interaction-enabled games, if the model omits an explicit empty punishment/reward line, the parser treats that as no interactions in that round.

## Compare `k=0` Rollouts To Human Treatments

Compare each generated `k=0` rollout to the distribution of real validation-wave games from the same `CONFIG_treatmentName`:

```bash
python -m trajectory_completion.analyze_k0_vs_human_treatments \
  --parsed-output-jsonl trajectory_completion/batch_inputs/.../parsed_output_batch_k0.jsonl \
  --request-manifest-csv trajectory_completion/batch_inputs/.../request_manifest.csv \
  --output-dir trajectory_completion/results/..._vs_human_treatments
```

This analysis:

- uses validation-wave games with `valid_number_of_starting_players == True`
- keeps the full treatment-level human distribution rather than a single gold game
- reports generated-vs-human comparisons for contribution, normalized efficiency, punishment/reward rates, chat intensity, and trajectory RMSE to the treatment mean
- writes a scatter-plot summary figure at `generated_vs_human_treatment_means.png`
