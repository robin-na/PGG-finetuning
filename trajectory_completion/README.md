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
