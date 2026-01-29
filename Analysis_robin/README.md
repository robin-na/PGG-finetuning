# Analysis_Robin

This folder contains the evaluation workflow to compare simulated PGG games with
human validation data.

## What it does
- Loads the latest simulation run per `VALIDATION_*` configuration from
  `output/<CONFIG>/<timestamp>/`.
- Loads human player-round data from
  `data/raw_data/validation_wave/player-rounds.csv` and maps each `gameId` to a
  configuration using `data/processed_data/df_analysis_val.csv`.
- Computes metrics per game, per player, and per round:
  - Contribution rate
  - Normalized efficiency
  - Punishment rate
  - Reward rate
  - Payoffs (per-row and averaged)
- Compares simulation vs human behavior by configuration pairing (e.g.
  `VALIDATION_10_C` vs `VALIDATION_10_T`), reporting RMSE/MAE/R² and plotting the
  human noise ceiling (±1 SD of human game-level means).

## Running the analysis
From the repo root:

```bash
python -m Analysis_robin.analyze
```

Optional overrides:

```bash
python -m Analysis_robin.analyze \
  --output-root /workspace/PGG-finetuning/output \
  --analysis-output-root /workspace/PGG-finetuning/analysis_output_Robin \
  --human-rounds /workspace/PGG-finetuning/data/raw_data/validation_wave/player-rounds.csv \
  --human-configs /workspace/PGG-finetuning/data/processed_data/df_analysis_val.csv
```

## Outputs
Each run writes to `analysis_output_Robin/<timestamp>/` and includes:
- Per-config CSV summaries for simulation and human data (game/player/round).
- `alignment_game_summary.csv` and `alignment_metric_summary.csv`.
- Noise ceiling plots (when `matplotlib` is available).
- `manifest.json` describing the analysis inputs and metrics.
