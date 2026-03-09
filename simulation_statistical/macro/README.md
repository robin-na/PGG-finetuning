# Statistical Macro Simulation

`simulation_statistical/macro` runs full-game PGG simulations numerically, using the CONFIG environment in `df_analysis_*` and a seeded random baseline policy.

Current policy:
- `random_baseline`
- contribution is sampled uniformly from the legal contribution space implied by `CONFIG_allOrNothing`
- punishment target is sampled with probability `0.10` when punishment exists
- reward target is sampled with probability `0.10` when reward exists
- punishment/reward magnitude is fixed at `1`
- chat is currently left blank even when `CONFIG_chat=true`

## Defaults

- `--data_root`: `benchmark_statistical/data`
- `--analysis_csv`: `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`
- `--output_root`: `benchmark_statistical/macro/runs`
- `--rows_out_path`: `output/macro_simulation_eval.csv`
- `--transcripts_out_path`: `output/history_transcripts.jsonl`
- `--debug_jsonl_path`: `output/macro_statistical_debug.jsonl`

Validation runs compare against the treatment-averaged benchmark in `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`.

Macro reports also fit a linear CONFIG baseline from the learning wave:
- train tables:
  - `benchmark_statistical/data/processed_data/df_analysis_learn.csv`
  - `benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv`
- eval tables:
  - `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`
  - `benchmark_statistical/data/raw_data/validation_wave/player-rounds.csv`

Feature set used by the linear baseline:
- `CONFIG_playerCount`
- `CONFIG_numRounds`
- `CONFIG_showNRounds`
- `CONFIG_allOrNothing`
- `CONFIG_chat`
- `CONFIG_defaultContribProp`
- `CONFIG_punishmentExists`
- `CONFIG_punishmentCost`
- `CONFIG_punishmentTech`
- `CONFIG_rewardExists`
- `CONFIG_rewardCost`
- `CONFIG_rewardTech`
- `CONFIG_showOtherSummaries`
- `CONFIG_showPunishmentId`
- `CONFIG_showRewardId`
- `CONFIG_MPCR`

## Run

```bash
python simulation_statistical/macro/run_macro_simulation.py
```

Subset example:

```bash
python simulation_statistical/macro/run_macro_simulation.py \
  --max_games 1 \
  --seed 7
```

## Output

Each run writes a timestamped folder under `benchmark_statistical/macro/runs/<run_id_or_timestamp>/` with:

- `macro_simulation_eval.csv`
- `history_transcripts.jsonl`
- `macro_statistical_debug.jsonl`
- `config.json`

For analysis compatibility, `data.punished` and `data.rewarded` are stored with `playerId` keys; avatar-keyed copies are also included in `data.punished_avatar` and `data.rewarded_avatar`.

## Analysis

Use the local wrapper:

```bash
python simulation_statistical/macro/analysis/run_analysis.py \
  --run_id <run_id>
```

Default report root:
- `benchmark_statistical/macro/reports`
