# Statistical Micro Simulation

`simulation_statistical/micro` generates micro-style prediction/evaluation rows without calling an LLM.

Current policy:
- `random_baseline`
- contribution is sampled uniformly from the legal contribution space implied by `CONFIG_allOrNothing`
- punishment target is sampled with probability `0.10` when punishment exists
- reward target is sampled with probability `0.10` when reward exists
- punishment/reward magnitude is fixed at `1`

## Defaults

- `--data_root`: `benchmark_statistical/data`
- `--analysis_csv`: `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`
- `--output_root`: `benchmark_statistical/micro/runs`
- `--rows_out_path`: `output/micro_behavior_eval.csv`
- `--transcripts_out_path`: `output/history_transcripts.jsonl`
- `--debug_jsonl_path`: `output/micro_statistical_debug.jsonl`

Validation runs use the averaged benchmark config table so one representative game is simulated per treatment.

## Run

```bash
python simulation_statistical/micro/run_micro_simulation.py
```

Subset example:

```bash
python simulation_statistical/micro/run_micro_simulation.py \
  --game_ids gucxYCpGb5Z3d39ya \
  --start_round 2 \
  --max_games 1 \
  --seed 7
```

## Output

Each run writes a timestamped folder under `benchmark_statistical/micro/runs/<run_id_or_timestamp>/` with:

- `micro_behavior_eval.csv`
- `history_transcripts.jsonl`
- `micro_statistical_debug.jsonl`
- `config.json`

The row schema intentionally matches `Micro_behavior_eval` so the same metric pipeline can score it.

## Analysis

Use the local wrapper:

```bash
python simulation_statistical/micro/analysis/run_analysis.py \
  --run_id <run_id>
```

Default report root:
- `benchmark_statistical/micro/reports`
