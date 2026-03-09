# Statistical Micro Analysis

This is a thin wrapper over `Micro_behavior_eval/analysis` with defaults pointed at the statistical micro runs.

## Run

```bash
python simulation_statistical/micro/analysis/run_analysis.py \
  --run_id <run_id>
```

Default roots:
- eval root: `benchmark_statistical/micro/runs`
- analysis root: `benchmark_statistical/micro/reports`

Validation runs are sourced from `benchmark_statistical/data`, with configs coming from `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`.

Because the statistical micro runner emits the same `micro_behavior_eval.csv` schema, all existing micro metrics and comparison reports remain available.

Statistical-micro specific supplement:
- `punishment_target_metrics_overall.csv`
- `punishment_target_f1_by_round.csv`
- `punishment_target_f1_by_game.csv`
- `punishment_target_f1_by_round.png`
- `punishment_target_f1_by_game.png`

The punishment-only supplement is computed on rows from games where `CONFIG_punishmentExists=true`.
