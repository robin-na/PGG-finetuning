# Statistical Macro Analysis

This wrapper keeps the `Macro_simulation_eval` report format but points it at the statistical macro runs.

## Run

```bash
python simulation_statistical/macro/analysis/run_analysis.py \
  --run_id <run_id>
```

Default roots:
- eval root: `benchmark_statistical/macro/runs`
- analysis root: `benchmark_statistical/macro/reports`

Validation evaluation is treatment-level: simulated representative games are compared against `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`, and the contribution / punishment / reward rate supplements aggregate the human reference over all raw validation games within each `CONFIG_treatmentName`.

Single-run reports also add a linear CONFIG baseline trained on the learning wave. Added outputs:
- `linear_config_baseline_metadata.json`
- `linear_config_baseline_summary.csv`

The macro figures now show:
- `macro_rmse_by_target.png`: simulation vs linear CONFIG baseline
- `macro_variance_across_players.png`: simulation vs linear CONFIG baseline vs human
- `macro_variance_across_games.png`: simulation vs linear CONFIG baseline vs human
- `macro_game_level_scatter_by_target.png`: human vs simulation and human vs linear CONFIG baseline
- `directional_effects_<run_id>.png`: human vs simulation vs linear CONFIG baseline

Generated reports mirror the macro evaluation package:
- `selected_runs.csv`
- `aggregate_efficiency_metrics.csv`
- `game_level_metrics.csv`
- `directional_effects.csv`
- `directional_sign_summary.csv`
- `analysis_manifest.json`

Single-run statistical-macro reports also add:
- `macro_rmse_summary.csv`
- `macro_variance_summary.csv`
- `macro_player_variance_by_game.csv`
- `figures/macro_rmse_by_target.png`
- `figures/macro_variance_across_players.png`
- `figures/macro_variance_across_games.png`
- `figures/macro_game_level_scatter_by_target.png`
