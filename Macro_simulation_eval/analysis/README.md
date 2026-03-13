# Macro Simulation Eval Analysis

## Purpose
`Macro_simulation_eval/analysis` is the analysis package for macro simulation outputs.

- Read-only input: `<eval_root>/<run_id>/macro_simulation_eval.csv`
- Write-only analysis output: `<analysis_root>/<analysis_run_id>/...`
- Default non-benchmark roots:
  - `eval_root=outputs/default/runs/source_default/macro_simulation_eval`
  - `analysis_root=reports/default/macro_simulation_eval`

For benchmark runs, use:
- `eval_root=outputs/benchmark/runs/<split-relative>/macro_simulation_eval`
- `analysis_root=reports/benchmark/macro_simulation_eval`

## Run

Single run:

```bash
python Macro_simulation_eval/analysis/run_analysis.py \
  --eval_root outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval \
  --run_id no_archetype/9999002_r0_macro_no_archetype \
  --human_analysis_csv benchmark/data/processed_data/df_analysis_val.csv \
  --analysis_root reports/benchmark/macro_simulation_eval
```

Comparison mode:

```bash
python Macro_simulation_eval/analysis/run_analysis.py \
  --eval_root outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval \
  --compare_run_ids no_archetype/9999002_r0_macro_no_archetype,retrieved_archetype/9999579_r0_macro_retrieved_archetype \
  --compare_labels "no archetype,retrieved archetype" \
  --human_analysis_csv benchmark/data/processed_data/df_analysis_val.csv \
  --analysis_root reports/benchmark/macro_simulation_eval \
  --analysis_run_id benchmark_filtered__macro_variants_latest
```

Directional CONFIG-effect comparison across 3 macro variants + linear CONFIG baseline:

```bash
python Macro_simulation_eval/analysis/analyze_config_directional_effects.py \
  --analysis_dir reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest \
  --learn_analysis_csv benchmark/data/processed_data/df_analysis_learn.csv \
  --val_analysis_csv benchmark/data/processed_data/df_analysis_val.csv
```

Four-variant macro summary plots (no/random/oracle/retrieved) + OLS CONFIG baseline:

```bash
python Macro_simulation_eval/analysis/plot_macro_four_variant_summary.py \
  --analysis_dir reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest \
  --summary_csv reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest/macro_four_variant_partial_rmse_variance_summary.csv \
  --shared_game_ids_csv reports/benchmark/macro_simulation_eval/benchmark_filtered__macro_variants_latest/macro_four_variant_shared4_game_ids.csv \
  --learn_rows_csv benchmark/data/raw_data/learning_wave/player-rounds.csv \
  --val_rows_csv benchmark/data/raw_data/validation_wave/player-rounds.csv \
  --scope shared_4way
```

Round-aware macro alignment comparison for specific runs:

```bash
python Macro_simulation_eval/analysis/compare_macro_alignment.py \
  --eval_root outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval \
  --run_ids no_archetype/10477689_r0_macro_no_archetype_local_12b,oracle_archetype/10478932_r0_macro_oracle_archetype_local_12b \
  --labels "no archetype,oracle archetype" \
  --analysis_root reports/benchmark/macro_simulation_eval \
  --analysis_run_id macro_alignment__local12b_no_vs_oracle
```

Optional: override punishment/reward magnitudes for selected simulated labels when
recomputing payoff-based metrics such as normalized efficiency:

```bash
python Macro_simulation_eval/analysis/compare_macro_alignment.py \
  --eval_root outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval \
  --run_ids oracle_archetype/10478932_r0_macro_oracle_archetype_local_12b,oracle_archetype/10034953_r0_macro_oracle_archetype \
  --labels "oracle archetype 12B,oracle archetype 22B mag1" \
  --magnitude_override_by_label "oracle archetype 22B mag1:1" \
  --analysis_root reports/benchmark/macro_simulation_eval \
  --analysis_run_id macro_alignment__oracle12b_vs_oracle22b_mag1
```

Optional: collapse each nonzero simulated punishment/reward edge to unit `1`
for selected labels. This is different from a plain magnitude override because
multiple units sent to the same target are treated as a single directed edge:

```bash
python Macro_simulation_eval/analysis/compare_macro_alignment.py \
  --eval_root outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval \
  --run_ids oracle_archetype/10478932_r0_macro_oracle_archetype_local_12b,oracle_archetype/10034953_r0_macro_oracle_archetype \
  --labels "oracle archetype 12B,oracle archetype 27B unit-edge" \
  --unit_edge_by_label "oracle archetype 27B unit-edge" \
  --analysis_root reports/benchmark/macro_simulation_eval \
  --analysis_run_id macro_alignment__oracle12b_vs_oracle27b_unit_edges
```

## Outputs
Under `<analysis_root>/<analysis_run_id>/`:

- `selected_runs.csv`
- `aggregate_efficiency_metrics.csv`
- `game_level_metrics.csv`
- `directional_effects.csv`
- `directional_sign_summary.csv`
- `analysis_manifest.json`
- `figures/` (if plotting enabled)

From `analyze_config_directional_effects.py`:
- `config_directional_effects_four_models_shared23.csv`
- `config_directional_effects_four_models_summary_shared23.csv`
- `config_directional_effects_four_models_summary_by_mode_shared23.csv`
- `config_directional_effects_four_models_wide_shared23.csv`
- `linear_config_baseline_shared23_predictions.csv`
- `config_directional_effects_four_models_manifest_shared23.json`
- `figures/config_directional_effects_four_models_sign_match_rate_shared23.png`
- `figures/config_directional_effects_four_models_delta_by_factor_shared23.png`

From `plot_macro_four_variant_summary.py`:
- `macro_four_variant_shared_4way_with_ols_table.csv`
- `macro_four_variant_shared_4way_with_ols_manifest.json`
- `figures/macro_four_variant_rmse_grouped_by_target_shared_4way_with_ols.png`

From `compare_macro_alignment.py`:
- `shared_game_ids.csv`
- `game_level_parity.csv`
- `game_level_alignment_summary.csv`
- `game_level_ols_baseline_summary.csv`
- `game_level_correlation_bootstrap.csv`
- `game_round_metrics.csv`
- `trajectory_alignment.csv`
- `event_study_alignment.csv`
- `targeting_alignment.csv`
- `player_game_distribution_metrics.csv`
- `player_heterogeneity_by_game.csv`
- `player_heterogeneity_summary.csv`
- `player_distribution_alignment_by_game.csv`
- `player_distribution_wasserstein_summary.csv`
- `contribution_shift_by_game.csv`
- `contribution_shift_parity.csv`
- `contribution_shift_alignment.csv`
- `figures/macro_alignment_game_level_parity.png`
- `figures/macro_alignment_game_level_correlation_bars.png`
- `figures/macro_alignment_game_level_rmse_bars.png`
- `figures/macro_alignment_round_trajectories.png`
- `figures/macro_alignment_event_study.png`
- `figures/macro_alignment_targeting.png`
- `figures/macro_alignment_player_distribution_contribution.png`
- `figures/macro_alignment_player_distribution_punishment.png`
- `figures/macro_alignment_player_distribution_reward.png`
- `figures/macro_alignment_player_heterogeneity.png`
- `figures/macro_alignment_distribution_wasserstein.png`
- `figures/macro_alignment_contribution_shift_parity.png`

Note: OLS baselines are CONFIG-only game-level regressions.
- Normalized efficiency target comes from `itt_relative_efficiency` in `df_analysis_*.csv`.
- Contribution/punishment/reward rate targets are aggregated from `raw_data/*/player-rounds.csv`.

## What It Computes

- Game-level simulated normalized efficiency (from macro row outputs and game config).
- Human normalized efficiency (`itt_relative_efficiency`) from the analysis CSV.
- Fit metrics: MAE, RMSE, correlation.
- Mechanism-aware punishment/reward parity, excluding games where the mechanism is absent.
- Round-normalized contribution / punishment / reward / efficiency trajectories.
- Per-game player distribution panels for contribution / punishment / reward rates.
- Early-to-late contribution shift parity by game.
- Next-round contribution response after punishment vs reward in enabled games only.
- Punishment/reward targeting rates by previous-round contribution percentile.
- Directional CONFIG effects:
  - For binary configs: `True mean - False mean`
  - For continuous configs: `above-median mean - below-median mean`
  - Sign agreement between human vs simulated deltas.
