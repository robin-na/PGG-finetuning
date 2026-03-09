# benchmark_statistical

`benchmark_statistical` is the benchmark root for the numerical PGG simulation baselines in `simulation_statistical`.

## Layout

- `benchmark_statistical/data`
  - validation-wave benchmark copied from `benchmark_sequential/data`
  - learning-wave files populated from the main `data/` root so the statistical baselines can train on the learning wave and evaluate on the validation benchmark
  - serves as the canonical data source for statistical simulations
  - includes `processed_data/df_analysis_val_averaged.csv`, the treatment-averaged validation benchmark
- `benchmark_statistical/micro`
  - run outputs and analysis reports for statistical micro simulations
- `benchmark_statistical/macro`
  - run outputs and analysis reports for statistical macro simulations

## Validation Benchmark

`benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv` is built from `benchmark_statistical/data/processed_data/df_analysis_val.csv` by grouping rows on `CONFIG_treatmentName`.

Aggregation rule:
- numeric columns are averaged within each treatment
- non-numeric columns keep the first non-null value from the treatment block
- the representative `gameId` is the first observed game for that treatment so the statistical simulators can still attach to a concrete round history

Regenerate it with:

```bash
python benchmark_statistical/build_validation_average_dataset.py
```

## Notes

- `simulation_statistical` now defaults to `benchmark_statistical/data` for validation runs.
- The macro report also fits a CONFIG-only linear regression baseline on `benchmark_statistical/data/processed_data/df_analysis_learn.csv` plus `benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv`.
- Statistical run artifacts are written under `benchmark_statistical/micro/...` and `benchmark_statistical/macro/...`, not `outputs/default` or `reports/default`.
