# Results: Ridge, GP-EI, Step 1

Learning-curve outputs for:

- Regression model: `Ridge(alpha=1.0)`
- Adaptive method: `GP-EI`
- Seed size: `10`
- Range: `n=10..150`
- Step: `1`
- Random baseline runs: `200`

## Files

- `run_summary.json`: run config + full-data anchor metrics
- `random_runs_raw.csv`: per-run random baseline metrics
- `random_runs_aggregated.csv`: random baseline mean/p10/p90 by `n_pairs`
- `adaptive_bo_gp_ei_curve.csv`: adaptive BO metrics by `n_pairs`
- `learning_curve_plot_data.csv`: combined plot-ready table (random mean + adaptive + anchor)
- `learning_curve_rmse_r2_random_vs_bo.png`: full-scale RMSE/R^2 plot
- `learning_curve_rmse_r2_random_vs_bo_zoom.png`: zoomed RMSE/R^2 plot

