# Forecasting Analysis Overview

## Purpose

This note explains how the forecasting runs are evaluated once full-rollout game transcripts have been generated and parsed.

The main analysis layers are:

1. per-run treatment-distribution comparison
2. macro pointwise alignment across validation configs
3. micro within-config distribution alignment
4. model-vs-human noise-ceiling comparison

This file is descriptive only. It should match the currently committed outputs in `forecasting/pgg/results/` and should not be treated as a second source of truth over the manifests.

## Human Reference Set

The main human comparison set is:

- validation-wave games only
- `valid_number_of_starting_players == True`
- still allowed to include incomplete games

That last point is important. It is how the committed per-run result manifests currently describe the human reference set, so it should be treated as an explicit evaluation choice.

## Per-Run Treatment Comparison

Script:

- [`forecasting/pgg/analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)

Outputs:

- `forecasting/pgg/results/<run_name>__vs_human_treatments/`

This is the base evaluation layer.

For each generated rollout and each human validation game, it builds:

- actor summaries
- round summaries
- game summaries

Then it compares generated vs human behavior within each `CONFIG_treatmentName`.

Main output families:

- `treatment_mean_alignment.csv` and `_summary.csv`
- `treatment_dispersion.csv` and `_summary.csv`
- `treatment_wasserstein_distance.csv` and `_summary.csv`
- `treatment_metric_comparison.csv`
- `overall_metric_summary.csv`

Interpretation:

- treatment mean alignment asks whether the model gets the average behavior right by design cell
- treatment dispersion asks whether it gets the spread across games right
- treatment Wasserstein asks whether the whole within-treatment distribution is close

## Macro Pointwise Alignment

Script:

- [`forecasting/pgg/exploratory/plot_macro_pointwise_alignment.py`](./exploratory/plot_macro_pointwise_alignment.py)

Outputs:

- `forecasting/pgg/results/macro_pointwise_alignment__llms/`

This is the cross-model macro view.

It works at the level of the `40` validation treatment cells and asks:

- how close are generated treatment means to human treatment means
- how close are generated within-treatment distributions to human within-treatment distributions

The main score families are:

- `rmse_of_config_means`
- `mean_wasserstein_distance`

The main config-level metrics shown are:

- mean contribution
- first-round contribution
- final-round contribution
- mean efficiency
- first-round efficiency
- final-round efficiency

Interpretation:

- lower is better
- this is the cleanest “did the model recover the config-to-config response surface?” view

## Micro Distribution Alignment

Script:

- [`forecasting/pgg/exploratory/analyze_micro_distribution_alignment.py`](./exploratory/analyze_micro_distribution_alignment.py)

Outputs:

- `forecasting/pgg/results/micro_distribution_alignment__llms/`

This is the within-config heterogeneity view.

It compares generated and human distributions for:

- player mean contribution
- player mean payoff
- round contribution
- round efficiency

And internally it also supports:

- round-to-round contribution change
- round-to-round efficiency change

The core score is within-config 1-Wasserstein distance.

Interpretation:

- macro scores can look decent even when the model collapses person-level or round-level variation
- this layer checks whether the within-treatment distributional texture is realistic

## Human Noise Ceiling

The noise ceiling is not a theoretical upper bound. It is an empirical human-vs-human reference built from repeated resampling within the same validation treatment cell.

That means:

- comparisons are always made within the same `CONFIG_treatmentName`
- the ceiling reflects how much disagreement you get even when both sides are human draws from the same design cell

### Model-Comparison Noise Ceiling

Script:

- [`forecasting/pgg/compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)

Outputs:

- `forecasting/pgg/results/model_comparison__noise_ceiling/`

Method:

1. for each treatment, take the pool of human validation games
2. sample one pseudo-model set with replacement
3. sample one pseudo-human set with replacement
4. size the pseudo-model set to the shared generated count across the compared models
5. size the pseudo-human set to the available human count
6. score pseudo-model vs pseudo-human using the same scalar score families used for the model comparison

Current committed scope:

- baseline
- Twin corrected

It does not currently include:

- demographic-only
- Twin unadjusted

So this committed combined output is narrower than the intended 4-mode benchmark.

### Macro Noise Ceiling

Script:

- [`forecasting/pgg/exploratory/plot_macro_pointwise_alignment.py`](./exploratory/plot_macro_pointwise_alignment.py)

Method:

- bootstrap two independent human resamples within each treatment
- compare their treatment means and within-treatment Wasserstein distances
- aggregate those components across the `40` treatment cells

This is the most directly matched ceiling for the macro pointwise plots.

### Micro Noise Ceiling

Script:

- [`forecasting/pgg/exploratory/analyze_micro_distribution_alignment.py`](./exploratory/analyze_micro_distribution_alignment.py)

Method:

- resample whole human games within each treatment
- derive pseudo-generated and pseudo-human actor and round summaries from those sampled games
- compare within-treatment Wasserstein distances over player-level and round-level summaries

Current committed status:

- the script supports this
- the committed `micro_distribution_alignment__llms` manifest has `bootstrap_iters = 0`
- so there is no materialized micro noise ceiling in the currently committed outputs

## Important Caveats

### 1. Baseline prompt construction is not identical to augmented prompt construction

Current manifests show:

- baseline runs use `selection_mode = one_per_treatment` with `selected_game_count = 40`
- augmented runs use `selection_mode = full` with `selected_game_count = 417`

The baseline is then repeated to match the valid-start treatment counts.

That may be acceptable if the goal is treatment-level alignment, but it is not the same prompt-construction path. That difference should stay explicit whenever baseline is compared to augmented modes.

### 2. Some runs produced fewer evaluable generated games than requested

Examples visible in committed manifests:

- `demographic_only_row_resampled_seed_0_gpt_5_1`: `414 / 417`
- `twin_sampled_unadjusted_seed_0_gpt_5_1`: `410 / 417`
- `twin_sampled_unadjusted_seed_0_gpt_5_mini`: `416 / 417`

Those gaps should remain visible in summaries because they affect strict comparability.

### 3. Legacy metadata still exists

The intended canonical comparison set is the 8 run names listed in the registry with `is_core_run = True`.

The repo also contains older legacy metadata directories, especially:

- long-form baseline directory names from before the canonical `baseline_gpt_5_1` / `baseline_gpt_5_mini` names
- manifests with noncanonical Twin variant strings like `twin-sampled_seed_0`

Those should be treated as historical artifacts, not primary benchmark entries.

## Registry

If you want the machine-readable lookup layer instead of this prose note, use:

- [`forecasting/pgg/registry/experiment_registry.csv`](./registry/experiment_registry.csv)
- [`forecasting/pgg/registry/analysis_registry.csv`](./registry/analysis_registry.csv)
