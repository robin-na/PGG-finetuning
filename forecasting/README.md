# Forecasting

Full-rollout (`k=0`) public-goods-game forecasting lives here.

This package is separate from `trajectory_completion`, which is for within-game continuation from an observed prefix.

## Layout

- `batch_input/`: raw OpenAI Batch request JSONL files
- `batch_output/`: raw completed OpenAI Batch output JSONL files
- `metadata/<run_name>/`: request manifest, selected games, gold transcript scaffold, parsed outputs, sample prompt, token estimates
- `results/`: evaluation outputs and treatment-level comparison outputs

The convention is:

- request file: `forecasting/batch_input/<run_name>.jsonl`
- batch output file: `forecasting/batch_output/<run_name>.jsonl`
- default run names are intentionally short, typically `<variant>_<model>`, with extra suffixes only when you deviate from the standard baseline settings
- the baseline direct-transcript variant is shortened to `baseline`, so the default files are things like `baseline_gpt_5_1.jsonl`

## Build A Batch

Example:

```bash
python -m forecasting.build_batch_inputs \
  --split val \
  --selection-mode one_per_treatment \
  --require-valid-starting-players \
  --k-values 0 \
  --min-num-rounds-exclusive 0 \
  --variant-name baseline_direct_transcript \
  --repeat-count-mode match_valid_start_treatment_counts \
  --model gpt-5.1
```

Notes:

- This builder is reserved for the `k=0` full-rollout task.
- `--repeat-count-mode match_valid_start_treatment_counts` repeats each treatment prompt to the number of validation-wave games with `valid_number_of_starting_players == True` for that treatment.
- The builder writes the request JSONL into `batch_input/` and the sidecar files into `metadata/<run_name>/`.

## Parse A Completed Batch

```bash
python -m forecasting.parse_outputs --run-name <run_name>
```

This expects:

- input at `forecasting/batch_output/<run_name>.jsonl`
- manifest at `forecasting/metadata/<run_name>/request_manifest.csv`

and writes:

- `forecasting/metadata/<run_name>/parsed_output.jsonl`

## Evaluate Against The Selected Gold Game

```bash
python -m forecasting.evaluate_outputs --run-name <run_name>
```

This writes:

- `forecasting/results/<run_name>__gold_eval/`

## Compare Against Human Treatment Distributions

```bash
python -m forecasting.analyze_vs_human_treatments --run-name <run_name>
```

This writes:

- `forecasting/results/<run_name>__vs_human_treatments/`

and compares each generated rollout to the distribution of real validation-wave games from the same `CONFIG_treatmentName`.

Key outputs now include:

- `generated_game_summary.csv` and `human_game_summary.csv` with per-game aggregate metrics, including decay slopes and mean within-round contribution variance
- `treatment_metric_comparison.csv` and `overall_metric_summary.csv` for generated-game-vs-human-distribution comparisons
- `treatment_mean_alignment.csv` and `treatment_mean_alignment_summary.csv` for treatment-mean MAE/RMSE/Spearman
- `treatment_dispersion.csv` and `treatment_dispersion_summary.csv` for across-game SD/IQR calibration within treatment
- `treatment_wasserstein_distance.csv` and `treatment_wasserstein_summary.csv` for per-treatment 1-Wasserstein distances

## Compare Models Against A Noise Ceiling

```bash
python -m forecasting.compare_models_with_noise_ceiling \
  --run-names baseline_gpt_5_1 baseline_gpt_5_mini
```

This writes:

- `forecasting/results/model_comparison__noise_ceiling/`

Key outputs include:

- `model_vs_noise_ceiling.png` with grouped bars for `gpt-5.1`, `gpt-5-mini`, and the matched human noise ceiling
- `model_vs_noise_ceiling_summary.csv` with raw scores, gap-to-ceiling, and ratio-to-ceiling values
- `noise_ceiling_summary.csv` and `noise_ceiling_bootstrap.csv` for the bootstrap human-vs-human reference
