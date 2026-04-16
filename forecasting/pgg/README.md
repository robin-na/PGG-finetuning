# PGG Forecasting

Full-rollout (`k=0`) public-goods-game forecasting lives here.

This package is separate from `trajectory_completion`, which is for within-game continuation from an observed prefix.

For the end-to-end experiment story, start with [PIPELINE_OVERVIEW.md](./PIPELINE_OVERVIEW.md).
For the evaluation layer, see [ANALYSIS_OVERVIEW.md](./ANALYSIS_OVERVIEW.md).
For project-level framing across PGG and non-PGG benchmarks, see [../CORE_NARRATIVE.md](../CORE_NARRATIVE.md).
For benchmark decisions, see [../DECISION_LOG.md](../DECISION_LOG.md).
For the current refactor plan, see [../REFACTOR_PLAN.md](../REFACTOR_PLAN.md).
For the PGG registry outputs, see [registry/README.md](./registry/README.md).

## Layout

- `batch_input/`: raw OpenAI Batch request JSONL files
- `batch_output/`: raw completed OpenAI Batch output JSONL files
- `metadata/<run_name>/`: request manifest, selected games, gold transcript scaffold, parsed outputs, sample prompt, token estimates
- `results/`: evaluation outputs and treatment-level comparison outputs
- `profile_sampling/`: PGG-specific Twin and demographic-only seat assignment scripts and artifacts
- `exploratory/`: non-mainline PGG analyses and registry builders

The convention is:

- request file: `forecasting/pgg/batch_input/<run_name>.jsonl`
- batch output file: `forecasting/pgg/batch_output/<run_name>.jsonl`
- metadata root: `forecasting/pgg/metadata/<run_name>/`
- results root: `forecasting/pgg/results/`

## API Key Loading

OpenAI-facing scripts auto-load `OPENAI_API_KEY` from the repo-root `.api_keys.env` if the variable is not already exported in your shell.

Example:

```bash
echo "OPENAI_API_KEY=sk-..." >> .api_keys.env
```

## Build A Batch

Example:

```bash
python -m forecasting.pgg.build_batch_inputs \
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

## Submit / Poll / Download A Batch

Submit a run from its manifest:

```bash
python -m forecasting.pgg.manage_openai_batch submit --run-name <run_name>
```

Check status:

```bash
python -m forecasting.pgg.manage_openai_batch status --run-name <run_name>
```

One-shot sync that submits if needed, refreshes the batch, and downloads the completed output into the manifest's expected `batch_output/<run_name>.jsonl` path:

```bash
python -m forecasting.pgg.manage_openai_batch sync --run-name <run_name>
```

Wait until completion and then download:

```bash
python -m forecasting.pgg.manage_openai_batch sync --run-name <run_name> --wait --poll-interval-sec 60
```

The batch manager writes local job state to `metadata/<run_name>/openai_batch_state.json`.

## Parse A Completed Batch

```bash
python -m forecasting.pgg.parse_outputs --run-name <run_name>
```

This expects:

- input at `forecasting/pgg/batch_output/<run_name>.jsonl`
- manifest at `forecasting/pgg/metadata/<run_name>/request_manifest.csv`

and writes:

- `forecasting/pgg/metadata/<run_name>/parsed_output.jsonl`

## Evaluate Against The Selected Gold Game

```bash
python -m forecasting.pgg.evaluate_outputs --run-name <run_name>
```

This writes:

- `forecasting/pgg/results/<run_name>__gold_eval/`

## Compare Against Human Treatment Distributions

```bash
python -m forecasting.pgg.analyze_vs_human_treatments --run-name <run_name>
```

This writes:

- `forecasting/pgg/results/<run_name>__vs_human_treatments/`

and compares each generated rollout to the distribution of real validation-wave games from the same `CONFIG_treatmentName`.

Key outputs include:

- `generated_game_summary.csv` and `human_game_summary.csv`
- `treatment_metric_comparison.csv` and `overall_metric_summary.csv`
- `treatment_mean_alignment.csv` and `treatment_mean_alignment_summary.csv`
- `treatment_dispersion.csv` and `treatment_dispersion_summary.csv`
- `treatment_wasserstein_distance.csv` and `treatment_wasserstein_summary.csv`

## Compare Models Against A Noise Ceiling

```bash
python -m forecasting.pgg.compare_models_with_noise_ceiling \
  --run-names baseline_gpt_5_1 baseline_gpt_5_mini
```

This writes:

- `forecasting/pgg/results/model_comparison__noise_ceiling/`

## Secondary PGG Analyses

These are not part of the main forecasting pipeline anymore:

- `forecasting/pgg/exploratory/plot_macro_pointwise_alignment.py`
- `forecasting/pgg/exploratory/analyze_micro_distribution_alignment.py`
- `forecasting/pgg/exploratory/random_action_baseline.py`
- `forecasting/pgg/exploratory/evaluate_config_linear_regression.py`
- `forecasting/pgg/exploratory/evaluate_config_supervised_models.py`
