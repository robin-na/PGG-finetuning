# benchmark_sequential

Sequential experiment-selection workspace for the paired learning/validation setup.

## Scope

- Build a paired learning pool from `data/processed_data/df_analysis_learn.csv`.
- Start from fixed seed pairs (`random_state=42`).
- Compare:
  - random experiment addition baseline
  - adaptive BO baseline (GP-EI or GP max-variance)
  - adaptive LLM rerank over a GP-EI shortlist (Condition C)
- Evaluate downstream prediction on validation paired data using:
  - target: `treatment_itt_efficiency`
  - inputs: fixed 15 `CONFIG_*` features + `control_itt_efficiency`
- Default regression model for analysis: `Ridge(alpha=1.0)`.

## Key Scripts

- `build_seed_dataset.py`
  - Creates the 300-row paired pool (150 pairs) and a 10-pair random seed set.
  - Output example:
    - `data_seed_random_learning_paired300_pairs10_rs42/`

- `evaluate_linear_seed_vs_full.py`
  - Trains on:
    - full learning pairs (150)
    - seed pairs (10)
  - Evaluates both on `benchmark_sequential/data/processed_data/df_paired_val.csv`.
  - Uses Ridge by default.

- `build_learning_curve_random_vs_bo.py`
  - Builds learning curves from `n_start` to `n_max` with configurable `--step`.
  - Methods:
    - `random_addition` (`--methods random`)
    - `adaptive_bo_gp_ei` (`--methods bo --bo-acquisition ei`)
    - `adaptive_llm_rerank_gp_ei` (`--methods llm`)
  - LLM path:
    - builds GP-EI shortlist (default `K=20`)
    - constructs compact structured prompt with history + shortlist
    - includes PGG/efficiency + CONFIG column semantics from `code_ref/build_positive_case_batch_input.py`
    - excludes validation-wave RMSE/R^2 from prompt context
    - includes BO shortlist score as `bo_score_ei` (single BO score column)
    - asks for reasoning first, then `selected_config_ids`
    - supports selecting a batch each step (`--llm-batch-size`, default `10`)
    - supports LLM-specific curve cap (`--llm-n-max`, default `0` meaning use global `--n-max`)
    - parses strict JSON response
    - validates selected IDs
    - falls back to shortlist top-EI on API/parse/validation failures
  - Supports seed scope:
    - `fixed42` (single seed set, smoke)
    - `multi20` (seed states `42..61`, full robustness benchmark)
  - Adds full-data anchor (`n=150`) in plots.
  - Default regression model is Ridge.

## Fixed Feature Set

`CONFIG_playerCount`, `CONFIG_numRounds`, `CONFIG_showNRounds`, `CONFIG_allOrNothing`, `CONFIG_chat`, `CONFIG_defaultContribProp`, `CONFIG_punishmentCost`, `CONFIG_punishmentTech`, `CONFIG_rewardExists`, `CONFIG_rewardCost`, `CONFIG_rewardTech`, `CONFIG_showOtherSummaries`, `CONFIG_showPunishmentId`, `CONFIG_showRewardId`, `CONFIG_MPCR`, plus `control_itt_efficiency`.

## Code Organization

- `benchmark_sequential/build_learning_curve_random_vs_bo.py`
  - Run orchestration, data loading, BO/random loops, output writing.
- `benchmark_sequential/code/llm_prompting.py`
  - LLM prompt construction, JSON parsing, selection validation/fallback.
- `benchmark_sequential/code/plot_utils.py`
  - Plotting and plot-table helpers.
- `benchmark_sequential/code/README.md`
  - Module-level description.

## Common Commands

```bash
# 1) Build paired pool + random seed pairs
python benchmark_sequential/build_seed_dataset.py

# 2) Seed vs full comparison (Ridge default)
python benchmark_sequential/evaluate_linear_seed_vs_full.py

# 3) Condition C smoke run (fixed42), step=1
python benchmark_sequential/build_learning_curve_random_vs_bo.py \
  --n-start 10 --n-max 150 --step 1 \
  --random-runs 200 --random-state 42 \
  --seed-scope fixed42 \
  --methods random,bo,llm \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --llm-batch-size 1

# 4) Condition C smoke run (fixed42), step=10
python benchmark_sequential/build_learning_curve_random_vs_bo.py \
  --n-start 10 --n-max 150 --step 10 \
  --random-runs 200 --random-state 42 \
  --seed-scope fixed42 \
  --methods random,bo,llm \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --llm-batch-size 10

# 5) Full benchmark (multi20), step=1
python benchmark_sequential/build_learning_curve_random_vs_bo.py \
  --n-start 10 --n-max 150 --step 1 \
  --random-runs 200 --random-state 42 \
  --seed-scope multi20 \
  --methods random,bo,llm \
  --llm-provider openai --llm-model gpt-4.1-mini

# 6) Full benchmark (multi20), step=10
python benchmark_sequential/build_learning_curve_random_vs_bo.py \
  --n-start 10 --n-max 150 --step 10 \
  --random-runs 200 --random-state 42 \
  --seed-scope multi20 \
  --methods random,bo,llm \
  --llm-provider openai --llm-model gpt-4.1-mini

# 7) Fixed42 run with x-axis to 150 but LLM curve capped at 60
python benchmark_sequential/build_learning_curve_random_vs_bo.py \
  --n-start 10 --n-max 150 --step 10 \
  --random-runs 200 --random-state 42 \
  --seed-scope fixed42 \
  --methods random,bo,llm \
  --llm-provider openai --llm-model gpt-4.1-mini \
  --llm-batch-size 10 --llm-n-max 60
```

## Output Layout

- Per-seed output folder pattern:
  - `results/learning_curve_ridge_random_bo_llm_gp_ei_rs<seed>_n10_to_150_step<step>_<scope>/run_YYYYMMDD_HHMMSS/`
- Multi20 batch output root pattern:
  - `results/learning_curve_ridge_random_bo_llm_gp_ei_rs42_to_61_n10_to_150_step<step>_multi20/run_YYYYMMDD_HHMMSS/`
  - Timestamp subfolders are auto-created to avoid accidental overwrite.
  - If a same-second folder already exists, suffix `_2`, `_3`, ... is appended.
  - If `--output-dir` is provided, that path is used directly (no extra timestamp subfolder).

Per-seed files:
- `run_summary.json`
- `random_runs_raw.csv`
- `random_runs_aggregated.csv`
- `adaptive_bo_gp_ei_curve.csv`
- `adaptive_llm_rerank_gp_ei_curve.csv`
- `learning_curve_plot_data.csv`
- `learning_curve_rmse_r2_all_methods.png`
- `learning_curve_rmse_r2_all_methods_zoom.png`
- `llm_decisions.csv`
- `llm_prompts_responses.jsonl`

Batch-level files (`multi20`):
- `batch_summary.json`
- `batch_aggregated_curves.csv`
- `batch_plots_rmse_r2_all_methods.png`
- `batch_plots_rmse_r2_all_methods_zoom.png`

## LLM Notes

- Expected env var for OpenAI runs: `OPENAI_API_KEY`.
- If `OPENAI_API_KEY` is missing and `llm` is requested in `--methods`, LLM is skipped entirely (only runnable methods execute), and this is logged as `llm_skipped_reason` in summaries.
- LLM response contract uses `reasoning` + `selected_config_ids` + `confidence` (no backup IDs).
- If key is present but an API/JSON/selection validation failure occurs, the LLM method falls back to shortlist top-EI and logs `fallback_reason` in `llm_decisions.csv`.
