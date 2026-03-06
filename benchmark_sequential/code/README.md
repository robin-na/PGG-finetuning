# benchmark_sequential/code

Modular components for the sequential learning-curve pipeline.

## Files

- `llm_prompting.py`
  - Builds the LLM reranker prompt.
  - Injects PGG + CONFIG semantics into the prompt context.
  - Handles prompt overflow trimming (`trim_oldest`).
  - Parses strict JSON responses and validates `selected_config_ids`.
  - Applies BO-score fallback when parsing/validation/API fails.

- `plot_utils.py`
  - Shared plotting helpers for per-run and batch outputs.
  - Generates RMSE/R^2 combined figures and zoomed variants.
  - Builds plot-data tables used for downstream charting.

- `__init__.py`
  - Package marker for imports from `benchmark_sequential.code`.

## Integration

Main orchestration remains in:
- `benchmark_sequential/build_learning_curve_random_vs_bo.py`

That script imports and uses the modular helpers above so prompt logic and plotting logic are separated from the run loop.
