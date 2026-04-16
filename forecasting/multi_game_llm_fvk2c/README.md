# Multi-Game Battery with LLM Delegation Forecasting

This subfolder is the active forecasting benchmark for the multi-game delegation dataset. The research question matches the top-level PGG benchmark: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

This benchmark is best treated as a subject-level experiment-session forecast rather than a transcript rollout. One LLM request should predict one participant's full battery for one randomized treatment assignment, including every scenario nested inside that same session.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the evaluation framing. The source dataset is documented in [../../non-PGG_generalization/data/multi_game_llm_fvk2c/README.md](../../non-PGG_generalization/data/multi_game_llm_fvk2c/README.md).

Critical unit definition:

- one row / one LLM request = one subject-level experiment session
- one design = one treatment arm (`TRP`, `TRU`, `TDP`, `TDU`, `ODP`, `ODU`)
- repeated rows within the same design differ by which participant was recruited into that treatment arm
- `Scenario` and `Case` are nested within the row and should be used for scoring after parsing, not for independent sampling

Active entrypoints:

- [`build_batch_inputs.py`](./build_batch_inputs.py)
- [`parse_outputs.py`](./parse_outputs.py)
- [`evaluate_outputs.py`](./evaluate_outputs.py)
- [`analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)
- [`compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)

Exploratory or paper-reproduction utilities now live under [`exploratory/`](./exploratory/).

Shared implementation:

- dataset adapter: [`forecasting/datasets/multi_game_llm_fvk2c.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/datasets/multi_game_llm_fvk2c.py)
- prompt builder: [`forecasting/prompts/multi_game_llm_fvk2c.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/prompts/multi_game_llm_fvk2c.py)
- run writer: [`forecasting/common/runs/non_pgg.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/common/runs/non_pgg.py)
