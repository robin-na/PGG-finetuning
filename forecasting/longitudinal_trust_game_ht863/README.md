# Longitudinal Trust Game Forecasting

This subfolder is the active forecasting benchmark for the longitudinal repeated trust game dataset. The research question is the same one the top-level PGG docs now emphasize: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

Here the target is not a multi-player transcript. The proposed benchmark simulates one participant's repeated trust ratings across 10 sessions, with 16 trust trials per session, using only the task rules and experiment design plus an optional synthetic profile card.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the evaluation framing. The source dataset is documented in [../../non-PGG_generalization/data/longitudinal_trust_game_ht863/README.md](../../non-PGG_generalization/data/longitudinal_trust_game_ht863/README.md).

Active entrypoint:

- [`build_batch_inputs.py`](./build_batch_inputs.py)

Shared implementation:

- dataset adapter: [`forecasting/datasets/longitudinal_trust_game_ht863.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/datasets/longitudinal_trust_game_ht863.py)
- prompt builder: [`forecasting/prompts/longitudinal_trust_game_ht863.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/prompts/longitudinal_trust_game_ht863.py)
- run writer: [`forecasting/common/runs/non_pgg.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/common/runs/non_pgg.py)
