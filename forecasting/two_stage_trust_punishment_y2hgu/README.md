# Two-Stage Trust / Punishment / Helping Forecasting

This subfolder is the active forecasting benchmark for the two-stage trust and punishment-helping dataset. The research question again mirrors the top-level PGG benchmark: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

This benchmark is not a transcript rollout. It is a structured stage-conditioned prediction task with role-specific outputs: Player A makes a deliberation and prosocial choice and then chooses a trust-game return, while Player B states conditional trust decisions given what Player A did.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the evaluation framing. The source dataset is documented in [../../non-PGG_generalization/data/two_stage_trust_punishment_y2hgu/README.md](../../non-PGG_generalization/data/two_stage_trust_punishment_y2hgu/README.md).

Active entrypoints:

- [`build_batch_inputs.py`](./build_batch_inputs.py)
- [`parse_outputs.py`](./parse_outputs.py)
- [`evaluate_outputs.py`](./evaluate_outputs.py)
- [`analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)
- [`compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)

Shared implementation:

- dataset adapter: [`forecasting/datasets/two_stage_trust_punishment_y2hgu.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/datasets/two_stage_trust_punishment_y2hgu.py)
- prompt builder: [`forecasting/prompts/two_stage_trust_punishment_y2hgu.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/prompts/two_stage_trust_punishment_y2hgu.py)
- run writer: [`forecasting/common/runs/non_pgg.py`](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/forecasting/common/runs/non_pgg.py)
