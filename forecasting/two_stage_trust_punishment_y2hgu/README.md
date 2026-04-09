# Two-Stage Trust / Punishment / Helping Forecasting

This subfolder sketches a `forecasting/` extension for the two-stage trust and punishment-helping dataset. The research question again mirrors the top-level PGG benchmark: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

This benchmark is not a transcript rollout. It is a structured stage-conditioned prediction task with role-specific outputs: Player A makes a deliberation and prosocial choice and then chooses a trust-game return, while Player B states conditional trust decisions given what Player A did.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the proposed benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the proposed evaluation. The source dataset is documented in [../../non-PGG_generalization/data/two_stage_trust_punishment_y2hgu/README.md](../../non-PGG_generalization/data/two_stage_trust_punishment_y2hgu/README.md).

Status: documentation only. No dataset-specific build, parse, or evaluation code lives in this subfolder yet.
