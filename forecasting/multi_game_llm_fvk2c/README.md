# Multi-Game Battery with LLM Delegation Forecasting

This subfolder sketches a `forecasting/` extension for the multi-game delegation dataset. The research question matches the top-level PGG benchmark: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

This benchmark is best treated as a subject-level experiment-session forecast rather than a transcript rollout. One LLM request should predict one participant's full battery for one randomized treatment assignment, including every scenario nested inside that same session.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the proposed benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the proposed evaluation. The source dataset is documented in [../../non-PGG_generalization/data/multi_game_llm_fvk2c/README.md](../../non-PGG_generalization/data/multi_game_llm_fvk2c/README.md).

Critical unit definition:

- one row / one LLM request = one subject-level experiment session
- one design = one treatment arm (`TRP`, `TRU`, `TDP`, `TDU`, `ODP`, `ODU`)
- repeated rows within the same design differ by which participant was recruited into that treatment arm
- `Scenario` and `Case` are nested within the row and should be used for scoring after parsing, not for independent sampling

Status: batch-input generation is implemented. Sampling should happen at the row/session level, never by resampling scenario content inside a row.
