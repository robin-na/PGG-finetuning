# Minority Game + BRET Forecasting

This subfolder sketches a `forecasting/` extension for the minority-game plus BRET dataset. The research question matches the top-level PGG benchmark: does sampling participant profile cards from the Twin dataset help LLMs make better from-scratch predictions of human behavior?

This dataset is the closest non-PGG match to the current full-rollout setup. The primary target is a full `11`-round sequence of `A/B` decisions in the bonus game, optionally paired with a separate BRET risk choice.

Start with [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the proposed benchmark shape and [ANALYSIS_OVERVIEW.md](ANALYSIS_OVERVIEW.md) for the proposed evaluation. The source dataset is documented in [../../non-PGG_generalization/data/minority_game_bret_njzas/README.md](../../non-PGG_generalization/data/minority_game_bret_njzas/README.md).

Status: documentation only. No dataset-specific build, parse, or evaluation code lives in this subfolder yet.
