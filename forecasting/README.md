# Forecasting

This directory is the forecasting benchmark root.

Each target experiment now has its own subfolder:

- [pgg](./pgg/README.md): public goods game full-rollout forecasting
- [chip_bargain](./chip_bargain/README.md): full-game bargaining benchmark with a dedicated multi-player builder
- [minority_game_bret_njzas](./minority_game_bret_njzas/README.md)
- [longitudinal_trust_game_ht863](./longitudinal_trust_game_ht863/README.md)
- [two_stage_trust_punishment_y2hgu](./two_stage_trust_punishment_y2hgu/README.md)
- [multi_game_llm_fvk2c](./multi_game_llm_fvk2c/README.md)

Shared infrastructure lives at the top level:

- `common/`: shared profile and run-writing code
- `datasets/`: canonical dataset adapters for non-PGG benchmarks
- `prompts/`: shared prompt builders for non-PGG benchmarks
- `non_pgg_batch_builder.py`: thin CLI entrypoint for non-PGG batch generation
- `kl_divergence_utils.py`: shared KL helpers used by multiple benchmarks

Project-level documents:

- [CORE_NARRATIVE.md](./CORE_NARRATIVE.md)
- [DECISION_LOG.md](./DECISION_LOG.md)
- [REFACTOR_PLAN.md](./REFACTOR_PLAN.md)

The active PGG pipeline no longer lives directly in `forecasting/`. Use `forecasting/pgg/` as the canonical home for:

- batch construction
- batch management
- parsing
- evaluation
- PGG metadata, batch files, and results

Secondary or non-mainline PGG analyses now live under `forecasting/pgg/exploratory/`.
