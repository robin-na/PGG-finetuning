# non-PGG_generalization

This folder is the upstream data and Twin-artifact side of the transportability pipeline.

Use it for:

- raw target datasets used by the current forecasting benchmarks
- raw Twin source data and derived deterministic Twin profile artifacts
- older or historical non-PGG experiments that are preserved for reference

Do not treat this folder as the active benchmark runner. The current forecasting and evaluation pipeline lives in:

- [`../forecasting/README.md`](../forecasting/README.md)

## Start Here

If you are trying to understand the current active pipeline, read these in order:

1. [`ACTIVE_PATHS.md`](./ACTIVE_PATHS.md)
2. [`data/README.md`](./data/README.md)
3. [`twin_profiles/README.md`](./twin_profiles/README.md)
4. [`../forecasting/README.md`](../forecasting/README.md)

## Current Role In The Pipeline

The split is:

- `non-PGG_generalization/`
  - upstream raw data
  - Twin profile/card construction
  - historical non-mainline experiments
- `forecasting/`
  - active benchmark assembly
  - prompt construction
  - batch input generation
  - parsing, evaluation, and plotting

## Main Folders

- [`data/`](./data/README.md)
  - raw target datasets and Twin source snapshots
- [`twin_profiles/`](./twin_profiles/README.md)
  - deterministic Twin profile/card build pipeline and related specifications
- [`archive/`](./archive/README.md)
  - historical prototype work, older transfer experiments, and draft paper assets that are not part of the active pipeline

## Active vs Historical

Active for the current forecasting pipeline:

- Twin source data under `data/Twin-2k-500/`
- deterministic Twin artifacts under `twin_profiles/output/`
- target datasets under:
  - `data/minority_game_bret_njzas/`
  - `data/longitudinal_trust_game_ht863/`
  - `data/two_stage_trust_punishment_y2hgu/`
  - `data/multi_game_llm_fvk2c/`

Historical or not on the active path:

- `archive/legacy_demographicsOnly/`
- `archive/pgg_transfer_eval/`
- `archive/pgg_archetype_transfer/`
- `archive/twin-20k-500/`
- `archive/paper/`
- the other parked datasets under `data/` that are not yet wired into `forecasting/`

## Notes

- Keep raw dataset locations stable unless the active adapters in `forecasting/datasets/` are updated too.
- `task_grounding/` is kept as a compatibility symlink to `twin_profiles/` so older generated artifacts do not need path rewrites.
- The active PGG benchmark does not read `non-PGG_generalization/data/PGG/`; it reads the repo-root [`../data/`](../data/) tree through [`../forecasting/pgg/`](../forecasting/pgg/).
