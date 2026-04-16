# Active Paths

This file maps the parts of `non-PGG_generalization/` that are actively used by the current forecasting pipeline.

## 1. Active Twin Source Data

The deterministic Twin profile pipeline currently reads from:

- [`data/Twin-2k-500/snapshot/question_catalog_and_human_response_csv/`](./data/Twin-2k-500/snapshot/question_catalog_and_human_response_csv/)
- [`data/Twin-2k-500/wave_split_dataset/`](./data/Twin-2k-500/wave_split_dataset/)

Important scripts:

- [`twin_profiles/build_twin_extended_profiles.py`](./twin_profiles/build_twin_extended_profiles.py)
- [`twin_profiles/render_twin_extended_profile_cards.py`](./twin_profiles/render_twin_extended_profile_cards.py)

## 2. Active Twin Derived Artifacts

The forecasting pipeline currently consumes these deterministic outputs:

- [`twin_profiles/output/twin_extended_profiles/twin_extended_profiles.jsonl`](./twin_profiles/output/twin_extended_profiles/twin_extended_profiles.jsonl)
- [`twin_profiles/output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`](./twin_profiles/output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl)

These are loaded by the shared forecasting run layer, not rebuilt at forecasting time.

Compatibility note:

- `task_grounding/` is kept as a symlink to `twin_profiles/`
- older generated manifests and summaries may still point at `task_grounding/`
- new code should treat `twin_profiles/` as canonical

## 3. Active Target Datasets

These raw datasets currently feed the active non-PGG forecasting benchmarks:

- [`data/minority_game_bret_njzas/`](./data/minority_game_bret_njzas/)
- [`data/longitudinal_trust_game_ht863/`](./data/longitudinal_trust_game_ht863/)
- [`data/two_stage_trust_punishment_y2hgu/`](./data/two_stage_trust_punishment_y2hgu/)
- [`data/multi_game_llm_fvk2c/`](./data/multi_game_llm_fvk2c/)

Note:

- the active PGG benchmark is intentionally not listed here
- active PGG data currently comes from the repo-root [`../data/`](../data/) tree through [`../forecasting/pgg/`](../forecasting/pgg/)
- [`data/PGG/`](./data/PGG/) is a parked local copy, not part of the active pipeline

Their active adapters live in:

- [`../forecasting/datasets/`](../forecasting/datasets/)

## 4. Active Consumer Side

The active benchmark pipeline that consumes the above inputs lives in:

- [`../forecasting/common/profiles/`](../forecasting/common/profiles/)
- [`../forecasting/common/runs/`](../forecasting/common/runs/)
- [`../forecasting/prompts/`](../forecasting/prompts/)
- [`../forecasting/non_pgg_batch_builder.py`](../forecasting/non_pgg_batch_builder.py)

Benchmark-specific entrypoints then live in each forecasting subfolder:

- [`../forecasting/minority_game_bret_njzas/`](../forecasting/minority_game_bret_njzas/)
- [`../forecasting/longitudinal_trust_game_ht863/`](../forecasting/longitudinal_trust_game_ht863/)
- [`../forecasting/two_stage_trust_punishment_y2hgu/`](../forecasting/two_stage_trust_punishment_y2hgu/)
- [`../forecasting/multi_game_llm_fvk2c/`](../forecasting/multi_game_llm_fvk2c/)

PGG-specific seat assignment artifacts are now maintained separately under:

- [`../forecasting/pgg/profile_sampling/`](../forecasting/pgg/profile_sampling/)

## 5. Not On The Active Path

These folders are preserved, but they are not part of the current mainline forecasting workflow:

- [`archive/`](./archive/README.md)
  - `archive/legacy_demographicsOnly/`
  - `archive/pgg_transfer_eval/`
  - `archive/pgg_archetype_transfer/`
  - `archive/twin-20k-500/`
  - `archive/paper/`
- parked datasets in `data/` that do not yet have active adapters in `forecasting/datasets/`
