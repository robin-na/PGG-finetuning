# Twin Profiles

This folder is the active deterministic Twin profile/card construction layer.

`non-PGG_generalization/twin_profiles/` is now the canonical path.

The legacy path `non-PGG_generalization/task_grounding/` is kept as a compatibility symlink so that older generated manifests and summaries do not need to be rewritten.

It is upstream of `forecasting/`. In other words:

- `twin_profiles/` builds reusable Twin-derived artifacts
- `forecasting/` samples from those artifacts and uses them in benchmark prompts
- PGG-specific seat assignment scripts and outputs now live under [`../../forecasting/pgg/profile_sampling/`](../../forecasting/pgg/profile_sampling/)

## Active Scripts

The scripts that matter for the current pipeline are:

- [`build_twin_extended_profiles.py`](./build_twin_extended_profiles.py)
  - builds deterministic extended Twin profiles from the raw Twin responses
- [`render_twin_extended_profile_cards.py`](./render_twin_extended_profile_cards.py)
  - renders prompt-facing profile cards from those profiles

Supporting specs and mappings:

- [`TWIN_EXTENDED_PROFILE_SPEC.md`](./TWIN_EXTENDED_PROFILE_SPEC.md)
- [`twin_extended_profile_mapping.csv`](./twin_extended_profile_mapping.csv)
- [`twin_extended_profile_schema.json`](./twin_extended_profile_schema.json)
- [`twin_extended_profile_card_schema.json`](./twin_extended_profile_card_schema.json)

## Active Outputs

The forecasting pipeline currently consumes:

- [`output/twin_extended_profiles/twin_extended_profiles.jsonl`](./output/twin_extended_profiles/twin_extended_profiles.jsonl)
- [`output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl`](./output/twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl)

The active consumer side is:

- [`../../forecasting/common/profiles/`](../../forecasting/common/profiles/)
- [`../../forecasting/common/runs/non_pgg.py`](../../forecasting/common/runs/non_pgg.py)

## PGG Compatibility Links

PGG-specific sampling scripts, notes, and output directories were moved to [`../../forecasting/pgg/profile_sampling/`](../../forecasting/pgg/profile_sampling/).

Any PGG-specific files that still appear here are compatibility symlinks only, kept so older manifests and summaries do not need path rewrites.

## Notes

- Treat this folder as deterministic artifact generation, not as the live benchmarking layer.
- If the active forecasting pipeline changes which card flavor it consumes, update this README and [`../ACTIVE_PATHS.md`](../ACTIVE_PATHS.md).
