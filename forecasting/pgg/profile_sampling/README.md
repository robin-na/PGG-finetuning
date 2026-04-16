# PGG Profile Sampling

This folder is the canonical home for PGG-specific profile assignment artifacts.

Use it for:

- Twin-to-PGG seat assignment scripts
- demographic-only PGG seat assignment scripts
- PGG-specific assignment outputs consumed by the active PGG batch builder

Do not use this folder as the source of shared Twin profiles or shared Twin prompt cards. Those live in:

- [`../../../non-PGG_generalization/twin_profiles/`](../../../non-PGG_generalization/twin_profiles/)

## Canonical Outputs

Active PGG build inputs live under:

- [`output/twin_to_pgg_validation_persona_sampling/`](./output/twin_to_pgg_validation_persona_sampling/)
- [`output/twin_to_pgg_validation_persona_sampling_unadjusted/`](./output/twin_to_pgg_validation_persona_sampling_unadjusted/)
- [`output/pgg_validation_demographic_only_sampling_row_resampled/`](./output/pgg_validation_demographic_only_sampling_row_resampled/)

Older demographic-only output is preserved here for reference:

- [`output/pgg_validation_demographic_only_sampling/`](./output/pgg_validation_demographic_only_sampling/)

## Compatibility

The older paths under:

- `non-PGG_generalization/twin_profiles/`
- `non-PGG_generalization/task_grounding/`

are kept as symlinks for backward compatibility with existing manifests, registries, and summaries. New code should point here instead.
