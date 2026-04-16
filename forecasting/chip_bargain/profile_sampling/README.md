# Chip Bargain Profile Sampling

This folder holds chip-bargain-specific profile assignment artifacts.

Current active use:

- `twin_to_chip_bargain_player_sampling_unadjusted/`
  - one unadjusted Twin persona sampled per unique chip-bargain player
  - assignments are then expanded to one `game_assignments.jsonl` row per bargaining game

Why this is separate from `non-PGG_generalization/twin_profiles/`:

- `twin_profiles/` is the shared Twin artifact factory
- this folder is benchmark-specific assignment output, analogous to [`forecasting/pgg/profile_sampling/`](../pgg/profile_sampling/)
