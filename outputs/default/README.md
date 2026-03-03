# Default Outputs

This directory stores generated artifacts from the original dataset root:

- `data/`
- `demographics/`

## Structure

- `runs/source_default/`
  - Outputs for the original, non-benchmark dataset.
  - Example:
    - `runs/source_default/micro_behavior_eval/no_archetype/<run_id>/...`
    - `runs/source_default/micro_behavior_eval/oracle_archetype/<run_id>/...`

Use `outputs/benchmark/` only for `benchmark/data` and `benchmark/data_ood_splits`.
