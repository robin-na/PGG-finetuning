# Archetype-Augmented Regression Outputs

This folder stores run artifacts (machine outputs) for archetype-augmented regression.

## OOD Runs

- Root:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/`
- Timestamped runs:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/runs/<run_id>/`
- Latest pointer:
  - `outputs/archetype_augmented_regression/ood_wave_anchored/latest_run.txt`

Each run folder contains:

- per-split `results.csv` and `summary.json`,
- aggregate comparison tables,
- `noise_ceiling_by_split.csv` and `noise_ceiling_summary.csv`,
- `run_manifest.json`.
