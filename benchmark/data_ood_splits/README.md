# OOD Split Datasets

Each split directory is a drop-in dataset root with the same structure as `benchmark/data`:

- `raw_data`
- `processed_data`
- `demographics`

Mapping inside each direction folder:

- `learning_wave`: train side
- `validation_wave`: test side

Rules used to build these splits:

- One CONFIG factor at a time (no multi-factor filtering)
- Both directions per factor
- Numeric factors use `low <= median` and `high > median`

Build:
```bash
python benchmark/build_ood_splits.py
```

See each direction folder's `summary.json` for counts/details.

## Archetype Banks for a Split

To build split-specific archetype retrieval banks:
```bash
python benchmark/scripts/build_split_archetype_banks.py \
  --split-root benchmark/data_ood_splits/<factor>/<direction>
```

Default output is outside `benchmark/`, at:

- `outputs/benchmark/runs/benchmark_ood/<factor>/<direction>/archetype_retrieval/learning_wave`
- `outputs/benchmark/runs/benchmark_ood/<factor>/<direction>/archetype_retrieval/validation_wave`

This keeps `benchmark/data_ood_splits/*` data-only.
