# OOD Split Datasets

Each split directory is a drop-in dataset root with the same structure as `benchmark/data`:
- `raw_data`
- `processed_data`
- `demographics`

Mapping used within each direction folder:
- `learning_wave` = train side of the split
- `validation_wave` = test side of the split

Rules:
- One CONFIG factor at a time (no multi-factor filtering).
- Both directions per factor.
- Numeric factors use `low <= median` and `high > median`.

Build command:
```bash
python benchmark/build_ood_splits.py
```

See `summary.json` for counts and per-split details.
