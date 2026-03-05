# Wave-Anchored OOD Split Datasets

Each split directory is a drop-in dataset root with the same structure as `benchmark/data`:
- `raw_data`
- `processed_data`
- `demographics`

Mapping used within each direction folder:
- `learning_wave` = train side sampled from original learning wave only
- `validation_wave` = test side sampled from original validation wave only

Rules:
- One CONFIG factor at a time (no multi-factor filtering).
- Both directions per factor.
- Numeric factors use per-wave medians (`<= median` vs `> median`).
- Boolean factors use per-wave `False` vs `True`.

Build command:
```bash
python benchmark/build_ood_splits_wave_anchored.py
```

See `summary.json` for counts and per-split selection shares.
