# non-PGG_generalization

This folder stores consolidated non-split assets for cross-task generalization workflows.

## Data layout

`non-PGG_generalization/data/` is organized by source family:

- `PGG/`
  - Consolidated PGG demographics and oracle archetype JSONL files.
- `Twin-2k-500/`
  - Placeholder folder for Twin-2k-500 data and metadata.

## Regenerate

From repo root:

```bash
python non-PGG_generalization/build_consolidated_data.py
```
