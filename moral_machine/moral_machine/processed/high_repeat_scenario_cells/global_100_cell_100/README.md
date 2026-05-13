# High-Repeat Moral Machine Scenario Cells

This directory contains benchmark-ready aggregate tables built from exact repeated
Moral Machine Stay/Swerve scenarios.

- Minimum global scenario count: `100`
- Minimum scenario-country cell count: `100`
- `scenarios.csv`: one row per retained exact scenario, including prompt-ready text.
- `cells.csv`: one row per retained `scenario_hash × UserCountry3` cell, including observed A/B shares.
- `scenarios.jsonl` and `cells.jsonl`: line-oriented copies of the same tables.
- `scenario_cells.sqlite`: indexed SQLite copy for fast sampling and joins.

Answer labels: `A = stay`, `B = swerve`.
