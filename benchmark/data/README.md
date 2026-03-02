# Filtered Dataset (Condition 2)

This folder contains a mirrored copy of the project dataset filtered with:

1. `valid_number_of_starting_players == TRUE` (from `data/processed_data/df_analysis_learn.csv` and `data/processed_data/df_analysis_val.csv`)
2. Keep only games where **every player in that game has at least one demographic field available** (`age` or `gender_code` or `education_code`) using:
   - `demographics/demographics_numeric_learn.csv`
   - `demographics/demographics_numeric_val.csv`

Builder script: `benchmark/build_filtered_dataset.py`

Run:

```bash
python benchmark/build_filtered_dataset.py
```

## Top-Level Summary

- Learning: 214 games, 26,979 actions (`player-rounds` rows)
- Validation: 249 games, 22,506 actions (`player-rounds` rows)
- Combined: 463 games, 49,485 actions

## File Row Counts

### `raw_data/learning_wave`

- `batches.csv`: 77
- `factor-types.csv`: 110
- `factors.csv`: 1005
- `game-lobbies.csv`: 214
- `games.csv`: 214
- `lobby-configs.csv`: 7
- `player-inputs.csv`: 2002
- `player-rounds.csv`: 26979
- `player-stages.csv`: 81743
- `players.csv`: 2002
- `rounds.csv`: 3017
- `stages.csv`: 9139
- `treatments.csv`: 206

### `raw_data/validation_wave`

- `batches.csv`: 69
- `factor-types.csv`: 66
- `factors.csv`: 300
- `game-lobbies.csv`: 249
- `games.csv`: 249
- `lobby-configs.csv`: 4
- `player-inputs.csv`: 1654
- `player-rounds.csv`: 22506
- `player-stages.csv`: 69178
- `players.csv`: 1654
- `rounds.csv`: 3404
- `stages.csv`: 10462
- `treatments.csv`: 102

### `processed_data`

- `df_analysis_learn.csv`: 214
- `df_analysis_val.csv`: 249
- `df_analysis_val_dedup.csv`: 25
- `df_paired_learn.csv`: 124
- `df_paired_val.csv`: 20
- `df_rounds_learn.csv`: 26979
- `df_rounds_val.csv`: 22506
- `prediction_survey.csv`: 10723

### `demographics`

- `demographics_numeric_learn.csv`: 2002
- `demographics_numeric_val.csv`: 1654

## Notes

- Raw and processed files keep the original names/structure under `benchmark/data`.
- `subset_summary_condition2.json` stores the main aggregate counts.
