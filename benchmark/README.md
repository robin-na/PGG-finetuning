# Benchmark Data Builders

This folder contains dataset builders for filtered benchmark data and one-factor OOD split datasets.

## 1) Build the condition-2 filtered base dataset

```bash
python benchmark/build_filtered_dataset.py
```

Output root:
- `benchmark/data`

Rule:
- Keep only games with `valid_number_of_starting_players == TRUE`
- Keep only games where every player has at least one demographic field available (`age` or `gender_code` or `education_code`)

## 2) Build one-factor OOD split datasets

```bash
python benchmark/build_ood_splits.py
```

Output root:
- `benchmark/data_ood_splits`

Layout:
- `benchmark/data_ood_splits/<factor>/<direction>/` is a drop-in data root
- Each direction folder contains:
  - `raw_data/learning_wave` (train side)
  - `raw_data/validation_wave` (test side)
  - `processed_data`
  - `demographics`

Factors:
- `player_count`
- `num_rounds`
- `all_or_nothing`
- `default_contrib_prop`
- `reward_exists`
- `show_n_rounds`
- `show_punishment_id`
- `show_other_summaries`
- `mpcr`
