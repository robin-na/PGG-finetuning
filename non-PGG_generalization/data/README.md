# Data

This folder contains raw or near-raw datasets.

The key distinction here is:

- active target datasets used by the current forecasting benchmarks
- parked datasets kept locally for possible future work

## Active Datasets

These are the datasets currently wired into `forecasting/`:

- [`minority_game_bret_njzas/`](./minority_game_bret_njzas/)
- [`longitudinal_trust_game_ht863/`](./longitudinal_trust_game_ht863/)
- [`two_stage_trust_punishment_y2hgu/`](./two_stage_trust_punishment_y2hgu/)
- [`multi_game_llm_fvk2c/`](./multi_game_llm_fvk2c/)

These are read by:

- [`../../forecasting/datasets/minority_game_bret_njzas.py`](../../forecasting/datasets/minority_game_bret_njzas.py)
- [`../../forecasting/datasets/longitudinal_trust_game_ht863.py`](../../forecasting/datasets/longitudinal_trust_game_ht863.py)
- [`../../forecasting/datasets/two_stage_trust_punishment_y2hgu.py`](../../forecasting/datasets/two_stage_trust_punishment_y2hgu.py)
- [`../../forecasting/datasets/multi_game_llm_fvk2c.py`](../../forecasting/datasets/multi_game_llm_fvk2c.py)

## Active Twin Source Data

The current Twin profile build path uses:

- [`Twin-2k-500/`](./Twin-2k-500/)

This feeds the deterministic profile/card pipeline under:

- [`../twin_profiles/README.md`](../twin_profiles/README.md)

## Other Local Datasets

These folders are currently parked. They are kept in the repo, but they are not on the active non-PGG forecasting path:

- `MobLab/`
- `PGG/`
- `chip_bargain/`
- `delay_discounting_age_income/`
- `information_sharing_motives/`
- `social_media_moral_judgment/`

Important note on `PGG/`:

- the active PGG benchmark does not currently read [`PGG/`](./PGG/)
- it reads the repo-root [`../../data/`](../../data/) tree through [`../../forecasting/pgg/`](../../forecasting/pgg/)
- the copy under `non-PGG_generalization/data/PGG/` should be treated as parked unless the active PGG pipeline is migrated intentionally

Keep these parked paths stable unless you are intentionally promoting one of them into the active forecasting pipeline and updating `forecasting/datasets/` or `forecasting/pgg/` accordingly.
