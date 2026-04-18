# Chip Bargaining

This folder is the active forecasting scaffold for the chip-bargain benchmark.

Current status:

- dataset adapter is implemented in [`../datasets/chip_bargain.py`](../datasets/chip_bargain.py)
- prompt builder is implemented in [`../prompts/chip_bargain.py`](../prompts/chip_bargain.py)
- dedicated multi-player batch generation is implemented in [`./build_batch_inputs.py`](./build_batch_inputs.py)
- parse/eval stack is implemented in:
  - [`./common.py`](./common.py)
  - [`./parse_outputs.py`](./parse_outputs.py)
  - [`./evaluate_outputs.py`](./evaluate_outputs.py)
  - [`./analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)
  - [`./compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)
- supported variants in the dedicated builder are:
  - `baseline`
  - `twin_sampled_unadjusted_seed_0`

## Prediction Unit

The current unit is:

- one request = one complete cohort-stage bargaining game

Each record contains:

- one chip family: `chip2`, `chip3`, or `chip4`
- one cohort of three players
- one standalone game stage
- 3 rounds
- 3 turns per round

The second stage in each cohort is labeled `Second negotiation game-negotiation2_alternate_profile` in the raw data. The manuscript states that participants completed two independent games with different chip valuations and profiles, and that the two games are evaluated independently.

## Current Design Cells

The scaffold currently treats `chip family × stage` as the repeated design cell:

- `CHIP2__GAME_1`
- `CHIP2__GAME_2_ALT_PROFILE`
- `CHIP3__GAME_1`
- `CHIP3__GAME_2_ALT_PROFILE`
- `CHIP4__GAME_1`
- `CHIP4__GAME_2_ALT_PROFILE`

Each cell has `24` human games.

## Variant Support

This dataset is structurally closer to PGG than the other non-PGG benchmarks because the raw logs contain full multi-player game trajectories.

Current support:

- `baseline` works directly
- `twin_sampled_unadjusted_seed_0` now works through a PGG-style assignment path:
  - one sampled Twin persona per player
  - the same sampled persona is reused for that player across both game stages within the same chip family
  - the active Twin card path is `chip_bargain_prompt_min`, not the older `pgg_prompt_min`
  - an experimental descriptive alternative also exists at `chip_bargain_descriptive_prompt_min`, which keeps direct Twin trust/ultimatum/dictator task summaries and avoids extra bargaining-specific cue weighting
  - two ablation alternatives also exist for Twin-only chip-bargain runs:
    - `chip_bargain_no_econ_games_prompt_min`: removes the direct trust/ultimatum/dictator task summaries
    - `chip_bargain_ultimatum_only_prompt_min`: keeps only the direct ultimatum-task summaries

Not supported yet:

- `demographic_only_row_resampled_seed_0`
- `twin_sampled_seed_0`

Reason:

- the checked chip-bargain files do not expose the standard demographic fields used by the adjusted non-PGG profile-sampling pipeline
- so only the unadjusted Twin variant is currently well-defined

## Entry Point

To build the default chip-bargain batch inputs:

```bash
python forecasting/chip_bargain/build_batch_inputs.py
```

This dedicated builder writes:

- `baseline`
- `twin_sampled_unadjusted_seed_0`

Before running the Twin-unadjusted variant, generate the player assignments:

```bash
python forecasting/chip_bargain/profile_sampling/sample_twin_personas_for_chip_bargain.py
```

## Data Sources

Raw source data live at:

- [`../../non-PGG_generalization/data/chip_bargain/`](../../non-PGG_generalization/data/chip_bargain/)

Useful references in that source folder:

- `manuscript.pdf`
- `chip_chat_4Chips_simulation.ipynb`

## Primary Evaluation Metrics

The primary paper-aligned metrics are:

- `final_surplus_ratio`
  - game-level Wasserstein distance on final total surplus divided by the Pareto-optimal surplus benchmark
  - aligns to Figure 2 / Table 3 in the manuscript
- `proposer_net_surplus`
  - turn-level Wasserstein distance on the proposer's surplus if the offered trade were accepted
  - aligns to the top panel of Figure 3
- `trade_ratio`
  - turn-level Wasserstein distance on `sell_quantity / buy_quantity`
  - aligns to the bottom panel of Figure 3
- `acceptance_rate`
  - turn-level Wasserstein distance on accepted vs declined proposals
  - aligns to the manuscript's emphasis that human and model proposals differ in how often they are accepted

Secondary metrics currently included in the treatment-level tables:

- `accepted_proposer_net_surplus`
- `declined_proposer_net_surplus`
- `accepted_trade_ratio`
- `declined_trade_ratio`
- `final_total_surplus`
- `mean_player_final_surplus`

## Important Caveat

The raw human logs do not perfectly preserve one fixed proposer order across all three rounds in every session. The active prompt/schema now carry the proposer schedule round by round, and the validator can enforce that schedule when it is present.

For backward compatibility, the eval stack still supports older manifests that only recorded one `turn_order` list. In that fallback mode it validates the weaker game-mechanics constraint:

- each round must contain exactly one turn from each player

The currently shipped `gpt-5-mini` runs predate this prompt correction, so those particular batch outputs were generated under the older fixed-order wording.

## Planned Next Steps

- add plotting scripts for the primary distance summaries
- decide whether adjusted demographic variants can be enabled if a separate demographic export is recovered
