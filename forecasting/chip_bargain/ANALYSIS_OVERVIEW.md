# Analysis Overview

The chip-bargain benchmark now has a PGG-style parse/eval stack.

Core files:

- [`./common.py`](./common.py)
- [`./parse_outputs.py`](./parse_outputs.py)
- [`./evaluate_outputs.py`](./evaluate_outputs.py)
- [`./analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)
- [`./compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)

## Canonical Unit

The scoring unit is:

- one complete 3-player game
- 3 rounds
- 3 turns per round

The parser validates model outputs against the bargaining mechanics, then re-simulates the game turn by turn to derive:

- per-turn proposal features
- per-turn group surplus trajectory
- final group surplus
- final player-level surplus

Human gold targets and model outputs are both normalized through the same simulator before scoring.

## Primary Metrics

The headline metrics are chosen from the manuscript's main performance and process analyses:

1. `final_surplus_ratio`
   - game-level WD on final total surplus divided by Pareto-optimal surplus
   - aligns to Figure 2 / Table 3
2. `proposer_net_surplus`
   - turn-level WD on the proposer-side surplus implied by the offer
   - aligns to the top panel of Figure 3
3. `trade_ratio`
   - turn-level WD on `sell_quantity / buy_quantity`
   - aligns to the bottom panel of Figure 3
4. `acceptance_rate`
   - turn-level WD on accepted vs declined proposals
   - aligns to the paper's emphasis on differences in offer acceptance

Secondary metrics in the full treatment tables:

- `accepted_proposer_net_surplus`
- `declined_proposer_net_surplus`
- `accepted_trade_ratio`
- `declined_trade_ratio`
- `final_total_surplus`
- `mean_player_final_surplus`

## Macro vs Micro Figures

The intended figure split now matches the PGG reporting style:

- macro figure:
  - game-level Wasserstein distances
  - `final_surplus_ratio`
  - `mean_trade_ratio`
  - `mean_acceptance_rate`
- micro figure:
  - player-level Wasserstein distances:
    - `player_final_surplus`
    - `player_proposer_mean_trade_ratio`
    - `player_proposer_acceptance_rate`
  - round-level Wasserstein distances:
    - `round_end_surplus_ratio`
    - `round_mean_trade_ratio`
    - `round_acceptance_rate`

Plot scripts:

- [`./plot_model_family_macro_panels.py`](./plot_model_family_macro_panels.py)
- [`./plot_model_family_micro_panels.py`](./plot_model_family_micro_panels.py)

## Noise Ceiling

The noise ceiling is bootstrapped at the game level within treatment cell:

- resample whole games within each `chip_family × stage` treatment
- expand sampled games back to their 9 turn rows
- compare pseudo-generated human samples to pseudo-human human samples

This preserves the within-game dependence structure, which is the chip-bargain analogue of the PGG clustered bootstrap.

Figure conventions:

- model bars:
  - bar height = mean score across treatment cells
  - error bar = standard error across treatment cells
- human ceiling bar:
  - bar height = bootstrap mean of the human-vs-human score
  - error bar = bootstrap 5th to 95th percentile interval

## Current Caveat

Some raw human sessions do not preserve one fixed proposer order across all three rounds. The eval stack therefore requires only:

- each round contains exactly one turn from each player

rather than enforcing one repeated sender order across rounds.
