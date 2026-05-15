# Persona Transfer Audit: Comprehensive Pilot Evaluation

Run: `nemotron_raw_fields_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2`
Parsed requests: 1280
Candidate observed players: 342
Returned top-k rows: 3810

## Concentration

Top-1 player coverage: 136 / 342 (39.8%)
Top-k probability player coverage: 236 / 342 (69.0%)
Top-1 effective number of observed player identities: 88.32
Probability-mass effective number of observed player identities: 140.59
Top 5% of matched identities capture 19.3% of probability mass.
Median within-game top-1 effective N / players: 31.8%
Median within-game top player share across personas: 62.5%

## Avatar-Label Diagnostic

Avatar labels are reused across games and are not stable person identities. This table is only a label/order diagnostic.

| Avatar | Top-1 share | Probability share |
|---|---:|---:|
| GORILLA | 0.235 | 0.228 |
| SLOTH | 0.234 | 0.204 |
| DUCK | 0.122 | 0.132 |
| CHICKEN | 0.088 | 0.094 |
| DOG | 0.047 | 0.056 |
| PARROT | 0.029 | 0.049 |
| MOOSE | 0.058 | 0.047 |
| COW | 0.045 | 0.038 |
| RABBIT | 0.033 | 0.038 |
| CHICK | 0.038 | 0.035 |
| OWL | 0.043 | 0.035 |
| MONKEY | 0.010 | 0.014 |
| CROCODILE | 0.007 | 0.010 |
| SNAKE | 0.009 | 0.009 |
| PINGUIN | 0.002 | 0.004 |
| FROG | 0.000 | 0.003 |
| ELEPHANT | 0.000 | 0.002 |
| WHALE | 0.000 | 0.001 |
| HORSE | 0.000 | 0.000 |

## Matched Behavior Skew

| Metric | Matched | Candidate-uniform | Difference | Std. diff |
|---|---:|---:|---:|---:|
| mean_contribution_rate | 0.866 | 0.753 | 0.113 | 0.437 |
| full_contribution_rate | 0.763 | 0.638 | 0.125 | 0.352 |
| zero_contribution_rate | 0.069 | 0.147 | -0.077 | -0.334 |
| contribution_sd | 3.110 | 4.616 | -1.507 | -0.429 |
| messages_per_round | 0.223 | 0.137 | 0.086 | 0.295 |
| reward_given_round_rate | 0.194 | 0.129 | 0.065 | 0.241 |
| punish_given_round_rate | 0.093 | 0.082 | 0.011 | 0.058 |
| punish_received_round_rate | 0.048 | 0.078 | -0.031 | -0.172 |

## Most-Matched Observed Players

| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |
|---|---|---:|---:|---:|---:|
| 3EG5RaDj4z7hkT2JD::GORILLA | VALIDATION_10_T | 23.16 | 29 | 1.000 | 1.000 |
| 65crAkEscRbHNJXbW::SLOTH | VALIDATION_10_C | 22.81 | 29 | 1.000 | 1.000 |
| 3EQGxadfbmAKaACgZ::GORILLA | VALIDATION_5_T | 21.93 | 32 | 1.000 | 1.000 |
| Fm7uYc98E2b6kcoDE::SLOTH | VALIDATION_1_T | 21.78 | 28 | 0.906 | 0.875 |
| 2TDnnb4wn9rwy3CYZ::GORILLA | VALIDATION_2_T | 21.45 | 27 | 0.645 | 0.000 |
| 2khRAszouj8vP8ZfP::DUCK | VALIDATION_18_C | 20.98 | 29 | 0.957 | 0.957 |
| 6Pxb9DsQpyk22cJNr::SLOTH | VALIDATION_1_C | 20.83 | 29 | 0.828 | 0.438 |
| ASQtYBWx6cGN27KxL::GORILLA | VALIDATION_12_C | 20.52 | 32 | 1.000 | 1.000 |
| 6sSM4xSyZQkaxyHav::DUCK | VALIDATION_5_C | 19.44 | 27 | 1.000 | 1.000 |
| 7ywewvEuctuz4Rq8j::SLOTH | VALIDATION_3_C | 18.87 | 29 | 0.913 | 0.654 |
