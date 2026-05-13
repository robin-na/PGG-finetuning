# Persona Transfer Audit: Comprehensive Pilot Evaluation

Run: `twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2`
Parsed requests: 1279
Candidate observed players: 342
Returned top-k rows: 3735

## Concentration

Top-1 player coverage: 200 / 342 (58.5%)
Top-k probability player coverage: 313 / 342 (91.5%)
Top-1 effective number of observed player identities: 131.55
Probability-mass effective number of observed player identities: 188.44
Top 5% of matched identities capture 20.3% of probability mass.
Median within-game top-1 effective N / players: 47.0%
Median within-game top player share across personas: 54.0%

## Avatar-Label Diagnostic

Avatar labels are reused across games and are not stable person identities. This table is only a label/order diagnostic.

| Avatar | Top-1 share | Probability share |
|---|---:|---:|
| SLOTH | 0.357 | 0.293 |
| GORILLA | 0.181 | 0.186 |
| DUCK | 0.103 | 0.121 |
| CHICKEN | 0.077 | 0.087 |
| DOG | 0.050 | 0.062 |
| PARROT | 0.032 | 0.047 |
| CHICK | 0.052 | 0.038 |
| MOOSE | 0.027 | 0.033 |
| OWL | 0.037 | 0.033 |
| RABBIT | 0.030 | 0.032 |
| COW | 0.022 | 0.023 |
| SNAKE | 0.014 | 0.013 |
| MONKEY | 0.005 | 0.011 |
| CROCODILE | 0.007 | 0.010 |
| ELEPHANT | 0.004 | 0.003 |
| PINGUIN | 0.000 | 0.003 |
| FROG | 0.000 | 0.002 |
| WHALE | 0.001 | 0.001 |
| HORSE | 0.001 | 0.001 |

## Matched Behavior Skew

| Metric | Matched | Candidate-uniform | Difference | Std. diff |
|---|---:|---:|---:|---:|
| mean_contribution_rate | 0.831 | 0.753 | 0.077 | 0.300 |
| full_contribution_rate | 0.726 | 0.638 | 0.089 | 0.250 |
| zero_contribution_rate | 0.095 | 0.147 | -0.052 | -0.223 |
| contribution_sd | 3.474 | 4.616 | -1.143 | -0.326 |
| messages_per_round | 0.179 | 0.137 | 0.042 | 0.144 |
| reward_given_round_rate | 0.165 | 0.129 | 0.036 | 0.134 |
| punish_given_round_rate | 0.093 | 0.082 | 0.011 | 0.061 |
| punish_received_round_rate | 0.056 | 0.078 | -0.022 | -0.124 |

## Most-Matched Observed Players

| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |
|---|---|---:|---:|---:|---:|
| 65crAkEscRbHNJXbW::SLOTH | VALIDATION_10_C | 19.79 | 25 | 1.000 | 1.000 |
| 3EQGxadfbmAKaACgZ::GORILLA | VALIDATION_5_T | 18.16 | 24 | 1.000 | 1.000 |
| 3EG5RaDj4z7hkT2JD::GORILLA | VALIDATION_10_T | 17.54 | 20 | 1.000 | 1.000 |
| ASQtYBWx6cGN27KxL::GORILLA | VALIDATION_12_C | 17.45 | 25 | 1.000 | 1.000 |
| PrfKWppTZdvPhe4oN::SLOTH | VALIDATION_19_T | 17.43 | 23 | 0.798 | 0.000 |
| Fm7uYc98E2b6kcoDE::SLOTH | VALIDATION_1_T | 17.43 | 20 | 0.906 | 0.875 |
| CoHyYysb84APn5gDE::SLOTH | VALIDATION_18_T | 17.05 | 22 | 0.261 | 0.261 |
| 2TDnnb4wn9rwy3CYZ::GORILLA | VALIDATION_2_T | 16.50 | 19 | 0.645 | 0.000 |
| 6Pxb9DsQpyk22cJNr::SLOTH | VALIDATION_1_C | 15.96 | 18 | 0.828 | 0.438 |
| 7oRFMBwqmesGX7t3H::SLOTH | VALIDATION_15_C | 15.21 | 20 | 1.000 | 1.000 |
