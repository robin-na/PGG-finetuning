# Persona Transfer Audit: Comprehensive Pilot Evaluation

Run: `argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2`
Parsed requests: 1280
Candidate observed players: 342
Returned top-k rows: 3789

## Concentration

Top-1 player coverage: 152 / 342 (44.4%)
Top-k probability player coverage: 247 / 342 (72.2%)
Top-1 effective number of observed player identities: 96.58
Probability-mass effective number of observed player identities: 151.53
Top 5% of matched identities capture 19.0% of probability mass.
Median within-game top-1 effective N / players: 35.7%
Median within-game top player share across personas: 59.4%

## Avatar-Label Diagnostic

Avatar labels are reused across games and are not stable person identities. This table is only a label/order diagnostic.

| Avatar | Top-1 share | Probability share |
|---|---:|---:|
| SLOTH | 0.287 | 0.242 |
| GORILLA | 0.205 | 0.203 |
| CHICKEN | 0.127 | 0.119 |
| DUCK | 0.090 | 0.115 |
| DOG | 0.045 | 0.059 |
| MOOSE | 0.038 | 0.040 |
| CHICK | 0.040 | 0.038 |
| PARROT | 0.022 | 0.037 |
| COW | 0.042 | 0.037 |
| RABBIT | 0.037 | 0.036 |
| OWL | 0.039 | 0.031 |
| MONKEY | 0.009 | 0.013 |
| SNAKE | 0.009 | 0.008 |
| CROCODILE | 0.005 | 0.008 |
| PINGUIN | 0.003 | 0.006 |
| FROG | 0.000 | 0.003 |
| ELEPHANT | 0.001 | 0.003 |
| WHALE | 0.000 | 0.001 |

## Matched Behavior Skew

| Metric | Matched | Candidate-uniform | Difference | Std. diff |
|---|---:|---:|---:|---:|
| mean_contribution_rate | 0.867 | 0.753 | 0.114 | 0.442 |
| full_contribution_rate | 0.765 | 0.638 | 0.127 | 0.358 |
| zero_contribution_rate | 0.069 | 0.147 | -0.077 | -0.334 |
| contribution_sd | 3.039 | 4.616 | -1.577 | -0.449 |
| messages_per_round | 0.218 | 0.137 | 0.081 | 0.279 |
| reward_given_round_rate | 0.168 | 0.129 | 0.039 | 0.144 |
| punish_given_round_rate | 0.097 | 0.082 | 0.015 | 0.079 |
| punish_received_round_rate | 0.048 | 0.078 | -0.030 | -0.172 |

## Most-Matched Observed Players

| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |
|---|---|---:|---:|---:|---:|
| Fm7uYc98E2b6kcoDE::SLOTH | VALIDATION_1_T | 21.91 | 30 | 0.906 | 0.875 |
| 3EG5RaDj4z7hkT2JD::GORILLA | VALIDATION_10_T | 21.87 | 28 | 1.000 | 1.000 |
| 65crAkEscRbHNJXbW::SLOTH | VALIDATION_10_C | 21.46 | 27 | 1.000 | 1.000 |
| 6Pxb9DsQpyk22cJNr::SLOTH | VALIDATION_1_C | 20.37 | 28 | 0.828 | 0.438 |
| 6sSM4xSyZQkaxyHav::DUCK | VALIDATION_5_C | 19.70 | 28 | 1.000 | 1.000 |
| 3v9ji6wNY7sXHe8cM::SLOTH | VALIDATION_4_C | 18.65 | 29 | 0.875 | 0.875 |
| 3EQGxadfbmAKaACgZ::GORILLA | VALIDATION_5_T | 18.50 | 23 | 1.000 | 1.000 |
| ASQtYBWx6cGN27KxL::GORILLA | VALIDATION_12_C | 18.22 | 26 | 1.000 | 1.000 |
| 7Y6PBvgZNqtriKzaZ::CHICKEN | VALIDATION_8_T | 17.21 | 28 | 1.000 | 1.000 |
| AJ4ikZuWJeJcEXhgZ::GORILLA | VALIDATION_3_T | 16.65 | 25 | 1.000 | 1.000 |
