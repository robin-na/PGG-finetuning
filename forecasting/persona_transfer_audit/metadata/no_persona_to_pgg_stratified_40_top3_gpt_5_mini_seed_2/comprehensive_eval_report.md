# Persona Transfer Audit: Comprehensive Pilot Evaluation

Run: `no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2`
Parsed requests: 40
Candidate observed players: 342
Returned top-k rows: 120

## Concentration

Top-1 player coverage: 40 / 342 (11.7%)
Top-k probability player coverage: 120 / 342 (35.1%)
Top-1 effective number of observed player identities: 40.00
Probability-mass effective number of observed player identities: 99.14
Top 5% of matched identities capture 10.9% of probability mass.
Median within-game top-1 effective N / players: 15.5%
Median within-game top player share across personas: 100.0%

## Avatar-Label Diagnostic

Avatar labels are reused across games and are not stable person identities. This table is only a label/order diagnostic.

| Avatar | Top-1 share | Probability share |
|---|---:|---:|
| GORILLA | 0.250 | 0.252 |
| DUCK | 0.275 | 0.210 |
| SLOTH | 0.125 | 0.135 |
| CHICKEN | 0.050 | 0.080 |
| PARROT | 0.050 | 0.067 |
| DOG | 0.050 | 0.055 |
| MOOSE | 0.050 | 0.036 |
| RABBIT | 0.025 | 0.026 |
| CHICK | 0.025 | 0.026 |
| MONKEY | 0.025 | 0.025 |
| OWL | 0.025 | 0.025 |
| CROCODILE | 0.025 | 0.021 |
| COW | 0.025 | 0.021 |
| FROG | 0.000 | 0.007 |
| SNAKE | 0.000 | 0.006 |
| WHALE | 0.000 | 0.005 |

## Matched Behavior Skew

| Metric | Matched | Candidate-uniform | Difference | Std. diff |
|---|---:|---:|---:|---:|
| mean_contribution_rate | 0.858 | 0.753 | 0.105 | 0.408 |
| full_contribution_rate | 0.746 | 0.638 | 0.109 | 0.306 |
| zero_contribution_rate | 0.073 | 0.147 | -0.073 | -0.316 |
| contribution_sd | 3.216 | 4.616 | -1.400 | -0.399 |
| messages_per_round | 0.202 | 0.137 | 0.065 | 0.225 |
| reward_given_round_rate | 0.206 | 0.129 | 0.077 | 0.285 |
| punish_given_round_rate | 0.109 | 0.082 | 0.027 | 0.143 |
| punish_received_round_rate | 0.042 | 0.078 | -0.036 | -0.203 |

## Most-Matched Observed Players

| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |
|---|---|---:|---:|---:|---:|
| 6sSM4xSyZQkaxyHav::DUCK | VALIDATION_5_C | 0.75 | 1 | 1.000 | 1.000 |
| 3EQGxadfbmAKaACgZ::GORILLA | VALIDATION_5_T | 0.75 | 1 | 1.000 | 1.000 |
| 2khRAszouj8vP8ZfP::DUCK | VALIDATION_18_C | 0.75 | 1 | 0.957 | 0.957 |
| HmA52oSS75D5Z2W8a::COW | VALIDATION_9_C | 0.75 | 1 | 1.000 | 1.000 |
| 6BxRfLcBtXYodAS8h::GORILLA | VALIDATION_8_C | 0.70 | 1 | 0.725 | 0.000 |
| 9oyspxPncwXdAkruR::DUCK | VALIDATION_2_C | 0.65 | 1 | 0.750 | 0.400 |
| CoHyYysb84APn5gDE::GORILLA | VALIDATION_18_T | 0.65 | 1 | 0.174 | 0.174 |
| 6Pxb9DsQpyk22cJNr::DUCK | VALIDATION_1_C | 0.65 | 1 | 0.584 | 0.312 |
| PmcxhmDRj2tN3cECC::GORILLA | VALIDATION_11_C | 0.65 | 1 | 1.000 | 1.000 |
| C5Cd8Yq5dDbdycXKB::CHICKEN | VALIDATION_14_C | 0.62 | 1 | 0.813 | 0.474 |
