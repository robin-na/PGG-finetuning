# Persona Transfer Audit: Comprehensive Pilot Evaluation

Run: `twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0`
Parsed requests: 40
Candidate observed players: 21
Returned top-k rows: 110

## Concentration

Top-1 effective number of observed player identities: 10.42
Probability-mass effective number of observed player identities: 13.95

## Avatar-Level Match Distribution

| Avatar | Top-1 share | Probability share |
|---|---:|---:|
| SLOTH | 0.525 | 0.467 |
| DUCK | 0.175 | 0.206 |
| GORILLA | 0.175 | 0.178 |
| CHICKEN | 0.125 | 0.145 |
| DOG | 0.000 | 0.004 |

## Matched Behavior Skew

| Metric | Matched | Candidate-uniform | Difference | Std. diff |
|---|---:|---:|---:|---:|
| mean_contribution_rate | 0.920 | 0.810 | 0.110 | 0.448 |
| full_contribution_rate | 0.846 | 0.709 | 0.138 | 0.391 |
| zero_contribution_rate | 0.026 | 0.061 | -0.035 | -0.380 |
| contribution_sd | 2.227 | 3.767 | -1.539 | -0.419 |
| messages_per_round | 0.340 | 0.275 | 0.065 | 0.118 |
| reward_given_round_rate | 0.410 | 0.296 | 0.114 | 0.342 |
| punish_given_round_rate | 0.172 | 0.129 | 0.043 | 0.145 |
| punish_received_round_rate | 0.116 | 0.162 | -0.046 | -0.161 |

## Most-Matched Observed Players

| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |
|---|---|---:|---:|---:|---:|
| T8RpnDhSrnavwJQL2::SLOTH | VALIDATION_8_T | 4.87 | 6 | 1.000 | 1.000 |
| JgQY6iNGkRDTwiYTC::SLOTH | VALIDATION_5_T | 4.67 | 6 | 1.000 | 1.000 |
| TvaK7RsiSd758hAmp::GORILLA | VALIDATION_1_T | 4.65 | 6 | 0.666 | 0.250 |
| 6sSM4xSyZQkaxyHav::SLOTH | VALIDATION_5_C | 3.70 | 5 | 1.000 | 1.000 |
| 6sSM4xSyZQkaxyHav::DUCK | VALIDATION_5_C | 3.60 | 3 | 1.000 | 1.000 |
| TvaK7RsiSd758hAmp::SLOTH | VALIDATION_1_T | 3.20 | 2 | 0.891 | 0.875 |
| 3Ce6KgbmNCQEwYBfk::CHICKEN | VALIDATION_19_C | 2.65 | 3 | 1.000 | 1.000 |
| 3Ce6KgbmNCQEwYBfk::DUCK | VALIDATION_19_C | 2.35 | 2 | 0.916 | 0.828 |
| T8RpnDhSrnavwJQL2::CHICKEN | VALIDATION_8_T | 2.33 | 2 | 1.000 | 1.000 |
| 3Ce6KgbmNCQEwYBfk::SLOTH | VALIDATION_19_C | 2.25 | 2 | 0.976 | 0.862 |
