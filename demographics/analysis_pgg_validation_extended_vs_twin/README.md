# PGG Validation Extended vs Twin Demographic Comparison

## Scope

This folder contains the validation-only extension of the Twin-vs-PGG demographic comparison.

It uses:

- Twin-2k wave 1-3 demographics from `wave1_3_response.csv`
- validation-wave rows only from `merged_demographcs_prolific.csv`

No learning-wave rows are included here.

Current coverage:

- PGG validation rows: `3639`
- Twin rows: `2058`
- PGG validation games covered: `470 / 470`

## Main Figures

Initial 3-dimension figure:

- `PGG-finetuning/demographics/analysis_pgg_vs_twin/combined_distribution_comparison.png`

Extended validation-only figure:

- `combined_extended_comparison.png`

The extended figure includes:

- age
- male/female
- completed-degree education
- race/ethnicity
- employment
- U.S. nationality proxy
- U.S. residence proxy

## Direct-Overlap Divergences

Total variation distance:

- age: `28.86` pp
- male/female: `1.87` pp
- education: `6.87` pp
- race/ethnicity: `9.55` pp
- employment: `23.88` pp

## Proxy Panels

Twin does not have direct country-of-residence, country-of-birth, or nationality fields.

So the figure uses two proxy comparisons:

- Twin `QID16` U.S. citizen vs PGG `PROLIFIC_Nationality == United States`
- Twin `QID11` current U.S. region vs PGG `PROLIFIC_Country of residence == United States`

These proxy divergences are large:

- U.S. nationality proxy: `46.54` pp
- U.S. residence proxy: `45.98` pp

## PGG-Only Country Summaries

There is no true Twin counterpart for:

- `PROLIFIC_Country of birth`
- `PROLIFIC_Country of residence`
- `PROLIFIC_Nationality`

They are summarized separately in `pgg_country_field_summary.csv`.

Top validation-only PGG categories:

- country of birth: `United States 51.08%`, `United Kingdom 33.59%`, `Canada 4.25%`
- country of residence: `United States 54.02%`, `United Kingdom 40.77%`, `Canada 5.21%`
- nationality: `United States 53.26%`, `United Kingdom 35.59%`, `Canada 4.53%`
