# PGG Extended vs Twin Demographic Comparison

## Purpose

This analysis compares the Twin-2k participant pool against the broader merged PGG demographic file:

- `PGG-finetuning/demographics/merged_demographcs_prolific.csv`

Unlike the earlier validation-only comparison, this file includes:

- exit-survey fields from `player-inputs.csv` style data
- matched Prolific profile fields when available

## Coverage

The merged PGG file contains `6689` participant rows, one per merged PGG participant/session.

Coverage against the processed PGG game tables:

- validation games covered: `470 / 470`
- learning games covered: `298 / 366`

So this file includes **all validation games** and a **subset of learning games**. It should be treated as an extended PGG sample, not as a complete learn+validation census.

## Source Variables

### Twin variables

From `wave1_3_response.csv`:

- `QID11`: current U.S. region
- `QID12`: sex assigned at birth
- `QID13`: age bracket
- `QID14`: highest completed education
- `QID15`: race or origin
- `QID16`: U.S. citizenship
- `QID24`: current employment status

### PGG extended variables

From `merged_demographcs_prolific.csv`:

- age: `PROLIFIC_Age`, with fallback to `PGGEXIT_data.age`
- sex: `PROLIFIC_Sex`, with fallback to `PGGEXIT_data.gender`
- education: `PGGEXIT_data.education`
- ethnicity: `PROLIFIC_Ethnicity simplified`
- employment: `PROLIFIC_Employment status` plus `PROLIFIC_Student status`
- country fields: `PROLIFIC_Country of birth`, `PROLIFIC_Country of residence`, `PROLIFIC_Nationality`

## Direct Overlap Comparisons

The direct harmonized comparisons are:

- age
- male/female
- completed-degree education
- race/ethnicity
- employment

These outputs are in:

- `direct_overlap_distribution_comparison.csv`
- `direct_overlap_divergence_metrics.csv`
- `direct_overlap_statistical_tests.csv`
- `combined_direct_overlap_comparison.png`

## Harmonization Rules

### Age

- PGG uses best available numeric age: Prolific age first, exit age fallback
- PGG age is bucketed to `18-29`, `30-49`, `50-64`, `65+`
- Twin uses `QID13` directly

### Male/Female

- PGG uses `PROLIFIC_Sex` when it is `Male` or `Female`
- otherwise it falls back to normalized `PGGEXIT_data.gender`
- Twin uses `QID12` (`Male` / `Female`)

### Education

Completed-degree harmonization:

- PGG `high-school` -> `high school`
- PGG `bachelor` -> `college/postsecondary`
- PGG `other` -> `college/postsecondary`
- PGG `master` -> `postgraduate`

Twin:

- `Less than high school` -> `high school`
- `High school graduate` -> `high school`
- `Some college, no degree` -> `high school`
- `Associate's degree` -> `college/postsecondary`
- `College graduate/some postgrad` -> `college/postsecondary`
- `Postgraduate` -> `postgraduate`

### Race / Ethnicity

The comparison is coarsened to:

- `white`
- `black`
- `asian`
- `other/mixed`

Twin:

- `White` -> `white`
- `Black` -> `black`
- `Asian` -> `asian`
- `Hispanic` -> `other/mixed`
- `Other` -> `other/mixed`

PGG:

- `White` -> `white`
- `Black` -> `black`
- `Asian` -> `asian`
- `Mixed` -> `other/mixed`
- `Other` -> `other/mixed`

This is imperfect because the Prolific simplified ethnicity field has no direct `Hispanic` bucket.

### Employment

The comparison is coarsened to:

- `full-time`
- `part-time`
- `unemployed`
- `student`
- `other/nonstandard`

Twin:

- `Full-time employment` -> `full-time`
- `Part-time employment` -> `part-time`
- `Unemployed` -> `unemployed`
- `Student` -> `student`
- `Self-employed`, `Home-maker`, `Retired` -> `other/nonstandard`

PGG:

- `Full-Time` -> `full-time`
- `Part-Time` -> `part-time`
- `Unemployed (and job seeking)` -> `unemployed`
- `Student status = Yes` with non-full-time / non-part-time employment -> `student`
- `Not in paid work ...`, `Other`, `Due to start a new job ...` -> `other/nonstandard`

This is also approximate because Prolific stores employment and student status as separate fields, while Twin uses one single employment-status question.

## Proxy-Only Comparisons

These are not direct variable matches, but the closest available U.S.-affiliation proxies:

- Twin `QID16` U.S. citizen vs PGG `PROLIFIC_Nationality == United States`
- Twin `QID11` current U.S. region vs PGG `PROLIFIC_Country of residence == United States`

Outputs:

- `proxy_us_alignment_comparison.csv`
- `proxy_us_alignment_divergence.csv`
- `proxy_us_alignment_tests.csv`

These should be interpreted cautiously.

## PGG-Only Country Fields

There is no true Twin counterpart for:

- country of birth
- country of residence
- nationality

So these are summarized separately in:

- `pgg_country_field_summary.csv`

## Main Takeaways

Using the direct harmonized overlap comparison:

- age shows a large shift
- employment also shows a large shift
- race/ethnicity shows a moderate shift
- education remains different, but less strongly than age or employment
- male/female remains closely aligned

The direct-overlap total variation distances are:

- age: `27.57` pp
- male/female: `1.66` pp
- education: `5.17` pp
- race/ethnicity: `11.71` pp
- employment: `23.07` pp

The PGG extended sample is also much less U.S.-concentrated than Twin:

- PGG U.S. residence: `46.55%`
- PGG U.S. nationality: `46.09%`
- PGG U.S. birth: `43.80%`

## Regeneration

Run:

```bash
python PGG-finetuning/demographics/compare_pgg_extended_vs_twin_demographics.py
```
