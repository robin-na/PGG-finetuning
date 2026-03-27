# PGG Validation vs Twin-2k Demographic Comparison

This folder documents how we compare the participant pools in the PGG validation data and the Twin-2k data.

## Goal

The comparison is meant to answer a narrow question:

- How similar are the two participant pools on age, sex/gender, and education?

This is a pool-level comparison only. It does not attempt to adjust the Twin sample to the PGG sample. Any later reweighting or matched sampling should build on these outputs rather than replace them.

## Data Sources

### PGG

The PGG side uses the validation-wave demographics table:

- `demographics/demographics_numeric_val.csv`

That file is generated from:

- `data/raw_data/validation_wave/player-inputs.csv`

The raw participant-input fields used are:

- `data.age`
- `data.gender`
- `data.education`

These are normalized by `demographics/generate_demo_table.py` into one row per participant with model-ready columns such as:

- `age`, `age_missing`
- `gender_man`, `gender_woman`, `gender_non_binary`, `gender_unknown`
- `education_high_school`, `education_bachelor`, `education_master`, `education_other`, `education_unknown`

Important PGG normalization details:

- `data.age` is parsed as numeric and retained only if `0 < age <= 100`.
- `data.gender` is an open-text field and is normalized into `man`, `woman`, `non_binary`, or `unknown`.
- `data.education` is normalized into `high_school`, `bachelor`, `master`, `other`, or `unknown`.
- In the raw validation data, the only observed education strings are `high-school`, `bachelor`, `master`, and `other`.

### Twin-2k

The Twin side uses:

- `non-PGG_generalization/data/Twin-2k-500/snapshot/question_catalog_and_human_response_csv/wave1_3_response.csv`

Question definitions come from:

- `non-PGG_generalization/data/Twin-2k-500/snapshot/question_catalog_and_human_response_csv/question_catalog.json`

The Twin variables used are:

- `QID12`: "What is the sex that you were assigned at birth?"
  - options: `Male`, `Female`
- `QID13`: "How old are you?"
  - options: `18-29`, `30-49`, `50-64`, `65+`
- `QID14`: "What is the highest level of schooling or degree that you have completed?"
  - options:
    - `Less than high school`
    - `High school graduate`
    - `Some college, no degree`
    - `Associate's degree`
    - `College graduate/some postgrad`
    - `Postgraduate`

Important measurement note:

- Twin records sex assigned at birth, not gender identity.
- PGG stores an open-text `data.gender` field, so the constructs are close but not identical.

## Sample Sizes

Base pool sizes:

- PGG validation participants: `3639`
- Twin participants: `2058`

Dimension-specific usable sample sizes differ because we exclude unknown categories where needed:

- Age comparison: PGG `3632`, Twin `2058`
- Male/Female comparison: PGG `3567`, Twin `2058`
- Harmonized education comparison: PGG `3630`, Twin `2058`

Why the PGG counts drop:

- Age: `5` missing ages and `1` valid age of `14`, which falls outside the shared `18+` comparison bins.
- Male/Female: excludes `58` PGG `non_binary` and `14` PGG `unknown`.
- Education: excludes `9` PGG `unknown`.

Twin has no missing values in `QID12-14` in this snapshot.

## How Each Dimension Is Compared

The comparison script is:

- `demographics/compare_pgg_vs_twin_demographics.py`

### Age

Age is compared in four bins:

- `18-29`
- `30-49`
- `50-64`
- `65+`

Implementation details:

- PGG numeric ages are binned into the same brackets with left-closed intervals `[18, 30)`, `[30, 50)`, `[50, 65)`, `[65, inf)`.
- Twin already reports age in those exact brackets through `QID13`.

### Sex / Gender

There are two sex/gender views in the outputs:

1. A raw audit view in `distribution_comparison.csv`
   - PGG categories: `man`, `woman`, `non_binary`, `unknown`
   - Twin categories: `man`, `woman`, with `0` inserted for `non_binary` and `unknown`

2. The harmonized comparison used in the combined figure
   - categories: `male`, `female`
   - PGG side uses only `gender_man` and `gender_woman`
   - Twin side uses `QID12 == Male` and `QID12 == Female`
   - PGG `non_binary` and `unknown` are excluded

This harmonized view is intentionally narrow because Twin has no gender-identity item beyond `Male/Female`.

### Education

There are three education views in the outputs.

1. Raw audit view in `distribution_comparison.csv`
   - PGG: `high_school`, `bachelor`, `master`, `other`, `unknown`
   - Twin recoded to the same labels using:
     - `Less than high school` -> `other`
     - `High school graduate` -> `high_school`
     - `Some college, no degree` -> `other`
     - `Associate's degree` -> `other`
     - `College graduate/some postgrad` -> `bachelor`
     - `Postgraduate` -> `master`

2. Coarse robustness views in `education_harmonized_comparison.csv`
   - `below_bachelor` vs `bachelor_or_higher`
   - `below_postgraduate` vs `postgraduate_or_higher`

3. Practical completed-degree harmonization used in the combined figure
   - `high school`
   - `college/postsecondary`
   - `postgraduate`

The current figure uses the following completed-degree mapping:

- PGG
  - `high school` = `education_high_school`
  - `college/postsecondary` = `education_bachelor + education_other`
  - `postgraduate` = `education_master`
- Twin
  - `high school` = `Less than high school + High school graduate + Some college, no degree`
  - `college/postsecondary` = `Associate's degree + College graduate/some postgrad`
  - `postgraduate` = `Postgraduate`

Rationale:

- `Some college, no degree` is grouped with `high school` because the reference point is highest completed degree.
- `Associate's degree` is grouped with `college/postsecondary`.
- `Less than high school` is merged into `high school` because it is very rare in Twin (`17 / 2058 = 0.826%`) and PGG has no matching category.
- PGG `other` is grouped with `college/postsecondary` as the most practical non-postgraduate catch-all, but this is an assumption and should be treated as such.

## Metrics and Tests

### Distribution tables

For each dimension, the script writes:

- counts in each category
- percentages within each dataset
- percentage-point differences (`PGG - Twin`)

### Divergence metric

Each subplot in the combined figure reports total variation distance (TVD):

`TVD(p, q) = 0.5 * sum_i |p_i - q_i|`

where `p_i` and `q_i` are category proportions for PGG and Twin.

Interpretation:

- `0 pp` means identical distributions
- larger values mean larger pool-level mismatch

Current TVD values:

- Age: `29.13 pp`
- Male/Female: `1.50 pp`
- Education (completed-degree harmonized): `6.86 pp`

### Statistical tests

The script writes chi-square tests to `statistical_tests.csv`.

Reported tests include:

- raw age comparison
- raw education comparison
- man/woman-only gender comparison
- coarse education robustness comparisons
- harmonized male/female comparison
- harmonized completed-degree education comparison

Current headline results:

- Age differs strongly between PGG and Twin.
- Male/Female does not show a statistically significant difference in the harmonized comparison.
- Education still differs after harmonization, though the mismatch is much smaller than the age shift.

## Current Harmonized Comparison Table

From `harmonized_distribution_comparison.csv`:

### Age

- `18-29`: PGG `33.87%`, Twin `18.85%`
- `30-49`: PGG `49.83%`, Twin `35.71%`
- `50-64`: PGG `14.01%`, Twin `31.97%`
- `65+`: PGG `2.29%`, Twin `13.46%`

### Male/Female

- `male`: PGG `47.77%`, Twin `49.27%`
- `female`: PGG `52.23%`, Twin `50.73%`

### Education

- `high school`: PGG `29.92%`, Twin `36.78%`
- `college/postsecondary`: PGG `48.76%`, Twin `48.01%`
- `postgraduate`: PGG `21.32%`, Twin `15.21%`

## Output Files

- `distribution_comparison.csv`
  - raw age / gender / education tables
- `harmonized_distribution_comparison.csv`
  - age + harmonized male/female + harmonized education
- `education_harmonized_comparison.csv`
  - coarse bachelor-or-higher and postgraduate-or-higher robustness tables
- `divergence_metrics.csv`
  - TVD and effective sample sizes used per dimension
- `statistical_tests.csv`
  - chi-square tests
- `summary_metrics.csv`
  - pool sizes, age summaries, and Twin less-than-high-school prevalence
- `combined_distribution_comparison.png`
  - main three-panel figure

## Regenerating

Run:

```bash
python demographics/compare_pgg_vs_twin_demographics.py
```

If the upstream PGG normalization changes, regenerate the PGG demographics first:

```bash
python demographics/generate_demo_table.py
```

## Caveats

- The age comparison is exact at the bracket level, but Twin ages are bracketed rather than numeric.
- The sex/gender comparison is only partially harmonized because Twin measures sex assigned at birth and PGG uses an open-text gender field.
- The education comparison depends on how PGG `other` is interpreted.
- This analysis is descriptive and diagnostic. It does not by itself correct the shift between pools.
