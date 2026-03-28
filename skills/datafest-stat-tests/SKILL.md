---
name: datafest-stat-tests
description: Use when a DataFest participant needs statistical tests, effect-size framing, slope interpretation, confidence intervals, or basic statistical reasoning added to an analysis, chart, or narrative.
---

# DataFest Stat Tests

Use this skill when the task is to add statistical testing or interpretation to an analysis in the DataFest project.

## Goal

Prefer simple, explainable statistics that support a presentation story:

- confidence intervals
- slopes and trend interpretation
- group comparisons
- association tests
- nonparametric backups when assumptions are weak

Do not add a statistical test just because one exists. Use tests when they clarify a real comparison or trend.

## Workflow

1. Define the question in plain language first.
2. Identify the variable types:
   - numeric vs numeric
   - numeric vs group
   - categorical vs categorical
   - time trend
3. Start with an effect summary before the test:
   - mean or median difference
   - slope
   - proportion difference
   - correlation
4. Choose the simplest defensible test.
5. Report:
   - effect size or slope
   - confidence interval when practical
   - p-value
   - plain-language interpretation
6. State at least one assumption or caveat when it matters.

## Preferred Test Map

- Numeric vs binary/two-group:
  - Welch t-test if distributions are reasonably well-behaved and sample sizes are not tiny
  - Mann-Whitney U if skew or outliers make rank-based comparison safer
- Numeric vs 3+ groups:
  - Welch ANOVA or standard ANOVA if assumptions are reasonable
  - Kruskal-Wallis as a nonparametric alternative
- Categorical vs categorical:
  - Chi-square test of independence
  - Fisher exact test for very small counts
- Numeric vs numeric:
  - Pearson correlation for linear association
  - Spearman correlation for monotonic but non-normal relationships
  - Simple linear regression when you want a slope and an interpretable change per unit
- Time trend:
  - Regress outcome on time index for a slope
  - Compare periods with a difference in means or proportions when that is easier to explain

## Reporting Style

Always include:

- what was compared
- sample sizes
- the direction of the result
- the size of the effect
- whether the uncertainty is small or large
- one sentence on whether the result is practically meaningful

Good example:

- "Monthly encounters increased by about 12,400 per year in the observed period, based on a simple linear trend fit. The trend was positive and statistically distinguishable from zero, but operational changes over time may also explain part of the increase."

## DataFest-Specific Cautions

- Large datasets can make tiny effects look statistically significant. Check practical size, not only p-values.
- Repeated encounters from the same patient break independence for many naive tests. When possible, aggregate to patient level first for patient-level claims.
- SDOH data is not a random sample of all patients, so significance there does not imply population prevalence.
- Encounter volume changes over time, so raw comparisons across years may reflect growth rather than a true clinical shift.
- One visit can create multiple encounter rows, so encounter-level tests can overstate evidence if the question is really about patients.

## Default Packages

Prefer:

- `scipy.stats` for classical tests
- `statsmodels` for regression and confidence intervals
- `pandas` and `numpy` for summaries

## Read As Needed

- Read `references/basic-tests.md` for a quick recipe table and reporting templates.
