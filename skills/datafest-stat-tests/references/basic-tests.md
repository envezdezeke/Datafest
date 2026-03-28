# Basic Tests Reference

## Quick Recipe Table

| Question | Good default | Report |
|---|---|---|
| Are two group means different? | Welch t-test | mean difference, CI, p-value |
| Are two skewed groups different? | Mann-Whitney U | median summary, rank-based p-value |
| Are 3+ groups different? | ANOVA or Kruskal-Wallis | group summaries, omnibus p-value |
| Are two categorical variables associated? | Chi-square | contingency table, p-value, standardized proportions if useful |
| Is there a relationship between two numeric variables? | Pearson or Spearman | correlation estimate, p-value |
| Is there a trend over time? | Simple linear regression | slope, CI, p-value |

## Simple Templates

### Two-group numeric comparison

1. Compute group means, medians, standard deviations, and sample sizes.
2. Check whether the question is about patients or encounters.
3. Use Welch t-test by default for two-group mean comparison.
4. If the data is extremely skewed or dominated by outliers, consider Mann-Whitney U.
5. Report the difference and not only significance.

### Slope or time trend

1. Aggregate to the right unit first, often month or patient.
2. Fit a simple regression of outcome on time index.
3. Interpret the slope in original units.
4. Mention if the trend may be confounded by rollout or volume growth.

### Categorical association

1. Build the contingency table.
2. Check for tiny cells.
3. Use chi-square unless counts are very small.
4. Describe which categories are more common, not just that an association exists.

## Interpretation Reminders

- Statistical significance is not the same as practical importance.
- Confidence intervals are often more useful than raw p-values for presentations.
- With large n, almost any nonzero effect can become significant.
- If observations are repeated within patients, aggregate first or clearly limit the claim to encounter-level patterns.
