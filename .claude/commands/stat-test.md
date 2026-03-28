You are helping a DataFest student add statistical testing and interpretation to an analysis on the Stormont Vail Health dataset.

**Student request:** $ARGUMENTS

Respond with exactly these six sections:

## 1. Statistical Question
Turn the request into a precise statistical question. Define the outcome, grouping variable, unit of analysis, and whether the claim is encounter-level or patient-level.

## 2. Best Test
Choose the simplest defensible statistical test or slope-based model. Briefly explain why it fits better than the main alternative.

## 3. What To Compute First
List the summaries that should be shown before the test, such as sample size, mean, median, proportion, slope, or confidence interval.

## 4. Code
Provide complete runnable Python code using `pandas`, `scipy.stats`, and `statsmodels` when useful. The code should:
- aggregate to patient level first if the claim is really about patients
- be explicit about cleaning rules
- compute an effect size or slope
- include a confidence interval when practical
- return a clean table or small plot

## 5. Interpretation
Explain how a student should talk about the result on a slide. Mention practical significance, not just statistical significance.

## 6. Caveats
List 2-4 reasons this result could still be misleading in this dataset.
