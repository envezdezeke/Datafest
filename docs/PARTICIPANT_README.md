# DataFest 2026 Participant Guide

## Start Here

This coding pack contains:

- `eda.py`: a working exploratory data analysis script
- `eda_notebook.ipynb`: notebook version for interactive exploration
- `generate_presentation.py`: regenerates the presentation HTML from the current EDA outputs
- `eda_presentation.html`: a presentation-ready summary of the current EDA
- `eda_output/`: pre-generated charts and summary CSVs
- `CLAUDE.md`: project-specific context and dataset caveats
- `.claude/commands/datafest.md`: a reusable AI prompt scaffold for analysis tasks
- `skills/`: local Codex skills for EDA, script understanding, and statistics
- `SCRIPT_GUIDE.md`: a quick explanation of how the project files fit together

## Recommended Order

1. Read `PARTICIPANT_README.md`, `SCRIPT_GUIDE.md`, and `CLAUDE.md`
2. Review `eda_presentation.html` for the big-picture story
3. Look through `eda_output/` to see the current charts and summary tables
4. Run or extend `eda.py` once the raw data folder has been provided separately
5. Build your own focused story around patient journeys

## Important Data Notes

- The raw data folder is not bundled in this copy.
- `encounters.csv` is the central table once the data is added back in.
- Use `PatientDurableKey` to connect encounters to patients.
- Use `PrimaryDiagnosisKey` to connect encounters to diagnoses.
- Exclude `PrimaryDiagnosisKey = -1` before diagnosis joins.
- Treat values such as `*Unspecified`, `*Unknown`, and `*Not Applicable` carefully.
- Filter placeholder diagnosis codes like `IMO0001` and `IMO0002` before clinical analysis.
- Filter SDOH rows with `DisplayName == '*Unspecified'` before interpreting SDOH patterns.

## If You're Using Codex

- Start by exploring the documentation and `eda_output/`
- If you have an EDA skill available in your Codex setup, use it as your first pass
- This pack includes local skill definitions in `skills/datafest-eda/`, `skills/script-explainer/`, and `skills/datafest-stat-tests/`
- Use the stats skill when you want Codex to add statistical tests, slope estimates, confidence intervals, and plain-language interpretation to an analysis
- Good follow-up directions:
  - patient journey definitions by diagnosis group
  - delayed follow-up and long-gap analysis
  - department transitions
  - subgroup comparisons by race/ethnicity or SDOH subset
  - chart cleanup for a final presentation

## Regeneration

To rebuild the presentation after updating the EDA outputs:

```bash
python3 generate_presentation.py
```
