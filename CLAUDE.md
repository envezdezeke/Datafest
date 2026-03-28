# ASA DataFest 2026 Student Guide

This project is a student-ready workspace for exploring the Stormont Vail Health dataset from ASA DataFest 2026.

## What This Project Is About

The central question is **patient journeys**:

- how patients move through the health system
- how encounters cluster around diagnoses
- where follow-up is quick, delayed, or fragmented
- how journey patterns differ by subgroup, department, or social context

The data covers **January 2022 through December 31, 2025** and is de-identified.

## Start Here

Recommended order:

1. Read this file once
2. Review `PARTICIPANT_README.md` and `SCRIPT_GUIDE.md`
3. Open `eda_presentation.html` for the big-picture story
4. Browse `eda_output/` to see what is already done
5. Extend `eda.py`, `eda_notebook.ipynb`, or create a smaller focused script

## Core Files

- `eda.py`: main batch EDA script
- `eda_notebook.ipynb`: notebook for interactive work
- `eda_output/`: generated charts and summary CSVs
- `generate_presentation.py`: builds the HTML presentation from the EDA outputs
- `skills/`: reusable local Codex skills
- `.claude/commands/`: prompt shortcuts for common student tasks

## Data Layout

The raw data folder is:

```text
2026-ASA-DataFest-Data-Files/
  encounters.csv
  patients.csv
  diagnosis.csv
  providers.csv
  departments.csv
  social_determinants.csv
  tigercensuscodes.csv
```

## Most Important Table

`encounters.csv` is the central table.

Each row is one patient-system interaction. Most useful joins:

- `encounters.PatientDurableKey -> patients.DurableKey`
- `encounters.PrimaryDiagnosisKey -> diagnosis.DiagnosisKey`
- `encounters.DepartmentKey -> departments.DepartmentKey`
- `encounters.AttendingProviderDurableKey -> providers.DurableKey`
- `patients.CensusBlockGroupFipsCode -> tigercensuscodes.GEOID`
- `social_determinants.EncounterKey -> encounters.EncounterKey`

## Student Rules Of Thumb

- Start with a narrow question instead of trying to explain the whole dataset at once.
- Reuse existing EDA outputs before creating duplicate charts.
- Prefer clear storytelling and interpretable analyses over overly complex modeling.
- When making patient-level claims, aggregate to patient level when possible.
- Report effect size or slope, not only p-values.
- Always connect the result back to the patient journey story.

## Data Cleaning Rules You Should Use

- Exclude `PrimaryDiagnosisKey = -1` before joining diagnosis.
- Exclude diagnosis placeholders `IMO0001` and `IMO0002` for clinical analysis.
- Filter SDOH rows where `DisplayName` starts with `*`.
- Treat `*Unspecified`, `*Unknown`, and `*Not Applicable` carefully.
- Prefer `AttendingProviderDurableKey` over `ProviderDurableKey` for clinical provider joins.
- Use `DiagnosisValue` to track conditions longitudinally.

## Variables That Are Usually Safe To Use

- `Date`
- `Type`
- `VisitTypeDescription`
- `DepartmentType`
- `DiagnosisValue`
- `GroupName`
- `PatientBirthYearBin`
- `OmbRace`
- `OmbEthnicity`
- `SmokingStatus` with caution
- selected cleaned SDOH questions

## Variables That Need Strong Caution

- `SexAssignedAtBirth`: mostly unspecified
- `SexualOrientation`: mostly unspecified
- `PrimarySpecialty`: mostly not useful
- `CensusBlockGroupFipsCode`: heavily suppressed
- raw `Domain` in SDOH: often missing
- encounter-level counts when the real question is about patients

## Precomputed Findings Worth Knowing

- Encounter volume grows each year from 2022 to 2025.
- Office visits, lab visits, and hospital encounters dominate the system.
- Fracture-related diagnosis groups are especially common.
- Patient utilization is highly skewed: many patients have a few encounters, a smaller set have very many.
- SDOH data is useful but not population-complete.
- Geography is incomplete because many census blocks are suppressed or unspecified.

## Good Question Types

- Which diagnosis groups have the longest or most complex journeys?
- Where are the biggest gaps between encounters?
- Which departments appear in short, intense journeys versus long, spread-out ones?
- Do some subgroups have longer delays to follow-up?
- Which SDOH factors line up with more complex or higher-utilization journeys?
- Which patterns are statistically significant and also practically meaningful?

## Statistical Guidance

If you are adding tests:

- start with a comparison or slope that matters substantively
- report sample size, effect size, and uncertainty
- use simple tests when possible
- avoid claiming causality
- remember that very large datasets make tiny effects look significant

Use the local stats skill when you want help choosing or interpreting tests.

## Skills In This Project

- `skills/datafest-eda/`: for dataset-aware exploratory analysis
- `skills/script-explainer/`: for understanding or extending project files
- `skills/datafest-stat-tests/`: for statistical tests, slopes, confidence intervals, and interpretation

## Useful Prompt Patterns

- "Use the EDA skill and help me analyze follow-up gaps after hospital encounters."
- "Use the stats skill and test whether encounter counts differ by race after aggregating to patient level."
- "Use the script explainer skill and show me how `eda.py` builds journey charts."
- "Give me one chart, one table, and one statistical test I can use on a slide."

## Minimal Python Starter

```python
import pandas as pd

DATA = "2026-ASA-DataFest-Data-Files/"

enc = pd.read_csv(DATA + "encounters.csv", low_memory=False, parse_dates=["Date"])
pat = pd.read_csv(DATA + "patients.csv", low_memory=False)
diag = pd.read_csv(DATA + "diagnosis.csv", low_memory=False)

enc = enc[enc["PrimaryDiagnosisKey"] != -1].copy()
base = enc.merge(
    diag[["DiagnosisKey", "DiagnosisValue", "GroupName"]],
    left_on="PrimaryDiagnosisKey",
    right_on="DiagnosisKey",
    how="left",
)
base = base[~base["DiagnosisValue"].isin(["IMO0001", "IMO0002"])].copy()
```

## Final Advice

A strong DataFest project usually has:

- one focused question
- one believable workflow
- two or three strong visuals
- one or two careful statistical checks
- a clear statement of caveats

Aim for a result that is easy to explain out loud in under two minutes.
