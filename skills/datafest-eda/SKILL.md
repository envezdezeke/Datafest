---
name: datafest-eda
description: Use when analyzing the ASA DataFest 2026 Stormont Vail Health project, especially for patient-journey EDA, chart creation, table joins, or extending the existing eda.py workflow.
---

# DataFest EDA

Use this skill when working inside the DataFest project folder and the goal is to explore, extend, or explain analysis on the Stormont Vail Health dataset.

## Focus

- Prefer EDA and storytelling over heavy modeling
- Reuse existing project outputs before creating new ones
- Keep memory usage in mind because the raw CSVs are large

## Workflow

1. Read `CLAUDE.md` first for the dataset map, join keys, and major caveats.
2. Inspect `eda_output/` before adding new analysis so you do not duplicate existing charts.
3. Extend `eda.py` or write a new focused script only when the current outputs do not answer the question.
4. Keep outputs presentation-friendly: clear titles, usable labels, and direct takeaways.

## Dataset Rules

- Treat `encounters.csv` as the central table.
- Exclude `PrimaryDiagnosisKey = -1` before joining to diagnosis.
- Use `DiagnosisValue` for longitudinal tracking.
- Exclude `DiagnosisValue` in `IMO0001` and `IMO0002` for clinical analysis.
- Filter SDOH rows where `DisplayName` starts with `*`.
- Prefer `AttendingProviderDurableKey` over `ProviderDurableKey` for clinical provider joins.

## Good Angles

- Patient journeys by diagnosis group
- Time gaps between related encounters
- Repeat-encounter patterns
- Department or care-setting transitions
- Subgroup differences using robust demographic fields

## Read As Needed

- Read `.claude/commands/datafest.md` if you want a strong prompt pattern for generating analysis code.
