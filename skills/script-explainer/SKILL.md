---
name: script-explainer
description: Use when explaining, teaching, or extending the DataFest project scripts, including eda.py, generate_presentation.py, notebook logic, and command prompts in .claude.
---

# Script Explainer

Use this skill when the user wants to understand how the project code works or wants help modifying it safely.

## Workflow

1. Start with the smallest relevant file.
2. Explain the script in terms of inputs, transformations, outputs, and assumptions.
3. Point out where dataset-specific logic appears so participants can adapt it confidently.
4. When editing code, preserve the existing output locations and naming conventions unless the user asks otherwise.

## Project Files To Know

- `eda.py`: batch EDA runner that loads the CSVs and writes charts and summary CSVs into `eda_output/`
- `generate_presentation.py`: builds an HTML presentation from the generated EDA artifacts
- `eda_notebook.ipynb`: interactive version of the analysis
- `.claude/commands/datafest.md`: a reusable prompt scaffold for DataFest analysis tasks
- `CLAUDE.md`: dataset-specific context and caveats

## Teaching Frame

When explaining a script, cover:

- what it reads
- what it computes
- what files it writes
- what assumptions could break it
- what a participant can safely customize first

## Safe First Customizations

- change chart selections
- add focused analyses by diagnosis or encounter type
- improve labels, captions, and exported tables
- add smaller helper scripts instead of making `eda.py` too large
