# Script Guide

## Core Files

- `eda.py`: the main batch analysis script. It loads the dataset, audits missingness, validates joins, and writes charts plus summary CSVs into `eda_output/`.
- `eda_notebook.ipynb`: a notebook version for exploratory work and live iteration.
- `generate_presentation.py`: converts the current EDA outputs into `eda_presentation.html`.
- `.claude/commands/datafest.md`: a structured prompt template for asking an AI assistant to propose a DataFest analysis and return runnable Python.

## How The Pieces Fit

1. Raw data is read by `eda.py`
2. Charts and summary tables are written to `eda_output/`
3. `generate_presentation.py` reads those generated artifacts
4. `eda_presentation.html` becomes a presentation-friendly summary

## Best Ways To Extend

- Add narrowly scoped analyses rather than turning `eda.py` into one huge script
- Reuse summary CSVs and generated plots where possible
- Keep dataset-specific filters explicit so participants learn the data caveats
- Prefer clear exported outputs over hidden notebook-only work
