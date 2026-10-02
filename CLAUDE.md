# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Entry for the Kaggle Playground Series Season 6, Episode 10 competition: https://www.kaggle.com/competitions/playground-series-s6e10

The repository is at its initial commit. It holds only `README.md` and the stock GitHub Python `.gitignore`, so there is no code, dependency manifest, build, lint, or test setup yet. Update this file once those exist.

## Notes

- The `.gitignore` is the generic Python template. It does **not** ignore competition data (`*.csv`, `data/`, `submission*.csv`) or model artifacts. Add those patterns before downloading data, so large files don't get committed.
- Development happens on Windows (PowerShell is the primary shell).
