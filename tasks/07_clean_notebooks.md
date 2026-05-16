# Improve reproducibility and project setup

## Goal
Make the project easier to run in Colab and locally.

## Files to modify
- requirements.txt
- README.md
- .env.example
- configs/
- notebooks/00_setup_colab.ipynb if present

## Requirements
- Add missing dependencies.
- Pin major library versions where reasonable.
- Add setup instructions for:
  - local environment
  - Google Colab
  - Google Drive mounting
- Add directory creation checks.
- Add seed configuration.
- Add a minimal "smoke test" command.

## Acceptance criteria
- A new user can set up the project from README.
- Dependencies match the actual code.
- There is a documented minimal run using sample data.