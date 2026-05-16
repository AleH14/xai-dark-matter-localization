# Implement PyTorch halo mass regression model

## Goal
Implement the main model for the project: galaxy image -> log10 halo mass regression.

## Context
This is the core model for the paper. Explainability will later be applied to this trained regressor.

## Files to create
- src/models/halo_mass_regressor.py
- src/training/train_regression.py
- src/training/evaluate_regression.py
- configs/regression_resnet18.yaml
- notebooks/06_train_halo_mass_regressor.ipynb

## Requirements
- Use PyTorch.
- Use ResNet18 initially.
- Modify the final layer for scalar regression.
- Use MSELoss or SmoothL1Loss.
- Report MAE, RMSE, and R2.
- Save:
  - model checkpoint
  - training history CSV
  - prediction vs true CSV
  - loss curve plot
  - predicted vs true scatter plot
- Support CPU/GPU automatically.
- Set random seed.
- Read configuration from YAML.

## Do not
- Do not use segmentation masks as targets.
- Do not hardcode Google Drive paths.
- Do not require the full dataset to fit in memory.

## Acceptance criteria
- Training runs on a small sample dataset.
- Evaluation outputs MAE, RMSE, R2.
- Checkpoint is saved.
- Results are reproducible using the YAML config.