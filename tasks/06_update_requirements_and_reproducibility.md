# Implement radial attribution analysis

## Goal
Quantify how much attribution falls in central, intermediate, and outer regions of each galaxy image.

## Context
The research question asks where the predictive information is located. Radial masks are analysis regions, not physical dark matter labels.

## Files to create
- src/analysis/radial_importance.py
- notebooks/09_radial_importance_analysis.ipynb

## Requirements
- Generate three radial masks:
  - center: r < 0.30
  - middle: 0.30 <= r < 0.70
  - outer: r >= 0.70
- For each attribution map, compute:
  - center_importance_percent
  - middle_importance_percent
  - outer_importance_percent
- Save results to CSV.
- Generate plots:
  - mean importance by region
  - boxplot by region
  - importance vs true halo mass
- Normalize attribution maps safely.
- Handle zero-sum maps.

## Do not
- Do not describe radial regions as dark matter ground truth.
- Do not threshold galaxy brightness to define dark matter.

## Acceptance criteria
- A CSV with radial importance metrics is produced.
- Plots are saved under results/analysis/.
- Notebook explains how the metrics answer the research question.