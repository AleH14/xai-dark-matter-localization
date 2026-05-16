# Build halo mass regression dataset

## Goal
Create a dataset builder that pairs each galaxy image with a physical halo-related label suitable for regression.

## Context
The main scientific task is not segmentation. The main task is:
image -> halo-related physical property

Possible target:
- log10(Group_M_Crit200)
- log10(halo_mass)
- another available IllustrisTNG halo mass proxy in the metadata

## Files to create or modify
- src/data/build_regression_dataset.py
- src/data/image_dataset.py
- notebooks/04_build_metadata_labels.ipynb
- README.md

## Requirements
- Read image paths and metadata CSV.
- Validate that each image has a matching physical label.
- Create train/val/test splits.
- Save a clean CSV with columns:
  - image_path
  - subhalo_id
  - halo_mass_raw
  - halo_mass_log10
  - split
- Add checks for missing images and missing labels.
- Print dataset statistics:
  - total samples
  - train/val/test count
  - min/max/mean/std of target
- Do not load all images into memory unnecessarily.

## Do not
- Do not create binary masks as labels.
- Do not call the label "dark matter location".
- Do not delete the existing segmentation code; leave it as optional.

## Acceptance criteria
- A clean regression metadata CSV is generated.
- The CSV can be used by a PyTorch Dataset.
- Missing data is handled safely.