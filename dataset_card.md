# XAI Halo-Property Prediction Dataset Card

## Dataset Summary

This dataset provides galaxy images, halo-related physical metadata, and spatial analysis masks derived from the TNG (IllustrisTNG) simulation. It is designed for explainable AI experiments that test whether image morphology contains predictive information associated with dark-matter-related halo properties.

This dataset does not provide dark matter detections or dark matter ground-truth segmentation labels. The project does not directly detect, segment, or localize dark matter.

## Dataset Features

- **Images**: 224x224 and 512x512 pixel astronomical simulation snapshots
- **Spatial analysis masks**: Binary masks defining radial or morphology-based image regions for attribution analysis
- **Splits**: Train/validation/test splits for model evaluation
- **Metadata**: Subhalo and halo-related properties including mass and redshift

## Data Samples

Total number of samples: [To be populated during dataset generation]

### Split Distribution

- Training: 70%
- Validation: 15%
- Test: 15%

## Image Properties

- **Source**: TNG Simulation snapshots
- **Resolution**: 224x224 and 512x512 pixels
- **Format**: PNG (normalized to [0, 1])
- **Channels**: Single channel (grayscale)

## Spatial Analysis Mask Properties

- **Type**: Binary spatial analysis masks
- **Format**: PNG
- **Foreground**: Selected analysis region (255)
- **Background**: Pixels outside the selected analysis region (0)

Radial masks are used to quantify attribution across center, middle, and outer image zones. They are not physical dark matter maps and should not be interpreted as dark matter locations.

## File Structure

```
TNG-DM-XAI-v1/
├── images_224/
│   ├── train/
│   ├── val/
│   └── test/
├── images_512/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
├── metadata.csv
├── train.csv
├── val.csv
└── test.csv
```

## Metadata Columns

- `subhalo_id`: Unique identifier for the subhalo
- `mass`: Total mass of the subhalo
- `redshift`: Redshift at which the snapshot was taken
- `simulation`: TNG simulation version (e.g., TNG100)
- `image_path`: Path to the corresponding image
- `mask_path`: Path to the corresponding spatial analysis mask

## Usage

See `README.md` for setup instructions and the image -> halo property regression -> XAI -> radial analysis pipeline.
