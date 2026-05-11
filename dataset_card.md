# Dark Matter Localization XAI Dataset Card

## Dataset Summary

This dataset provides images and segmentation masks for dark matter localization using the TNG (Illustris TNG) simulation. It's designed for training explainable AI models to identify and localize dark matter in simulated astronomical observations.

## Dataset Features

- **Images**: 224x224 and 512x512 pixel astronomical simulation snapshots
- **Masks**: Binary segmentation masks indicating dark matter locations
- **Splits**: Train/validation/test splits for model evaluation
- **Metadata**: Subhalo properties including mass and redshift

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

## Mask Properties

- **Type**: Binary segmentation masks
- **Format**: PNG
- **Foreground**: Dark matter regions (255)
- **Background**: Non-dark matter regions (0)

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
- `mask_path`: Path to the corresponding mask

## Usage

See `README.md` for setup instructions and dataset pipeline information.
