# XAI for Halo-Related Information in Galaxy Images

This project studies whether galaxy images from the TNG simulation contain spatially localized information that helps predict physical properties associated with dark-matter-related halos.

The project does not directly detect, segment, or localize dark matter. Instead, it identifies spatial regions containing predictive information associated with dark-matter-related halo properties.

## Project Structure

- **data/**: Dataset storage
  - `raw/tng/`: Raw data from TNG simulation
  - `processed/TNG-DM-XAI-v1/`: Processed dataset with galaxy images, metadata, and spatial analysis masks
- **notebooks/**: Jupyter notebooks for data processing pipeline
- **src/**: Python modules for API integration and data processing

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create necessary directories (already created in this structure)

## Dataset Pipeline

1. `01_api_test.ipynb`: Test TNG API connectivity
2. `02_select_subhalos.ipynb`: Select subhalos and collect halo-related physical metadata
3. `03_download_images.ipynb`: Download images from TNG
4. `04_build_metadata_labels.ipynb`: Build clean regression metadata with halo-mass labels
5. `04_preprocess_images.ipynb`: Preprocess and resize images
6. `05_build_masks.ipynb`: Create optional spatial analysis masks
7. `06_dataset_statistics.ipynb`: Analyze dataset statistics

The research pipeline is:

```txt
galaxy image -> halo-property regression -> XAI attribution maps -> radial analysis
```

Radial masks are used to quantify attribution in center, middle, and outer image regions. These masks are analysis regions, not dark matter ground-truth labels.

## Halo Mass Regression Metadata

The main supervised task is image -> halo-related physical property regression. To build a clean metadata CSV:

```bash
python -m src.data.build_regression_dataset \
  --input-csv data/processed/TNG-DM-XAI-v1/metadata_with_images.csv \
  --output-csv data/processed/TNG-DM-XAI-v1/regression_metadata.csv \
  --label-column group_m_crit200 \
  --resolution 224
```

The generated CSV contains:

- `image_path`
- `subhalo_id`
- `halo_mass_raw`
- `halo_mass_log10`
- `split`

The builder checks for missing images and invalid labels, creates train/validation/test splits by `subhalo_id`, and prints target statistics. The CSV can be consumed lazily with `src.data.image_dataset.HaloMassImageDataset` for PyTorch training without loading all images into memory.

## Data Organization

Images and spatial analysis masks are organized at two resolutions:
- 224x224 pixels (for efficient training)
- 512x512 pixels (for high-resolution analysis)

Each resolution contains train/val/test splits.

## Configuration

Edit `src/config.py` to customize:
- Data directories
- Image sizes
- Train/val/test ratios
- API settings
