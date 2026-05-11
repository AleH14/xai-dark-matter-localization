# Dark Matter XAI Dataset

This project builds an explainable AI (XAI) dataset for dark matter localization using the TNG simulation.

## Project Structure

- **data/**: Dataset storage
  - `raw/tng/`: Raw data from TNG simulation
  - `processed/TNG-DM-XAI-v1/`: Processed dataset with images and masks
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
2. `02_select_subhalos.ipynb`: Select dark matter subhalos
3. `03_download_images.ipynb`: Download images from TNG
4. `04_preprocess_images.ipynb`: Preprocess and resize images
5. `05_build_masks.ipynb`: Create segmentation masks
6. `06_dataset_statistics.ipynb`: Analyze dataset statistics

## Data Organization

Images and masks are organized at two resolutions:
- 224x224 pixels (for efficient training)
- 512x512 pixels (for high-resolution analysis)

Each resolution contains train/val/test splits.

## Configuration

Edit `src/config.py` to customize:
- Data directories
- Image sizes
- Train/val/test ratios
- API settings
