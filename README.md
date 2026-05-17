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

### Local Environment

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On Linux/macOS:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create local configuration:

   ```bash
   copy .env.example .env
   ```

   On Linux/macOS:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set `TNG_API_KEY`.

5. Create/check project directories:

   ```bash
   python -m src.setup_project
   ```

### Google Colab

1. Mount Google Drive in the first notebook cell:

   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=False)
   ```

2. Clone the repository into Drive:

   ```bash
   %cd /content/drive/MyDrive
   !git clone https://github.com/AleH14/xai-dark-matter-localization.git
   %cd xai-dark-matter-localization
   ```

3. Install dependencies:

   ```bash
   !pip install -r requirements.txt
   ```

4. Create `.env` from `.env.example` and set:

   ```txt
   DATA_ROOT=/content/drive/MyDrive/xai-dark-matter-data
   USE_GOOGLE_DRIVE=True
   TNG_API_KEY=your_api_key_here
   SEED=42
   ```

5. Create/check directories:

   ```bash
   !python -m src.setup_project
   ```

### Smoke Test

Before downloading TNG data, run a minimal synthetic test of the regression stack:

```bash
python -m src.training.smoke_test_regression --output-dir outputs/smoke_test
```

This creates tiny synthetic images, builds a regression CSV, runs one ResNet18 forward/backward pass, computes MAE/RMSE/R2, and saves a small checkpoint. It is only a setup check, not a scientific result.

## Dataset Pipeline

0. `00_setup_colab.ipynb`: Optional Colab setup, dependency installation, and directory checks
1. `01_api_test.ipynb`: Test TNG API connectivity
2. `02_select_subhalos.ipynb`: Select subhalos and collect halo-related physical metadata
3. `03_download_images.ipynb`: Download images from TNG
4. `04_build_metadata_labels.ipynb`: Build clean regression metadata with halo-mass labels
5. `04_preprocess_images.ipynb`: Preprocess and resize images
6. `06_train_halo_mass_regressor.ipynb`: Train the PyTorch halo-mass regression model
7. `05_build_masks.ipynb`: Create optional spatial analysis masks
8. `08_generate_xai_maps.ipynb`: Generate Grad-CAM and Integrated Gradients attribution maps
9. `09_radial_importance_analysis.ipynb`: Quantify attribution by center, middle, and outer regions
10. `06_dataset_statistics.ipynb`: Analyze dataset statistics

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

## Halo Mass Regressor

The initial model is a PyTorch ResNet18 with its classification head replaced by a single scalar output for `halo_mass_log10`.

Train from the YAML config:

```bash
python -m src.training.train_regression --config configs/regression_resnet18.yaml
```

Evaluate a saved checkpoint:

```bash
python -m src.training.evaluate_regression \
  --config configs/regression_resnet18.yaml \
  --checkpoint outputs/regression_resnet18/checkpoints/best_model.pt \
  --split test
```

Training automatically uses GPU when available and saves:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `training_history.csv`
- `predictions_test.csv`
- `metrics_test.json`
- `loss_curve.png`
- `predicted_vs_true_test.png`

Evaluation reports MAE, RMSE, and R2.

## Explainability Maps

After training a checkpoint, generate attribution maps in Colab with:

```bash
jupyter notebook notebooks/08_generate_xai_maps.ipynb
```

The notebook loads `outputs/regression_resnet18/checkpoints/best_model.pt`, applies Grad-CAM and Integrated Gradients to selected test images, and saves outputs under `results/xai/`:

- raw attribution arrays as `.npy`
- heatmap PNGs
- heatmap overlays on the original galaxy images
- `xai_summary.csv`

Attribution maps show regions influential for the model prediction, not direct dark matter detections.

## Radial Importance Analysis

After generating XAI maps, quantify where attribution is concentrated:

```bash
python -m src.analysis.radial_importance \
  --xai-dir results/xai \
  --summary-csv results/xai/xai_summary.csv \
  --output-csv results/analysis/radial_importance.csv \
  --output-dir results/analysis
```

The analysis uses three geometric radial regions:

- center: `r < 0.30`
- middle: `0.30 <= r < 0.70`
- outer: `r >= 0.70`

It saves `center_importance_percent`, `middle_importance_percent`, and `outer_importance_percent` for each attribution map, plus plots under `results/analysis/`. These regions are analysis masks, not dark matter ground truth.

## Data Organization

Images and spatial analysis masks are organized at two resolutions:
- 224x224 pixels (for efficient training)
- 512x512 pixels (for high-resolution analysis)

Each resolution contains train/val/test splits.

## Configuration

Runtime settings live in `.env` and experiment settings live in YAML configs under `configs/`.

Key `.env` values:

- `TNG_API_KEY`: IllustrisTNG API key
- `DATA_ROOT`: local or Drive-backed data root
- `DATASET_NAME`: processed dataset name
- `SEED`: reproducibility seed
- `OUTPUT_ROOT`: training output root
- `RESULTS_ROOT`: analysis output root

Main configs:

- `configs/regression_resnet18.yaml`: full halo-mass regression experiment
- `configs/regression_resnet18_smoke.yaml`: tiny smoke-test configuration

The training scripts set Python, NumPy, and PyTorch seeds from the YAML config. Directory checks can be rerun safely with `python -m src.setup_project`.
