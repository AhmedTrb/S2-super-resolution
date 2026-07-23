# S2 Super-Resolution Repository

This repository contains a parcel-based Sentinel-2 workflow for observation-driven dataset building, super-resolution inference, feature extraction, and PlanetScope-based evaluation.

## Scope

The repository is organized around four active tasks:

1. Build LR/SR parcel datasets from Sentinel-2 stacks and field observations.
2. Run SEN2SRLite-based super-resolution and save georeferenced LR/SR patches.
3. Compute parcel-level features and statistics directly during dataset generation.
4. Evaluate SR outputs against PlanetScope reference imagery.

## Current Workflow

### 1. Observation Preparation

Generate LR/SR parcel patches, masks, and an observation inventory:

```bash
python scripts/data_preparation/prepare_s2_observation.py
```

This path writes normalized reflectance LR patches, corrected SR patches, parcel masks, and a CSV inventory under `yellowness_dataset/`.

### 2. Dataset Builder

Build the larger observation-driven dataset used for training and analysis:

```bash
python scripts/data_preparation/build_obs_dataset.py
```

This builder:

- reads the multi-temporal Sentinel-2 stack
- snaps field observations to the nearest Sentinel-2 acquisition
- extracts LR parcel patches
- runs super-resolution
- rasterizes LR/SR parcel masks
- computes parcel statistics and advanced LR/SR feature families directly in the pipeline
- writes `lr_inventory.csv` and `sr_inventory.csv`

### 3. SR Inference

Run direct SEN2SRLite inference on Sentinel-2 imagery:

```bash
python scripts/sr_inference/inference-SEN2SRLite.py
```

### 4. Evaluation

Use the main evaluation notebook:

```bash
jupyter notebook notebooks/evaluation_unified.ipynb
```

This notebook is the active evaluation entry point for multi-zone, multi-model comparison against PlanetScope references.

## Repository Layout

```text
s2-super-resolution/
├── configs/                         YAML configuration files
├── data/                            Parcel data, zones, calendars, and source imagery metadata
├── docs/                            Supplemental project notes
├── model/                           SEN2SRLite model assets
├── notebooks/
│   ├── evaluation_unified.ipynb     Main evaluation notebook
│   ├── data_preparation_validation.ipynb
│   └── exploratory/                 Exploratory analysis notebooks
├── outputs/                         Evaluation outputs and audit artifacts
├── scripts/
│   ├── data_preparation/
│   │   ├── audit_observation_pipeline.py
│   │   ├── build_obs_dataset.py
│   │   ├── build_temporal_dataset.py
│   │   ├── enrich_obs_features.py
│   │   └── prepare_s2_observation.py
│   ├── sr_inference/
│   │   └── inference-SEN2SRLite.py
│   └── run_pipeline.py
├── src/
│   └── s2_pipeline/                 Reusable processing classes
├── yellowness_dataset/              Generated observation patches and inventories
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Core Modules

### `src/s2_pipeline/`

Reusable building blocks for the processing pipeline:

- `date_matcher.py`: observation-to-Sentinel-2 date matching
- `patching.py`: parcel-centered patch extraction
- `masks.py`: LR/SR parcel mask rasterization
- `indices.py`: base spectral-index statistics
- `super_resolution.py`: reflectance normalization and SR inference wrapper
- `pipeline.py`: end-to-end observation-driven pipeline classes plus integrated advanced feature extraction

### `scripts/data_preparation/`

Active data-preparation entry points:

- `prepare_s2_observation.py`: direct observation patch extraction into `yellowness_dataset/`
- `build_obs_dataset.py`: larger dataset builder for training/evaluation workflows
- `build_temporal_dataset.py`: temporal dataset variant
- `audit_observation_pipeline.py`: four-patch audit for reflectance and save-path verification
- `enrich_obs_features.py`: retained as a standalone feature-enrichment utility, though equivalent feature logic is now integrated into the main pipeline classes

## Features Computed In-Pipeline

The active pipeline now computes parcel-level LR and SR features directly during processing:

- band-wise mean/std for all 10 Sentinel-2 bands
- base indices: NDRE, NDYVI, GCC, NDYI, mNDblue
- additional indices: NDVI, GNDVI, NDMI, NBR
- red-edge slopes and curvature
- selected band differences and ratios
- optional texture features when `scikit-image` is available and enabled

## Data Conventions

- Sentinel-2 LR patches are normalized to reflectance in `[0, 1]` before use.
- SR outputs are normalized/clipped back to `[0, 1]` before save and statistics.
- LR and SR patches are stored as `float32` GeoTIFF.
- Parcel masks are stored as `uint8` GeoTIFF.
- Inventory rows use parcel-level masked statistics only.

## Main Outputs

### `yellowness_dataset/`

- observation patch GeoTIFFs
- LR/SR parcel masks
- `observations_inventory.csv`

### `torchgeo_yellowness_dataset/`

- `lr_inventory.csv`
- `sr_inventory.csv`
- `images_lr/`, `images_sr/`
- `masks_lr/`, `masks_sr/`

### `outputs/evaluation/`

- per-zone metrics CSV files
- summary HTML tables
- evaluation artifacts and figures

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Script Commands

Use these commands from the repository root after activating the virtual environment.

### Observation Preparation And Extraction

Generate the observation-level LR and SR dataset in `yellowness_dataset/`:

```bash
python scripts/data_preparation/prepare_s2_observation.py
```

Build the TorchGeo-style training dataset in `torchgeo_yellowness_dataset/`:

```bash
python scripts/data_preparation/build_obs_dataset.py
```

Build the multi-temporal dataset variant:

```bash
python scripts/data_preparation/build_temporal_dataset.py
```

Run the modular pipeline entry point:

```bash
python scripts/run_pipeline.py
```

### Feature Calculation And Inventory Enrichment

Enrich the observation inventory with advanced parcel, spectral, and optional texture features:

```bash
python scripts/data_preparation/enrich_obs_features.py --csv yellowness_dataset/observations_inventory.csv --output yellowness_dataset/observations_inventory_enriched.csv
```

Overwrite the source inventory in place:

```bash
python scripts/data_preparation/enrich_obs_features.py --csv yellowness_dataset/observations_inventory.csv --in-place
```

Use a different parcel shapefile or apply a parcel buffer during feature extraction:

```bash
python scripts/data_preparation/enrich_obs_features.py --csv yellowness_dataset/observations_inventory.csv --shapefile data/shapefiles/2020_SEPIM.shp --buffer-meters 1.0
```

### Super-Resolution Inference And Audit

Run the direct SEN2SRLite inference script:

```bash
python scripts/sr_inference/inference-SEN2SRLite.py
```

Audit a few observation patches to verify reflectance handling and optional GeoTIFF saves:

```bash
python scripts/data_preparation/audit_observation_pipeline.py --limit 4 --save
```

### Deep Learning Training

Train the yellowness regressor on your single combined inventory CSV (recommended for small data: frozen backbone first):

```bash
python scripts/train_yellowness_regressor.py --inventory yellowness_dataset/observations_inventory_all_feature_new.csv --root-dir yellowness_dataset --resolution sr --backbone auto --torchgeo-weight ResNet50_Weights.SENTINEL2_ALL_DINO --mask-fusion mask_as_channel --freeze-backbone --unfreeze-epoch 0 --epochs 60 --batch-size 8 --learning-rate 1e-3 --backbone-learning-rate 1e-4 --output-dir outputs/yellowness_regression/resnet50_dino_frozen
```

Train on LR paths from the same CSV:

```bash
python scripts/train_yellowness_regressor.py --inventory yellowness_dataset/observations_inventory_all_feature_new.csv --root-dir yellowness_dataset --resolution lr --backbone auto --torchgeo-weight ResNet50_Weights.SENTINEL2_ALL_DINO --mask-fusion mask_as_channel --freeze-backbone --unfreeze-epoch 0 --epochs 60 --batch-size 8 --learning-rate 1e-3 --backbone-learning-rate 1e-4 --output-dir outputs/yellowness_regression/resnet50_dino_lr_frozen
```

Optional gradual unfreezing (start unfreezing at epoch 25):

```bash
python scripts/train_yellowness_regressor.py --inventory yellowness_dataset/observations_inventory_all_feature_new.csv --root-dir yellowness_dataset --resolution sr --backbone auto --torchgeo-weight ViTBase16_Weights.SENTINEL2_ALL_MAE --mask-fusion late_mask --freeze-backbone --unfreeze-epoch 25 --epochs 60 --batch-size 8 --learning-rate 1e-3 --backbone-learning-rate 1e-4 --output-dir outputs/yellowness_regression/vitb16_mae_unfreeze25
```

TorchGeo pretrained weight enums supported in your experiments:

```text
ResNet50_Weights.SENTINEL2_ALL_DINO
ResNet50_Weights.SENTINEL2_ALL_MOCO
ViTSmall16_Weights.SENTINEL2_ALL_DINO
ViTSmall16_Weights.SENTINEL2_ALL_MOCO
ViTBase16_Weights.SENTINEL2_ALL_MAE
ViTBase16_Weights.SENTINEL2_ALL_FGMAE
FGMAEEarthLoc_Weights.SENTINEL2_RESNET50
```

Available mask-fusion modes:

```text
image_only
masked_image
mask_as_channel
image_and_mask
late_mask
```

Supported backbone pattern:

```text
auto
simple_cnn
torchgeo:<TorchGeoModelName>
```

### Backbone And Mask-Fusion Benchmarking

Benchmark one backbone across all mask-integration modes:

```bash
python scripts/benchmark_yellowness_models.py --inventory yellowness_dataset/observations_inventory_all_feature_new.csv --root-dir yellowness_dataset --resolution sr --backbones auto --torchgeo-weights ResNet50_Weights.SENTINEL2_ALL_DINO ResNet50_Weights.SENTINEL2_ALL_MOCO ViTSmall16_Weights.SENTINEL2_ALL_DINO ViTSmall16_Weights.SENTINEL2_ALL_MOCO ViTBase16_Weights.SENTINEL2_ALL_MAE ViTBase16_Weights.SENTINEL2_ALL_FGMAE FGMAEEarthLoc_Weights.SENTINEL2_RESNET50 --mask-fusions image_only masked_image mask_as_channel image_and_mask late_mask --freeze-backbone --unfreeze-epoch 0 --epochs 40 --batch-size 8 --learning-rate 1e-3 --backbone-learning-rate 1e-4 --output outputs/yellowness_benchmark_torchgeo_weights.csv
```

Benchmark mixed baselines (simple CNN and TorchGeo):

```bash
python scripts/benchmark_yellowness_models.py --inventory yellowness_dataset/observations_inventory_all_feature_new.csv --root-dir yellowness_dataset --resolution sr --backbones simple_cnn auto --torchgeo-weights ResNet50_Weights.SENTINEL2_ALL_DINO --mask-fusions mask_as_channel late_mask --freeze-backbone --epochs 40 --batch-size 8 --learning-rate 1e-3 --backbone-learning-rate 1e-4 --output outputs/yellowness_benchmark_mixed.csv
```

### Evaluation

Open the main evaluation notebook:

```bash
jupyter notebook notebooks/evaluation_unified.ipynb
```

Open the data-preparation validation notebook:

```bash
jupyter notebook notebooks/data_preparation_validation.ipynb
```

### Important Notes

- Several preparation scripts still use hardcoded local paths for the Sentinel-2 stack, parcel files, and model directory.
- The training scripts expect inventories under `torchgeo_yellowness_dataset/`.
- The benchmark script writes a CSV table that you can use as the comparison output across backbone and mask-integration combinations.

Primary dependencies include:

- `numpy`, `pandas`
- `geopandas`, `shapely`, `rasterio`
- `torch`, `mlstac`, `sen2sr`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `lpips` for evaluation
- optional `scikit-image` for texture features

## Notes

- Several scripts still use project-local absolute paths and should be adjusted before reuse on another machine.
- Large data artifacts and generated outputs are expected to live outside version-control discipline even if they are present locally in this workspace.
- Exploratory notebooks are kept only for active analysis; archived notebook duplicates were removed during cleanup.
