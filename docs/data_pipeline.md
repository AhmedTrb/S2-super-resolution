# Data Pipeline: Time Series Reading, Patch Extraction, LR/SR Dataset Building

This document describes the steps actually followed to go from a raw multi-temporal
Sentinel-2 stack + field observations to a ready-to-use LR/SR parcel patch dataset.
It reflects the logic implemented in [src/s2_pipeline](../src/s2_pipeline) and
exercised by [scripts/data_preparation](../scripts/data_preparation) and
[notebooks/01_data_preparation_validation.ipynb](../notebooks/01_data_preparation_validation.ipynb).

## 1. Inputs

| Input | Description | Example path |
|---|---|---|
| Sentinel-2 time series stack | Multi-date, 10-band GeoTIFF stack (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) per zone | `Sentinel2_T31UEQ_TSG.tif` |
| Acquisition calendar | Text file listing the Sentinel-2 acquisition dates that index the stack's time axis | `data/Sentinel2_T31UEQ_interpolation_dates.txt` |
| Parcel geometries | Shapefile/GeoPackage with one polygon per parcel | `data/shapefiles/2020_SEPIM.shp`, `data/parcels/*.gpkg` |
| Field observations | CSV/table of yellowness (or other trait) observations with a date and a parcel id | project-specific, loaded per script |
| SEN2SRLite model | Pretrained 4x super-resolution model, auto-downloaded via `mlstac` | `model/SEN2SRLite/` |

> ⚠️ **Paths must be edited per machine.** The stack path, parcel shapefile path, and
> model directory are hardcoded at the top of each script/notebook (not read from a
> shared config). See [docs/paths_to_update.md](paths_to_update.md) for the exact
> list of lines to edit before running anything.

## 2. Step-by-step pipeline

### Step 1 — Match each observation to a Sentinel-2 acquisition date
`src/s2_pipeline/date_matcher.py`
- Reads the acquisition calendar (`Sentinel2_T*_interpolation_dates.txt`).
- For each field observation date, finds the nearest Sentinel-2 acquisition date
  (nearest-neighbor date snapping), so every observation can be linked to a
  specific time-index (band-group) in the stack.

### Step 2 — Build a parcel-centered patch window
`src/s2_pipeline/patching.py`
- Reads the parcel geometry for the observation's parcel id.
- Computes a fixed-size pixel window (`PatchWindow`: row/col min/max) centered on
  the parcel centroid/bounds, in the stack's raster grid (affine transform).
- Ensures the window is large enough to contain the parcel polygon; larger LR
  patch sizes are tried if the parcel doesn't fully fit (see
  `select_lr_patch_size` in `scripts/run_paper_representation_cv.py` for the
  automatic sizing logic used later in training).

### Step 3 — Read the LR (Sentinel-2) patch for the matched date
- Using the patch window and the matched acquisition's band-group index, a
  `(10, H, W)` array is read directly from the multi-temporal stack with
  `rasterio` (windowed read, no full-image load).
- Reflectance is normalized to `[0, 1]` before any further processing.

### Step 4 — Run SEN2SRLite super-resolution on the LR patch
`src/s2_pipeline/super_resolution.py`
- Loads the SEN2SRLite model once (`mlstac.download` + `mlstac.load`, cached
  under `model/SEN2SRLite/`).
- Verifies patch dimensions are compatible with the model's tiling requirement
  (multiples of 256 px for full-image inference; small patches are inferred
  directly).
- Runs the model to obtain a `(10, 4H, 4W)` SR patch, then re-clips/normalizes
  back to `[0, 1]`.

### Step 5 — Rasterize LR/SR parcel masks
`src/s2_pipeline/masks.py`
- Rasterizes the parcel polygon onto the LR grid and, separately, onto the 4x
  SR grid (affine transform scaled accordingly), producing `uint8` binary masks
  used to restrict statistics/pooling to in-parcel pixels.

### Step 6 — Compute parcel-level spectral indices/statistics
`src/s2_pipeline/indices.py` (+ additional families integrated in `pipeline.py`)
- Band-wise mean/std over the masked parcel pixels, for both LR and SR.
- Base indices: NDRE, NDYVI, GCC, NDYI, mNDblue.
- Extended indices (added in the dataset-builder scripts): NDVI, GNDVI, NDMI,
  NBR, red-edge slope/curvature, selected band ratios/differences, and
  optional GLCM texture features (`scikit-image`, opt-in).

### Step 7 — Write outputs and inventory rows
`src/s2_pipeline/pipeline.py`, `scripts/data_preparation/*.py`
- LR and SR patches are saved as georeferenced `float32` GeoTIFFs.
- LR and SR masks are saved as `uint8` GeoTIFFs.
- One CSV row per observation is appended to an inventory file
  (`lr_inventory.csv` / `sr_inventory.csv` or `observations_inventory.csv`
  depending on the script), recording: parcel id, observation date, matched
  Sentinel-2 date, file paths, and computed statistics/indices.

## 3. Entry points that run this pipeline

| Script | Output location | Notes |
|---|---|---|
| `scripts/data_preparation/prepare_s2_observation.py` | `yellowness_dataset/` | Direct, single-purpose observation extraction. |
| `scripts/data_preparation/build_obs_dataset.py` | `yellowness_dataset/` (`lr_inventory.csv`, `sr_inventory.csv`, `images_lr/`, `images_sr/`, `masks_lr/`, `masks_sr/`) | Production builder used for the dataset committed under `yellowness_dataset/`; adds the full feature-family computation. |
| `scripts/data_preparation/build_temporal_dataset.py` | project output dir | Multi-temporal variant; also exposes a PyTorch `Dataset` wrapper directly. |
| `scripts/data_preparation/audit_observation_pipeline.py` | `outputs/audit_observation_pipeline/` | Runs the pipeline on a handful of parcels/dates and dumps diagnostic plots + logs to sanity-check reflectance ranges, SR outputs, and file paths before a full run. |
| `scripts/data_preparation/enrich_obs_features.py` | enriched CSV copy | Standalone re-computation of the extended feature set on an existing inventory CSV (kept for cases where the inline pipeline features need to be regenerated without re-running SR inference). |
| `notebooks/01_data_preparation_validation.ipynb` | ad hoc | Interactive, cell-by-cell walkthrough of steps 1-7 above with visual checks (LR vs SR patch previews, mask overlays) — the recommended way to validate the pipeline on a new zone before running a full batch script. |

## 4. Resulting dataset layout

```
yellowness_dataset/
├── images_lr/                 # LR GeoTIFF patches (float32, [0,1] reflectance)
├── images_sr/                 # SR GeoTIFF patches (float32, [0,1] reflectance, 4x resolution)
├── masks_lr/                  # LR parcel masks (uint8)
├── masks_sr/                  # SR parcel masks (uint8)
├── lr_inventory.csv           # one row per observation, LR paths + stats
├── sr_inventory.csv           # one row per observation, SR paths + stats
└── observations_inventory.csv # combined/raw inventory (depending on script used)
```

This inventory is the direct input to every ML/DL training script described in
[docs/detection_methods.md](detection_methods.md).
