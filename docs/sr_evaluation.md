# SR Model Evaluation Methodology

This document describes the evaluation steps implemented in
[notebooks/02_sr_evaluation_unified.ipynb](../notebooks/02_sr_evaluation_unified.ipynb),
the main entry point for comparing super-resolution model outputs against
PlanetScope reference imagery.

## Goal

Quantify how close SEN2SRLite (and any other candidate SR model) outputs are to
a higher-resolution independent reference (PlanetScope, ~3 m), and compare
against a naive bicubic-upsampling baseline, across multiple zones.

## Step-by-step evaluation pipeline

### 1. Discover PlanetScope reference scenes
`find_ps_files`, `extract_date_from_path`
- Lists all PlanetScope scenes available for a zone and parses their
  acquisition date from the filename.

### 2. Clip references to the zone of interest
`clip_ps_to_zone`
- Clips each PlanetScope scene to the study zone polygon so comparisons only
  use the relevant spatial extent.

### 3. Screen references by spatial coverage
`coverage_fraction`, `screen_ps_scenes` (threshold: `COVERAGE_THRESHOLD`)
- Rejects PlanetScope scenes with too much missing/cloud data over the zone
  (coverage below threshold), keeping only usable reference dates.

### 4. Discover candidate SR outputs
`find_sr_files`, `extract_date_from_sr`, `debug_sr_date_parsing`
- Lists SR GeoTIFFs produced by `scripts/sr_inference/inference-SEN2SRLite.py`
  (or another candidate model) and parses their acquisition date.

### 5. Pair each SR output with the closest-in-time PlanetScope reference
- For each screened PlanetScope date, the temporally-nearest SR output is
  selected as its counterpart for comparison.

### 6. Geometric alignment
`warp_sr_to_ps`, `align_arrays` (`center_crop`), `shared_valid_mask`
- Warps the SR raster onto the PlanetScope grid/CRS/resolution (`rasterio`
  reprojection), then center-crops both arrays to a common shape.
- Builds a shared valid-data mask (excludes nodata/cloud pixels present in
  either image) so metrics are only computed on jointly-valid pixels.

### 7. Radiometric alignment
`match_histograms_bandwise`
- Applies band-wise histogram matching between the SR image and the
  PlanetScope reference to reduce sensor/atmospheric-correction radiometric
  offsets before computing similarity metrics (this normalizes brightness/
  contrast differences that are not spatial-detail errors).

### 8. Compute the bicubic baseline
`build_baseline_from_stack`
- Builds a naive baseline by bicubically upsampling the original LR Sentinel-2
  patch to the PlanetScope grid resolution, to be scored with the same metric
  suite as the SR model. This is the reference point that quantifies how much
  the SR model actually adds over simple interpolation.

### 9. Compute similarity/error metrics
`compute_sam`, `compute_metrics`, `ssim_fn`, `to_lpips_tensor`
- **RMSE / MAE**: per-band pixel-wise error against the reference.
- **R²**: variance explained.
- **SSIM**: structural similarity (windowed), on the shared valid mask.
- **SAM** (Spectral Angle Mapper): spectral-shape fidelity, independent of
  brightness scaling.
- **LPIPS** (optional): perceptual similarity using a pretrained network, on
  an RGB composite.

### 10. Per-zone orchestration
`run_zone_evaluation`
For each zone, and for each screened PlanetScope reference/SR pair:
1. Geometric alignment (warp SR to PS grid, center-crop).
2. Valid mask + histogram matching.
3. Metrics for the SR model.
4. Metrics for the bicubic baseline (computed once per reference date).

Results are aggregated into `outputs/evaluation/evaluation_results_all_zones.csv`
(per-zone, per-date metrics for SR model vs. baseline) plus summary plots under
`outputs/evaluation_plots/` and `outputs/comparison_plots/`.

## Related notebook: strict two-zone pairing

`notebooks copy/evaluation_pairs_two_zones.ipynb` (kept in the legacy
`notebooks copy/` folder — not moved per current project decision) implements a
stricter variant of the same idea: it enforces exactly one SR-to-reference pair
per PlanetScope date (no many-to-one duplication), and exports the aligned
LR/SR/reference GeoTIFF triplets for each compared pair, in addition to the
same metric suite.

## Interpreting results

- Compare **SR metrics vs. bicubic-baseline metrics** for the same reference —
  the SR model is only useful where it clearly beats the baseline.
- SSIM/SAM are more informative than RMSE alone for judging whether SR
  recovers real spatial/spectral structure rather than just smoothing.
- Coverage screening (step 3) matters: scenes with heavy cloud/nodata can
  silently bias metrics if not filtered out.
