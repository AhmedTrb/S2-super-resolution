# Yellowness Detection Methods: ML and DL Approaches

Both families of methods consume the same LR/SR inventory produced by the
[data pipeline](data_pipeline.md) (`lr_inventory.csv` / `sr_inventory.csv` and
their associated patch/mask GeoTIFFs) and predict a continuous parcel-level
yellowness value. Two complementary approaches are implemented and compared:
**classical ML on extracted/backbone features**, and **end-to-end deep
learning** on raw patches.

## A. Classical ML methods

### A.1 Feature-based regression (`scripts/train_yellowness_regressor.py`)
- Input: the precomputed statistical/spectral-index features already stored in
  the inventory CSV (band means/stds, NDRE/NDYVI/GCC/NDYI/mNDblue, etc. — see
  [data_pipeline.md](data_pipeline.md#step-6--compute-parcel-level-spectral-indicesstatistics)).
- Optional dimensionality reduction: PCA (`pca`, `pca_var95`), mutual-info
  top-k (`mi_topk`), F-regression top-k (`f_reg_topk`), variance top-k
  (`var_topk`), or PLS (feature ranked by |correlation| with target, top 8 by
  default, component count auto-capped via `SafePLSRegressor`).
- Regressors: Bayesian Ridge, Ridge, PLS, Random Forest, Extra Trees, XGBoost.
- Optional target transform, grouped train/val split (`make_group_split`,
  stratified by parcel-level yellowness quantiles so a parcel's observations
  never leak across the split).

### A.2 Backbone feature extraction + ML regressor (`scripts/train_yellowness_backbone_ml.py`)
- Input: raw LR or SR image patches (not the tabular features).
- A pretrained **TorchGeo** backbone (ResNet50, ViT-S/B, DINOv2, Swin v2,
  auto-resolved from `--torchgeo-weights`, e.g.
  `ResNet50_Weights.SENTINEL2_ALL_NDVI_SECO_ECO`) extracts a feature vector per
  patch, optionally restricted to a subset of Sentinel-2 bands
  (`backbone_band_indices`, e.g. dropping B8A for SATLAS MI_MS weights).
- The extracted features are then reduced (`--feature-reduction-method`:
  `none`, `pca`, `pca_var95`, `mi_topk`, `f_reg_topk`, `var_topk`) and fed into
  the same family of regressors as A.1 (ridge, pls, random_forest,
  extra_trees, xgboost).
- Logs a `[band-audit]` line per run to confirm which bands/channels were
  actually fed to the backbone.

### A.3 Representation benchmark (`scripts/run_yellowness_representation_benchmark.py`)
- Sweeps many regressor + dimensionality-reduction combinations (Ridge, PLS,
  Random Forest, HistGradientBoosting, XGBoost, LightGBM, CatBoost x PCA/SVD/
  PLS) over LR vs. SR features to systematically identify the best-performing
  ML configuration. Produces comparison plots/tables under
  `outputs/ML_regression/`.

## B. Deep learning methods

### B.1 Small masked CNN (`scripts/train_small_masked_cnn.py`, `src/s2_pipeline/small_masked_cnn.py`)
End-to-end CNN trained directly on LR or SR patches + parcel mask:
1. Convolutional stem (24 channels, GroupNorm, SiLU).
2. Three residual (or, with `--architecture plain_cnn`, plain non-residual)
   blocks with channel progression 24 → 48 → 64 and stride-2 downsampling.
3. **Masked average pooling**: only in-parcel pixels of the final feature map
   are pooled, using the parcel mask.
4. Pooled features concatenated with the parcel mask area fraction.
5. Regression head: 65 → 32 → 1 (SiLU, dropout 0.2).

Supports:
- Band recipes: `raw10` (all 10 S2 bands) or `paper_indices` (`B2,B3,B4,H,S,
  GLI,OSAVI,NDRE`, computed on the fly in `_apply_band_recipe`).
- Mask fusion modes (mask as extra channel, masked-image, feature-mask-pool,
  etc.).
- Loss: Huber. Optimizer: AdamW + `ReduceLROnPlateau`. Best checkpoint
  selected on validation MAE. Early stopping on non-improving epochs.
- Training-only augmentation: horizontal/vertical flip + random 90° rotation,
  applied identically to image and mask.

### B.2 Fixed representation experiments E1–E4 (the paper-replication protocol)
Two dedicated runners reuse the exact same `SmallMaskedCNNRegressor` to answer
one question in isolation: **does spatial resolution (LR vs SR) or spectral
representation (raw bands vs. derived indices) matter more?**

| Experiment | Resolution | Representation | Patch size | Channels |
|---|---|---|---:|---:|
| E1 | LR | `raw10` | 128×128 | 10 |
| E2 | LR | `paper_indices` | 128×128 | 8 |
| E3 | SR | `raw10` | 256×256 | 10 |
| E4 | SR | `paper_indices` | 256×256 | 8 |

- `scripts/run_paper_representation_experiments.py`: single grouped
  train/val/test split (70/15/15 at parcel level, seed 42), no
  hyperparameter search, test set never used for model selection. Outputs
  `comparison_table.csv` (MAE/RMSE/R2 per split, best epoch, param count).
- `scripts/run_paper_representation_cv.py`: K-fold (default 5) grouped-
  stratified cross-validation version of the same 4 experiments, with
  automatic LR patch-size selection (`select_lr_patch_size`, smallest of
  {32,64,128}px that fully contains ~100% of parcel masks) and a fixed SR
  patch size (256px). Supports `--mask-channel-variants both` (with/without
  the mask channel) and `--architecture {residual, plain_cnn}`.
- Full methodology (splits, augmentation, LR schedule, checkpoint selection,
  reproducibility): [paper_representation_training_protocol.md](paper_representation_training_protocol.md).
- Full results write-up (quantitative comparison, learning curves, residual
  analysis, conclusions): [analyse resultats DL.md](analyse%20resultats%20DL.md).
  **Headline result: E4 (SR + paper_indices) is the best configuration**
  (test MAE 17.50, RMSE 22.84, R² 0.674), showing that super-resolution helps
  most when combined with derived spectral indices rather than raw bands.

### B.3 Ablations (`scripts/run_small_cnn_ablations.py`)
- Materializes one shared train/val split (`*_shared_split`) and reuses it
  across 8 ablation configurations of the small masked CNN (varying band
  recipe / mask fusion / architecture), to isolate the effect of each design
  choice without split-related variance. Merges per-fold histories into
  summary CSVs under `outputs/DL_training_results/` and
  `outputs/small_masked_cnn/`.

### B.4 Interactive training notebook (`notebooks/03_dl_models_training.ipynb`)
- The Kaggle-oriented notebook that runs the full backbone sweep (TorchGeo
  ResNet/ViT/DINO/Swin, mask fusion modes, spectral indices) and, in later
  cells, calls `run_paper_representation_experiments.py` /
  `run_paper_representation_cv.py` via `subprocess` for the fixed E1-E4
  comparison, plus masked-pooling feature-map visualizations.

## C. Which script to run for what

| Goal | Script |
|---|---|
| Quick baseline on tabular features | `scripts/train_yellowness_regressor.py` |
| Try a pretrained backbone + ML head | `scripts/train_yellowness_backbone_ml.py` |
| Sweep many ML configurations at once | `scripts/run_yellowness_representation_benchmark.py` |
| Train one small CNN configuration | `scripts/train_small_masked_cnn.py` |
| Reproduce the LR/SR × raw/indices comparison (single split) | `scripts/run_paper_representation_experiments.py` |
| Reproduce the LR/SR × raw/indices comparison (cross-validated) | `scripts/run_paper_representation_cv.py` |
| Compare CNN design choices with one fixed split | `scripts/run_small_cnn_ablations.py` |
| Benchmark backbone architectures under a fixed protocol | `scripts/benchmark_yellowness_models.py` |
