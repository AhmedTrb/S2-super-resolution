# Kaggle Backbone Experiments

This sheet groups the Sentinel-2 backbone experiments into runnable Kaggle notebook commands.

Assumptions:
- Kaggle notebook has GPU accelerator enabled.
- The repository is mounted at `/kaggle/working/S2-super-resolution`.
- The training script is `scripts/train_yellowness_backbone_ml.py`.
- Multi-GPU extraction is handled automatically when Kaggle exposes 2 GPUs.

## Common Setup

```python
from pathlib import Path
import os
import subprocess

REPO = Path('/kaggle/working/S2-super-resolution')
python_bin = 'python'
INVENTORY = REPO / 'yellowness_dataset' / 'observations_inventory.csv'
DATASET_DIR = REPO / 'yellowness_dataset'
DEEP_OUT = REPO / 'outputs' / 'yellowness_regression'
ML_OUT = REPO / 'outputs' / 'yellowness_backbone_ml'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

```python
def run_cmd(cmd):
    print('RUN:', ' '.join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], check=True, cwd=REPO)
```

## One-Run Template

Use this template for any single experiment.

```python
cmd = [
    python_bin,
    'scripts/train_yellowness_backbone_ml.py',
    '--inventory', str(INVENTORY),
    '--root-dir', str(DATASET_DIR),
    '--resolution', 'sr',
    '--backbone', 'torchgeo:vit_base_patch16_224',
    '--torchgeo-weight', 'ViTBase16_Weights.SENTINEL2_ALL_MAE',
    '--mask-fusion', 'feature_mask_pool',
    '--regressors', 'ridge', 'elastic_net', 'pls', 'random_forest', 'extra_trees', 'xgboost',
    '--dr-methods', 'none', 'pca',
    '--pca-variance', '0.95',
    '--pls-top-k', '8',
    '--batch-size', '32',
    '--num-workers', '4',
    '--save-feature-csv',
    '--feature-csv-name', 'backbone_embeddings.csv',
    '--export-deep-runs-summary',
    '--deep-runs-root', str(DEEP_OUT),
    '--output-dir', str(ML_OUT / 'sr_vitb16_mae'),
]
run_cmd(cmd)
```

## Grouped SR Experiments

### 1. ResNet50 And ViT-Small 16 Group

```python
sr_group_1 = [
    ('torchgeo:resnet50', 'ResNet50_Weights.SENTINEL2_ALL_DINO', 'sr_resnet50_all_dino'),
    ('torchgeo:resnet50', 'ResNet50_Weights.SENTINEL2_ALL_MOCO', 'sr_resnet50_all_moco'),
    ('torchgeo:resnet50', 'ResNet50_Weights.SENTINEL2_ALL_DECUR', 'sr_resnet50_all_decur'),
    ('torchgeo:vit_small_patch16_224', 'ViTSmall16_Weights.SENTINEL2_ALL_DINO', 'sr_vitsmall16_all_dino'),
    ('torchgeo:vit_small_patch16_224', 'ViTSmall16_Weights.SENTINEL2_ALL_MOCO', 'sr_vitsmall16_all_moco'),
]

for backbone, weight, run_name in sr_group_1:
    cmd = [
        python_bin,
        'scripts/train_yellowness_backbone_ml.py',
        '--inventory', str(INVENTORY),
        '--root-dir', str(DATASET_DIR),
        '--resolution', 'sr',
        '--backbone', backbone,
        '--torchgeo-weight', weight,
        '--mask-fusion', 'feature_mask_pool',
        '--regressors', 'ridge', 'elastic_net', 'pls', 'random_forest', 'extra_trees', 'xgboost',
        '--dr-methods', 'none', 'pca',
        '--pca-variance', '0.95',
        '--pls-top-k', '8',
        '--batch-size', '32',
        '--num-workers', '4',
        '--save-feature-csv',
        '--feature-csv-name', 'backbone_embeddings.csv',
        '--export-deep-runs-summary',
        '--deep-runs-root', str(DEEP_OUT),
        '--output-dir', str(ML_OUT / run_name),
    ]
    run_cmd(cmd)
```

### 2. MAE And DINOv2 SoftCon Group

```python
sr_group_2 = [
    ('torchgeo:vit_base_patch16_224', 'ViTBase16_Weights.SENTINEL2_ALL_MAE', 'sr_vitbase16_all_mae'),
    ('torchgeo:vit_small_patch14_dinov2', 'ViTSmall14_DINOv2_Weights.SENTINEL2_ALL_SOFTCON', 'sr_vitsmall14_all_softcon'),
    ('torchgeo:vit_base_patch14_dinov2', 'ViTBase14_DINOv2_Weights.SENTINEL2_ALL_SOFTCON', 'sr_vitbase14_all_softcon'),
]

for backbone, weight, run_name in sr_group_2:
    cmd = [
        python_bin,
        'scripts/train_yellowness_backbone_ml.py',
        '--inventory', str(INVENTORY),
        '--root-dir', str(DATASET_DIR),
        '--resolution', 'sr',
        '--backbone', backbone,
        '--torchgeo-weight', weight,
        '--mask-fusion', 'feature_mask_pool',
        '--regressors', 'ridge', 'elastic_net', 'pls', 'random_forest', 'extra_trees', 'xgboost',
        '--dr-methods', 'none', 'pca',
        '--pca-variance', '0.95',
        '--pls-top-k', '8',
        '--batch-size', '32',
        '--num-workers', '4',
        '--save-feature-csv',
        '--feature-csv-name', 'backbone_embeddings.csv',
        '--export-deep-runs-summary',
        '--deep-runs-root', str(DEEP_OUT),
        '--output-dir', str(ML_OUT / run_name),
    ]
    run_cmd(cmd)
```

### 3. SATLAS MI-MS Group

```python
sr_group_3 = [
    ('torchgeo:resnet50', 'ResNet50_Weights.SENTINEL2_MI_MS_SATLAS', 'sr_resnet50_mi_ms_satlas'),
    ('torchgeo:swin_v2_t', 'Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS', 'sr_swinv2t_mi_ms_satlas'),
]

for backbone, weight, run_name in sr_group_3:
    cmd = [
        python_bin,
        'scripts/train_yellowness_backbone_ml.py',
        '--inventory', str(INVENTORY),
        '--root-dir', str(DATASET_DIR),
        '--resolution', 'sr',
        '--backbone', backbone,
        '--torchgeo-weight', weight,
        '--mask-fusion', 'feature_mask_pool',
        '--regressors', 'ridge', 'elastic_net', 'pls', 'random_forest', 'extra_trees', 'xgboost',
        '--dr-methods', 'none', 'pca',
        '--pca-variance', '0.95',
        '--pls-top-k', '8',
        '--batch-size', '32',
        '--num-workers', '4',
        '--save-feature-csv',
        '--feature-csv-name', 'backbone_embeddings.csv',
        '--export-deep-runs-summary',
        '--deep-runs-root', str(DEEP_OUT),
        '--output-dir', str(ML_OUT / run_name),
    ]
    run_cmd(cmd)
```

## Grouped LR Experiments

Use the same groups for LR by changing `--resolution sr` to `--resolution lr` and the output folder prefix from `sr_...` to `lr_...`.

```python
for backbone, weight, run_name in sr_group_1:
    cmd = [
        python_bin,
        'scripts/train_yellowness_backbone_ml.py',
        '--inventory', str(INVENTORY),
        '--root-dir', str(DATASET_DIR),
        '--resolution', 'lr',
        '--backbone', backbone,
        '--torchgeo-weight', weight,
        '--mask-fusion', 'feature_mask_pool',
        '--regressors', 'ridge', 'elastic_net', 'pls', 'random_forest', 'extra_trees', 'xgboost',
        '--dr-methods', 'none', 'pca',
        '--pca-variance', '0.95',
        '--pls-top-k', '8',
        '--batch-size', '32',
        '--num-workers', '4',
        '--save-feature-csv',
        '--feature-csv-name', 'backbone_embeddings.csv',
        '--export-deep-runs-summary',
        '--deep-runs-root', str(DEEP_OUT),
        '--output-dir', str(ML_OUT / run_name.replace('sr_', 'lr_')),
    ]
    run_cmd(cmd)
```

## Optional Fast Sweep

If you want the fastest sweep first, drop `pls` from `--regressors` and keep only:

```python
'--regressors', 'ridge', 'elastic_net', 'random_forest', 'extra_trees', 'xgboost'
```

## Notes

- The script now uses `DataParallel` automatically when Kaggle exposes more than one CUDA device.
- SATLAS MI-MS weights automatically use the 9-band subset `B02, B03, B04, B05, B06, B07, B08, B11, B12`.
- Feature exports now include `id_plot`, `observation_date`, `yellowness`, `backbone_name`, and the embedding vector, plus a compressed `.npz` companion file.
- PCA defaults to 95% explained variance and PLS is capped to a safe component count at fit time.
