# Deep Learning Training Protocol

## Objective

The objective is to compare two Sentinel-2 input representations with the same
`SmallMaskedCNNRegressor` architecture and the same training procedure:

| Experiment | Resolution | Input representation | Image channels | Patch size |
|---|---:|---|---:|---:|
| E1 | LR | `raw10`: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 | 128 x 128 |
| E2 | LR | `paper_indices`: B2, B3, B4, H, S, GLI, OSAVI, NDRE | 8 | 128 x 128 |
| E3 | SR | `raw10` | 10 | 256 x 256 |
| E4 | SR | `paper_indices` | 8 | 256 x 256 |

Only the input representation and spatial resolution vary. The CNN architecture,
parcel-mask integration, loss, augmentation, optimizer, learning-rate schedule,
and stopping procedure remain fixed across all experiments.

## Data Split

- A single split is created once with random seed `42` and reused for E1-E4.
- Splitting is performed at parcel level using the resolved parcel identifier,
  so observations from the same parcel cannot occur in different subsets.
- Parcel-level mean yellowness is divided into up to five quantile bins before
  sampling groups. This preserves the target distribution approximately across
  train, validation, and test groups.
- The default proportions are 70% train, 15% validation, and 15% test at group
  level. The exact row and parcel counts are saved in `split_ids.json`.
- The test set is held out until final evaluation. It is never used for epoch
  selection, early stopping, learning-rate decisions, or any other model
  selection decision.

## Input Preparation

Sentinel-2 reflectance is normalized to the range [0, 1]. For `paper_indices`,
the following bands are used:

- R = B4, G = B3, B = B2, NIR = B8, and RE = B5.
- H and S are computed with the standard HSV formulas from R, G, and B; H is
  normalized from degrees to [0, 1].
- GLI = (2G - R - B) / (2G + R + B).
- OSAVI = (NIR - R) / (NIR + R + 0.16).
- NDRE = (NIR - RE) / (NIR + RE).

All divisions use an epsilon-protected safe division. NaN and Inf values are
replaced by zero. The parcel mask is kept as a separate one-channel tensor and
is also concatenated to the image input, giving 11 effective CNN input channels
for `raw10` and 9 for `paper_indices`.

Before training each experiment, the pipeline verifies:

- expected number of image channels and patch dimensions;
- image and mask spatial alignment;
- sampled channel means, standard deviations, minima, and maxima;
- absence of NaN and Inf values;
- non-empty parcel masks; and
- total and trainable parameter counts.

## Model Architecture

The same `SmallMaskedCNNRegressor` is trained from scratch in all four runs.
It consists of:

1. A convolutional stem with 24 channels, GroupNorm, and SiLU activation.
2. A 24-channel residual block.
3. A stride-2 downsampling block from 24 to 48 channels.
4. A 48-channel residual block.
5. A stride-2 downsampling block from 48 to 64 channels.
6. A 64-channel residual block.
7. Parcel-masked average pooling of the feature map.
8. Concatenation of the pooled features with parcel mask area fraction.
9. A fully connected head: 65 -> 32 -> 1, with SiLU and dropout 0.2.

The model uses adaptive spatial behavior through convolutional downsampling and
masked pooling, allowing both 128 x 128 and 256 x 256 patches without resizing.
No pretrained weights are used.

## Loss and Optimization

- Loss: Huber loss.
- Optimizer: AdamW.
- Initial learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `16` in the Kaggle notebook configuration; the standalone script
  default is `8`.
- Training augmentation: enabled for training samples only.
- Validation and test augmentation: disabled.

The training augmentation applies synchronized geometric transformations to the
image and parcel mask:

- horizontal flip with probability 0.5;
- vertical flip with probability 0.5; and
- a random 90-degree rotation selected from 0, 90, 180, or 270 degrees.

The mask receives exactly the same transformation as its image, preserving
pixel-level alignment.

## Learning-Rate Scheduling and Early Stopping

After each epoch, the model is evaluated on the validation set. A
`ReduceLROnPlateau` scheduler monitors validation Huber loss:

- reduction factor: `0.5`;
- scheduler patience: `10` epochs without validation-loss improvement.

The best checkpoint is selected using validation MAE only. The default maximum
number of epochs is `120` in the Kaggle cell and `200` in the standalone script.
Early stopping terminates training after `20` consecutive non-improving epochs
in the Kaggle configuration, or `30` epochs with the standalone defaults. The
checkpoint from the best validation-MAE epoch is restored before train,
validation, and test predictions are generated.

## Evaluation

For each experiment, predictions are generated with the restored best
validation checkpoint. The following metrics are reported independently for
train, validation, and test sets:

- MAE: mean absolute error;
- RMSE: root mean squared error; and
- R2: coefficient of determination.

The best epoch and total/trainable parameter counts are also reported. Training
histories are saved in `history.csv`, predictions in
`predictions_train.csv`, `predictions_val.csv`, and `predictions_test.csv`, and
the final four-experiment comparison is saved in `comparison_table.csv`.

## Reproducibility

Python, NumPy, and PyTorch random generators are seeded with `42`. The same
split metadata, model settings, input verification reports, histories, model
checkpoints, predictions, and comparison table are saved under the experiment
output directory.