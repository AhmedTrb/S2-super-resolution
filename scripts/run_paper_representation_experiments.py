#!/usr/bin/env python3
"""Run the 4 fixed input-representation experiments with the unchanged SmallMaskedCNNRegressor.

The architecture, mask integration, loss (Huber), augmentation, optimizer (AdamW),
scheduler (ReduceLROnPlateau) and training procedure are identical to
`train_small_masked_cnn.py`. Only the input representation and the resolution
change across the 4 experiments, using ONE shared grouped parcel train/val/test
split and ONE seed:

    E1: LR (128x128) + raw10          -> [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
    E2: LR (128x128) + paper_indices  -> [B2,B3,B4,H,S,GLI,OSAVI,NDRE]
    E3: SR (256x256) + raw10
    E4: SR (256x256) + paper_indices

No cross-validation, no hyperparameter search, no additional indices, no
pretrained models. Test performance is never used for model/epoch selection
(selection is on validation MAE only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.s2_pipeline.small_masked_cnn import SmallMaskedCNNConfig, SmallMaskedCNNRegressor
from src.s2_pipeline.yellowness_training import ParcelYellownessDataset

import train_small_masked_cnn as tcnn

EXPERIMENTS: tuple[dict[str, str], ...] = (
    {"name": "E1", "resolution": "lr", "band_recipe": "raw10", "patch_size": "128"},
    {"name": "E2", "resolution": "lr", "band_recipe": "paper_indices", "patch_size": "128"},
    {"name": "E3", "resolution": "sr", "band_recipe": "raw10", "patch_size": "256"},
    {"name": "E4", "resolution": "sr", "band_recipe": "paper_indices", "patch_size": "256"},
)

CHANNELS_BY_RECIPE: dict[str, int] = {"raw10": 10, "paper_indices": 8}


def make_grouped_train_val_test_split(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Grouped (no parcel leakage) split, stratified by parcel-level yellowness quantiles.

    Preserves the same balancing strategy as the existing CV split (`make_group_bins`)
    but produces a single train/val/test partition instead of K folds.
    """
    group_col = tcnn.resolve_group_col(df)
    bins_map = tcnn.make_group_bins(df, group_col, n_bins=n_bins)

    groups = df[group_col].astype(str)
    unique_groups = groups.drop_duplicates().to_numpy()
    group_bins = pd.Series(unique_groups, index=unique_groups).map(bins_map).fillna(0).astype(int)

    rng = np.random.RandomState(seed)
    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []

    for b in sorted(group_bins.unique().tolist()):
        bin_groups = group_bins[group_bins == b].index.to_numpy().copy()
        rng.shuffle(bin_groups)
        n = len(bin_groups)

        n_test = max(1, int(round(n * test_frac))) if n >= 3 else (1 if n >= 2 else 0)
        n_test = min(n_test, max(0, n - 1))
        remaining = n - n_test
        n_val = max(1, int(round(n * val_frac))) if remaining >= 2 else (1 if remaining >= 1 else 0)
        n_val = min(n_val, max(0, remaining - 1)) if remaining >= 1 else 0

        test_groups.extend(bin_groups[:n_test].tolist())
        val_groups.extend(bin_groups[n_test : n_test + n_val].tolist())
        train_groups.extend(bin_groups[n_test + n_val :].tolist())

    train_idx = np.where(groups.isin(train_groups).to_numpy())[0]
    val_idx = np.where(groups.isin(val_groups).to_numpy())[0]
    test_idx = np.where(groups.isin(test_groups).to_numpy())[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"Degenerate split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} rows. "
            "Increase dataset size or reduce n_bins/val_frac/test_frac."
        )

    def _bin_distribution(ids: list[str]) -> dict[str, int]:
        counts = pd.Series(ids).map(bins_map).value_counts().sort_index()
        return {str(k): int(v) for k, v in counts.items()}

    meta = {
        "group_column": group_col,
        "seed": int(seed),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "n_bins": int(n_bins),
        "method": "grouped_stratified_train_val_test",
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_bin_distribution": _bin_distribution(train_groups),
        "val_bin_distribution": _bin_distribution(val_groups),
        "test_bin_distribution": _bin_distribution(test_groups),
    }
    return train_idx, val_idx, test_idx, meta


def build_raw_datasets(
    cfg: "ExperimentConfig",
    resolution: str,
    band_recipe: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> tuple[ParcelYellownessDataset, ParcelYellownessDataset, ParcelYellownessDataset]:
    dataset_band_recipe = None if band_recipe == "raw10" else band_recipe
    kwargs = dict(
        root_dir=cfg.root_dir,
        resolution=resolution,
        band_indices=None,
        band_recipe=dataset_band_recipe,
        shapefile_path=cfg.shapefile_path,
        shapefile_buffer_m=cfg.mask_buffer_m,
    )
    train_ds = ParcelYellownessDataset(cfg.inventory, rows=train_rows, **kwargs)
    val_ds = ParcelYellownessDataset(cfg.inventory, rows=val_rows, **kwargs)
    test_ds = ParcelYellownessDataset(cfg.inventory, rows=test_rows, **kwargs)
    return train_ds, val_ds, test_ds


def verify_experiment_inputs(
    exp_name: str,
    train_ds: ParcelYellownessDataset,
    val_ds: ParcelYellownessDataset,
    test_ds: ParcelYellownessDataset,
    patch_size: int,
    expected_channels: int,
    n_stat_samples: int = 24,
) -> dict[str, Any]:
    """Verify input shape, channel statistics, NaN/Inf, mask alignment before training."""
    sample = train_ds[0]
    image = sample["image"].float()
    mask = sample["mask"].float()
    cropped_image, cropped_mask = tcnn.crop_image_and_mask(image, mask, patch_size)

    if cropped_image.shape[0] != expected_channels:
        raise ValueError(
            f"[{exp_name}] expected {expected_channels} channels, got {cropped_image.shape[0]}."
        )
    if cropped_mask.shape[-2:] != cropped_image.shape[-2:]:
        raise ValueError(
            f"[{exp_name}] mask/image spatial mismatch: mask={tuple(cropped_mask.shape)} "
            f"image={tuple(cropped_image.shape)}."
        )

    n_check = min(len(train_ds), n_stat_samples)
    stacked_images = []
    stacked_masks = []
    for i in range(n_check):
        s = train_ds[i]
        im, mk = tcnn.crop_image_and_mask(s["image"].float(), s["mask"].float(), patch_size)
        stacked_images.append(im)
        stacked_masks.append(mk)
    images = torch.stack(stacked_images)
    masks = torch.stack(stacked_masks)

    has_nan = bool(torch.isnan(images).any().item())
    has_inf = bool(torch.isinf(images).any().item())
    if has_nan or has_inf:
        raise ValueError(f"[{exp_name}] NaN/Inf detected in {n_check} sampled inputs.")
    if bool((masks.sum(dim=(-3, -2, -1)) <= 0).any().item()):
        raise ValueError(f"[{exp_name}] empty mask found among sampled inputs.")

    report = {
        "experiment": exp_name,
        "patch_size": int(patch_size),
        "raw_shape": tuple(int(x) for x in image.shape),
        "cropped_image_shape": tuple(int(x) for x in cropped_image.shape),
        "cropped_mask_shape": tuple(int(x) for x in cropped_mask.shape),
        "expected_channels": int(expected_channels),
        "n_stat_samples": int(n_check),
        "channel_mean": [float(x) for x in images.mean(dim=(0, 2, 3)).tolist()],
        "channel_std": [float(x) for x in images.std(dim=(0, 2, 3)).tolist()],
        "channel_min": [float(x) for x in images.amin(dim=(0, 2, 3)).tolist()],
        "channel_max": [float(x) for x in images.amax(dim=(0, 2, 3)).tolist()],
        "has_nan": has_nan,
        "has_inf": has_inf,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    print(
        f"[verify] {exp_name}: raw_shape={report['raw_shape']} cropped={report['cropped_image_shape']} "
        f"nan={has_nan} inf={has_inf} train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )
    return report


@dataclass
class ExperimentConfig:
    inventory: str
    root_dir: str
    output_dir: str
    shapefile_path: str
    mask_buffer_m: float
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    seed: int
    val_frac: float
    test_frac: float
    use_mask_channel: bool
    pooling_mode: str
    loss_name: str
    augmentation: bool
    select_on: str


def train_and_evaluate_experiment(
    exp_name: str,
    resolution: str,
    band_recipe: str,
    patch_size: int,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    cfg: ExperimentConfig,
    device: torch.device,
    out_root: Path,
) -> dict[str, Any]:
    exp_dir = out_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_raw, val_raw, test_raw = build_raw_datasets(
        cfg, resolution=resolution, band_recipe=band_recipe,
        train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
    )

    expected_channels = CHANNELS_BY_RECIPE[band_recipe]
    verify_report = verify_experiment_inputs(
        exp_name, train_raw, val_raw, test_raw, patch_size=patch_size, expected_channels=expected_channels
    )

    mean, std = tcnn.compute_train_band_stats(train_raw, patch_size=patch_size)

    train_ds = tcnn.FoldDataset(train_raw, mean=mean, std=std, patch_size=patch_size, augment=cfg.augmentation)
    val_ds = tcnn.FoldDataset(val_raw, mean=mean, std=std, patch_size=patch_size, augment=False)
    test_ds = tcnn.FoldDataset(test_raw, mean=mean, std=std, patch_size=patch_size, augment=False)

    use_cuda_loader = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda_loader,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = SmallMaskedCNNRegressor(
        SmallMaskedCNNConfig(
            image_channels=expected_channels,
            use_mask_channel=cfg.use_mask_channel,
            pooling_mode=cfg.pooling_mode,
            dropout=0.2,
        )
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] {exp_name}: total={total_params} trainable={trainable_params}")

    loss_fn = tcnn.get_loss(cfg.loss_name)
    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    history: list[dict[str, Any]] = []
    best_value = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_y, tr_p = tcnn.run_epoch(model, train_loader, optimizer=opt, loss_fn=loss_fn, device=device)
        va_loss, va_y, va_p = tcnn.run_epoch(model, val_loader, optimizer=None, loss_fn=loss_fn, device=device)

        tr_m = tcnn.metric_dict(tr_y, tr_p)
        va_m = tcnn.metric_dict(va_y, va_p)
        lr_now = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "validation_loss": va_loss,
            "train_MAE": tr_m["MAE"],
            "validation_MAE": va_m["MAE"],
            "train_RMSE": tr_m["RMSE"],
            "validation_RMSE": va_m["RMSE"],
            "train_R2": tr_m["R2"],
            "validation_R2": va_m["R2"],
            "lr": lr_now,
        }
        history.append(row)
        print(
            f"[{exp_name}] epoch={epoch:03d} train_loss={tr_loss:.5f} val_loss={va_loss:.5f} "
            f"train_mae={tr_m['MAE']:.5f} val_mae={va_m['MAE']:.5f} val_r2={va_m['R2']:.4f} lr={lr_now:.2e}"
        )

        # Selection uses validation only; test set is never inspected for early stopping/checkpointing.
        monitor = va_m["MAE"] if cfg.select_on == "val_mae" else va_loss
        scheduler.step(va_loss)
        if monitor < best_value:
            best_value = float(monitor)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {"model": model.state_dict(), "epoch": epoch, "monitor": best_value}
        else:
            stale_epochs += 1

        if stale_epochs >= cfg.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f"[{exp_name}] training finished without a valid checkpoint.")

    torch.save(best_state, exp_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
    (exp_dir / "input_verification.json").write_text(json.dumps(verify_report, indent=2), encoding="utf-8")

    model.load_state_dict(best_state["model"])
    pred_tr = tcnn.predict_full(model, train_loader, device)
    pred_va = tcnn.predict_full(model, val_loader, device)
    pred_te = tcnn.predict_full(model, test_loader, device)
    pred_tr.to_csv(exp_dir / "predictions_train.csv", index=False)
    pred_va.to_csv(exp_dir / "predictions_val.csv", index=False)
    pred_te.to_csv(exp_dir / "predictions_test.csv", index=False)

    tr_metrics = tcnn.metric_dict(pred_tr["y_true"].to_numpy(), pred_tr["y_pred"].to_numpy())
    va_metrics = tcnn.metric_dict(pred_va["y_true"].to_numpy(), pred_va["y_pred"].to_numpy())
    te_metrics = tcnn.metric_dict(pred_te["y_true"].to_numpy(), pred_te["y_pred"].to_numpy())

    result = {
        "experiment": exp_name,
        "resolution": resolution,
        "band_recipe": band_recipe,
        "patch_size": int(patch_size),
        "image_channels": int(expected_channels),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "best_epoch": int(best_epoch),
        "train_MAE": tr_metrics["MAE"], "train_RMSE": tr_metrics["RMSE"], "train_R2": tr_metrics["R2"],
        "val_MAE": va_metrics["MAE"], "val_RMSE": va_metrics["RMSE"], "val_R2": va_metrics["R2"],
        "test_MAE": te_metrics["MAE"], "test_RMSE": te_metrics["RMSE"], "test_R2": te_metrics["R2"],
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 4 fixed E1-E4 input-representation experiments with SmallMaskedCNNRegressor."
    )
    p.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    p.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_representation_experiments")
    p.add_argument("--shapefile-path", type=Path, default=REPO_ROOT / "data" / "shapefiles" / "2020_SEPIM.shp")
    p.add_argument("--mask-buffer-m", type=float, default=-20.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--use-mask-channel", dest="use_mask_channel", action="store_true")
    p.add_argument("--no-use-mask-channel", dest="use_mask_channel", action="store_false")
    p.add_argument("--pooling-mode", choices=["global_avg", "masked_avg"], default="masked_avg")
    p.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    p.add_argument("--augmentation", dest="augmentation", action="store_true")
    p.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    p.add_argument("--select-on", choices=["val_mae", "val_loss"], default="val_mae")
    p.add_argument("--require-gpu", action="store_true")
    p.set_defaults(use_mask_channel=True, augmentation=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tcnn.set_seed(args.seed)

    cuda_available = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda_available={cuda_available}")
    if args.require_gpu and not cuda_available:
        raise RuntimeError("GPU is required (--require-gpu) but CUDA is not available.")
    device = torch.device("cuda" if cuda_available else "cpu")

    cfg = ExperimentConfig(
        inventory=str(args.inventory),
        root_dir=str(args.root_dir),
        output_dir=str(args.output_dir),
        shapefile_path=str(args.shapefile_path),
        mask_buffer_m=float(args.mask_buffer_m),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        use_mask_channel=bool(args.use_mask_channel),
        pooling_mode=str(args.pooling_mode),
        loss_name=str(args.loss),
        augmentation=bool(args.augmentation),
        select_on=str(args.select_on),
    )

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tcnn.save_yaml(out_root / "config.yaml", asdict(cfg))

    full_df = pd.read_csv(cfg.inventory)
    train_idx, val_idx, test_idx, split_meta = make_grouped_train_val_test_split(
        full_df, seed=cfg.seed, val_frac=cfg.val_frac, test_frac=cfg.test_frac
    )
    (out_root / "split_ids.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    print(
        f"[split] group_column={split_meta['group_column']} "
        f"train_rows={split_meta['train_rows']} val_rows={split_meta['val_rows']} test_rows={split_meta['test_rows']}"
    )

    train_rows = full_df.iloc[train_idx].reset_index(drop=True)
    val_rows = full_df.iloc[val_idx].reset_index(drop=True)
    test_rows = full_df.iloc[test_idx].reset_index(drop=True)

    results: list[dict[str, Any]] = []
    for exp in EXPERIMENTS:
        print(f"\n{'=' * 80}\nRunning {exp['name']}: resolution={exp['resolution']} band_recipe={exp['band_recipe']}\n{'=' * 80}")
        result = train_and_evaluate_experiment(
            exp_name=exp["name"],
            resolution=exp["resolution"],
            band_recipe=exp["band_recipe"],
            patch_size=int(exp["patch_size"]),
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            cfg=cfg,
            device=device,
            out_root=out_root,
        )
        results.append(result)

    comparison = pd.DataFrame(results)
    comparison_path = out_root / "comparison_table.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\nComparison table (E1-E4):")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
