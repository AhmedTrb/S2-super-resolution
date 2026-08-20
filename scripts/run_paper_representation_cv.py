#!/usr/bin/env python3
"""Separate command: grouped stratified K-fold CV for the fixed E1-E4 experiments.

Reuses the unchanged `SmallMaskedCNNRegressor`, mask integration, Huber loss,
AdamW/ReduceLROnPlateau, and validation-MAE checkpoint selection from
`train_small_masked_cnn.py` (`train_one_fold`), but trains/evaluates with
K-fold (default 5) grouped, target-stratified cross-validation instead of a
single train/val split. Data augmentation (flip/rotate) is enabled by default.

    E1: LR + raw10          E2: LR + paper_indices
    E3: SR + raw10          E4: SR + paper_indices

LR patch size is auto-selected as the smallest of {32, 64, 128} px that fully
contains the parcel mask (no truncation) for (nearly) all samples. The SR
patch size is fixed at 256px (`--sr-patch-size`) so the parcel reliably fits
in the SR crop regardless of the chosen LR size.

Use `--mask-channel-variants both` to additionally run every experiment
without the mask channel (`use_mask_channel=False`) alongside the default
with-mask-channel run, to see the effect of the mask channel. Use
`--architecture plain_cnn` to replace the 3 residual blocks with 3 normal
(non-residual) 2D conv blocks, still followed by masked feature pooling and
the same regression head.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

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
    {"name": "E1", "resolution": "lr", "band_recipe": "raw10"},
    {"name": "E2", "resolution": "lr", "band_recipe": "paper_indices"},
    {"name": "E3", "resolution": "sr", "band_recipe": "raw10"},
    {"name": "E4", "resolution": "sr", "band_recipe": "paper_indices"},
)

CHANNELS_BY_RECIPE: dict[str, int] = {"raw10": 10, "paper_indices": 8}
DEFAULT_LR_PATCH_CANDIDATES: tuple[int, ...] = (32, 64, 128)


def select_lr_patch_size(
    inventory: Path,
    root_dir: Path,
    shapefile_path: Path,
    mask_buffer_m: float,
    candidates: tuple[int, ...] = DEFAULT_LR_PATCH_CANDIDATES,
    containment_threshold: float = 1.0,
    max_samples: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Pick the smallest LR patch size (from `candidates`) that fully contains the parcel mask.

    For each candidate size, checks the fraction of samples where the centered crop
    (`crop_image_and_mask`) retains 100% of the full-image parcel-mask pixels (no
    truncation). Returns the smallest candidate whose containment rate reaches
    `containment_threshold`; falls back to the largest candidate otherwise.
    """
    dataset = ParcelYellownessDataset(
        inventory, root_dir=root_dir, resolution="lr",
        shapefile_path=shapefile_path, shapefile_buffer_m=mask_buffer_m,
    )
    n_check = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)

    report: dict[str, Any] = {"candidates": {}, "containment_threshold": containment_threshold, "checked_samples": n_check}
    chosen: Optional[int] = None

    for size in sorted(candidates):
        n_total = 0
        n_full = 0
        for i in range(n_check):
            sample = dataset[i]
            mask = sample["mask"].float()
            full_pixels = float(mask.sum().item())
            if full_pixels <= 0:
                continue
            _, cropped_mask = tcnn.crop_image_and_mask(sample["image"].float(), mask, size)
            crop_pixels = float(cropped_mask.sum().item())
            n_total += 1
            if crop_pixels >= full_pixels - 1e-6:
                n_full += 1
        rate = (n_full / n_total) if n_total else 0.0
        report["candidates"][str(size)] = {
            "checked_samples": n_total,
            "fully_contained": n_full,
            "containment_rate": rate,
        }
        print(f"[patch-select] size={size}px containment_rate={rate:.4f} ({n_full}/{n_total})")
        if chosen is None and rate >= containment_threshold:
            chosen = size

    if chosen is None:
        chosen = max(candidates)
        report["fallback_used"] = True
        print(f"[patch-select] no candidate reached threshold={containment_threshold}; falling back to largest={chosen}px")

    report["chosen_lr_patch_size"] = chosen
    return chosen, report


def run_cv_experiment(
    exp_name: str,
    resolution: str,
    band_recipe: str,
    patch_size: int,
    folds: list[tuple[Any, Any]],
    split_meta: dict[str, Any],
    full_df: pd.DataFrame,
    cfg_kwargs: dict[str, Any],
    device: torch.device,
    use_mask_channel: bool,
    architecture: str,
) -> dict[str, Any]:
    cfg = tcnn.TrainConfig(
        experiment_name=f"paper_cv_{exp_name}",
        resolution=resolution,
        band_recipe=band_recipe,
        model_kind="cnn",
        patch_size=int(patch_size),
        architecture=architecture,
        use_mask_channel=use_mask_channel,
        **cfg_kwargs,
    )

    out_root = Path(cfg.output_dir) / cfg.experiment_name / cfg.resolution
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "split_ids.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    tcnn.save_yaml(out_root / "config.yaml", asdict(cfg))

    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (tr, va) in enumerate(folds, start=1):
        train_rows = full_df.iloc[tr].reset_index(drop=True)
        val_rows = full_df.iloc[va].reset_index(drop=True)
        print(f"\n{'-' * 80}\n[{exp_name}] fold {fold_idx}/{len(folds)} resolution={resolution} band_recipe={band_recipe} patch_size={patch_size}\n{'-' * 80}")
        tcnn.train_one_fold(fold_idx=fold_idx, train_rows=train_rows, val_rows=val_rows, cfg=cfg, device=device)

        fold_dir = out_root / f"fold_{fold_idx:02d}"
        fold_metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold": fold_idx,
                "best_epoch": fold_metrics["best_epoch"],
                "train_MAE": fold_metrics["train"]["MAE"],
                "train_RMSE": fold_metrics["train"]["RMSE"],
                "train_R2": fold_metrics["train"]["R2"],
                "val_MAE": fold_metrics["validation"]["MAE"],
                "val_RMSE": fold_metrics["validation"]["RMSE"],
                "val_R2": fold_metrics["validation"]["R2"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_root / "fold_metrics.csv", index=False)

    image_channels = CHANNELS_BY_RECIPE[band_recipe]
    param_model = SmallMaskedCNNRegressor(
        SmallMaskedCNNConfig(
            image_channels=image_channels,
            use_mask_channel=cfg.use_mask_channel,
            pooling_mode=cfg.pooling_mode,
            architecture=cfg.architecture,
            dropout=0.2,
        )
    )
    total_params = sum(p.numel() for p in param_model.parameters())
    trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)

    result: dict[str, Any] = {
        "experiment": exp_name,
        "resolution": resolution,
        "band_recipe": band_recipe,
        "architecture": architecture,
        "use_mask_channel": bool(use_mask_channel),
        "patch_size": int(patch_size),
        "folds": len(folds),
        "image_channels": int(image_channels),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    for metric in ["train_MAE", "train_RMSE", "train_R2", "val_MAE", "val_RMSE", "val_R2", "best_epoch"]:
        result[f"{metric}_mean"] = float(fold_df[metric].mean())
        result[f"{metric}_std"] = float(fold_df[metric].std(ddof=0))

    (out_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{exp_name}] val_MAE={result['val_MAE_mean']:.4f}+/-{result['val_MAE_std']:.4f} val_R2={result['val_R2_mean']:.4f}+/-{result['val_R2_std']:.4f}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 5-fold (default) grouped stratified CV for the fixed E1-E4 experiments.")
    p.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    p.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_representation_cv")
    p.add_argument("--shapefile-path", type=Path, default=REPO_ROOT / "data" / "shapefiles" / "2020_SEPIM.shp")
    p.add_argument("--mask-buffer-m", type=float, default=-20.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--augmentation", dest="augmentation", action="store_true")
    p.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    p.set_defaults(augmentation=True)
    p.add_argument("--lr-patch-size", type=int, default=None, help="Skip auto-selection and force this LR patch size.")
    p.add_argument("--lr-patch-candidates", type=int, nargs="+", default=list(DEFAULT_LR_PATCH_CANDIDATES))
    p.add_argument("--patch-containment-threshold", type=float, default=1.0)
    p.add_argument("--patch-select-max-samples", type=int, default=0, help="0 = check all samples.")
    p.add_argument("--sr-patch-size", type=int, default=256, help="Fixed SR patch size in pixels (parcel must fit at SR resolution).")
    p.add_argument("--architecture", choices=["residual", "plain_cnn"], default="residual")
    p.add_argument(
        "--mask-channel-variants", choices=["with", "without", "both"], default="with",
        help="'with' (default, mask channel on), 'without' (mask channel off), or 'both' to run each experiment twice.",
    )
    p.add_argument("--multi-gpu", choices=["auto", "single", "dp"], default="auto")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tcnn.set_seed(args.seed)

    cuda_available = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda_available={cuda_available}")
    if args.require_gpu and not cuda_available:
        raise RuntimeError("GPU is required (--require-gpu) but CUDA is not available.")
    device = torch.device("cuda" if cuda_available else "cpu")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.lr_patch_size is not None:
        lr_patch_size = int(args.lr_patch_size)
        patch_report = {"chosen_lr_patch_size": lr_patch_size, "forced": True}
    else:
        lr_patch_size, patch_report = select_lr_patch_size(
            inventory=args.inventory,
            root_dir=args.root_dir,
            shapefile_path=args.shapefile_path,
            mask_buffer_m=args.mask_buffer_m,
            candidates=tuple(args.lr_patch_candidates),
            containment_threshold=args.patch_containment_threshold,
            max_samples=args.patch_select_max_samples,
        )
    sr_patch_size = int(args.sr_patch_size)
    patch_report["sr_patch_size"] = sr_patch_size
    (out_root / "patch_size_selection.json").write_text(json.dumps(patch_report, indent=2), encoding="utf-8")
    print(f"[patch-select] chosen lr_patch_size={lr_patch_size}px -> fixed sr_patch_size={sr_patch_size}px")

    patch_size_by_resolution = {"lr": lr_patch_size, "sr": sr_patch_size}

    full_df = pd.read_csv(args.inventory)
    folds, split_meta = tcnn.make_grouped_folds(full_df, seed=args.seed, requested_folds=args.folds)
    print(f"[split] group_column={split_meta['group_column']} effective_folds={split_meta['effective_folds']} method={split_meta['method']}")

    cfg_kwargs = dict(
        inventory=str(args.inventory),
        root_dir=str(args.root_dir),
        output_dir=str(out_root),
        shapefile_path=str(args.shapefile_path),
        mask_buffer_m=float(args.mask_buffer_m),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        folds=int(args.folds),
        pooling_mode="masked_avg",
        loss_name="huber",
        augmentation=bool(args.augmentation),
        select_on="val_mae",
        multi_gpu=str(args.multi_gpu),
        log_jsonl=True,
    )

    mask_channel_options = {"with": [True], "without": [False], "both": [True, False]}[args.mask_channel_variants]

    results: list[dict[str, Any]] = []
    for exp in EXPERIMENTS:
        patch_size = patch_size_by_resolution[exp["resolution"]]
        for use_mask_channel in mask_channel_options:
            exp_name = exp["name"] if len(mask_channel_options) == 1 else f"{exp['name']}_{'mask' if use_mask_channel else 'nomask'}"
            result = run_cv_experiment(
                exp_name=exp_name,
                resolution=exp["resolution"],
                band_recipe=exp["band_recipe"],
                patch_size=patch_size,
                folds=folds,
                split_meta=split_meta,
                full_df=full_df,
                cfg_kwargs=cfg_kwargs,
                device=device,
                use_mask_channel=use_mask_channel,
                architecture=str(args.architecture),
            )
            results.append(result)

    comparison = pd.DataFrame(results)
    comparison_path = out_root / "comparison_table.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n{args.folds}-fold CV comparison table (E1-E4):")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
