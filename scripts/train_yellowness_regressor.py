from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import fit_regressor, make_dataloader, make_group_split


def infer_torchgeo_backbone_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "torchgeo:resnet50"
    if weight_name.startswith("ResNet152_Weights."):
        return "torchgeo:resnet152"
    if weight_name.startswith("ViTSmall16_Weights."):
        return "torchgeo:vit_small_patch16_224"
    if weight_name.startswith("ViTBase16_Weights."):
        return "torchgeo:vit_base_patch16_224"
    if weight_name.startswith("ViTSmall14_DINOv2_Weights."):
        return "torchgeo:vit_small_patch14_dinov2"
    if weight_name.startswith("ViTBase14_DINOv2_Weights."):
        return "torchgeo:vit_base_patch14_dinov2"
    if weight_name.startswith("Swin_V2_T_Weights."):
        return "torchgeo:swin_v2_t"
    if weight_name.startswith("Swin_V2_B_Weights."):
        return "torchgeo:swin_v2_b"
    if weight_name.startswith("FGMAEEarthLoc_Weights.") and "RESNET50" in weight_name.upper():
        return "torchgeo:resnet50"
    raise ValueError("Could not infer a TorchGeo backbone from --torchgeo-weight.")


def resolve_backbone(backbone: str, torchgeo_weight: Optional[str]) -> str:
    if backbone != "auto":
        return backbone
    if torchgeo_weight is None:
        return "simple_cnn"
    return infer_torchgeo_backbone_from_weight(torchgeo_weight)


def resolve_backbone_band_indices(weight_name: Optional[str]) -> Optional[list[int]]:
    if not weight_name:
        return None

    satlas_9band_prefixes = (
        "ResNet50_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet152_Weights.SENTINEL2_SI_MS_SATLAS",
        
        "Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
        
        "Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS",
        
    )
    if weight_name.startswith(satlas_9band_prefixes):
        return None
    return None


def resolve_band_recipe(weight_name: Optional[str]) -> Optional[str]:
    if not weight_name:
        return None

    satlas_9band_prefixes = (
        "ResNet50_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet152_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS",
        
    )
    if weight_name.startswith(satlas_9band_prefixes):
        return "satlas_l1c_9_from_l2a"

    if weight_name.startswith("SENTINEL2_ALL_NDVI_SECO_ECO"):
        return "ndvi_seco_eco_9"

    return None


def _center_crop_last2d(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    h, w = x.shape[-2], x.shape[-1]
    if crop_size is None or crop_size <= 0 or (crop_size == h and crop_size == w):
        return x
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} is larger than sample size {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return x[..., top : top + crop_size, left : left + crop_size]


def _rotate_last2d(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.ndim < 2 or k % 4 == 0:
        return x
    return torch.rot90(x, k=k % 4, dims=(-2, -1))


def _transform_sample(obj: Any, crop_size: int, rot_k: int) -> Any:
    if torch.is_tensor(obj):
        y = _center_crop_last2d(obj, crop_size)
        y = _rotate_last2d(y, rot_k)
        return y
    if isinstance(obj, dict):
        return {k: _transform_sample(v, crop_size, rot_k) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_transform_sample(v, crop_size, rot_k) for v in obj)
    if isinstance(obj, list):
        return [_transform_sample(v, crop_size, rot_k) for v in obj]
    return obj


class CenterCropRotateDataset(Dataset):
    """
    - Center-crops all tensor-like fields on spatial dims (last two dims).
    - Optionally expands each sample with 4 rotations (0/90/180/270).
    """
    def __init__(self, base: Dataset, crop_size: int, enable_rotations: bool) -> None:
        self.base = base
        self.crop_size = crop_size
        self.enable_rotations = enable_rotations
        self.multiplier = 4 if enable_rotations else 1

    def __len__(self) -> int:
        return len(self.base) * self.multiplier

    def __getitem__(self, idx: int) -> Any:
        base_idx = idx % len(self.base)
        rot_k = (idx // len(self.base)) if self.enable_rotations else 0
        sample = self.base[base_idx]
        return _transform_sample(sample, self.crop_size, rot_k)


def _rebuild_loader(
    base_loader: DataLoader,
    dataset: Dataset,
    *,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=getattr(base_loader, "pin_memory", False),
        drop_last=getattr(base_loader, "drop_last", False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a patch+mask yellowness regressor.")
    parser.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory_all_feature_new.csv")
    parser.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    parser.add_argument("--resolution", choices=("lr", "sr"), default="sr")
    parser.add_argument("--backbone", default="auto")
    parser.add_argument("--torchgeo-weight", default=None)
    parser.add_argument("--mask-fusion", default="mask_as_channel")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=0)
    parser.add_argument("--plateau-unfreeze-patience", type=int, default=4)
    parser.add_argument("--plateau-unfreeze-min-delta", type=float, default=0.0)
    parser.add_argument("--unfreeze-last-stages", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-patch-size", type=int, default=256)
    parser.add_argument("--shapefile-path", type=Path, default=None)
    parser.add_argument("--mask-buffer-m", type=float, default=-20.0)

    # New: center crop + rotation augmentation
    parser.add_argument("--center-crop-size", type=int, default=224)
    parser.add_argument("--rotate-augment", dest="rotate_augment", action="store_true")
    parser.add_argument("--no-rotate-augment", dest="rotate_augment", action="store_false")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "yellowness_regression")

    # NEW
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-early-stopping", dest="use_early_stopping", action="store_false")
    parser.add_argument("--data-parallel", action="store_true")

    parser.set_defaults(freeze_backbone=True, rotate_augment=True, use_early_stopping=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = resolve_backbone(args.backbone, args.torchgeo_weight)
    backbone_band_indices = resolve_backbone_band_indices(args.torchgeo_weight)
    band_recipe = resolve_band_recipe(args.torchgeo_weight)
    if band_recipe in {"satlas_l1c_9_from_l2a", "ndvi_seco_eco_9"}:
        backbone_image_channels = 9
    else:
        backbone_image_channels = len(backbone_band_indices) if backbone_band_indices is not None else 10

    split = make_group_split(
        inventory_csv=args.inventory,
        root_dir=args.root_dir,
        resolution=args.resolution,
        test_size=args.test_size,
        random_state=args.random_state,
        band_indices=backbone_band_indices,
        band_recipe=band_recipe,
        shapefile_path=args.shapefile_path,
        shapefile_buffer_m=args.mask_buffer_m,
    )

    # Build base loaders first
    train_base_loader = make_dataloader(split.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_base_loader = make_dataloader(split.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Wrap datasets: train = crop + rotations, val = crop only
    train_dataset = CenterCropRotateDataset(
        train_base_loader.dataset,
        crop_size=args.center_crop_size,
        enable_rotations=args.rotate_augment,
    )
    val_dataset = CenterCropRotateDataset(
        val_base_loader.dataset,
        crop_size=args.center_crop_size,
        enable_rotations=False,
    )

    train_loader = _rebuild_loader(
        train_base_loader,
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = _rebuild_loader(
        val_base_loader,
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_yellowness_model(
        backbone_name=backbone_name,
        mask_fusion=args.mask_fusion,
        torchgeo_weight=args.torchgeo_weight,
        freeze_backbone=args.freeze_backbone,
        image_channels=backbone_image_channels,
        sample_patch_size=args.center_crop_size,  # use cropped spatial size
        backbone_band_indices=backbone_band_indices,
    )

    # NEW
    use_data_parallel = args.data_parallel or torch.cuda.device_count() > 1
    if use_data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    result = fit_regressor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch if args.unfreeze_epoch > 0 else None,
        backbone_learning_rate=args.backbone_learning_rate,
        early_stopping_patience=(args.early_stopping_patience if args.use_early_stopping else None),
        early_stopping_min_delta=args.early_stopping_min_delta,
        plateau_unfreeze_patience=args.plateau_unfreeze_patience,
        plateau_unfreeze_min_delta=args.plateau_unfreeze_min_delta,
        unfreeze_last_stages=args.unfreeze_last_stages,
        verbose=True,
    )

    checkpoint_path = args.output_dir / "best_model.pt"
    # NEW: save wrapped model correctly
    model_to_save = result["model"].module if isinstance(result["model"], torch.nn.DataParallel) else result["model"]
    torch.save(model_to_save.state_dict(), checkpoint_path)
    history_path = args.output_dir / "training_history.json"
    run_config_path = args.output_dir / "run_config.json"
    history_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(
            {
                "inventory": str(args.inventory),
                "root_dir": str(args.root_dir),
                "resolution": args.resolution,
                "backbone": backbone_name,
                "torchgeo_weight": args.torchgeo_weight,
                "band_recipe": band_recipe,
                "mask_fusion": args.mask_fusion,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "backbone_learning_rate": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
                "freeze_backbone": args.freeze_backbone,
                "unfreeze_epoch": args.unfreeze_epoch,
                "plateau_unfreeze_patience": args.plateau_unfreeze_patience,
                "plateau_unfreeze_min_delta": args.plateau_unfreeze_min_delta,
                "unfreeze_last_stages": args.unfreeze_last_stages,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "sample_patch_size": args.sample_patch_size,
                "center_crop_size": args.center_crop_size,
                "rotate_augment": args.rotate_augment,
                "early_stopping_patience": args.early_stopping_patience if args.use_early_stopping else None,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "best_metrics": result["best_metrics"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved run config: {run_config_path}")
    print("Best validation metrics:")
    print(json.dumps(result["best_metrics"], indent=2))


if __name__ == "__main__":
    main()