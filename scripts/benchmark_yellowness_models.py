#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import fit_regressor, make_dataloader, make_group_split


def infer_torchgeo_backbone_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "torchgeo:resnet50"
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
    if weight_name.startswith("FGMAEEarthLoc_Weights.") and "RESNET50" in weight_name.upper():
        return "torchgeo:resnet50"
    raise ValueError("Could not infer a TorchGeo backbone from --torchgeo-weights.")


def resolve_backbone(backbone: str, torchgeo_weight: Optional[str]) -> str:
    if backbone != "auto":
        return backbone
    if torchgeo_weight is None:
        return "simple_cnn"
    return infer_torchgeo_backbone_from_weight(torchgeo_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple yellowness regression backbones and mask fusion modes.")
    parser.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory_all_feature_new.csv")
    parser.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    parser.add_argument("--resolution", choices=("lr", "sr"), default="sr")
    parser.add_argument("--backbones", nargs="+", default=["auto"])
    parser.add_argument("--torchgeo-weights", nargs="*", default=[])
    parser.add_argument(
        "--mask-fusions",
        nargs="+",
        default=["image_only", "masked_image", "mask_as_channel", "image_and_mask", "late_mask"],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze-epoch", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-patch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "yellowness_benchmark.csv")
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split = make_group_split(
        args.inventory,
        root_dir=args.root_dir,
        resolution=args.resolution,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_loader = make_dataloader(split.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(split.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weighted_runs = list(args.torchgeo_weights) if args.torchgeo_weights else [None]
    rows = []
    for backbone in args.backbones:
        for torchgeo_weight in weighted_runs:
            resolved_backbone = resolve_backbone(backbone, torchgeo_weight)
            for mask_fusion in args.mask_fusions:
                model = build_yellowness_model(
                    backbone_name=resolved_backbone,
                    torchgeo_weight=torchgeo_weight,
                    freeze_backbone=args.freeze_backbone,
                    mask_fusion=mask_fusion,
                    image_channels=10,
                    sample_patch_size=args.sample_patch_size,
                )
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
                )
                rows.append(
                    {
                        "backbone": resolved_backbone,
                        "torchgeo_weight": torchgeo_weight or "none",
                        "mask_fusion": mask_fusion,
                        **result["best_metrics"],
                    }
                )
                print(json.dumps(rows[-1], indent=2))

    benchmark_df = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    benchmark_df.to_csv(args.output, index=False)
    print(f"Saved benchmark table: {args.output}")
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    main()