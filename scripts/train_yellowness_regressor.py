#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import fit_regressor, make_dataloader, make_group_split


def infer_torchgeo_backbone_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "torchgeo:resnet50"
    if weight_name.startswith("ViTSmall16_Weights."):
        return "torchgeo:vit_small_patch16_224"
    if weight_name.startswith("ViTBase16_Weights."):
        return "torchgeo:vit_base_patch16_224"
    if weight_name.startswith("FGMAEEarthLoc_Weights.") and "RESNET50" in weight_name.upper():
        return "torchgeo:resnet50"
    raise ValueError("Could not infer a TorchGeo backbone from --torchgeo-weight.")


def resolve_backbone(backbone: str, torchgeo_weight: Optional[str]) -> str:
    if backbone != "auto":
        return backbone
    if torchgeo_weight is None:
        return "simple_cnn"
    return infer_torchgeo_backbone_from_weight(torchgeo_weight)


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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-patch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "yellowness_regression")
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = resolve_backbone(args.backbone, args.torchgeo_weight)

    split = make_group_split(
        inventory_csv=args.inventory,
        root_dir=args.root_dir,
        resolution=args.resolution,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_loader = make_dataloader(split.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(split.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_yellowness_model(
        backbone_name=backbone_name,
        mask_fusion=args.mask_fusion,
        torchgeo_weight=args.torchgeo_weight,
        freeze_backbone=args.freeze_backbone,
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

    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "training_history.json"
    run_config_path = args.output_dir / "run_config.json"
    torch.save(result["model"].state_dict(), checkpoint_path)
    history_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    run_config_path.write_text(
        json.dumps(
            {
                "inventory": str(args.inventory),
                "root_dir": str(args.root_dir),
                "resolution": args.resolution,
                "backbone": backbone_name,
                "torchgeo_weight": args.torchgeo_weight,
                "mask_fusion": args.mask_fusion,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "backbone_learning_rate": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
                "freeze_backbone": args.freeze_backbone,
                "unfreeze_epoch": args.unfreeze_epoch,
                "test_size": args.test_size,
                "random_state": args.random_state,
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