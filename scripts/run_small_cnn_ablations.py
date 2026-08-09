#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": "exp1_baselines",
            "flags": ["--model-kind", "baseline", "--no-augmentation"],
        },
        {
            "name": "exp2_cnn_global_nomask",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "global_avg", "--no-use-mask-channel", "--no-augmentation"],
        },
        {
            "name": "exp3_cnn_global_mask",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "global_avg", "--use-mask-channel", "--no-augmentation"],
        },
        {
            "name": "exp4_cnn_masked_nomask",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--no-use-mask-channel", "--no-augmentation"],
        },
        {
            "name": "exp5_cnn_masked_mask",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--no-augmentation"],
        },
        {
            "name": "exp6_cnn_masked_mask_aug",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation"],
        },
        {
            "name": "exp7_loss_mse",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation", "--loss", "mse"],
        },
        {
            "name": "exp8_loss_mae",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation", "--loss", "mae"],
        },
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run compact small-CNN ablations with shared split assignments.")
    p.add_argument("--resolution", choices=["lr", "sr"], required=True)
    p.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    p.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "small_masked_cnn")
    p.add_argument("--shapefile-path", type=Path, default=Path("data") / "shapefiles" / "2020_SEPIM.shp")
    p.add_argument("--mask-buffer-m", type=float, default=-20.0)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python", default="python")
    p.add_argument("--base-name", default="small_cnn_ablation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    train_script = repo / "scripts" / "train_small_masked_cnn.py"

    grid = _experiment_grid()
    summary_rows: list[dict[str, Any]] = []

    split_exp_name = f"{args.base_name}_shared_split"
    split_cmd = [
        args.python,
        str(train_script),
        "--experiment-name",
        split_exp_name,
        "--resolution",
        args.resolution,
        "--inventory",
        str(args.inventory),
        "--root-dir",
        str(args.root_dir),
        "--output-dir",
        str(args.output_dir),
        "--shapefile-path",
        str(args.shapefile_path),
        "--mask-buffer-m",
        str(args.mask_buffer_m),
        "--patch-size",
        str(args.patch_size),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-epochs",
        "1",
        "--early-stopping-patience",
        "1",
        "--folds",
        str(args.folds),
        "--seed",
        str(args.seed),
        "--model-kind",
        "baseline",
        "--no-augmentation",
    ]
    _run(split_cmd, cwd=repo)
    shared_split = Path(args.output_dir) / split_exp_name / args.resolution / "split_ids.json"

    for exp in grid:
        exp_name = f"{args.base_name}_{exp['name']}"
        cmd = [
            args.python,
            str(train_script),
            "--experiment-name",
            exp_name,
            "--resolution",
            args.resolution,
            "--inventory",
            str(args.inventory),
            "--root-dir",
            str(args.root_dir),
            "--output-dir",
            str(args.output_dir),
            "--shapefile-path",
            str(args.shapefile_path),
            "--mask-buffer-m",
            str(args.mask_buffer_m),
            "--patch-size",
            str(args.patch_size),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--max-epochs",
            str(args.max_epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
            "--split-ids-path",
            str(shared_split),
        ] + exp["flags"]

        _run(cmd, cwd=repo)

        metrics_path = Path(args.output_dir) / exp_name / args.resolution / "metrics.json"
        metrics = _load_metrics(metrics_path)
        agg = metrics.get("aggregate", {})

        summary_rows.append(
            {
                "experiment_name": exp_name,
                "resolution": args.resolution,
                "val_MAE_mean": agg.get("val_MAE", {}).get("mean"),
                "val_MAE_std": agg.get("val_MAE", {}).get("std"),
                "val_RMSE_mean": agg.get("val_RMSE", {}).get("mean"),
                "val_RMSE_std": agg.get("val_RMSE", {}).get("std"),
                "val_R2_mean": agg.get("val_R2", {}).get("mean"),
                "val_R2_std": agg.get("val_R2", {}).get("std"),
                "val_Spearman_mean": agg.get("val_Spearman", {}).get("mean"),
                "val_Spearman_std": agg.get("val_Spearman", {}).get("std"),
            }
        )

    out_dir = Path(args.output_dir) / f"{args.base_name}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{args.resolution}.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
