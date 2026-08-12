#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _export_experiment_fold_evolution(exp_dir: Path, exp_name: str, resolution: str) -> dict[str, Any]:
    """Collect per-fold epoch histories into one CSV for downstream plotting."""
    fold_dirs = sorted(path for path in exp_dir.glob("fold_*") if path.is_dir())
    rows: list[pd.DataFrame] = []

    for fold_dir in fold_dirs:
        fold_label = fold_dir.name
        history_path = fold_dir / "history.csv"
        if history_path.exists():
            hist = pd.read_csv(history_path)
            hist.insert(0, "fold", fold_label)
            hist.insert(0, "resolution", resolution)
            hist.insert(0, "experiment_name", exp_name)
            rows.append(hist)

    if not rows:
        # Baseline-only runs do not have epoch history; return lightweight metadata.
        return {
            "history_rows": 0,
            "history_path": None,
        }

    merged = pd.concat(rows, ignore_index=True)
    out_path = exp_dir / "history_all_folds.csv"
    merged.to_csv(out_path, index=False)
    return {
        "history_rows": int(len(merged)),
        "history_path": str(out_path),
    }


def _experiment_grid(resolution: str) -> list[dict[str, Any]]:
    common = [
        # {
        #     "name": "exp1_baseline_masked_band_means",
        #     "flags": ["--model-kind", "baseline", "--no-augmentation"],
        # },
        {
            "name": "exp5_cnn_masked_mask_noaug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--no-augmentation", "--loss", "huber"],
        },
        {
            "name": "exp6_cnn_masked_mask_aug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation", "--loss", "huber"],
        },
        {
            "name": "exp8_cnn_masked_mask_aug_mse",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation", "--loss", "mse"],
        },
        {
            "name": "exp9_cnn_masked_mask_aug_mae",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--use-mask-channel", "--augmentation", "--loss", "mae"],
        },
    ]

    if resolution == "sr":
        # SR is the expensive pass. Keep only the strongest masked-pooling variants from LR
        # and drop weak global/no-mask configs to get faster results.
        return common

    return [
        *common,
        # Lower-value experiments retained for LR exploration only.
        # Commented as weak on the latest ranking, so they are excluded from SR.
        {
            "name": "exp2_cnn_global_nomask_noaug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "global_avg", "--no-use-mask-channel", "--no-augmentation", "--loss", "huber"],
        },
        {
            "name": "exp3_cnn_global_mask_noaug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "global_avg", "--use-mask-channel", "--no-augmentation", "--loss", "huber"],
        },
        {
            "name": "exp4_cnn_masked_nomask_noaug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "masked_avg", "--no-use-mask-channel", "--no-augmentation", "--loss", "huber"],
        },
        {
            "name": "exp7_cnn_global_mask_aug_huber",
            "flags": ["--model-kind", "cnn", "--pooling-mode", "global_avg", "--use-mask-channel", "--augmentation", "--loss", "huber"],
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
    p.add_argument("--multi-gpu", choices=["auto", "single", "dp"], default="dp")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--require-two-gpus", action="store_true")
    p.add_argument("--fast", action="store_true", help="Use faster settings (fewer folds/epochs, larger batch size).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    train_script = repo / "scripts" / "train_small_masked_cnn.py"

    effective_folds = args.folds
    effective_max_epochs = args.max_epochs
    effective_patience = args.early_stopping_patience
    effective_batch_size = args.batch_size
    if args.fast:
        effective_folds = min(args.folds, 3)
        effective_max_epochs = min(args.max_epochs, 80)
        effective_patience = min(args.early_stopping_patience, 15)
        effective_batch_size = max(args.batch_size, 16)
        print(
            f"[fast-mode] folds={effective_folds} max_epochs={effective_max_epochs} "
            f"patience={effective_patience} batch_size={effective_batch_size}"
        )

    grid = _experiment_grid(args.resolution)
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
        str(effective_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-epochs",
        "1",
        "--early-stopping-patience",
        "1",
        "--folds",
        str(effective_folds),
        "--seed",
        str(args.seed),
        "--model-kind",
        "baseline",
        "--no-augmentation",
        "--multi-gpu",
        str(args.multi_gpu),
    ]
    if args.require_gpu:
        split_cmd.append("--require-gpu")
    if args.require_two_gpus:
        split_cmd.append("--require-two-gpus")
    _run(split_cmd, cwd=repo)
    shared_split = Path(args.output_dir) / split_exp_name / args.resolution / "split_ids.json"

    for exp in grid:
        exp_name = f"{args.base_name}_{exp['name']}"
        exp_dir = Path(args.output_dir) / exp_name / args.resolution
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
            str(effective_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--max-epochs",
            str(effective_max_epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--early-stopping-patience",
            str(effective_patience),
            "--folds",
            str(effective_folds),
            "--seed",
            str(args.seed),
            "--split-ids-path",
            str(shared_split),
            "--multi-gpu",
            str(args.multi_gpu),
        ] + exp["flags"]
        if args.require_gpu:
            cmd.append("--require-gpu")
        if args.require_two_gpus:
            cmd.append("--require-two-gpus")

        _run(cmd, cwd=repo)

        history_meta = _export_experiment_fold_evolution(exp_dir=exp_dir, exp_name=exp_name, resolution=args.resolution)

        metrics_path = exp_dir / "metrics.json"
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
                "history_rows": history_meta["history_rows"],
                "history_path": history_meta["history_path"],
            }
        )

    out_dir = Path(args.output_dir) / f"{args.base_name}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{args.resolution}.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    # Also save a single CSV with all experiment fold-evolution rows for plotting later.
    combined_rows: list[pd.DataFrame] = []
    for row in summary_rows:
        history_path = row.get("history_path")
        if not history_path:
            continue
        path = Path(history_path)
        if path.exists():
            combined_rows.append(pd.read_csv(path))
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(out_dir / f"history_all_experiments_{args.resolution}.csv", index=False)

    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
