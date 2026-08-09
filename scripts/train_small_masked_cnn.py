#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline.small_masked_cnn import SmallMaskedCNNConfig, SmallMaskedCNNRegressor
from src.s2_pipeline.yellowness_training import ParcelYellownessDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_group_col(df: pd.DataFrame) -> str:
    for c in ("id_plot", "ID_PARCEL", "parcel_id", "PARCEL_ID", "parcel_index"):
        if c in df.columns:
            return c
    raise ValueError("No group column found. Expected one of id_plot/ID_PARCEL/parcel_id/PARCEL_ID/parcel_index.")


def make_group_bins(df: pd.DataFrame, group_col: str, n_bins: int = 5) -> dict[str, int]:
    agg = df.groupby(group_col, dropna=False)["yellowness"].median().reset_index(name="group_y")
    q = int(min(n_bins, max(2, agg["group_y"].nunique())))
    if q < 2:
        agg["bin"] = 0
    else:
        agg["bin"] = pd.qcut(agg["group_y"].rank(method="first"), q=q, labels=False, duplicates="drop")
        agg["bin"] = agg["bin"].fillna(0).astype(int)
    return {str(k): int(v) for k, v in zip(agg[group_col].astype(str), agg["bin"])}


def make_grouped_folds(df: pd.DataFrame, seed: int, requested_folds: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    group_col = resolve_group_col(df)
    bins_map = make_group_bins(df, group_col, n_bins=5)

    groups = df[group_col].astype(str)
    y_bins = groups.map(bins_map).fillna(0).astype(int)

    unique_groups = groups.drop_duplicates().to_numpy()
    group_bins = pd.Series(unique_groups).map(bins_map).fillna(0).astype(int)
    max_folds = int(group_bins.value_counts().min()) if not group_bins.empty else 0
    n_splits = min(requested_folds, max_folds)

    split_meta: dict[str, Any] = {
        "group_column": group_col,
        "requested_folds": int(requested_folds),
        "effective_folds": int(max(n_splits, 0)),
        "strat_bins": {k: int(v) for k, v in bins_map.items()},
        "method": "",
    }

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if n_splits >= 2:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr, va in splitter.split(df, y=y_bins, groups=groups):
                folds.append((tr, va))
            split_meta["method"] = "StratifiedGroupKFold"
        except Exception:
            folds = []

    if not folds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, va = next(splitter.split(df, groups=groups))
        folds = [(tr, va)]
        split_meta["effective_folds"] = 1
        split_meta["method"] = "GroupShuffleSplit"

    split_meta["folds"] = []
    for idx, (tr, va) in enumerate(folds, start=1):
        tr_ids = sorted(df.iloc[tr][group_col].astype(str).unique().tolist())
        va_ids = sorted(df.iloc[va][group_col].astype(str).unique().tolist())
        tr_bins = pd.Series(tr_ids).map(bins_map).value_counts().sort_index().to_dict()
        va_bins = pd.Series(va_ids).map(bins_map).value_counts().sort_index().to_dict()
        split_meta["folds"].append(
            {
                "fold": idx,
                "train_ids": tr_ids,
                "val_ids": va_ids,
                "train_bin_distribution": {str(k): int(v) for k, v in tr_bins.items()},
                "val_bin_distribution": {str(k): int(v) for k, v in va_bins.items()},
            }
        )

    return folds, split_meta


def load_folds_from_split_ids(df: pd.DataFrame, split_ids_path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    data = json.loads(split_ids_path.read_text(encoding="utf-8"))
    group_col = data.get("group_column")
    if not group_col or group_col not in df.columns:
        raise ValueError(f"Invalid split_ids.json: group_column={group_col!r} missing from inventory.")

    group_values = df[group_col].astype(str)
    folds_payload = data.get("folds", [])
    if not folds_payload:
        raise ValueError("Invalid split_ids.json: no folds found.")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in folds_payload:
        tr_ids = set(str(x) for x in f.get("train_ids", []))
        va_ids = set(str(x) for x in f.get("val_ids", []))
        tr_idx = np.where(group_values.isin(tr_ids).to_numpy())[0]
        va_idx = np.where(group_values.isin(va_ids).to_numpy())[0]
        if tr_idx.size == 0 or va_idx.size == 0:
            raise ValueError("Invalid split_ids.json: empty train/val index set for one fold.")
        folds.append((tr_idx, va_idx))

    data["method"] = f"{data.get('method', 'unknown')}+reused"
    data["source_split_ids"] = str(split_ids_path)
    return folds, data


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if size is None or size <= 0:
        return x
    if h < size or w < size:
        raise ValueError(f"Requested patch size {size} but got sample size {(h, w)}")
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top : top + size, left : left + size]


def crop_image_and_mask(image: torch.Tensor, mask: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[-2], image.shape[-1]
    if size is None or size <= 0:
        return image, mask
    if h < size or w < size:
        raise ValueError(f"Requested patch size {size} but got sample size {(h, w)}")

    nonzero = torch.nonzero(mask[0] > 0, as_tuple=False)
    if nonzero.numel() > 0:
        center_y = int(nonzero[:, 0].float().mean().round().item())
        center_x = int(nonzero[:, 1].float().mean().round().item())
        top = max(0, min(center_y - size // 2, h - size))
        left = max(0, min(center_x - size // 2, w - size))
    else:
        top = (h - size) // 2
        left = (w - size) // 2

    return (
        image[..., top : top + size, left : left + size],
        mask[..., top : top + size, left : left + size],
    )


def apply_train_aug(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        image = torch.rot90(image, k=k, dims=(-2, -1))
        mask = torch.rot90(mask, k=k, dims=(-2, -1))
    return image, mask


class FoldDataset(Dataset):
    def __init__(
        self,
        base_dataset: ParcelYellownessDataset,
        mean: torch.Tensor,
        std: torch.Tensor,
        patch_size: int,
        augment: bool,
    ) -> None:
        self.base = base_dataset
        self.mean = mean
        self.std = std
        self.patch_size = int(patch_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        image = sample["image"].float()
        mask = sample["mask"].float()

        image, mask = crop_image_and_mask(image, mask, self.patch_size)

        if mask.sum().item() <= 0:
            raise ValueError(
                f"Empty mask found at row_index={sample.get('row_index')} id_plot={sample.get('id_plot')}"
            )

        if self.augment:
            image, mask = apply_train_aug(image, mask)

        image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return {
            "image": image,
            "mask": mask,
            "target": sample["target"].float(),
            "id_plot": str(sample["id_plot"]),
            "row_index": int(sample["row_index"]),
        }


def compute_train_band_stats(dataset: ParcelYellownessDataset, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    sum_c: Optional[torch.Tensor] = None
    sum_sq_c: Optional[torch.Tensor] = None
    count = 0

    for i in range(len(dataset)):
        s = dataset[i]
        img = center_crop(s["image"].float(), patch_size)
        if sum_c is None:
            c = img.shape[0]
            sum_c = torch.zeros(c, dtype=torch.float64)
            sum_sq_c = torch.zeros(c, dtype=torch.float64)
        flat = img.reshape(img.shape[0], -1).double()
        sum_c += flat.sum(dim=1)
        sum_sq_c += (flat * flat).sum(dim=1)
        count += flat.shape[1]

    if sum_c is None or sum_sq_c is None or count == 0:
        raise RuntimeError("Failed to compute train-band normalization stats.")

    mean = (sum_c / count).float()
    var = (sum_sq_c / count - mean.double() * mean.double()).clamp_min(1e-8).float()
    std = torch.sqrt(var).clamp_min(1e-4)
    return mean, std


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    try:
        from scipy.stats import spearmanr

        out["Spearman"] = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        out["Spearman"] = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    if np.isnan(out["Spearman"]):
        out["Spearman"] = 0.0
    return out


def get_loss(name: str) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss={name}")


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], loss_fn: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    losses: list[float] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for batch in loader:
        img = batch["image"].to(device)
        msk = batch["mask"].to(device)
        y = batch["target"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(img, msk)
            loss = loss_fn(pred, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list).reshape(-1)
    y_pred = np.concatenate(y_pred_list).reshape(-1)
    return float(np.mean(losses)), y_true, y_pred


def predict_full(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            pred = model(img, msk).cpu().numpy().reshape(-1)
            y = batch["target"].cpu().numpy().reshape(-1)
            ids = batch["id_plot"]
            row_idx = batch["row_index"].cpu().numpy().reshape(-1)
            for i in range(len(pred)):
                rows.append(
                    {
                        "id_plot": str(ids[i]),
                        "row_index": int(row_idx[i]),
                        "y_true": float(y[i]),
                        "y_pred": float(pred[i]),
                    }
                )
    return pd.DataFrame(rows)


def run_baselines(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    def to_parcel_avg(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        feats: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        ids_all: list[str] = []
        rows_all: list[int] = []
        for batch in loader:
            img = batch["image"].numpy()
            msk = batch["mask"].numpy()
            y = batch["target"].numpy()
            ids = batch["id_plot"]
            rows = batch["row_index"].numpy()

            for i in range(img.shape[0]):
                m = (msk[i, 0] > 0).astype(np.float32)
                denom = float(m.sum())
                if denom <= 0:
                    continue
                feat = (img[i] * m[None, ...]).reshape(img.shape[1], -1).sum(axis=1) / denom
                feats.append(feat.astype(np.float32))
                targets.append(np.array([y[i]], dtype=np.float32))
                ids_all.append(str(ids[i]))
                rows_all.append(int(rows[i]))
        return np.vstack(feats), np.concatenate(targets), ids_all, rows_all

    Xtr, ytr, idtr, rowtr = to_parcel_avg(train_loader)
    Xva, yva, idva, rowva = to_parcel_avg(val_loader)

    models: dict[str, Any] = {
        "linear_regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        out_dir = fold_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        model.fit(Xtr, ytr)
        p_tr = model.predict(Xtr).reshape(-1)
        p_va = model.predict(Xva).reshape(-1)

        tr_m = metric_dict(ytr, p_tr)
        va_m = metric_dict(yva, p_va)
        rows.append(
            {
                "model": name,
                "train_MAE": tr_m["MAE"],
                "val_MAE": va_m["MAE"],
                "train_RMSE": tr_m["RMSE"],
                "val_RMSE": va_m["RMSE"],
                "train_R2": tr_m["R2"],
                "val_R2": va_m["R2"],
                "train_Spearman": tr_m["Spearman"],
                "val_Spearman": va_m["Spearman"],
            }
        )

        pd.DataFrame({"id_plot": idtr, "row_index": rowtr, "y_true": ytr, "y_pred": p_tr}).to_csv(out_dir / "predictions_train.csv", index=False)
        pd.DataFrame({"id_plot": idva, "row_index": rowva, "y_true": yva, "y_pred": p_va}).to_csv(out_dir / "predictions_val.csv", index=False)

    pd.DataFrame(rows).to_csv(fold_dir / "metrics_baselines.csv", index=False)
    return rows


@dataclass
class TrainConfig:
    experiment_name: str
    resolution: str
    inventory: str
    root_dir: str
    output_dir: str
    shapefile_path: str
    mask_buffer_m: float
    patch_size: int
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    seed: int
    folds: int
    model_kind: str
    use_mask_channel: bool
    pooling_mode: str
    loss_name: str
    augmentation: bool
    select_on: str
    multi_gpu: str
    log_jsonl: bool


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_one_fold(
    fold_idx: int,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    fold_dir = Path(cfg.output_dir) / cfg.experiment_name / cfg.resolution / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_raw = ParcelYellownessDataset(
        cfg.inventory,
        root_dir=cfg.root_dir,
        resolution=cfg.resolution,
        rows=train_rows,
        band_indices=None,
        band_recipe=None,
        shapefile_path=cfg.shapefile_path,
        shapefile_buffer_m=cfg.mask_buffer_m,
    )
    val_raw = ParcelYellownessDataset(
        cfg.inventory,
        root_dir=cfg.root_dir,
        resolution=cfg.resolution,
        rows=val_rows,
        band_indices=None,
        band_recipe=None,
        shapefile_path=cfg.shapefile_path,
        shapefile_buffer_m=cfg.mask_buffer_m,
    )

    mean, std = compute_train_band_stats(train_raw, patch_size=cfg.patch_size)

    train_ds = FoldDataset(train_raw, mean=mean, std=std, patch_size=cfg.patch_size, augment=cfg.augmentation)
    val_ds = FoldDataset(val_raw, mean=mean, std=std, patch_size=cfg.patch_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if cfg.model_kind == "baseline":
        baseline_rows = run_baselines(train_loader, val_loader, fold_dir=fold_dir, seed=cfg.seed + fold_idx)
        summary = pd.DataFrame(baseline_rows)
        best_row = summary.sort_values("val_MAE").iloc[0].to_dict()
        return {
            "fold": fold_idx,
            "best": best_row,
            "type": "baseline",
        }

    base_model = SmallMaskedCNNRegressor(
        SmallMaskedCNNConfig(
            image_channels=10,
            use_mask_channel=cfg.use_mask_channel,
            pooling_mode=cfg.pooling_mode,
            dropout=0.2,
        )
    ).to(device)

    use_data_parallel = (
        cfg.multi_gpu in {"auto", "dp"}
        and device.type == "cuda"
        and torch.cuda.device_count() >= 2
    )
    model: nn.Module = nn.DataParallel(base_model) if use_data_parallel else base_model
    if use_data_parallel:
        print(f"[multi-gpu] Using DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        print(f"[multi-gpu] Running on device={device}")

    loss_fn = get_loss(cfg.loss_name)
    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    history: list[dict[str, Any]] = []
    best_value = float("inf")
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    stale_epochs = 0
    history_jsonl_path = fold_dir / "history.jsonl"
    if history_jsonl_path.exists():
        history_jsonl_path.unlink()

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, optimizer=opt, loss_fn=loss_fn, device=device)
        va_loss, va_y, va_p = run_epoch(model, val_loader, optimizer=None, loss_fn=loss_fn, device=device)

        tr_m = metric_dict(tr_y, tr_p)
        va_m = metric_dict(va_y, va_p)
        lr_now = float(opt.param_groups[0]["lr"])
        duration = float(time.perf_counter() - t0)

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
            "train_Spearman": tr_m["Spearman"],
            "validation_Spearman": va_m["Spearman"],
            "lr": lr_now,
            "epoch_duration_sec": duration,
        }
        history.append(row)
        if cfg.log_jsonl:
            with history_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        print(
            f"[fold {fold_idx:02d}] epoch={epoch:03d} "
            f"train_loss={tr_loss:.5f} val_loss={va_loss:.5f} "
            f"train_mae={tr_m['MAE']:.5f} val_mae={va_m['MAE']:.5f} "
            f"train_r2={tr_m['R2']:.4f} val_r2={va_m['R2']:.4f} lr={lr_now:.2e}"
        )

        monitor = va_m["MAE"] if cfg.select_on == "val_mae" else va_loss
        scheduler.step(va_loss)
        if monitor < best_value:
            best_value = float(monitor)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                "model": _unwrap_model(model).state_dict(),
                "epoch": epoch,
                "monitor": best_value,
                "config": asdict(cfg),
                "mean": mean.cpu(),
                "std": std.cpu(),
                "multi_gpu_used": bool(use_data_parallel),
                "gpu_count": int(torch.cuda.device_count() if device.type == "cuda" else 0),
            }
        else:
            stale_epochs += 1

        if stale_epochs >= cfg.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    torch.save(best_state, fold_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    _unwrap_model(model).load_state_dict(best_state["model"])
    pred_tr = predict_full(model, train_loader, device)
    pred_va = predict_full(model, val_loader, device)
    pred_tr.to_csv(fold_dir / "predictions_train.csv", index=False)
    pred_va.to_csv(fold_dir / "predictions_val.csv", index=False)

    tr_best = metric_dict(pred_tr["y_true"].to_numpy(), pred_tr["y_pred"].to_numpy())
    va_best = metric_dict(pred_va["y_true"].to_numpy(), pred_va["y_pred"].to_numpy())

    metrics = {
        "best_epoch": int(best_epoch),
        "selection_metric": cfg.select_on,
        "best_monitor_value": float(best_value),
        "train": tr_best,
        "validation": va_best,
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        hist_df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
        axes[0].plot(hist_df["epoch"], hist_df["validation_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(hist_df["epoch"], hist_df["train_MAE"], label="train")
        axes[1].plot(hist_df["epoch"], hist_df["validation_MAE"], label="val")
        axes[1].set_title("MAE")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(fold_dir / "plots_loss_mae.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    return {
        "fold": fold_idx,
        "best": {
            "val_MAE": float(va_best["MAE"]),
            "val_RMSE": float(va_best["RMSE"]),
            "val_R2": float(va_best["R2"]),
            "val_Spearman": float(va_best["Spearman"]),
            "train_MAE": float(tr_best["MAE"]),
            "train_RMSE": float(tr_best["RMSE"]),
            "train_R2": float(tr_best["R2"]),
            "train_Spearman": float(tr_best["Spearman"]),
        },
        "type": "cnn",
    }


def aggregate_fold_metrics(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat: list[dict[str, Any]] = []
    for row in fold_rows:
        if row["type"] == "baseline":
            b = row["best"]
            flat.append(
                {
                    "fold": row["fold"],
                    "val_MAE": float(b.get("val_MAE", np.nan)),
                    "val_RMSE": float(b.get("val_RMSE", np.nan)),
                    "val_R2": float(b.get("val_R2", np.nan)),
                    "val_Spearman": float(b.get("val_Spearman", np.nan)),
                }
            )
        else:
            b = row["best"]
            flat.append(
                {
                    "fold": row["fold"],
                    "val_MAE": float(b["val_MAE"]),
                    "val_RMSE": float(b["val_RMSE"]),
                    "val_R2": float(b["val_R2"]),
                    "val_Spearman": float(b["val_Spearman"]),
                }
            )

    df = pd.DataFrame(flat)
    agg = {}
    for col in ["val_MAE", "val_RMSE", "val_R2", "val_Spearman"]:
        agg[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=0)),
        }
    return {"fold_metrics": flat, "aggregate": agg}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate compact masked CNN regressors with grouped-stratified folds.")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--resolution", choices=["lr", "sr"], required=True)
    p.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    p.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "small_masked_cnn")
    p.add_argument("--shapefile-path", type=Path, default=REPO_ROOT / "data" / "shapefiles" / "2020_SEPIM.shp")
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
    p.add_argument("--split-ids-path", type=Path, default=None)

    p.add_argument("--model-kind", choices=["baseline", "cnn"], default="cnn")
    p.add_argument("--use-mask-channel", dest="use_mask_channel", action="store_true")
    p.add_argument("--no-use-mask-channel", dest="use_mask_channel", action="store_false")
    p.add_argument("--pooling-mode", choices=["global_avg", "masked_avg"], default="masked_avg")
    p.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    p.add_argument("--augmentation", dest="augmentation", action="store_true")
    p.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    p.add_argument("--select-on", choices=["val_mae", "val_loss"], default="val_mae")
    p.add_argument("--multi-gpu", choices=["auto", "single", "dp"], default="auto")
    p.add_argument("--log-jsonl", dest="log_jsonl", action="store_true")
    p.add_argument("--no-log-jsonl", dest="log_jsonl", action="store_false")

    p.set_defaults(use_mask_channel=True, augmentation=True, log_jsonl=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = TrainConfig(
        experiment_name=args.experiment_name,
        resolution=args.resolution,
        inventory=str(args.inventory),
        root_dir=str(args.root_dir),
        output_dir=str(args.output_dir),
        shapefile_path=str(args.shapefile_path),
        mask_buffer_m=float(args.mask_buffer_m),
        patch_size=int(args.patch_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        folds=int(args.folds),
        model_kind=str(args.model_kind),
        use_mask_channel=bool(args.use_mask_channel),
        pooling_mode=str(args.pooling_mode),
        loss_name=str(args.loss),
        augmentation=bool(args.augmentation),
        select_on=str(args.select_on),
        multi_gpu=str(args.multi_gpu),
        log_jsonl=bool(args.log_jsonl),
    )

    out_root = Path(cfg.output_dir) / cfg.experiment_name / cfg.resolution
    out_root.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(cfg.inventory)
    if args.split_ids_path is not None:
        folds, split_meta = load_folds_from_split_ids(full_df, args.split_ids_path)
    else:
        folds, split_meta = make_grouped_folds(full_df, seed=cfg.seed, requested_folds=cfg.folds)
    (out_root / "split_ids.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    save_yaml(out_root / "config.yaml", asdict(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (tr, va) in enumerate(folds, start=1):
        train_rows = full_df.iloc[tr].reset_index(drop=True)
        val_rows = full_df.iloc[va].reset_index(drop=True)
        fold_result = train_one_fold(
            fold_idx=fold_idx,
            train_rows=train_rows,
            val_rows=val_rows,
            cfg=cfg,
            device=device,
        )
        fold_rows.append(fold_result)

    agg = aggregate_fold_metrics(fold_rows)
    (out_root / "metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
