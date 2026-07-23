from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _resolve_first(columns: Iterable[str], options: Sequence[str]) -> Optional[str]:
    return next((name for name in options if name in columns), None)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    raw_is_integer = np.issubdtype(image.dtype, np.integer)
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_is_integer or float(np.max(image)) > 2.0:
        image = image / 10_000.0
    return np.clip(image, 0.0, 1.0)


def _resolve_group_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("id_plot", "parcel_ref_id", "parcel_index")
    return next((column for column in candidates if column in df.columns), None)


class ParcelYellownessDataset(Dataset):
    """Return image patch, binary parcel mask, and yellowness target for regression."""

    def __init__(
        self,
        inventory_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        resolution: str = "sr",
        rows: Optional[pd.DataFrame] = None,
    ) -> None:
        self.inventory_path = Path(inventory_csv)
        self.root_dir = Path(root_dir) if root_dir else self.inventory_path.parent
        self.resolution = resolution.lower()
        self.inventory = rows.copy().reset_index(drop=True) if rows is not None else pd.read_csv(self.inventory_path)

        image_options = {
            "lr": ("lr_patch", "lr_image_path", "lr_image", "lr_patch_path"),
            "sr": ("sr_patch", "sr_image_path", "sr_image", "sr_patch_path"),
        }
        mask_options = {
            "lr": ("lr_mask", "lr_mask_path", "mask_path"),
            "sr": ("sr_mask", "sr_mask_path", "mask_path"),
        }
        self.image_col = _resolve_first(self.inventory.columns, image_options[self.resolution])
        self.mask_col = _resolve_first(self.inventory.columns, mask_options[self.resolution])
        if self.image_col is None or self.mask_col is None:
            raise ValueError(
                f"Could not resolve {self.resolution.upper()} image/mask columns in inventory."
            )

        if "yellowness" not in self.inventory.columns:
            raise ValueError("Inventory must include a 'yellowness' column.")

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.inventory.iloc[index]
        image_path = self.root_dir / str(row[self.image_col])
        mask_path = self.root_dir / str(row[self.mask_col])

        with rasterio.open(image_path) as src:
            image = _normalize_image(src.read())

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "target": torch.tensor(float(row["yellowness"]), dtype=torch.float32),
            "id_plot": str(row.get("id_plot", "")),
            "row_index": int(index),
        }


@dataclass(frozen=True)
class SplitDatasets:
    train: ParcelYellownessDataset
    val: ParcelYellownessDataset
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame


def make_group_split(
    inventory_csv: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    resolution: str = "sr",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitDatasets:
    inventory_path = Path(inventory_csv)
    full_df = pd.read_csv(inventory_path)
    group_column = _resolve_group_column(full_df)
    groups = full_df[group_column] if group_column else np.arange(len(full_df))
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, val_idx = next(splitter.split(full_df, groups=groups))
    train_rows = full_df.iloc[train_idx].reset_index(drop=True)
    val_rows = full_df.iloc[val_idx].reset_index(drop=True)
    return SplitDatasets(
        train=ParcelYellownessDataset(inventory_csv, root_dir=root_dir, resolution=resolution, rows=train_rows),
        val=ParcelYellownessDataset(inventory_csv, root_dir=root_dir, resolution=resolution, rows=val_rows),
        train_rows=train_rows,
        val_rows=val_rows,
    )


def make_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(image, mask)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_regressor(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].cpu().numpy()
            pred = model(image, mask).cpu().numpy()
            preds.append(pred)
            targets.append(target)

    if not preds:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    unfreeze_epoch: Optional[int] = None,
    backbone_learning_rate: Optional[float] = None,
    early_stopping_patience: int | None = None,   # NEW
    early_stopping_min_delta: float = 0.0,        # NEW
    verbose: bool = True,                          # NEW
) -> Dict[str, Any]:
    def set_backbone_trainable(trainable: bool) -> None:
        setter = getattr(model, "set_backbone_trainable", None)
        if callable(setter):
            setter(trainable)

    def build_optimizer(backbone_lr: Optional[float]) -> torch.optim.Optimizer:
        if backbone_lr is None:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        backbone_params: list[torch.nn.Parameter] = []
        head_params: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(parameter)
            else:
                head_params.append(parameter)

        param_groups: list[Dict[str, Any]] = []
        if head_params:
            param_groups.append({"params": head_params, "lr": learning_rate})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer construction.")
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    model = model.to(device)
    if freeze_backbone:
        set_backbone_trainable(False)
    optimizer = build_optimizer(backbone_learning_rate)
    loss_fn = nn.SmoothL1Loss()
    history: list[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_rmse = float("inf")
    backbone_unfrozen = False

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        if freeze_backbone and unfreeze_epoch and (not backbone_unfrozen) and epoch >= unfreeze_epoch:
            set_backbone_trainable(True)
            optimizer = build_optimizer(backbone_learning_rate)
            backbone_unfrozen = True

        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate_regressor(model, val_loader, device)
        record = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(record)

        if verbose:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_metrics['rmse']:.6f} "
                f"train_mae={val_metrics['mae']:.6f} val_mae={val_metrics['rmse']:.6f}"
            )

        # Early stopping on val_loss
        if val_metrics["rmse"] < (best_val - early_stopping_min_delta):
            best_val = val_metrics["rmse"]
            epochs_no_improve = 0
            # ...existing best checkpoint logic...
        else:
            epochs_no_improve += 1

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_loss improvement for {early_stopping_patience} epochs)."
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_metrics": evaluate_regressor(model, val_loader, device),
    }