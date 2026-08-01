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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy


EXPECTED_S2_BAND_ORDER: tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)


def _normalize_band_name(name: str) -> str:
    text = str(name).strip().upper()
    if not text:
        return text
    if text.startswith("B") and len(text) == 2 and text[1].isdigit():
        return f"B0{text[1]}"
    return text


def _align_s2_band_order(image: np.ndarray, descriptions: Sequence[str]) -> np.ndarray:
    if image.shape[0] == len(EXPECTED_S2_BAND_ORDER) and not any(descriptions):
        return image

    normalized = [_normalize_band_name(name) for name in descriptions]
    index_by_name = {name: idx for idx, name in enumerate(normalized) if name}

    if all(name in index_by_name for name in EXPECTED_S2_BAND_ORDER):
        ordered_indices = [index_by_name[name] for name in EXPECTED_S2_BAND_ORDER]
        return image[ordered_indices]

    if image.shape[0] != len(EXPECTED_S2_BAND_ORDER):
        raise ValueError(
            "Expected 10 Sentinel-2 bands ordered as "
            f"{EXPECTED_S2_BAND_ORDER}, but got {image.shape[0]} bands and no usable descriptions."
        )

    return image


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


def _resolve_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("observation_date", "date", "acquisition_date", "capture_date")
    return next((column for column in candidates if column in df.columns), None)


def _resolve_path_columns(df: pd.DataFrame, resolution: str) -> tuple[Optional[str], Optional[str]]:
    image_options = {
        "lr": ("lr_patch", "lr_image_path", "lr_image", "lr_patch_path"),
        "sr": ("sr_patch", "sr_image_path", "sr_image", "sr_patch_path"),
    }
    mask_options = {
        "lr": ("lr_mask", "lr_mask_path", "mask_path"),
        "sr": ("sr_mask", "sr_mask_path", "mask_path"),
    }
    image_col = _resolve_first(df.columns, image_options[resolution])
    mask_col = _resolve_first(df.columns, mask_options[resolution])
    return image_col, mask_col


def _filter_missing_rows(df: pd.DataFrame, root_dir: Path, resolution: str) -> pd.DataFrame:
    image_col, mask_col = _resolve_path_columns(df, resolution)
    if image_col is None or mask_col is None:
        raise ValueError(f"Could not resolve {resolution.upper()} image/mask columns in inventory.")

    image_exists = df[image_col].astype(str).map(lambda value: (root_dir / value).exists())
    mask_exists = df[mask_col].astype(str).map(lambda value: (root_dir / value).exists())
    keep_mask = image_exists & mask_exists
    skipped = int((~keep_mask).sum())
    if skipped:
        print(f"Skipping {skipped} rows with missing {resolution.upper()} image or mask files.")

    filtered = df.loc[keep_mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No valid {resolution.upper()} rows remain after removing missing files.")
    return filtered


class ParcelYellownessDataset(Dataset):
    """Return image patch, binary parcel mask, and yellowness target for regression."""

    def __init__(
        self,
        inventory_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        resolution: str = "sr",
        rows: Optional[pd.DataFrame] = None,
        band_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.inventory_path = Path(inventory_csv)
        self.root_dir = Path(root_dir) if root_dir else self.inventory_path.parent
        self.resolution = resolution.lower()
        raw_inventory = rows.copy().reset_index(drop=True) if rows is not None else pd.read_csv(self.inventory_path)
        self.inventory = _filter_missing_rows(raw_inventory, self.root_dir, self.resolution)
        self.band_indices = tuple(int(index) for index in band_indices) if band_indices is not None else None
        self.date_col = _resolve_date_column(self.inventory)

        self.image_col, self.mask_col = _resolve_path_columns(self.inventory, self.resolution)
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
            image = _normalize_image(_align_s2_band_order(src.read(), src.descriptions))
        if self.band_indices is not None:
            image = image[list(self.band_indices), ...]

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "target": torch.tensor(float(row["yellowness"]), dtype=torch.float32),
            "id_plot": str(row.get("id_plot", "")),
            "observation_date": str(row.get(self.date_col, "")) if self.date_col is not None else "",
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
    band_indices: Optional[Sequence[int]] = None,
) -> SplitDatasets:
    inventory_path = Path(inventory_csv)
    resolved_root_dir = Path(root_dir) if root_dir else inventory_path.parent
    full_df = _filter_missing_rows(pd.read_csv(inventory_path), resolved_root_dir, resolution)
    group_column = _resolve_group_column(full_df)
    if group_column is None:
        groups = np.arange(len(full_df))
        split_labels = np.zeros(len(full_df), dtype=int)
        train_idx, val_idx = train_test_split(
            np.arange(len(full_df)),
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=split_labels,
        )
    else:
        parcel_means = full_df.groupby(group_column)["yellowness"].mean().reset_index(name="parcel_mean_yellowness")
        quantiles = min(5, parcel_means["parcel_mean_yellowness"].nunique())
        if quantiles < 2:
            parcel_means["stratum"] = 0
        else:
            parcel_means["stratum"] = pd.qcut(
                parcel_means["parcel_mean_yellowness"],
                q=quantiles,
                labels=False,
                duplicates="drop",
            )

        group_to_stratum = parcel_means.set_index(group_column)["stratum"].to_dict()
        groups = full_df[group_column].astype(str)
        group_rows = pd.DataFrame({"group": groups}).reset_index().rename(columns={"index": "row_index"})
        group_labels = group_rows["group"].map(group_to_stratum).fillna(0).astype(int)

        unique_groups = group_rows["group"].drop_duplicates().to_numpy()
        unique_labels = pd.Series(unique_groups).map(group_to_stratum).fillna(0).astype(int).to_numpy()
        train_groups, val_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=unique_labels,
        )
        train_idx = full_df[full_df[group_column].astype(str).isin(train_groups)].index.to_numpy()
        val_idx = full_df[full_df[group_column].astype(str).isin(val_groups)].index.to_numpy()
    train_rows = full_df.iloc[train_idx].reset_index(drop=True)
    val_rows = full_df.iloc[val_idx].reset_index(drop=True)
    return SplitDatasets(
        train=ParcelYellownessDataset(inventory_csv, root_dir=root_dir, resolution=resolution, rows=train_rows, band_indices=band_indices),
        val=ParcelYellownessDataset(inventory_csv, root_dir=root_dir, resolution=resolution, rows=val_rows, band_indices=band_indices),
        train_rows=train_rows,
        val_rows=val_rows,
    )


def make_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


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


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            pred = model(image, mask)
            loss = loss_fn(pred, target)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


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
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    plateau_unfreeze_patience: int | None = None,
    plateau_unfreeze_min_delta: float = 0.0,
    unfreeze_last_stages: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    def set_backbone_trainable(trainable: bool) -> None:
        setter = getattr(base_model, "set_backbone_trainable", None)
        if callable(setter):
            setter(trainable)

    def unfreeze_backbone_for_finetune(num_stages: int) -> None:
        unfreezer = getattr(base_model, "unfreeze_last_backbone_stages", None)
        if callable(unfreezer):
            unfreezer(num_stages)
        else:
            set_backbone_trainable(True)

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
    backbone_unfrozen = False

    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    stop_epochs_no_improve = 0
    unfreeze_epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        if freeze_backbone and unfreeze_epoch and (not backbone_unfrozen) and epoch >= unfreeze_epoch:
            unfreeze_backbone_for_finetune(unfreeze_last_stages)
            optimizer = build_optimizer(backbone_learning_rate)
            backbone_unfrozen = True

        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = evaluate_loss(model, val_loader, device, loss_fn)
        val_metrics = evaluate_regressor(model, val_loader, device)
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "backbone_unfrozen": float(backbone_unfrozen),
        }
        history.append(record)

        if verbose:
            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f}"
            )

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            best_val_rmse = val_metrics["rmse"]
            stop_epochs_no_improve = 0
            best_state = deepcopy(model.state_dict())
        else:
            stop_epochs_no_improve += 1

        if val_metrics["rmse"] < (best_val_rmse - plateau_unfreeze_min_delta):
            best_val_rmse = val_metrics["rmse"]
            unfreeze_epochs_no_improve = 0
        else:
            unfreeze_epochs_no_improve += 1

        if (
            freeze_backbone
            and not backbone_unfrozen
            and plateau_unfreeze_patience is not None
            and unfreeze_epochs_no_improve >= plateau_unfreeze_patience
        ):
            unfreeze_backbone_for_finetune(unfreeze_last_stages)
            optimizer = build_optimizer(backbone_learning_rate)
            backbone_unfrozen = True
            unfreeze_epochs_no_improve = 0
            if verbose:
                stage_msg = "all stages" if unfreeze_last_stages <= 0 else f"last {unfreeze_last_stages} stages"
                print(
                    f"Validation RMSE plateau detected at epoch {epoch}; unfreezing {stage_msg} for fine-tuning."
                )

        if early_stopping_patience is not None and stop_epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_loss improvement for {early_stopping_patience} epochs)."
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_regressor(model, val_loader, device)
    final_metrics["val_loss"] = evaluate_loss(model, val_loader, device, loss_fn)

    return {
        "model": model,
        "history": history,
        "best_metrics": final_metrics,
    }