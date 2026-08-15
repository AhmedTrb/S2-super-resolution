from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARCELS_SHP = REPO_ROOT / "data" / "shapefiles" / "2020_SEPIM.shp"


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

SATLAS_L1C_9_FROM_L2A_INDICES: tuple[int, ...] = (2, 1, 0, 3, 4, 5, 6, 8, 9)


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


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Elementwise division that avoids NaN/Inf: entries with |denominator| <= eps are set to 0."""
    result = np.zeros_like(numerator, dtype=np.float32)
    valid = np.abs(denominator) > eps
    result[valid] = numerator[valid] / denominator[valid]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _rgb_to_hue_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Standard HSV hue/saturation from normalized reflectance triplets; hue is returned normalized to [0, 1]."""
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    saturation = _safe_divide(delta, cmax, eps=eps)

    delta_safe = np.where(np.abs(delta) > eps, delta, eps)
    is_r_max = (cmax == r) & (delta > eps)
    is_g_max = (cmax == g) & (delta > eps) & ~is_r_max
    is_b_max = (cmax == b) & (delta > eps) & ~is_r_max & ~is_g_max

    hue = np.zeros_like(cmax, dtype=np.float32)
    hue = np.where(is_r_max, 60.0 * (((g - b) / delta_safe) % 6.0), hue)
    hue = np.where(is_g_max, 60.0 * (((b - r) / delta_safe) + 2.0), hue)
    hue = np.where(is_b_max, 60.0 * (((r - g) / delta_safe) + 4.0), hue)
    hue = np.mod(hue, 360.0)

    hue_norm = np.nan_to_num((hue / 360.0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    saturation = np.nan_to_num(saturation.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return hue_norm, saturation


def _compute_paper_indices(image: np.ndarray) -> np.ndarray:
    """Compute [B2, B3, B4, H, S, GLI, OSAVI, NDRE] from a normalized 10-band L2A image."""
    if image.shape[0] < 8:
        raise ValueError(
            "paper_indices recipe requires bands B02..B8A in standard order "
            f"{EXPECTED_S2_BAND_ORDER[:8]}, got shape={image.shape}."
        )

    b2, b3, b4, b5 = image[0], image[1], image[2], image[3]
    b8 = image[6]

    r, g, b = b4, b3, b2
    nir, red_edge = b8, b5

    hue, saturation = _rgb_to_hue_saturation(r, g, b)
    gli = _safe_divide(2.0 * g - r - b, 2.0 * g + r + b)
    osavi = _safe_divide(nir - r, nir + r + 0.16)
    ndre = _safe_divide(nir - red_edge, nir + red_edge)

    stacked = np.stack([b2, b3, b4, hue, saturation, gli, osavi, ndre], axis=0).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_plot_ids_from_path(value: Any) -> list[str]:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return []

    matches = re.findall(r"(?:^|/)(?:lr|sr)_plot_(\d+)", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"plot_(\d+)", text, flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    raw_is_integer = np.issubdtype(image.dtype, np.integer)
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_is_integer or float(np.max(image)) > 2.0:
        image = image / 10_000.0
    return np.clip(image, 0.0, 1.0)


def _resolve_group_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer parcel identifiers for group-wise split to avoid parcel leakage.
    candidates = ("PARCEL_ID", "ID_PARCEL", "parcel_id", "id_plot", "parcel_ref_id", "parcel_index")
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


class ParcelYellownessDataset(Dataset):
    """Return image patch, binary parcel mask, and yellowness target for regression."""

    def __init__(
        self,
        inventory_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        resolution: str = "sr",
        rows: Optional[pd.DataFrame] = None,
        band_indices: Optional[Sequence[int]] = None,
        band_recipe: Optional[str] = None,
        shapefile_path: Optional[Union[str, Path]] = None,
        shapefile_buffer_m: float = -20.0,
    ) -> None:
        self.inventory_path = Path(inventory_csv)
        self.root_dir = Path(root_dir) if root_dir else self.inventory_path.parent
        self.resolution = resolution.lower()
        self.inventory = rows.copy().reset_index(drop=True) if rows is not None else pd.read_csv(self.inventory_path)
        self.band_indices = tuple(int(index) for index in band_indices) if band_indices is not None else None
        self.band_recipe = str(band_recipe).strip().lower() if band_recipe else None
        self.date_col = _resolve_date_column(self.inventory)
        self.shapefile_path = Path(shapefile_path) if shapefile_path else DEFAULT_PARCELS_SHP
        self.shapefile_buffer_m = float(shapefile_buffer_m)
        self._parcels_gdf: Any = None
        self._parcel_lookup_by_crs: dict[str, dict[str, Any]] = {}
        self._parcel_id_column: Optional[str] = None

        self.image_col, self.mask_col = _resolve_path_columns(self.inventory, self.resolution)
        if self.image_col is None or self.mask_col is None:
            raise ValueError(
                f"Could not resolve {self.resolution.upper()} image/mask columns in inventory."
            )

        if "yellowness" not in self.inventory.columns:
            raise ValueError("Inventory must include a 'yellowness' column.")

    def _apply_band_recipe(self, image: np.ndarray) -> np.ndarray:
        if self.band_recipe is None:
            return image

        if self.band_recipe == "satlas_l1c_9_from_l2a":
            if image.shape[0] < 10:
                raise ValueError(
                    "SATLAS adaptation requires 10-band L2A input in standard order "
                    f"{EXPECTED_S2_BAND_ORDER}, got shape={image.shape}."
                )
            # SATLAS S2 weights expect pseudo-RGB + SWIR order:
            # [B04, B03, B02, B05, B06, B07, B08, B11, B12].
            return image[list(SATLAS_L1C_9_FROM_L2A_INDICES), ...]

        if self.band_recipe == "ndvi_seco_eco_9":
            if image.shape[0] < 8:
                raise ValueError(
                    "NDVI SECO ECO recipe requires at least bands B02..B8A in standard order."
                )
            base = image[:8, ...]
            b4 = image[2, ...]
            b8 = image[6, ...]
            ndvi = (b8 - b4) / np.clip(b8 + b4, 1e-6, None)
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return np.concatenate([base, ndvi[None, ...]], axis=0)

        if self.band_recipe in ("paper_indices", "raw10"):
            if self.band_recipe == "raw10":
                return image
            return _compute_paper_indices(image)

        raise ValueError(f"Unsupported band_recipe='{self.band_recipe}'.")

    @staticmethod
    def _parcel_key_candidates(value: Any) -> list[str]:
        candidates: list[str] = []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return candidates

        candidates.append(text)
        try:
            as_float = float(text)
            if as_float.is_integer():
                candidates.append(str(int(as_float)))
        except Exception:
            pass
        return list(dict.fromkeys(candidates))

    def _candidate_paths(self, raw_path: Any) -> list[Path]:
        text = str(raw_path).strip()
        if not text or text.lower() == "nan":
            return []

        normalized = text.replace("\\", "/")
        stripped = normalized.lstrip("./")
        basename = stripped.split("/")[-1]

        paths: list[Path] = []
        raw = Path(text)
        if raw.is_absolute():
            paths.append(raw)
        paths.extend(
            [
                self.root_dir / text,
                self.root_dir / normalized,
                self.root_dir / stripped,
                self.root_dir / basename,
            ]
        )

        deduped: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key not in seen:
                deduped.append(path)
                seen.add(key)
        return deduped

    def _resolve_existing_path(self, raw_path: Any, kind: str) -> Path:
        candidates = self._candidate_paths(raw_path)
        for path in candidates:
            if path.exists():
                return path
        if candidates:
            raise FileNotFoundError(f"{kind} file not found. Tried: {candidates[0]} (and {len(candidates)-1} alternatives)")
        raise FileNotFoundError(f"{kind} file path is empty in inventory.")

    def _ensure_parcels_loaded(self) -> None:
        if self._parcels_gdf is not None:
            return

        try:
            import geopandas as gpd
        except Exception as exc:
            raise ImportError("geopandas is required to generate masks from shapefile when mask files are missing.") from exc

        if not self.shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found for mask generation: {self.shapefile_path}")

        self._parcels_gdf = gpd.read_file(self.shapefile_path)
        # Prefer the authoritative parcel identifier first.
        id_candidates = ("ID_PARCEL", "id_plot", "PARCEL_ID", "parcel_id")
        self._parcel_id_column = next((column for column in id_candidates if column in self._parcels_gdf.columns), None)
        if self._parcel_id_column is None:
            raise ValueError("Could not find a parcel identifier column in shapefile. Expected one of: id_plot, ID_PARCEL, PARCEL_ID.")

    def _row_parcel_keys(self, row: pd.Series) -> list[str]:
        keys: list[str] = []
        for column in ("ID_PARCEL", "id_plot", "PARCEL_ID", "parcel_id", "plot_id", "parcel_ref_id", "parcel_index"):
            if column not in row.index:
                continue
            keys.extend(self._parcel_key_candidates(row.get(column)))

        # Fallback for inventories that only encode parcel id in image/mask filename.
        for column in (self.image_col, self.mask_col):
            if column is None or column not in row.index:
                continue
            for plot_id in _extract_plot_ids_from_path(row.get(column)):
                keys.extend(self._parcel_key_candidates(plot_id))
        return list(dict.fromkeys(keys))

    def _parcel_lookup_for_crs(self, target_crs: Any) -> dict[str, Any]:
        self._ensure_parcels_loaded()
        assert self._parcels_gdf is not None
        assert self._parcel_id_column is not None

        crs_key = str(target_crs) if target_crs is not None else "none"
        if crs_key in self._parcel_lookup_by_crs:
            return self._parcel_lookup_by_crs[crs_key]

        parcels = self._parcels_gdf
        if target_crs is not None and getattr(parcels, "crs", None) is not None and parcels.crs != target_crs:
            parcels = parcels.to_crs(target_crs)

        lookup: dict[str, Any] = {}
        for _, row in parcels.iterrows():
            geometry = row.get("geometry")
            if geometry is None or getattr(geometry, "is_empty", True):
                continue
            for key in self._parcel_key_candidates(row.get(self._parcel_id_column)):
                lookup[key] = geometry

        self._parcel_lookup_by_crs[crs_key] = lookup
        return lookup

    def _build_mask_from_shapefile(self, row: pd.Series, image_src: Any) -> np.ndarray:
        lookup = self._parcel_lookup_for_crs(image_src.crs)
        row_keys = self._row_parcel_keys(row)
        for key in row_keys:
            geometry = lookup.get(key)
            if geometry is None:
                continue
            candidates: list[Any] = []
            buffered_geometry = geometry
            if self.shapefile_buffer_m != 0.0:
                try:
                    candidate = geometry.buffer(self.shapefile_buffer_m)
                    if not getattr(candidate, "is_empty", True):
                        buffered_geometry = candidate
                except Exception:
                    buffered_geometry = geometry

            # Prefer buffered geometry (parcel interior), but fallback to original geometry if empty after rasterization.
            candidates.append(buffered_geometry)
            if buffered_geometry is not geometry:
                candidates.append(geometry)

            for geom in candidates:
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(image_src.height, image_src.width),
                    transform=image_src.transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(np.float32)
                if float(mask.sum()) > 0.0:
                    return mask

        raise FileNotFoundError(
            "Mask file is missing and corresponding parcel geometry could not be found in shapefile "
            f"{self.shapefile_path} for row keys={row_keys}"
        )

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.inventory.iloc[index]
        image_path = self._resolve_existing_path(row[self.image_col], kind="image")

        with rasterio.open(image_path) as src:
            image = _normalize_image(_align_s2_band_order(src.read(), src.descriptions))
            try:
                mask_path = self._resolve_existing_path(row[self.mask_col], kind="mask")
            except FileNotFoundError:
                mask = self._build_mask_from_shapefile(row, src)
            else:
                with rasterio.open(mask_path) as mask_src:
                    mask = mask_src.read(1).astype(np.float32)
                if float(np.nansum(mask > 0)) <= 0.0:
                    # Some inventories contain placeholder/empty mask rasters; regenerate from shapefile.
                    mask = self._build_mask_from_shapefile(row, src)
        image = self._apply_band_recipe(image)
        if self.band_indices is not None:
            image = image[list(self.band_indices), ...]
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
    band_recipe: Optional[str] = None,
    shapefile_path: Optional[Union[str, Path]] = None,
    shapefile_buffer_m: float = -20.0,
) -> SplitDatasets:
    inventory_path = Path(inventory_csv)
    full_df = pd.read_csv(inventory_path)
    group_column = _resolve_group_column(full_df)
    if group_column is not None:
        print(f"Using group split column: {group_column}")
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
        train=ParcelYellownessDataset(
            inventory_csv,
            root_dir=root_dir,
            resolution=resolution,
            rows=train_rows,
            band_indices=band_indices,
            band_recipe=band_recipe,
            shapefile_path=shapefile_path,
            shapefile_buffer_m=shapefile_buffer_m,
        ),
        val=ParcelYellownessDataset(
            inventory_csv,
            root_dir=root_dir,
            resolution=resolution,
            rows=val_rows,
            band_indices=band_indices,
            band_recipe=band_recipe,
            shapefile_path=shapefile_path,
            shapefile_buffer_m=shapefile_buffer_m,
        ),
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