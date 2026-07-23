from __future__ import annotations

import gc
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from torch.utils.data import Dataset
from tqdm import tqdm

from .date_matcher import TemporalDateMatcher
from .indices import SpectralIndexComputer
from .masks import ParcelMaskGenerator
from .patching import ParcelPatchExtractor, PatchWindow
from .super_resolution import SuperResolutionProcessor

logger = logging.getLogger(__name__)

BAND_NAMES: Tuple[str, ...] = (
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
BAND_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(BAND_NAMES)}

WAVELENGTHS_NM: Dict[str, float] = {
    "B05": 705.0,
    "B06": 740.0,
    "B07": 783.0,
    "B8A": 865.0,
}

BASE_INDEX_DEFS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "NDRE": lambda c: (b(c, "B08") - b(c, "B05")) / (b(c, "B08") + b(c, "B05") + 1e-6),
    "NDYVI": lambda c: (b(c, "B08") - (b(c, "B04") + b(c, "B05")))
    / (b(c, "B08") + b(c, "B04") + b(c, "B05") + 1e-6),
    "GCC": lambda c: b(c, "B03") / (b(c, "B04") + b(c, "B03") + b(c, "B02") + 1e-6),
    "NDYI": lambda c: (b(c, "B03") - b(c, "B02")) / (b(c, "B03") + b(c, "B02") + 1e-6),
    "mNDblue": lambda c: (b(c, "B02") - b(c, "B06")) / (b(c, "B02") + b(c, "B08") + 1e-6),
    "NDVI": lambda c: (b(c, "B08") - b(c, "B04")) / (b(c, "B08") + b(c, "B04") + 1e-6),
    "GNDVI": lambda c: (b(c, "B08") - b(c, "B03")) / (b(c, "B08") + b(c, "B03") + 1e-6),
    "NDMI": lambda c: (b(c, "B08") - b(c, "B11")) / (b(c, "B08") + b(c, "B11") + 1e-6),
    "NBR": lambda c: (b(c, "B08") - b(c, "B12")) / (b(c, "B08") + b(c, "B12") + 1e-6),
}

BAND_PAIR_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("B08", "B04"),
    ("B8A", "B05"),
    ("B11", "B08"),
    ("B12", "B11"),
    ("B03", "B02"),
    ("B07", "B05"),
)

TEXTURE_PROPS: Tuple[str, ...] = ("contrast", "homogeneity", "energy", "correlation", "ASM")


def b(cube: np.ndarray, name: str) -> np.ndarray:
    return cube[BAND_INDEX[name]].astype(np.float32)


def normalize_reflectance(cube: np.ndarray) -> np.ndarray:
    raw_is_integer = np.issubdtype(cube.dtype, np.integer)
    cube = cube.astype(np.float32)
    if raw_is_integer or float(np.nanmax(cube)) > 2.0:
        cube = cube / 10_000.0
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cube, 0.0, 1.0)


def compute_summary_stats(name: str, arr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = arr[mask]
    if values.size == 0:
        return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
    }


def red_edge_features(cube: np.ndarray) -> Dict[str, np.ndarray]:
    b05 = b(cube, "B05")
    b06 = b(cube, "B06")
    b07 = b(cube, "B07")
    b8a = b(cube, "B8A")

    s_705_740 = (b06 - b05) / (WAVELENGTHS_NM["B06"] - WAVELENGTHS_NM["B05"])
    s_740_783 = (b07 - b06) / (WAVELENGTHS_NM["B07"] - WAVELENGTHS_NM["B06"])
    s_783_865 = (b8a - b07) / (WAVELENGTHS_NM["B8A"] - WAVELENGTHS_NM["B07"])

    c_705_740_783 = b05 - 2.0 * b06 + b07
    c_740_783_865 = b06 - 2.0 * b07 + b8a

    return {
        "RE_slope_705_740": s_705_740,
        "RE_slope_740_783": s_740_783,
        "RE_slope_783_865": s_783_865,
        "RE_curvature_705_740_783": c_705_740_783,
        "RE_curvature_740_783_865": c_740_783_865,
    }


def pair_features(cube: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for left, right in BAND_PAIR_FEATURES:
        left_band = b(cube, left)
        right_band = b(cube, right)
        out[f"diff_{left}_{right}"] = left_band - right_band
        out[f"ratio_{left}_{right}"] = left_band / (right_band + 1e-6)
    return out


def maybe_texture_features(cube: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    try:
        from skimage.feature import graycomatrix, graycoprops
    except Exception:
        return {}

    def _quantize(img: np.ndarray, img_min: float, img_max: float, levels: int = 32) -> np.ndarray:
        scaled = (img - img_min) / (img_max - img_min + 1e-6)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * (levels - 1)).astype(np.uint8)

    def _crop_to_mask(img: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return img[y0:y1, x0:x1], m[y0:y1, x0:x1]

    def _glcm_props(img: np.ndarray, m: np.ndarray, prefix: str, lo: float, hi: float) -> Dict[str, float]:
        if not np.any(m):
            return {f"{prefix}_{prop}": np.nan for prop in TEXTURE_PROPS}

        img_crop, m_crop = _crop_to_mask(img, m)
        if float(np.mean(m_crop)) < 0.60:
            return {f"{prefix}_{prop}": np.nan for prop in TEXTURE_PROPS}

        fill_value = float(np.median(img_crop[m_crop]))
        img_filled = img_crop.copy()
        img_filled[~m_crop] = fill_value
        quant = _quantize(img_filled, lo, hi)

        glcm = graycomatrix(
            quant,
            distances=[1, 2],
            angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
            levels=32,
            symmetric=True,
            normed=True,
        )

        out: Dict[str, float] = {}
        for prop in TEXTURE_PROPS:
            out[f"{prefix}_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
        return out

    nir = b(cube, "B08")
    ndvi = (b(cube, "B08") - b(cube, "B04")) / (b(cube, "B08") + b(cube, "B04") + 1e-6)

    out: Dict[str, float] = {}
    out.update(_glcm_props(nir, mask, "tex_nir", 0.0, 1.0))
    out.update(_glcm_props(ndvi, mask, "tex_ndvi", -1.0, 1.0))
    return out


def compute_feature_block(cube: np.ndarray, mask: np.ndarray, include_texture: bool) -> Dict[str, float]:
    normalized = normalize_reflectance(cube)
    cube_invalid = (normalized == 0.0) | np.isnan(normalized) | np.isinf(normalized)
    cube_valid_mask = ~cube_invalid.any(axis=0)
    valid = mask.astype(bool) & cube_valid_mask

    features: Dict[str, float] = {}

    for band_name in BAND_NAMES:
        features.update(compute_summary_stats(f"band_{band_name}", b(normalized, band_name), valid))

    for idx_name, fn in BASE_INDEX_DEFS.items():
        features.update(compute_summary_stats(idx_name, fn(normalized), valid))

    for name, arr in red_edge_features(normalized).items():
        features.update(compute_summary_stats(name, arr, valid))

    return features


def extract_crop_metadata(parcel_row: pd.Series) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    field_map = {
        "crop_code": ("CODE_CULTU",),
        "crop_group": ("CODE_GROUP",),
        "crop_name_1": ("CULTURE_D1",),
        "crop_name_2": ("CULTURE_D2",),
        "region": ("REGION", "REGION_ext"),
        "source": ("source",),
        "parcel_id": ("PARCEL_ID",),
    }

    for target_name, candidates in field_map.items():
        for candidate in candidates:
            if candidate in parcel_row.index and pd.notna(parcel_row.get(candidate)):
                metadata[target_name] = parcel_row.get(candidate)
                break

    return metadata


def compute_percentile_features(name: str, arr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = arr[mask]
    if values.size == 0:
        return {f"{name}_p{pct}": np.nan for pct in (10, 25, 50, 75, 90)}
    p10, p25, p50, p75, p90 = np.nanpercentile(values, [10, 25, 50, 75, 90])
    return {
        f"{name}_p10": float(p10),
        f"{name}_p25": float(p25),
        f"{name}_p50": float(p50),
        f"{name}_p75": float(p75),
        f"{name}_p90": float(p90),
    }


def compute_spatial_extras(cube: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    normalized = normalize_reflectance(cube)
    valid = mask.astype(bool) & np.isfinite(normalized).all(axis=0)
    features: Dict[str, float] = {}

    if not np.any(valid):
        features[f"{prefix}yellow_fraction"] = np.nan
        features[f"{prefix}yellow_cluster_count"] = np.nan
        features[f"{prefix}yellow_largest_cluster_fraction"] = np.nan
        return features

    for band_name in BAND_NAMES:
        features.update(compute_percentile_features(f"{prefix}band_{band_name}", b(normalized, band_name), valid))

    ndvi = (b(normalized, "B08") - b(normalized, "B04")) / (b(normalized, "B08") + b(normalized, "B04") + 1e-6)
    ndre = (b(normalized, "B08") - b(normalized, "B05")) / (b(normalized, "B08") + b(normalized, "B05") + 1e-6)

    features.update(compute_percentile_features(f"{prefix}NDVI", ndvi, valid))
    features.update(compute_percentile_features(f"{prefix}NDRE", ndre, valid))

    yellow_threshold = float(np.nanpercentile(ndvi[valid], 25))
    yellow_mask = valid & (ndvi <= yellow_threshold)
    yellow_fraction = float(np.sum(yellow_mask) / np.sum(valid))

    try:
        from scipy import ndimage
    except Exception:
        features[f"{prefix}yellow_fraction"] = yellow_fraction
        features[f"{prefix}yellow_cluster_count"] = np.nan
        features[f"{prefix}yellow_largest_cluster_fraction"] = np.nan
        return features

    labeled, cluster_count = ndimage.label(yellow_mask)
    if cluster_count == 0:
        largest_fraction = 0.0
    else:
        sizes = ndimage.sum(yellow_mask, labeled, index=range(1, cluster_count + 1))
        largest_fraction = float(np.max(sizes) / np.sum(valid)) if np.size(sizes) else 0.0

    features[f"{prefix}yellow_fraction"] = yellow_fraction
    features[f"{prefix}yellow_cluster_count"] = float(cluster_count)
    features[f"{prefix}yellow_largest_cluster_fraction"] = largest_fraction
    return features


@dataclass(frozen=True)
class ObservationEvent:
    parcel_index: int
    parcel_row: pd.Series
    observation_column: str
    observation_date: object
    yellowness: float
    s2_date_str: str
    day_offset: int


class ObservationDrivenS2Pipeline:
    """Process one Sentinel-2 stack and one shapefile into parcel-level LR/SR outputs."""

    def __init__(
        self,
        stack_path: Union[str, Path],
        parcels: gpd.GeoDataFrame,
        date_matcher: TemporalDateMatcher,
        output_dir: Union[str, Path],
        sr_processor: SuperResolutionProcessor,
        crs: Any,
        observation_columns: Optional[Sequence[str]] = None,
        spectral_indexer: Optional[SpectralIndexComputer] = None,
        min_patch_size: int = 256,
        parcel_buffer_meters: float = 0.0,
        include_advanced_features: bool = True,
        include_texture_features: bool = False,
        include_spatial_features: bool = True,
    ) -> None:
        self.stack_path = Path(stack_path)
        self.parcels = parcels
        self.date_matcher = date_matcher
        self.output_dir = Path(output_dir)
        self.sr_processor = sr_processor
        self.crs = crs
        self.observation_columns = observation_columns or self._discover_observation_columns(parcels)
        self.spectral_indexer = spectral_indexer or SpectralIndexComputer()
        self.patch_extractor = ParcelPatchExtractor(min_patch_size=min_patch_size)
        self.parcel_buffer_meters = float(parcel_buffer_meters)
        self.include_advanced_features = include_advanced_features
        self.include_texture_features = include_texture_features
        self.include_spatial_features = include_spatial_features

        self.images_lr_dir = self.output_dir / "images_lr"
        self.images_sr_dir = self.output_dir / "images_sr"
        self.masks_lr_dir = self.output_dir / "masks_lr"
        self.masks_sr_dir = self.output_dir / "masks_sr"
        for directory in (self.images_lr_dir, self.images_sr_dir, self.masks_lr_dir, self.masks_sr_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._lr_inventory_rows: List[Dict[str, Any]] = []
        self._sr_inventory_rows: List[Dict[str, Any]] = []
        self._combined_inventory_rows: List[Dict[str, Any]] = []
        self._lr_masks_written: set[str] = set()
        self._sr_masks_written: set[str] = set()

    def _geometry_for_processing(self, geometry: BaseGeometry, parcel_index: int) -> BaseGeometry:
        if self.parcel_buffer_meters == 0.0:
            return geometry

        buffered = geometry.buffer(self.parcel_buffer_meters)
        if buffered.is_empty:
            logger.warning(
                "Buffered geometry is empty for parcel index %s with buffer %s. Falling back to original geometry.",
                parcel_index,
                self.parcel_buffer_meters,
            )
            return geometry
        return buffered

    @staticmethod
    def _discover_observation_columns(gdf: gpd.GeoDataFrame, start_col: str = "12_05", end_col: str = "18_11") -> List[str]:
        columns = gdf.columns.tolist()
        if start_col in columns and end_col in columns:
            start_idx = columns.index(start_col)
            end_idx = columns.index(end_col) + 1
            return columns[start_idx:end_idx]

        # Fallback for shapefiles where campaign anchors are absent: infer observation columns
        # from day_month labels such as 12_05, 7_6, or 07_06.
        observation_cols = [
            col for col in columns if re.match(r"^\d{1,2}_\d{1,2}$", str(col).strip())
        ]
        if not observation_cols:
            raise ValueError(
                "Could not discover observation columns. Expected either anchor columns "
                f"'{start_col}'..'{end_col}' or date-like columns matching DD_MM."
            )

        def _month_day_key(col_name: str) -> Tuple[int, int]:
            day_str, month_str = str(col_name).split("_", 1)
            return int(month_str), int(day_str)

        return sorted(observation_cols, key=_month_day_key)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = self._enumerate_observation_events()
        if not events:
            return pd.DataFrame(), pd.DataFrame()

        # Load existing combined-inventory observation keys to prevent duplicate reprocessing
        existing_observation_keys: set[Tuple[str, str, str]] = set()
        combined_csv_path = self.output_dir / "observations_inventory.csv"
        if combined_csv_path.exists():
            try:
                existing_df = pd.read_csv(combined_csv_path)
                required_cols = {"id_plot", "observation_date", "s2_acquisition_date"}
                if required_cols.issubset(existing_df.columns):
                    for _, row in existing_df.iterrows():
                        obs_key = (
                            str(row.get("id_plot", "")).strip(),
                            str(row.get("observation_date", "")).strip(),
                            str(row.get("s2_acquisition_date", "")).strip(),
                        )
                        if all(obs_key):
                            existing_observation_keys.add(obs_key)
                    logger.info(
                        f"Loaded {len(existing_observation_keys)} existing observation keys from {combined_csv_path.name}"
                    )
            except Exception as e:
                logger.warning(f"Could not read existing {combined_csv_path.name}: {e}")

        with rasterio.open(self.stack_path) as src:
            for event in tqdm(events, desc="Processing observations", unit="parcel"):
                plot_id = str(event.parcel_row.get("ID_PARCEL", event.parcel_index)).strip()
                obs_date = event.observation_date.isoformat()
                obs_key = (plot_id, obs_date, event.s2_date_str)
                if obs_key in existing_observation_keys:
                    logger.info(f"Skipping already-processed observation: id_plot={plot_id}, obs={obs_date}")
                    continue

                processing_geometry = self._geometry_for_processing(event.parcel_row.geometry, event.parcel_index)

                window = self.patch_extractor.compute_window(
                    geom=processing_geometry,
                    img_height=src.height,
                    img_width=src.width,
                    raster_transform=src.transform,
                    parcel_index=event.parcel_index,
                )
                if window is None:
                    continue

                band_offset = self.date_matcher.get_band_offset(event.s2_date_str)
                lr_patch = self._read_lr_patch(src, window, band_offset)
                self._process_observation(
                    event=event,
                    window=window,
                    lr_patch=lr_patch,
                    parcel_geometry=processing_geometry,
                )

        # Merge new rows with preexisting rows if they existed, keeping all unique id_plots
        lr_inventory = pd.DataFrame(self._lr_inventory_rows)
        sr_inventory = pd.DataFrame(self._sr_inventory_rows)
        combined_inventory = pd.DataFrame(self._combined_inventory_rows)

        lr_csv_path = self.output_dir / "lr_inventory.csv"
        sr_csv_path = self.output_dir / "sr_inventory.csv"

        dedupe_subset = ["id_plot", "observation_date", "s2_acquisition_date"]

        if lr_csv_path.exists() and not lr_inventory.empty:
            try:
                lr_inventory = pd.concat([pd.read_csv(lr_csv_path), lr_inventory], ignore_index=True)
                if all(col in lr_inventory.columns for col in dedupe_subset):
                    lr_inventory = lr_inventory.drop_duplicates(subset=dedupe_subset, keep="last")
                elif "id_plot" in lr_inventory.columns:
                    lr_inventory = lr_inventory.drop_duplicates(subset=["id_plot"], keep="last")
            except Exception:
                pass
        if sr_csv_path.exists() and not sr_inventory.empty:
            try:
                sr_inventory = pd.concat([pd.read_csv(sr_csv_path), sr_inventory], ignore_index=True)
                if all(col in sr_inventory.columns for col in dedupe_subset):
                    sr_inventory = sr_inventory.drop_duplicates(subset=dedupe_subset, keep="last")
                elif "id_plot" in sr_inventory.columns:
                    sr_inventory = sr_inventory.drop_duplicates(subset=["id_plot"], keep="last")
            except Exception:
                pass
        if combined_csv_path.exists() and not combined_inventory.empty:
            try:
                combined_inventory = pd.concat([pd.read_csv(combined_csv_path), combined_inventory], ignore_index=True)
                if all(col in combined_inventory.columns for col in dedupe_subset):
                    combined_inventory = combined_inventory.drop_duplicates(subset=dedupe_subset, keep="last")
                elif "id_plot" in combined_inventory.columns:
                    combined_inventory = combined_inventory.drop_duplicates(subset=["id_plot"], keep="last")
            except Exception:
                pass

        if not lr_inventory.empty:
            lr_inventory.to_csv(lr_csv_path, index=False)
        if not sr_inventory.empty:
            sr_inventory.to_csv(sr_csv_path, index=False)
        if not combined_inventory.empty:
            combined_inventory.to_csv(combined_csv_path, index=False)

        return lr_inventory, sr_inventory

    def _enumerate_observation_events(self) -> List[ObservationEvent]:
        events: List[ObservationEvent] = []
        for parcel_index, row in self.parcels.iterrows():
            for column in self.observation_columns:
                value = row.get(column)
                if pd.isna(value) or value == 0:
                    continue
                obs_date = TemporalDateMatcher.parse_observation_column(column)
                s2_date_str = self.date_matcher.find_nearest_s2_date(obs_date)
                day_offset = self.date_matcher.days_to_nearest_s2(obs_date)
                events.append(
                    ObservationEvent(
                        parcel_index=parcel_index,
                        parcel_row=row,
                        observation_column=column,
                        observation_date=obs_date,
                        yellowness=float(value),
                        s2_date_str=s2_date_str,
                        day_offset=day_offset,
                    )
                )
        return events

    def _read_lr_patch(self, src: rasterio.io.DatasetReader, window: PatchWindow, band_offset: int) -> np.ndarray:
        band_start = band_offset * 10 + 1
        band_indexes = list(range(band_start, band_start + 10))
        return src.read(
            band_indexes,
            window=rasterio.windows.Window(
                col_off=window.col_min,
                row_off=window.row_min,
                width=window.col_max - window.col_min,
                height=window.row_max - window.row_min,
            ),
        )

    def _process_observation(
        self,
        event: ObservationEvent,
        window: PatchWindow,
        lr_patch: np.ndarray,
        parcel_geometry: BaseGeometry,
    ) -> None:
        plot_id = str(event.parcel_row.get("ID_PARCEL", event.parcel_index))
        zone_bounds = window.zone.bounds
        normalized_lr_patch = self.sr_processor.normalize_cube(lr_patch)
        sr_cube = self.sr_processor.run(normalized_lr_patch)
        _, h_lr, w_lr = normalized_lr_patch.shape
        bands_sr, h_sr, w_sr = sr_cube.shape

        lr_mask, sr_mask, lr_transform, sr_transform = ParcelMaskGenerator.create_masks(
            zone_bounds=zone_bounds,
            parcel_geometry=parcel_geometry,
            lr_shape=(h_lr, w_lr),
            sr_shape=(h_sr, w_sr),
        )
        lr_index_stats = self.spectral_indexer.compute_stats(normalized_lr_patch, lr_mask)
        sr_index_stats = self.spectral_indexer.compute_stats(sr_cube, sr_mask)
        lr_feature_stats = compute_feature_block(normalized_lr_patch, lr_mask, self.include_texture_features) if self.include_advanced_features else {}
        sr_feature_stats = compute_feature_block(sr_cube, sr_mask, self.include_texture_features) if self.include_advanced_features else {}

        s2_date_dt = pd.to_datetime(event.s2_date_str, errors="coerce")
        observation_date_dt = pd.to_datetime(event.observation_date, errors="coerce")
        geometry = event.parcel_row.geometry
        parcel_area_m2 = float(event.parcel_row.get("area_m2", event.parcel_row.get("SURF_PARC", geometry.area)))
        parcel_area_ha = float(event.parcel_row.get("area_ha", parcel_area_m2 / 10_000.0))
        common_row = {
            "year": int(s2_date_dt.year) if not pd.isna(s2_date_dt) else np.nan,
            "parcel_area_m2": parcel_area_m2,
            "parcel_area_ha": parcel_area_ha,
            "month": int(s2_date_dt.month) if not pd.isna(s2_date_dt) else np.nan,
            "day_of_year": int(s2_date_dt.dayofyear) if not pd.isna(s2_date_dt) else np.nan,
            "s2_date_difference_days": int(abs((s2_date_dt - observation_date_dt).days))
            if not pd.isna(s2_date_dt) and not pd.isna(observation_date_dt)
            else np.nan,
            "id_plot": plot_id,
            "observation_date": event.observation_date.isoformat(),
            "s2_acquisition_date": event.s2_date_str,
            "day_offset": event.day_offset,
            "yellowness": event.yellowness,
        }

        obs_date_str = event.observation_date.isoformat()
        lr_image_path = self.images_lr_dir / f"lr_plot_{plot_id}_obs_{obs_date_str}.tif"
        sr_image_path = self.images_sr_dir / f"sr_plot_{plot_id}_obs_{obs_date_str}.tif"
        lr_mask_path = self.masks_lr_dir / f"mask_lr_plot_{plot_id}_static.tif"
        sr_mask_path = self.masks_sr_dir / f"mask_sr_plot_{plot_id}_static.tif"

        self._write_geotiff(lr_image_path, normalized_lr_patch.astype(np.float32), h_lr, w_lr, 10, lr_transform)
        self._write_geotiff(sr_image_path, sr_cube, h_sr, w_sr, bands_sr, sr_transform)

        if plot_id not in self._lr_masks_written and not lr_mask_path.exists():
            self._write_geotiff(lr_mask_path, lr_mask.astype(np.uint8), h_lr, w_lr, 1, lr_transform)
            self._lr_masks_written.add(plot_id)

        if plot_id not in self._sr_masks_written and not sr_mask_path.exists():
            self._write_geotiff(sr_mask_path, sr_mask.astype(np.uint8), h_sr, w_sr, 1, sr_transform)
            self._sr_masks_written.add(plot_id)

        lr_row = {
            **common_row,
            "lr_image_path": os.path.relpath(lr_image_path, self.output_dir),
            "lr_mask_path": os.path.relpath(lr_mask_path, self.output_dir),
        }
        lr_row.update({f"lr_{name}": value for name, value in lr_index_stats.items()})
        if lr_feature_stats:
            lr_row.update({f"lr_{name}": value for name, value in lr_feature_stats.items()})
        self._lr_inventory_rows.append(lr_row)

        sr_row = {
            **common_row,
            "sr_image_path": os.path.relpath(sr_image_path, self.output_dir),
            "sr_mask_path": os.path.relpath(sr_mask_path, self.output_dir),
        }
        sr_row.update({f"sr_{name}": value for name, value in sr_index_stats.items()})
        if sr_feature_stats:
            sr_row.update({f"sr_{name}": value for name, value in sr_feature_stats.items()})
        self._sr_inventory_rows.append(sr_row)

        combined_row = {
            **common_row,
            "lr_patch": os.path.relpath(lr_image_path, self.output_dir),
            "sr_patch": os.path.relpath(sr_image_path, self.output_dir),
            "lr_mask": os.path.relpath(lr_mask_path, self.output_dir),
            "sr_mask": os.path.relpath(sr_mask_path, self.output_dir),
        }
        combined_row.update({f"lr_{name}": value for name, value in lr_index_stats.items()})
        combined_row.update({f"sr_{name}": value for name, value in sr_index_stats.items()})
        if lr_feature_stats:
            combined_row.update({f"lr_{name}": value for name, value in lr_feature_stats.items()})
        if sr_feature_stats:
            combined_row.update({f"sr_{name}": value for name, value in sr_feature_stats.items()})
        self._combined_inventory_rows.append(combined_row)

    def _write_geotiff(self, path: Path, array: np.ndarray, height: int, width: int, count: int, transform: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": str(array.dtype),
            "crs": self.crs,
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256 if count <= 1 else 512,
            "blockysize": 256 if count <= 1 else 512,
        }
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array.astype(np.float32), 0.0, 1.0)
        with rasterio.open(path, "w", **profile) as dst:
            if array.ndim == 2:
                dst.write(array.astype(np.float32 if np.issubdtype(array.dtype, np.floating) else np.uint8), 1)
            else:
                dst.write(array)

    @staticmethod
    def _release_gpu_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TorchGeoParcelDataset(Dataset):
    """Simple dataset that stacks an SR image and a parcel mask for training."""

    def __init__(self, inventory_csv: Union[str, Path], root_dir: Optional[Union[str, Path]] = None, drop_missing_targets: bool = True) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(inventory_csv).parent
        self.inventory = pd.read_csv(inventory_csv)
        if drop_missing_targets and "yellowness" in self.inventory.columns:
            self.inventory = self.inventory[self.inventory["yellowness"].notna()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.inventory.iloc[index]
        image_path = self.root_dir / str(row["sr_image_path"])
        mask_path = self.root_dir / str(row["sr_mask_path"])

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        stacked = np.concatenate([image, mask[np.newaxis, ...]], axis=0)
        tensor = torch.from_numpy(stacked)
        if "yellowness" not in row or pd.isna(row["yellowness"]):
            target = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            target = torch.tensor(float(row["yellowness"]), dtype=torch.float32)
        return tensor, target
