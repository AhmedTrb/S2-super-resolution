#!/usr/bin/env python3
"""
Production observation-driven dataset builder for TorchGeo foundation-model training.

This is a refinement of the original "exhaustive" pipeline (which super-resolved
every parcel x every S2 date). Here, the pipeline is driven by the field
observations themselves:

  1. For every parcel, scan its yellowness observation columns.
  2. For each non-null observation, snap its date to the nearest S2 overpass.
  3. Extract the LR patch ONLY for that (parcel, snapped_date) pair, lazily,
     directly from the multi-temporal stack (no full date x parcel cache).
  4. Build LR + SR binary parcel masks (mask cached/reused per parcel).
  5. Run SEN2SRLite super-resolution on that single LR patch.
  6. Compute spectral indices (NDRE, NDYVI, GCC, NDYI, mNDblue) on the masked
     LR patch -> per-index mean/std.
  7. Persist:
       - LR GeoTIFF  (images_lr/)
       - SR GeoTIFF  (images_sr/)
       - LR mask     (masks_lr/, written once per parcel)
       - SR mask     (masks_sr/, written once per parcel)
       - lr_inventory.csv : id_plot, observation_date, s2_acquisition_date,
                            yellowness, day_offset, lr_image_path, lr_mask_path,
                            <index>_mean, <index>_std (for every index)
       - sr_inventory.csv : id_plot, observation_date, s2_acquisition_date,
                            yellowness, day_offset, sr_image_path, sr_mask_path
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import mlstac
import numpy as np
import pandas as pd
import rasterio
import sen2sr
import torch
import torch.nn.functional as F
from rasterio.features import rasterize
from rasterio.transform import Affine, from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from torch.utils.data import Dataset
from tqdm import tqdm

# Headless execution: never open interactive figure windows.
import matplotlib

matplotlib.use("Agg")

from src.s2_pipeline.pipeline import compute_feature_block

logger = logging.getLogger(__name__)


def log_stage(message: str) -> None:
    """Print and log a pipeline stage message without breaking tqdm bars."""
    tqdm.write(message)
    logger.info(message)


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Constants (mirrors verified notebook / inference script settings)
# ---------------------------------------------------------------------------

BANDS_PER_IMAGE: int = 10
BAND_NAMES: Sequence[str] = (
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

# Band -> channel-index lookup, used by the spectral-index registry below.
BAND_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(BAND_NAMES)}

UPSCALE: int = 4
OVERLAP: int = 32
MODEL_MULTIPLE: int = 256
MIN_PATCH_SIZE: int = 256

OBSERVATION_YEAR: int = 2020
OBSERVATION_START_COL: str = "12_05"
OBSERVATION_END_COL: str = "18_11"

MODEL_URL: str = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
)

EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Spectral index registry
# ---------------------------------------------------------------------------
#
# Each index is a function of a (10, H, W) reflectance cube (band order =
# BAND_NAMES) and returns an (H, W) array. Reflectance is expected in the
# same units the cube already carries (the cube is NOT rescaled here -- the
# caller decides whether to feed raw DN or /10000 reflectance; this pipeline
# feeds raw-DN-scale-consistent cubes the same way for every index so ratios
# are unaffected by the scale factor).

def _band(cube: np.ndarray, name: str) -> np.ndarray:
    return cube[BAND_INDEX[name]].astype(np.float32)


def index_ndre(cube: np.ndarray) -> np.ndarray:
    """NDRE = (B8 - B5) / (B8 + B5)"""
    b8, b5 = _band(cube, "B08"), _band(cube, "B05")
    return (b8 - b5) / (b8 + b5 + EPS)


def index_ndyvi(cube: np.ndarray) -> np.ndarray:
    """NDYVI = (B8 - (B4 + B5)) / (B8 + (B4 + B5))"""
    b8, b4, b5 = _band(cube, "B08"), _band(cube, "B04"), _band(cube, "B05")
    summed = b4 + b5
    return (b8 - summed) / (b8 + summed + EPS)


def index_gcc(cube: np.ndarray) -> np.ndarray:
    """GCC = B3 / (B4 + B3 + B2)"""
    b3, b4, b2 = _band(cube, "B03"), _band(cube, "B04"), _band(cube, "B02")
    return b3 / (b4 + b3 + b2 + EPS)


def index_ndyi(cube: np.ndarray) -> np.ndarray:
    """NDYI = (B3 - B2) / (B3 + B2)"""
    b3, b2 = _band(cube, "B03"), _band(cube, "B02")
    return (b3 - b2) / (b3 + b2 + EPS)


def index_mndblue(cube: np.ndarray) -> np.ndarray:
    """mNDblue = (B2 - B6) / (B2 + B8)"""
    b2, b6, b8 = _band(cube, "B02"), _band(cube, "B06"), _band(cube, "B08")
    return (b2 - b6) / (b2 + b8 + EPS)


SPECTRAL_INDICES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "NDRE": index_ndre,
    "NDYVI": index_ndyvi,
    "GCC": index_gcc,
    "NDYI": index_ndyi,
    "mNDblue": index_mndblue,
}


def compute_index_stats(
    cube: np.ndarray,
    mask: np.ndarray,
    indices: Mapping[str, Callable[[np.ndarray], np.ndarray]] = SPECTRAL_INDICES,
) -> Dict[str, float]:
    """
    Compute mean/std for every registered spectral index, restricted to
    pixels inside the binary parcel mask. Returns a flat dict:
    {"NDRE_mean": ..., "NDRE_std": ..., ...}
    """
    # Scale to reflectance space (DN / 10000) if raw DN is passed
    is_dn = np.issubdtype(cube.dtype, np.integer) or (np.issubdtype(cube.dtype, np.floating) and cube.max() > 10.0)
    cube = cube.astype(np.float32)
    if is_dn:
        cube = cube / 10_000.0

    # Create valid mask combining the parcel mask and valid pixels in the cube
    # (excluding nodata = 0.0, NaN, and Inf)
    cube_invalid = (cube == 0.0) | np.isnan(cube) | np.isinf(cube)
    cube_valid_mask = ~cube_invalid.any(axis=0)
    valid = mask.astype(bool) & cube_valid_mask

    stats: Dict[str, float] = {}

    if not valid.any():
        for name in indices:
            stats[f"{name}_mean"] = float("nan")
            stats[f"{name}_std"] = float("nan")
        return stats

    for name, fn in indices.items():
        band_image = fn(cube)
        band_image = np.nan_to_num(band_image, nan=0.0, posinf=0.0, neginf=0.0)
        values = band_image[valid]
        stats[f"{name}_mean"] = float(np.mean(values))
        stats[f"{name}_std"] = float(np.std(values))

    return stats


# ---------------------------------------------------------------------------
# TemporalDateMatcher
# ---------------------------------------------------------------------------


class TemporalDateMatcher:
    """
    Calendar utility that maps irregular field-observation dates onto the
    nearest available Sentinel-2 overpass date from a timeline anchor file.
    """

    def __init__(self, calendar_path: Union[str, Path]) -> None:
        self.calendar_path = Path(calendar_path)
        self._s2_dates: List[date] = self._load_calendar(self.calendar_path)
        self._s2_date_strings: List[str] = [d.isoformat() for d in self._s2_dates]
        self._s2_date_to_index: Dict[str, int] = {
            d: i for i, d in enumerate(self._s2_date_strings)
        }

    @staticmethod
    def _load_calendar(calendar_path: Path) -> List[date]:
        """Read one ``YYYY-MM-DD`` or ``YYYYMMDD`` date per line."""
        dates: List[date] = []
        with calendar_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                token = raw_line.strip()
                if not token:
                    continue
                dates.append(TemporalDateMatcher._parse_date_token(token))
        if not dates:
            raise ValueError(f"No dates found in calendar file: {calendar_path}")
        dates.sort()
        return dates

    @staticmethod
    def _parse_date_token(token: str) -> date:
        token = token.strip()
        if "-" in token:
            return datetime.strptime(token, "%Y-%m-%d").date()
        if len(token) == 8 and token.isdigit():
            return datetime.strptime(token, "%Y%m%d").date()
        raise ValueError(f"Unsupported date token format: {token!r}")

    @staticmethod
    def parse_observation_column(column_name: str, year: int = OBSERVATION_YEAR) -> date:
        """
        Convert shapefile observation columns such as ``'12_05'`` (DD_MM) into
        a concrete calendar date in the campaign year.
        """
        day_str, month_str = column_name.split("_")
        return date(year, int(month_str), int(day_str))

    @property
    def s2_dates(self) -> List[date]:
        return list(self._s2_dates)

    @property
    def s2_date_strings(self) -> List[str]:
        return list(self._s2_date_strings)

    def find_nearest_s2_date(
        self,
        observation_date: Union[date, datetime, str],
    ) -> str:
        """
        Snap an irregular observation date to the chronologically closest
        Sentinel-2 overpass, returned as ``YYYY-MM-DD``.
        """
        obs = self._coerce_date(observation_date)
        nearest = min(self._s2_dates, key=lambda s2: abs((s2 - obs).days))
        return nearest.isoformat()

    def days_to_nearest_s2(self, observation_date: Union[date, datetime, str]) -> int:
        """Absolute day offset between an observation and its snapped S2 date."""
        obs = self._coerce_date(observation_date)
        snapped = datetime.strptime(self.find_nearest_s2_date(obs), "%Y-%m-%d").date()
        return abs((snapped - obs).days)

    def s2_band_index_for_date(self, s2_date_str: str) -> int:
        """
        Timestamp position (0-based) of an S2 date within the calendar, used
        to compute the band offset for that timestamp inside the stack.
        """
        try:
            return self._s2_date_to_index[s2_date_str]
        except KeyError as exc:
            raise ValueError(f"S2 date {s2_date_str!r} not found in calendar.") from exc

    @staticmethod
    def _coerce_date(value: Union[date, datetime, str]) -> date:
        if isinstance(value, str):
            return TemporalDateMatcher._parse_date_token(value)
        if isinstance(value, datetime):
            return value.date()
        return value


# ---------------------------------------------------------------------------
# SpatialMaskGenerator
# ---------------------------------------------------------------------------


class SpatialMaskGenerator:
    """
    Static rasterization helpers that produce LR / SR binary parcel masks
    aligned to patch bounding boxes.
    """

    @staticmethod
    def build_transform(
        zone_bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Affine:
        xmin, ymin, xmax, ymax = zone_bounds
        return from_bounds(xmin, ymin, xmax, ymax, width=width, height=height)

    @staticmethod
    def rasterize_masks(
        zone_bounds: Tuple[float, float, float, float],
        parcel_geometry: BaseGeometry,
        lr_shape: Tuple[int, int],
        sr_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Affine, Affine]:
        """
        Return LR and SR binary masks (inside = 1, outside = 0) plus their
        georeferencing transforms.
        """
        h_lr, w_lr = lr_shape
        h_sr, w_sr = sr_shape

        lr_transform = SpatialMaskGenerator.build_transform(zone_bounds, w_lr, h_lr)
        sr_transform = SpatialMaskGenerator.build_transform(zone_bounds, w_sr, h_sr)

        shapes = [(parcel_geometry, 1)]

        lr_mask = rasterize(
            shapes,
            out_shape=(h_lr, w_lr),
            transform=lr_transform,
            fill=0,
            dtype=np.uint8,
        )
        sr_mask = rasterize(
            shapes,
            out_shape=(h_sr, w_sr),
            transform=sr_transform,
            fill=0,
            dtype=np.uint8,
        )
        return lr_mask, sr_mask, lr_transform, sr_transform


# ---------------------------------------------------------------------------
# Observation enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationEvent:
    """One (parcel, field-observation-date) pair to be processed."""

    parcel_index: int
    parcel_row: pd.Series
    observation_column: str
    observation_date: date
    yellowness: float
    s2_date_str: str
    day_offset: int


def discover_observation_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    """Return yellowness observation columns bounded by campaign start/end."""
    columns = gdf.columns.tolist()
    start_idx = columns.index(OBSERVATION_START_COL)
    end_idx = columns.index(OBSERVATION_END_COL) + 1
    return columns[start_idx:end_idx]


def filter_parcels_in_raster(
    gdf: gpd.GeoDataFrame,
    raster_crs: Any,
    raster_bounds: Tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Keep parcels intersecting the Sentinel-2 tile extent."""
    parcels = gdf.to_crs(raster_crs)
    image_extent = box(*raster_bounds)
    return parcels[parcels.intersects(image_extent)].reset_index(drop=True)


def enumerate_observation_events(
    parcels: gpd.GeoDataFrame,
    date_matcher: TemporalDateMatcher,
    observation_columns: Optional[Sequence[str]] = None,
) -> List[ObservationEvent]:
    """
    Walk every parcel x observation-column cell, keep the non-null/non-zero
    ones, and snap each to its nearest S2 acquisition date. This is the
    observation-driven replacement for the old "every date x every parcel"
    cache: only cells with a real field measurement produce work downstream.
    """
    columns = observation_columns or discover_observation_columns(parcels)
    events: List[ObservationEvent] = []

    for parcel_index, row in parcels.iterrows():
        for column in columns:
            value = row.get(column)
            if pd.isna(value) or value == 0:
                continue

            obs_date = TemporalDateMatcher.parse_observation_column(column)
            s2_date_str = date_matcher.find_nearest_s2_date(obs_date)
            day_offset = date_matcher.days_to_nearest_s2(obs_date)

            events.append(
                ObservationEvent(
                    parcel_index=int(parcel_index),
                    parcel_row=row,
                    observation_column=column,
                    observation_date=obs_date,
                    yellowness=float(value),
                    s2_date_str=s2_date_str,
                    day_offset=day_offset,
                )
            )

    return events


# ---------------------------------------------------------------------------
# Patch window extraction (single-parcel, on demand)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchWindow:
    """Pixel window metadata for one parcel, computed once and reused across
    every observation event that touches that parcel."""

    parcel_index: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    zone: box


def compute_patch_window(
    geom: BaseGeometry,
    img_height: int,
    img_width: int,
    raster_transform: Affine,
    min_size: int = MIN_PATCH_SIZE,
) -> Optional[PatchWindow]:
    """
    Derive a stable parcel-centered pixel window (verified cell-7 logic),
    for a single parcel geometry. Returns None if the window would fall
    outside the raster bounds.
    """
    p_xmin, p_ymin, p_xmax, p_ymax = geom.bounds

    p_row_min, p_col_min = rasterio.transform.rowcol(raster_transform, p_xmin, p_ymax)
    p_row_max, p_col_max = rasterio.transform.rowcol(raster_transform, p_xmax, p_ymin)

    p_width = p_col_max - p_col_min
    p_height = p_row_max - p_row_min

    p_center_col = p_col_min + (p_width // 2)
    p_center_row = p_row_min + (p_height // 2)

    target_width = max(min_size, p_width)
    target_height = max(min_size, p_height)

    col_min = p_center_col - (target_width // 2)
    col_max = col_min + target_width
    row_min = p_center_row - (target_height // 2)
    row_max = row_min + target_height

    if col_min < 0:
        col_max -= col_min
        col_min = 0
    if col_max > img_width:
        col_min -= col_max - img_width
        col_max = img_width

    if row_min < 0:
        row_max -= row_min
        row_min = 0
    if row_max > img_height:
        row_min -= row_max - img_height
        row_max = img_height

    if col_min < 0 or row_min < 0:
        return None

    ul_x, ul_y = rasterio.transform.xy(raster_transform, row_min, col_min, offset="ul")
    lr_x, lr_y = rasterio.transform.xy(raster_transform, row_max, col_max, offset="lr")
    adjusted_zone = box(ul_x, lr_y, lr_x, ul_y)

    return PatchWindow(
        parcel_index=-1,  # filled in by caller
        row_min=row_min,
        row_max=row_max,
        col_min=col_min,
        col_max=col_max,
        zone=adjusted_zone,
    )


class PatchWindowCache:
    """
    Lazily computes and memoizes one PatchWindow per parcel index, so that a
    parcel touched by multiple observation dates only has its window solved
    once.
    """

    def __init__(self, img_height: int, img_width: int, raster_transform: Affine) -> None:
        self.img_height = img_height
        self.img_width = img_width
        self.raster_transform = raster_transform
        self._cache: Dict[int, Optional[PatchWindow]] = {}

    def get(self, parcel_index: int, geom: BaseGeometry) -> Optional[PatchWindow]:
        if parcel_index not in self._cache:
            window = compute_patch_window(
                geom=geom,
                img_height=self.img_height,
                img_width=self.img_width,
                raster_transform=self.raster_transform,
            )
            if window is not None:
                window = PatchWindow(
                    parcel_index=parcel_index,
                    row_min=window.row_min,
                    row_max=window.row_max,
                    col_min=window.col_min,
                    col_max=window.col_max,
                    zone=window.zone,
                )
            self._cache[parcel_index] = window
        return self._cache[parcel_index]


def read_lr_patch(
    src: rasterio.io.DatasetReader,
    window: PatchWindow,
    band_offset: int,
) -> np.ndarray:
    """
    Read a single (BANDS_PER_IMAGE, H, W) LR patch for one timestamp from the
    open multi-temporal stack. ``band_offset`` is the 0-based timestamp index
    (i.e. the position of the S2 date within the calendar).
    """
    band_start = band_offset * BANDS_PER_IMAGE + 1
    band_indexes = list(range(band_start, band_start + BANDS_PER_IMAGE))
    return src.read(
        band_indexes,
        window=rasterio.windows.Window(
            col_off=window.col_min,
            row_off=window.row_min,
            width=window.col_max - window.col_min,
            height=window.row_max - window.row_min,
        ),
    )


# ---------------------------------------------------------------------------
# SEN2SRLite inference helpers
# ---------------------------------------------------------------------------


def load_sr_model(
    model_url: str,
    model_dir: Union[str, Path],
    device: torch.device,
) -> torch.nn.Module:
    model_dir = Path(model_dir)
    log_stage(f"[MODEL] Loading SEN2SRLite from {model_dir} on {device}...")
    load_start = time.perf_counter()
    mlstac.download(file=model_url, output_dir=str(model_dir))
    model = mlstac.load(str(model_dir)).compiled_model(device=str(device))
    model.eval()
    log_stage(f"[MODEL] SEN2SRLite ready in {format_elapsed(time.perf_counter() - load_start)}")
    return model


def is_model_compatible(height: int, width: int, model_multiple: int = MODEL_MULTIPLE) -> bool:
    return (height % model_multiple == 0) and (width % model_multiple == 0)


def bicubic_upsample(cube: np.ndarray, scale: int = UPSCALE) -> np.ndarray:
    tensor = torch.from_numpy(cube).unsqueeze(0)
    upsampled = F.interpolate(
        tensor,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    )
    return upsampled.squeeze(0).numpy()


def run_super_resolution(
    lr_patch: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    overlap: int = OVERLAP,
) -> np.ndarray:
    """
    Upscale a ``(10, H, W)`` LR patch to SR using SEN2SRLite with bicubic
    fallback for non-compatible border dimensions.
    """
    cube = lr_patch.astype(np.float32) / 10_000.0
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    cube = np.clip(cube, 0.0, 1.0)
    height, width = cube.shape[1:]

    if is_model_compatible(height, width):
        tensor = torch.from_numpy(cube).to(device)
        with torch.no_grad():
            sr = sen2sr.predict_large(model=model, X=tensor, overlap=overlap)
        if isinstance(sr, torch.Tensor):
            sr = sr.cpu().numpy()
        del tensor
    else:
        sr = bicubic_upsample(cube, UPSCALE)

    sr = np.clip(sr, 0.0, 1.0)
    return np.nan_to_num(sr.astype(np.float32))


def write_geotiff(
    path: Path,
    array: np.ndarray,
    profile: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **dict(profile)) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)
            for band_idx, band_name in enumerate(BAND_NAMES, start=1):
                if band_idx <= dst.count:
                    dst.set_band_description(band_idx, band_name)


# ---------------------------------------------------------------------------
# ObservationDrivenDatasetBuilder
# ---------------------------------------------------------------------------


class ObservationDrivenDatasetBuilder:
    """
    Core orchestrator: for every field observation (not every date x parcel
    combination), read the matching LR patch, build masks, super-resolve,
    compute spectral indices, persist GeoTIFFs, and compile the two
    inventory CSVs (LR-side and SR-side).
    """

    def __init__(
        self,
        stack_path: Union[str, Path],
        parcels: gpd.GeoDataFrame,
        date_matcher: TemporalDateMatcher,
        output_dir: Union[str, Path],
        model: torch.nn.Module,
        device: torch.device,
        crs: Any,
        observation_columns: Optional[Sequence[str]] = None,
        spectral_indices: Mapping[str, Callable[[np.ndarray], np.ndarray]] = SPECTRAL_INDICES,
        include_advanced_features: bool = True,
        include_texture_features: bool = False,
    ) -> None:
        self.stack_path = Path(stack_path)
        self.parcels = parcels
        self.date_matcher = date_matcher
        self.output_dir = Path(output_dir)
        self.images_lr_dir = self.output_dir / "images_lr"
        self.images_sr_dir = self.output_dir / "images_sr"
        self.masks_lr_dir = self.output_dir / "masks_lr"
        self.masks_sr_dir = self.output_dir / "masks_sr"
        self.model = model
        self.device = device
        self.crs = crs
        self.observation_columns = observation_columns or discover_observation_columns(parcels)
        self.spectral_indices = spectral_indices
        self.include_advanced_features = include_advanced_features
        self.include_texture_features = include_texture_features

        for directory in (
            self.images_lr_dir,
            self.images_sr_dir,
            self.masks_lr_dir,
            self.masks_sr_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        # mask dedup: a parcel's mask geometry never changes across dates,
        # so write it once and reuse across every observation for that parcel.
        self._lr_masks_written: set[str] = set()
        self._sr_masks_written: set[str] = set()

        self._lr_inventory_rows: List[Dict[str, Any]] = []
        self._sr_inventory_rows: List[Dict[str, Any]] = []

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process only the (parcel, observation-date) pairs that have a real
        field measurement. Returns (lr_inventory, sr_inventory) DataFrames.
        """
        phase_start = time.perf_counter()

        log_stage("[OBS] Enumerating field observations...")
        events = enumerate_observation_events(
            parcels=self.parcels,
            date_matcher=self.date_matcher,
            observation_columns=self.observation_columns,
        )
        log_stage(
            f"[OBS] Found {len(events)} valid observations across "
            f"{self.parcels.shape[0]} parcels "
            f"({len(self.observation_columns)} observation columns scanned)."
        )

        if not events:
            log_stage("[OBS] No observations found -- nothing to process.")
            return pd.DataFrame(), pd.DataFrame()

        with rasterio.open(self.stack_path) as src:
            if src.count % BANDS_PER_IMAGE != 0:
                raise ValueError(
                    f"Expected band count divisible by {BANDS_PER_IMAGE}, got {src.count}."
                )
            num_timestamps = src.count // BANDS_PER_IMAGE
            log_stage(
                f"[OBS] Stack metadata: {src.width}x{src.height}px | "
                f"{src.count} bands | {num_timestamps} timestamps | CRS={src.crs}"
            )

            window_cache = PatchWindowCache(
                img_height=src.height,
                img_width=src.width,
                raster_transform=src.transform,
            )

            event_pbar = tqdm(
                events,
                desc="Observations",
                unit="obs",
                dynamic_ncols=True,
            )

            skipped_no_window = 0
            skipped_no_band = 0

            for event in event_pbar:
                plot_id = str(event.parcel_row.get("ID_PARCEL", event.parcel_index))
                event_pbar.set_postfix(
                    plot=plot_id, date=event.s2_date_str, refresh=False
                )

                window = window_cache.get(event.parcel_index, event.parcel_row.geometry)
                if window is None:
                    logger.warning(
                        "Skipping observation for parcel %s on %s: window exceeds raster bounds.",
                        plot_id,
                        event.s2_date_str,
                    )
                    skipped_no_window += 1
                    continue

                try:
                    timestamp_idx = self.date_matcher.s2_band_index_for_date(event.s2_date_str)
                except ValueError:
                    logger.warning(
                        "Skipping observation for parcel %s: snapped date %s not in calendar.",
                        plot_id,
                        event.s2_date_str,
                    )
                    skipped_no_band = 1
                    continue

                if timestamp_idx >= num_timestamps:
                    logger.warning(
                        "Skipping observation for parcel %s on %s: "
                        "timestamp index %d exceeds stack's %d timestamps.",
                        plot_id,
                        event.s2_date_str,
                        timestamp_idx,
                        num_timestamps,
                    )
                    skipped_no_band += 1
                    continue

                lr_patch = read_lr_patch(src, window, band_offset=timestamp_idx)
                if lr_patch is None or lr_patch.size == 0:
                    continue

                self._process_observation(event=event, window=window, lr_patch=lr_patch)

            if skipped_no_window:
                log_stage(f"[OBS] Skipped {skipped_no_window} observations: window out of bounds.")
            if skipped_no_band:
                log_stage(f"[OBS] Skipped {skipped_no_band} observations: date not in stack.")

        lr_inventory = pd.DataFrame(self._lr_inventory_rows)
        sr_inventory = pd.DataFrame(self._sr_inventory_rows)

        lr_csv_path = self.output_dir / "lr_inventory.csv"
        sr_csv_path = self.output_dir / "sr_inventory.csv"
        lr_inventory.to_csv(lr_csv_path, index=False)
        sr_inventory.to_csv(sr_csv_path, index=False)

        log_stage(
            f"[OBS] Dataset compiled in {format_elapsed(time.perf_counter() - phase_start)}: "
            f"{len(lr_inventory)} LR rows -> {lr_csv_path} | "
            f"{len(sr_inventory)} SR rows -> {sr_csv_path}"
        )

        self._release_gpu_memory()
        return lr_inventory, sr_inventory

    def _process_observation(
        self,
        event: ObservationEvent,
        window: PatchWindow,
        lr_patch: np.ndarray,
    ) -> None:
        plot_id = str(event.parcel_row.get("ID_PARCEL", event.parcel_index))
        zone_bounds = window.zone.bounds
        xmin, ymin, xmax, ymax = zone_bounds

        # --- super-resolution ---
        sr_cube = run_super_resolution(lr_patch, self.model, self.device)
        _, h_lr, w_lr = lr_patch.shape
        bands_sr, h_sr, w_sr = sr_cube.shape

        lr_mask, sr_mask, lr_transform, sr_transform = SpatialMaskGenerator.rasterize_masks(
            zone_bounds=zone_bounds,
            parcel_geometry=event.parcel_row.geometry,
            lr_shape=(h_lr, w_lr),
            sr_shape=(h_sr, w_sr),
        )

        # --- spectral indices on the masked LR patch ---
        index_stats = compute_index_stats(
            cube=lr_patch,
            mask=lr_mask,
            indices=self.spectral_indices,
        )

        # --- filenames ---
        obs_date_str = event.observation_date.isoformat()
        lr_image_filename = f"lr_plot_{plot_id}_obs_{obs_date_str}.tif"
        sr_image_filename = f"sr_plot_{plot_id}_obs_{obs_date_str}.tif"
        lr_mask_filename = f"mask_lr_plot_{plot_id}_static.tif"
        sr_mask_filename = f"mask_sr_plot_{plot_id}_static.tif"

        lr_image_path = self.images_lr_dir / lr_image_filename
        sr_image_path = self.images_sr_dir / sr_image_filename
        lr_mask_path = self.masks_lr_dir / lr_mask_filename
        sr_mask_path = self.masks_sr_dir / sr_mask_filename

        lr_profile = {
            "driver": "GTiff",
            "height": h_lr,
            "width": w_lr,
            "count": lr_patch.shape[0],
            "dtype": "float32",
            "crs": self.crs,
            "transform": lr_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        sr_profile = {
            "driver": "GTiff",
            "height": h_sr,
            "width": w_sr,
            "count": bands_sr,
            "dtype": "float32",
            "crs": self.crs,
            "transform": sr_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        lr_reflectance = np.clip(lr_patch.astype(np.float32) / 10_000.0, 0.0, 1.0)
        write_geotiff(lr_image_path, lr_reflectance, lr_profile)
        write_geotiff(sr_image_path, sr_cube, sr_profile)

        lr_feature_stats = (
            compute_feature_block(lr_reflectance, lr_mask, include_texture=self.include_texture_features)
            if self.include_advanced_features
            else {}
        )
        sr_feature_stats = (
            compute_feature_block(sr_cube, sr_mask, include_texture=self.include_texture_features)
            if self.include_advanced_features
            else {}
        )

        if plot_id not in self._lr_masks_written and not lr_mask_path.exists():
            lr_mask_profile = dict(lr_profile)
            lr_mask_profile.update({"count": 1, "dtype": "uint8"})
            write_geotiff(lr_mask_path, lr_mask, lr_mask_profile)
            self._lr_masks_written.add(plot_id)

        if plot_id not in self._sr_masks_written and not sr_mask_path.exists():
            sr_mask_profile = dict(sr_profile)
            sr_mask_profile.update({"count": 1, "dtype": "uint8"})
            write_geotiff(sr_mask_path, sr_mask, sr_mask_profile)
            self._sr_masks_written.add(plot_id)

        # --- LR inventory row: indices + stats, no SR info ---
        lr_row: Dict[str, Any] = {
            "id_plot": plot_id,
            "observation_date": obs_date_str,
            "s2_acquisition_date": event.s2_date_str,
            "day_offset": event.day_offset,
            "yellowness": event.yellowness,
            "lr_image_path": os.path.relpath(lr_image_path, self.output_dir),
            "lr_mask_path": os.path.relpath(lr_mask_path, self.output_dir),
        }
        lr_row.update(index_stats)
        if lr_feature_stats:
            lr_row.update({f"lr_{name}": value for name, value in lr_feature_stats.items()})
        self._lr_inventory_rows.append(lr_row)

        # --- SR inventory row: paths only, as requested ---
        sr_row = {
            "id_plot": plot_id,
            "observation_date": obs_date_str,
            "s2_acquisition_date": event.s2_date_str,
            "day_offset": event.day_offset,
            "yellowness": event.yellowness,
            "sr_image_path": os.path.relpath(sr_image_path, self.output_dir),
            "sr_mask_path": os.path.relpath(sr_mask_path, self.output_dir),
        }
        if sr_feature_stats:
            sr_row.update({f"sr_{name}": value for name, value in sr_feature_stats.items()})
        self._sr_inventory_rows.append(sr_row)

    @staticmethod
    def _release_gpu_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# TorchGeo dataset (SR-side, unchanged contract: 10-band SR image + mask)
# ---------------------------------------------------------------------------


class TorchGeoYellownessDataset(Dataset):
    """
    PyTorch dataset that stacks a 10-band SR image with a 1-band parcel mask
    into an 11-channel tensor for TorchGeo foundation-model fine-tuning.
    Reads from ``sr_inventory.csv`` produced by ObservationDrivenDatasetBuilder.
    """

    def __init__(
        self,
        inventory_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        drop_missing_targets: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(inventory_csv).parent
        self.inventory = pd.read_csv(inventory_csv)

        if drop_missing_targets and "yellowness" in self.inventory.columns:
            self.inventory = self.inventory[self.inventory["yellowness"].notna()].reset_index(
                drop=True
            )

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.inventory.iloc[index]

        image_path = self.root_dir / str(row["sr_image_path"])
        mask_path = self.root_dir / str(row["mask_path"])

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """End-to-end dataset build using project-local hardcoded paths."""
    _configure_logging()
    pipeline_start = time.perf_counter()

    repo_root = Path(__file__).resolve().parent

    stack_path = Path(r"C:/Users/ahtrabelsi/Desktop/stage/S2 data/Sentinel2_T31UEQ_TSG.tif")
    parcels_path = repo_root / "data" / "2020_buf20" / "2020_buf20.shp"
    calendar_path = repo_root / "data" / "Sentinel2_T31UEQ_interpolation_dates.txt"
    output_dir = repo_root / "torchgeo_yellowness_dataset"
    model_dir = repo_root / "model" / "SEN2SRLite"

    log_stage("=" * 72)
    log_stage("Observation-driven TorchGeo yellowness dataset pipeline")
    log_stage(f"  stack     -> {stack_path}")
    log_stage(f"  parcels   -> {parcels_path}")
    log_stage(f"  calendar  -> {calendar_path}")
    log_stage(f"  output    -> {output_dir}")
    log_stage(f"  model     -> {model_dir}")
    log_stage("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_stage(f"[INIT] Compute device: {device}")

    log_stage("[INIT] Loading S2 calendar and parcel shapefile...")
    date_matcher = TemporalDateMatcher(calendar_path)
    gdf = gpd.read_file(parcels_path)
    log_stage(
        f"[INIT] Calendar dates: {len(date_matcher.s2_dates)} | "
        f"parcels in shapefile: {len(gdf)}"
    )

    if not stack_path.is_file():
        raise FileNotFoundError(f"Sentinel-2 stack not found: {stack_path}")

    with rasterio.open(stack_path) as src:
        parcels = filter_parcels_in_raster(gdf, src.crs, src.bounds)
        crs = src.crs

    log_stage(f"[INIT] Parcels intersecting S2 tile: {len(parcels)}")

    model = load_sr_model(MODEL_URL, model_dir, device)

    builder = ObservationDrivenDatasetBuilder(
        stack_path=stack_path,
        parcels=parcels,
        date_matcher=date_matcher,
        output_dir=output_dir,
        model=model,
        device=device,
        crs=crs,
    )
    lr_inventory, sr_inventory = builder.run()

    log_stage("[DATASET] Validating TorchGeoYellownessDataset (SR-side)...")
    dataset = TorchGeoYellownessDataset(
        inventory_csv=output_dir / "sr_inventory.csv",
        root_dir=output_dir,
        drop_missing_targets=True,
    )
    log_stage(f"[DATASET] Ready with {len(dataset)} labeled training samples.")

    if len(dataset) > 0:
        sample_tensor, sample_target = dataset[0]
        log_stage(
            f"[DATASET] Sample tensor shape: {tuple(sample_tensor.shape)} | "
            f"yellowness target: {float(sample_target):.4f}"
        )

    if not lr_inventory.empty:
        log_stage(f"[DATASET] LR inventory columns: {list(lr_inventory.columns)}")

    log_stage(
        f"Pipeline finished in {format_elapsed(time.perf_counter() - pipeline_start)}"
    )


if __name__ == "__main__":
    main()