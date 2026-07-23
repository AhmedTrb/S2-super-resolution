#!/usr/bin/env python3
"""
Production multi-temporal dataset builder for TorchGeo foundation-model training.

Translates the verified sandbox logic from ``data-exploration.ipynb`` into a
headless, GPU-ready pipeline that:
  1. Matches irregular yellowness field observations to Sentinel-2 overpass dates.
  2. Runs SEN2SRLite super-resolution on cached LR patches.
  3. Rasterizes parcel masks aligned to SR grids (saved once per parcel).
  4. Writes georeferenced GeoTIFF assets and a CSV inventory ledger.
  5. Exposes a PyTorch ``Dataset`` for downstream model fine-tuning.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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

ImageryCache = Dict[str, Dict[int, Dict[str, Any]]]


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
        if isinstance(observation_date, str):
            obs = self._parse_date_token(observation_date)
        elif isinstance(observation_date, datetime):
            obs = observation_date.date()
        else:
            obs = observation_date

        nearest = min(self._s2_dates, key=lambda s2: abs((s2 - obs).days))
        return nearest.isoformat()

    def days_to_nearest_s2(self, observation_date: Union[date, datetime, str]) -> int:
        """Absolute day offset between an observation and its snapped S2 date."""
        if isinstance(observation_date, str):
            obs = self._parse_date_token(observation_date)
        elif isinstance(observation_date, datetime):
            obs = observation_date.date()
        else:
            obs = observation_date
        snapped = datetime.strptime(self.find_nearest_s2_date(obs), "%Y-%m-%d").date()
        return abs((snapped - obs).days)


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
# Patch extraction / imagery cache (verified notebook logic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchWindow:
    """Pixel window metadata reused across all S2 timestamps for one parcel."""

    parcel_index: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    zone: box
    parcel_row: pd.Series


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


def extract_patch_windows(
    parcels: gpd.GeoDataFrame,
    img_height: int,
    img_width: int,
    raster_transform: Affine,
    min_size: int = MIN_PATCH_SIZE,
) -> List[PatchWindow]:
    """
    Derive stable parcel-centered pixel windows (verified cell-7 logic).
    """
    windows: List[PatchWindow] = []

    for parcel_index, row in parcels.iterrows():
        geom = row.geometry
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
            logger.warning("Skipping parcel %s: window exceeds raster bounds.", row.get("ID_PARCEL"))
            continue

        ul_x, ul_y = rasterio.transform.xy(raster_transform, row_min, col_min, offset="ul")
        lr_x, lr_y = rasterio.transform.xy(raster_transform, row_max, col_max, offset="lr")
        adjusted_zone = box(ul_x, lr_y, lr_x, ul_y)

        windows.append(
            PatchWindow(
                parcel_index=int(parcel_index),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                zone=adjusted_zone,
                parcel_row=row,
            )
        )

    return windows


def build_imagery_cache_from_stack(
    stack_path: Union[str, Path],
    parcels: gpd.GeoDataFrame,
    date_matcher: TemporalDateMatcher,
) -> ImageryCache:
    """
    Build ``imagery_cache[date_str][parcel_index] = {"patch", "zone"}`` from a
    multi-temporal Sentinel-2 stack (10 bands per timestamp).
    """
    cache: ImageryCache = {}
    stack_path = Path(stack_path)
    phase_start = time.perf_counter()

    log_stage(f"[READ] Opening Sentinel-2 stack: {stack_path}")

    with rasterio.open(stack_path) as src:
        if src.count % BANDS_PER_IMAGE != 0:
            raise ValueError(
                f"Expected band count divisible by {BANDS_PER_IMAGE}, got {src.count}."
            )

        num_timestamps = src.count // BANDS_PER_IMAGE
        s2_dates = date_matcher.s2_date_strings
        if num_timestamps != len(s2_dates):
            log_stage(
                f"[READ] Warning: stack has {num_timestamps} timestamps but calendar "
                f"lists {len(s2_dates)} dates; using the minimum of both."
            )

        active_dates = s2_dates[:num_timestamps]

        log_stage(
            f"[READ] Raster metadata: {src.width}x{src.height}px | "
            f"{src.count} bands | {num_timestamps} timestamps | CRS={src.crs}"
        )

        log_stage("[READ] Computing parcel patch windows...")
        windows = extract_patch_windows(
            parcels=parcels,
            img_height=src.height,
            img_width=src.width,
            raster_transform=src.transform,
        )
        num_parcels = len(windows)
        total_reads = num_timestamps * num_parcels

        log_stage(
            f"[READ] Extracting {total_reads} LR patches "
            f"({len(active_dates)} dates x {num_parcels} parcels) — this is the slow step."
        )

        date_pbar = tqdm(
            enumerate(active_dates),
            total=len(active_dates),
            desc="S2 read dates",
            unit="date",
            dynamic_ncols=True,
        )

        for ts_idx, date_str in date_pbar:
            date_start = time.perf_counter()
            band_start = ts_idx * BANDS_PER_IMAGE + 1
            band_indexes = list(range(band_start, band_start + BANDS_PER_IMAGE))
            cache[date_str] = {}

            date_pbar.set_postfix(current=date_str, refresh=False)

            parcel_pbar = tqdm(
                windows,
                desc=f"  parcels {date_str}",
                unit="parcel",
                leave=False,
                dynamic_ncols=True,
            )

            for window in parcel_pbar:
                plot_id = window.parcel_row.get("ID_PARCEL", window.parcel_index)
                parcel_pbar.set_postfix(plot=plot_id, refresh=False)

                patch = src.read(
                    band_indexes,
                    window=rasterio.windows.Window(
                        col_off=window.col_min,
                        row_off=window.row_min,
                        width=window.col_max - window.col_min,
                        height=window.row_max - window.row_min,
                    ),
                )
                cache[date_str][window.parcel_index] = {
                    "patch": patch,
                    "zone": window.zone,
                    "target_parcel": window.parcel_row,
                }

            date_elapsed = time.perf_counter() - date_start
            tqdm.write(
                f"[READ] Finished {date_str} ({ts_idx + 1}/{len(active_dates)}) "
                f"— {num_parcels} parcels in {format_elapsed(date_elapsed)}"
            )

    total_elapsed = time.perf_counter() - phase_start
    log_stage(
        f"[READ] Imagery cache complete: {len(cache)} dates, "
        f"{num_parcels} parcels/date, total time {format_elapsed(total_elapsed)}"
    )
    return cache


# ---------------------------------------------------------------------------
# Yellowness matching helpers
# ---------------------------------------------------------------------------


def match_yellowness_to_s2_date(
    parcel_row: pd.Series,
    s2_date_str: str,
    observation_columns: Sequence[str],
    date_matcher: TemporalDateMatcher,
) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """
    For a parcel and S2 acquisition date, pick the nearest valid field
    observation and return ``(yellowness, observation_date, day_offset)``.
    """
    s2_date = datetime.strptime(s2_date_str, "%Y-%m-%d").date()
    best: Optional[Tuple[int, float, date, str]] = None

    for column in observation_columns:
        value = parcel_row.get(column)
        if pd.isna(value) or value == 0:
            continue

        obs_date = TemporalDateMatcher.parse_observation_column(column)
        offset = abs((s2_date - obs_date).days)
        candidate = (offset, float(value), obs_date, column)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return None, None, None

    offset_days, yellowness, obs_date, _ = best
    return yellowness, obs_date.isoformat(), offset_days


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
# MultiTemporalDatasetBuilder
# ---------------------------------------------------------------------------


class MultiTemporalDatasetBuilder:
    """
    Core orchestrator: iterate all S2 dates and parcels, run SR inference,
    persist GeoTIFF assets, and compile the inventory CSV.
    """

    def __init__(
        self,
        imagery_cache: ImageryCache,
        parcels: gpd.GeoDataFrame,
        date_matcher: TemporalDateMatcher,
        output_dir: Union[str, Path],
        model: torch.nn.Module,
        device: torch.device,
        crs: Any,
        observation_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.imagery_cache = imagery_cache
        self.parcels = parcels
        self.date_matcher = date_matcher
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images_sr"
        self.masks_dir = self.output_dir / "masks_sr"
        self.model = model
        self.device = device
        self.crs = crs
        self.observation_columns = observation_columns or discover_observation_columns(parcels)

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self._masks_written: set[str] = set()
        self._inventory_rows: List[Dict[str, Any]] = []

    @staticmethod
    def build_imagery_cache_from_stack(
        stack_path: Union[str, Path],
        parcels: gpd.GeoDataFrame,
        date_matcher: TemporalDateMatcher,
    ) -> ImageryCache:
        return build_imagery_cache_from_stack(stack_path, parcels, date_matcher)

    def run(self, s2_dates: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Process the full timeline. Returns the compiled inventory DataFrame.
        """
        timeline = list(s2_dates or self.date_matcher.s2_date_strings)
        phase_start = time.perf_counter()

        total_jobs = sum(
            len(self.imagery_cache.get(date_str, {}))
            for date_str in timeline
            if date_str in self.imagery_cache
        )
        skipped_dates = sum(1 for date_str in timeline if date_str not in self.imagery_cache)

        log_stage(
            f"[SR] Starting super-resolution on {self.device}: "
            f"{len(timeline)} dates, {total_jobs} parcel-date jobs"
        )
        if skipped_dates:
            log_stage(f"[SR] Warning: {skipped_dates} dates missing from imagery cache.")

        overall_pbar = tqdm(
            total=total_jobs,
            desc="SR total progress",
            unit="patch",
            dynamic_ncols=True,
        )

        for date_idx, date_str in enumerate(timeline, start=1):
            if date_str not in self.imagery_cache:
                tqdm.write(f"[SR] Skipping {date_str}: not found in imagery cache.")
                self._release_gpu_memory()
                continue

            date_cache = self.imagery_cache[date_str]
            date_start = time.perf_counter()
            tqdm.write(
                f"[SR] Date {date_idx}/{len(timeline)}: {date_str} "
                f"({len(date_cache)} parcels)"
            )

            parcel_pbar = tqdm(
                sorted(date_cache.items()),
                desc=f"SR {date_str}",
                unit="parcel",
                leave=False,
                dynamic_ncols=True,
            )

            for parcel_index, payload in parcel_pbar:
                plot_id = str(
                    payload.get("target_parcel", self.parcels.iloc[parcel_index]).get(
                        "ID_PARCEL", parcel_index
                    )
                )
                parcel_pbar.set_postfix(plot=plot_id, refresh=False)

                self._process_parcel_date(
                    date_str=date_str,
                    parcel_index=parcel_index,
                    payload=payload,
                )
                overall_pbar.update(1)
                overall_pbar.set_postfix(date=date_str, plot=plot_id, refresh=False)

            self._release_gpu_memory()
            date_elapsed = time.perf_counter() - date_start
            tqdm.write(
                f"[SR] Completed {date_str} in {format_elapsed(date_elapsed)} "
                f"| inventory rows so far: {len(self._inventory_rows)}"
            )

        overall_pbar.close()

        inventory = pd.DataFrame(self._inventory_rows)
        csv_path = self.output_dir / "dataset_inventory.csv"
        inventory.to_csv(csv_path, index=False)
        log_stage(
            f"[SR] Dataset compiled in {format_elapsed(time.perf_counter() - phase_start)}: "
            f"{len(inventory)} rows, {len(self._masks_written)} unique masks -> {csv_path}"
        )
        return inventory

    def _process_parcel_date(
        self,
        date_str: str,
        parcel_index: int,
        payload: Mapping[str, Any],
    ) -> None:
        lr_patch: np.ndarray = payload["patch"]
        zone = payload["zone"]
        parcel_row: pd.Series = payload.get("target_parcel", self.parcels.iloc[parcel_index])

        if lr_patch is None or lr_patch.size == 0:
            return

        plot_id = str(parcel_row["ID_PARCEL"])
        zone_bounds = zone.bounds
        xmin, ymin, xmax, ymax = zone_bounds

        sr_cube = run_super_resolution(lr_patch, self.model, self.device)
        _, h_lr, w_lr = lr_patch.shape
        bands_sr, h_sr, w_sr = sr_cube.shape

        _, sr_mask, _, sr_transform = SpatialMaskGenerator.rasterize_masks(
            zone_bounds=zone_bounds,
            parcel_geometry=parcel_row.geometry,
            lr_shape=(h_lr, w_lr),
            sr_shape=(h_sr, w_sr),
        )

        image_filename = f"image_plot_{plot_id}_date_{date_str}.tif"
        mask_filename = f"mask_plot_{plot_id}_static.tif"
        image_path = self.images_dir / image_filename
        mask_path = self.masks_dir / mask_filename

        image_profile = {
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
        write_geotiff(image_path, sr_cube, image_profile)

        if plot_id not in self._masks_written and not mask_path.exists():
            mask_profile = dict(image_profile)
            mask_profile.update({"count": 1, "dtype": "uint8"})
            write_geotiff(mask_path, sr_mask, mask_profile)
            self._masks_written.add(plot_id)

        yellowness, obs_date, day_offset = match_yellowness_to_s2_date(
            parcel_row=parcel_row,
            s2_date_str=date_str,
            observation_columns=self.observation_columns,
            date_matcher=self.date_matcher,
        )

        self._inventory_rows.append(
            {
                "plot_id": plot_id,
                "patch_index": parcel_index,
                "s2_acquisition_date": date_str,
                "image_path": os.path.relpath(image_path, self.output_dir),
                "mask_path": os.path.relpath(mask_path, self.output_dir),
                "crop_code": parcel_row.get("CODE_CULTU", "Unknown"),
                "crop_group": parcel_row.get("CODE_GROUP", -1),
                "surface_area_ha": float(parcel_row.get("area_ha", 0.0)),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "yellowness": yellowness,
                "yellowness_obs_date": obs_date,
                "yellowness_day_offset": day_offset,
            }
        )

    @staticmethod
    def _release_gpu_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# TorchGeoYellownessDataset
# ---------------------------------------------------------------------------


class TorchGeoYellownessDataset(Dataset):
    """
    PyTorch dataset that stacks a 10-band SR image with a 1-band attention mask
    into an 11-channel tensor for TorchGeo foundation-model fine-tuning.
    """

    def __init__(
        self,
        inventory_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        drop_missing_targets: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(inventory_csv).parent
        self.inventory = pd.read_csv(inventory_csv)

        date_col = "s2_acquisition_date"
        if date_col not in self.inventory.columns and "s2_image_date" in self.inventory.columns:
            self.inventory = self.inventory.rename(columns={"s2_image_date": date_col})

        if drop_missing_targets and "yellowness" in self.inventory.columns:
            self.inventory = self.inventory[self.inventory["yellowness"].notna()].reset_index(
                drop=True
            )

    def __len__(self) -> int:
        return len(self.inventory)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.inventory.iloc[index]

        image_path = self.root_dir / str(row["image_path"])
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
    log_stage("TorchGeo yellowness dataset pipeline")
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

    imagery_cache = MultiTemporalDatasetBuilder.build_imagery_cache_from_stack(
        stack_path=stack_path,
        parcels=parcels,
        date_matcher=date_matcher,
    )

    model = load_sr_model(MODEL_URL, model_dir, device)

    builder = MultiTemporalDatasetBuilder(
        imagery_cache=imagery_cache,
        parcels=parcels,
        date_matcher=date_matcher,
        output_dir=output_dir,
        model=model,
        device=device,
        crs=crs,
    )
    inventory = builder.run()

    log_stage("[DATASET] Validating TorchGeoYellownessDataset...")
    dataset = TorchGeoYellownessDataset(
        inventory_csv=output_dir / "dataset_inventory.csv",
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

    log_stage(
        f"Pipeline finished in {format_elapsed(time.perf_counter() - pipeline_start)}"
    )


if __name__ == "__main__":
    main()
