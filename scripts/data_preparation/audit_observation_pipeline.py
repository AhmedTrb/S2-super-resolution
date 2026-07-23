from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import mlstac
import numpy as np
import pandas as pd
import rasterio
import sen2sr
import torch
from rasterio.transform import Affine
from shapely.geometry import box

repo_root = Path.cwd().resolve()
if not (repo_root / "src").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))

from s2_pipeline import (
    ParcelMaskGenerator,
    ParcelPatchExtractor,
    SpectralIndexComputer,
    SuperResolutionProcessor,
    TemporalDateMatcher,
)


STACK_PATH = Path(r"C:/Users/ahtrabelsi/Desktop/stage/S2 data/Sentinel2_T31UEQ_TSG.tif")
PARCELS_PATH = repo_root / "data" / "2020_buf20" / "2020_buf20.shp"
CALENDAR_PATH = repo_root / "data" / "Sentinel2_T31UEQ_interpolation_dates.txt"
MODEL_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
MODEL_DIR = repo_root / "model" / "SEN2SRLite"
OUTPUT_DIR = repo_root / "outputs" / "audit_observation_pipeline"


def log(message: str) -> None:
    print(message)


def fmt_range(array: np.ndarray) -> str:
    if array.size == 0:
        return "empty"
    return f"min={float(np.nanmin(array)):.6f} max={float(np.nanmax(array)):.6f}"


def load_model(model_url: str, model_dir: Path, device: torch.device) -> torch.nn.Module:
    log(f"[MODEL] Loading SEN2SRLite into {model_dir} on {device}...")
    start = time.perf_counter()
    mlstac.download(file=model_url, output_dir=str(model_dir))
    model = mlstac.load(str(model_dir)).compiled_model(device=str(device))
    model.eval()
    log(f"[MODEL] Ready in {time.perf_counter() - start:.1f}s")
    return model


def write_geotiff(
    path: Path,
    array: np.ndarray,
    height: int,
    width: int,
    count: int,
    transform: Affine,
    crs: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(array.dtype),
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256 if count <= 1 else 512,
        "blockysize": 256 if count <= 1 else 512,
    }
    with rasterio.open(path, "w", **profile) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def read_back_stats(path: Path) -> Tuple[str, str, Tuple[int, ...]]:
    with rasterio.open(path) as src:
        arr = src.read()
        return src.dtypes[0], src.compression.name if src.compression else "none", arr.shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reflectance handling on four observation patches.")
    parser.add_argument("--limit", type=int, default=4, help="Number of valid observation patches to audit.")
    parser.add_argument("--save", action="store_true", help="Write audit GeoTIFFs and verify roundtrip reads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_URL, MODEL_DIR, device)
    matcher = TemporalDateMatcher(CALENDAR_PATH)
    patch_extractor = ParcelPatchExtractor(min_patch_size=256)
    indexer = SpectralIndexComputer()

    observations: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    with rasterio.open(STACK_PATH) as src:
        parcels = gpd.read_file(PARCELS_PATH).to_crs(src.crs)
        parcels = parcels[parcels.intersects(box(*src.bounds))].reset_index(drop=True)
        observation_columns = [col for col in parcels.columns if re.match(r"^\d{2}_\d{2}$", str(col))]

        for parcel_idx, parcel_row in parcels.iterrows():
            window = patch_extractor.compute_window(
                geom=parcel_row.geometry,
                img_height=src.height,
                img_width=src.width,
                raster_transform=src.transform,
                parcel_index=parcel_idx,
            )
            if window is None:
                continue

            for obs_col in observation_columns:
                value = parcel_row.get(obs_col)
                if pd.isna(value) or float(value) == 0:
                    continue

                obs_date = TemporalDateMatcher.parse_observation_column(obs_col)
                try:
                    s2_date = matcher.find_nearest_s2_date(obs_date)
                    band_offset = matcher.get_band_offset(s2_date)
                except ValueError:
                    continue

                band_start = band_offset * 10 + 1
                band_indexes = list(range(band_start, band_start + 10))
                lr_patch = src.read(
                    band_indexes,
                    window=rasterio.windows.Window(
                        col_off=window.col_min,
                        row_off=window.row_min,
                        width=window.col_max - window.col_min,
                        height=window.row_max - window.row_min,
                    ),
                )

                observations.append(
                    {
                        "parcel_index": int(parcel_idx),
                        "obs_col": obs_col,
                        "obs_date": obs_date.isoformat(),
                        "s2_date": s2_date,
                        "yellowness": float(value),
                        "geometry": parcel_row.geometry,
                        "window": window,
                        "lr_patch": lr_patch,
                        "crs": src.crs,
                    }
                )
                if len(observations) >= args.limit:
                    break
            if len(observations) >= args.limit:
                break

    if len(observations) < args.limit:
        raise RuntimeError(f"Only found {len(observations)} valid observations, expected {args.limit}.")

    log(f"[AUDIT] Selected {len(observations)} observation patches.")

    for sample_idx, sample in enumerate(observations, start=1):
        lr_patch = sample["lr_patch"]
        window = sample["window"]
        crs = sample["crs"]

        lr_reflectance = lr_patch.astype(np.float32) / 10_000.0
        lr_reflectance = np.nan_to_num(lr_reflectance, nan=0.0, posinf=0.0, neginf=0.0)
        lr_reflectance = np.clip(lr_reflectance, 0.0, 1.0)

        # Build SR input from the same normalized reflectance cube that the model sees.
        X = torch.from_numpy(lr_reflectance).to(device)
        with torch.no_grad():
            sr_raw = sen2sr.predict_large(model=model, X=X, overlap=32)
        if isinstance(sr_raw, torch.Tensor):
            sr_raw = sr_raw.cpu().numpy()
        sr_raw = np.asarray(sr_raw, dtype=np.float32)
        sr_corrected = SuperResolutionProcessor.normalize_cube(sr_raw)

        lr_mask, sr_mask, lr_transform, sr_transform = ParcelMaskGenerator.create_masks(
            parcel_geometry=sample["geometry"],
            zone_bounds=window.zone.bounds,
            lr_shape=lr_reflectance.shape[1:],
            sr_shape=sr_corrected.shape[1:],
        )

        lr_stats = indexer.compute_stats(lr_reflectance, lr_mask)
        sr_stats = indexer.compute_stats(sr_corrected, sr_mask)
        mean_deltas = {name: float(sr_stats[f"{name}_mean"] - lr_stats[f"{name}_mean"]) for name in indexer.indices}

        log(f"\n[AUDIT #{sample_idx}] parcel={sample['parcel_index']} obs={sample['obs_col']} s2={sample['s2_date']}")
        log(f"  raw LR DN: {fmt_range(lr_patch)} dtype={lr_patch.dtype}")
        log(f"  LR reflectance normalized: {fmt_range(lr_reflectance)} dtype={lr_reflectance.dtype}")
        log(f"  LR mask unique: {np.unique(lr_mask).tolist()}")
        log(f"  SR raw model output: {fmt_range(sr_raw)} dtype={sr_raw.dtype}")
        log(f"  SR corrected/clipped: {fmt_range(sr_corrected)} dtype={sr_corrected.dtype}")
        log(f"  SR mask unique: {np.unique(sr_mask).tolist()}")
        log(f"  mean deltas (SR-LR) on parcel mask: {mean_deltas}")

        if args.save:
            stem = f"audit_{sample_idx:02d}_parcel_{sample['parcel_index']:05d}_{sample['obs_col']}"
            lr_path = output_dir / f"{stem}_lr.tif"
            sr_path = output_dir / f"{stem}_sr.tif"
            lr_mask_path = output_dir / f"{stem}_lr_mask.tif"
            sr_mask_path = output_dir / f"{stem}_sr_mask.tif"

            write_geotiff(
                lr_path,
                lr_reflectance,
                lr_reflectance.shape[1],
                lr_reflectance.shape[2],
                lr_reflectance.shape[0],
                lr_transform,
                crs,
            )
            write_geotiff(
                sr_path,
                sr_corrected,
                sr_corrected.shape[1],
                sr_corrected.shape[2],
                sr_corrected.shape[0],
                sr_transform,
                crs,
            )
            write_geotiff(lr_mask_path, lr_mask.astype(np.uint8), lr_mask.shape[0], lr_mask.shape[1], 1, lr_transform, crs)
            write_geotiff(sr_mask_path, sr_mask.astype(np.uint8), sr_mask.shape[0], sr_mask.shape[1], 1, sr_transform, crs)

            lr_dtype, lr_compression, lr_shape = read_back_stats(lr_path)
            sr_dtype, sr_compression, sr_shape = read_back_stats(sr_path)
            audit_rows.append(
                {
                    "sample_idx": sample_idx,
                    "parcel_index": sample["parcel_index"],
                    "obs_col": sample["obs_col"],
                    "lr_min": float(np.min(lr_reflectance)),
                    "lr_max": float(np.max(lr_reflectance)),
                    "sr_raw_min": float(np.min(sr_raw)),
                    "sr_raw_max": float(np.max(sr_raw)),
                    "sr_corr_min": float(np.min(sr_corrected)),
                    "sr_corr_max": float(np.max(sr_corrected)),
                    "lr_dtype": lr_dtype,
                    "sr_dtype": sr_dtype,
                    "lr_compression": lr_compression,
                    "sr_compression": sr_compression,
                    "lr_shape": str(lr_shape),
                    "sr_shape": str(sr_shape),
                }
            )

    if args.save:
        report = pd.DataFrame(audit_rows)
        report_path = output_dir / "audit_report.csv"
        report.to_csv(report_path, index=False)
        log(f"\n[AUDIT] Saved audit files to {output_dir}")
        log(f"[AUDIT] Report written to {report_path}")


if __name__ == "__main__":
    main()