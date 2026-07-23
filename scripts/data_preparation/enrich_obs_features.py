#!/usr/bin/env python3
"""Enrich observation inventory with advanced LR/SR parcel features.

This script reads an existing observation CSV (for example
`yellowness_dataset/observations_inventory.csv`), loads LR/SR patches,
rasterizes the exact parcel geometry from a shapefile using `id_plot`,
computes additional feature families, and writes an enriched CSV.

Feature families:
- vegetation indices (existing + additional)
- red-edge slopes
- red-edge curvature
- band differences and ratios
- optional texture features (GLCM on NIR and NDVI when scikit-image is available)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

repo_root = Path.cwd().resolve()
if not (repo_root / "src").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))

from s2_pipeline.pipeline import (
    compute_feature_block as pipeline_compute_feature_block,
    compute_spatial_extras as pipeline_compute_spatial_extras,
    extract_crop_metadata as pipeline_extract_crop_metadata,
)

EPS = 1e-6

BAND_NAMES = (
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
BAND_INDEX = {name: idx for idx, name in enumerate(BAND_NAMES)}

WAVELENGTHS_NM = {
    "B05": 705.0,
    "B06": 740.0,
    "B07": 783.0,
    "B8A": 865.0,
}

BASE_INDEX_DEFS = {
    "NDRE": lambda c: (b(c, "B08") - b(c, "B05")) / (b(c, "B08") + b(c, "B05") + EPS),
    "NDYVI": lambda c: (b(c, "B08") - (b(c, "B04") + b(c, "B05")))
    / (b(c, "B08") + b(c, "B04") + b(c, "B05") + EPS),
    "GCC": lambda c: b(c, "B03") / (b(c, "B04") + b(c, "B03") + b(c, "B02") + EPS),
    "NDYI": lambda c: (b(c, "B03") - b(c, "B02")) / (b(c, "B03") + b(c, "B02") + EPS),
    "mNDblue": lambda c: (b(c, "B02") - b(c, "B06")) / (b(c, "B02") + b(c, "B08") + EPS),
    "NDVI": lambda c: (b(c, "B08") - b(c, "B04")) / (b(c, "B08") + b(c, "B04") + EPS),
    "GNDVI": lambda c: (b(c, "B08") - b(c, "B03")) / (b(c, "B08") + b(c, "B03") + EPS),
    "NDMI": lambda c: (b(c, "B08") - b(c, "B11")) / (b(c, "B08") + b(c, "B11") + EPS),
    "NBR": lambda c: (b(c, "B08") - b(c, "B12")) / (b(c, "B08") + b(c, "B12") + EPS),
}

BAND_PAIR_FEATURES = (
    ("B08", "B04"),
    ("B8A", "B05"),
    ("B11", "B08"),
    ("B12", "B11"),
    ("B03", "B02"),
    ("B07", "B05"),
)

TEXTURE_PROPS = ("contrast", "homogeneity", "energy", "correlation", "ASM")

NON_FEATURE_COLUMNS = {
    "year",
    "stack_name",
    "parcel_index",
    "id_plot",
    "observation_column",
    "observation_date",
    "s2_date",
    "yellowness",
    "lr_patch",
    "sr_patch",
    "lr_mask",
    "sr_mask",
    "lr_image_path",
    "sr_image_path",
    "lr_mask_path",
    "sr_mask_path",
    "mask_path",
    "crop_code",
    "crop_group",
    "crop_name_1",
    "crop_name_2",
    "region",
    "parcel_ref_id",
    "source",
    "parcel_area_m2",
    "parcel_area_ha",
    "parcel_surface_declared",
    "s2_acquisition_date",
    "month",
    "day_of_year",
    "s2_date_difference_days",
}


def b(cube: np.ndarray, name: str) -> np.ndarray:
    return cube[BAND_INDEX[name]].astype(np.float32)


def normalize_reflectance(cube: np.ndarray) -> np.ndarray:
    raw_is_integer = np.issubdtype(cube.dtype, np.integer)
    cube = cube.astype(np.float32)
    if raw_is_integer or float(np.nanmax(cube)) > 2.0:
        cube = cube / 10000.0
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cube, 0.0, 1.0)


def canonical_id(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        as_int = int(value)
        return str(as_int) if float(as_int) == float(value) else str(value)
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def load_parcel_geometries(
    shapefile_path: Path,
    inventory: pd.DataFrame,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]], Any]:
    parcels = gpd.read_file(shapefile_path)

    id_candidates = ["id_plot", "ID_PARCEL", "PARCEL_ID", "id", "plot_id"]
    by_lower = {str(c).lower(): c for c in parcels.columns}

    available = []
    for candidate in id_candidates:
        col = by_lower.get(candidate.lower())
        if col is not None and col not in available:
            available.append(col)

    if not available:
        raise ValueError("Could not find a parcel id column in shapefile. Expected one of: " + ", ".join(id_candidates))

    inventory_ids = {canonical_id(v) for v in inventory["id_plot"].tolist()}

    best_col = None
    best_overlap = -1
    for col in available:
        parcel_ids = {canonical_id(v) for v in parcels[col].tolist()}
        overlap = len(inventory_ids & parcel_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    id_col = best_col
    print(f"[INFO] Using shapefile id column '{id_col}' ({best_overlap} matches with inventory id_plot)")
    if best_overlap == 0:
        print("[WARN] No overlap found between inventory id_plot values and shapefile id values.")

    geometries: Dict[str, object] = {}
    attributes: Dict[str, Dict[str, object]] = {}
    attribute_columns = [c for c in parcels.columns if c != "geometry"]

    for _, row in parcels.iterrows():
        key = canonical_id(row[id_col])
        if key:
            geometries[key] = row.geometry
            attributes[key] = {col: row[col] for col in attribute_columns}

    missing = sorted(inventory_ids - set(geometries.keys()))
    if missing:
        preview = ", ".join(missing[:10])
        print(f"[WARN] {len(missing)} id_plot values were not found in shapefile ids. First missing: {preview}")

    return geometries, attributes, parcels.crs


def add_selected_parcel_metadata(df: pd.DataFrame, attributes_by_id: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    if not attributes_by_id:
        return df

    selected_rows = []
    for value in df["id_plot"].tolist():
        attrs = attributes_by_id.get(canonical_id(value), {})
        selected_rows.append(
            {
                "crop_code": attrs.get("CODE_CULTU"),
                "crop_group": attrs.get("CODE_GROUP"),
                "crop_name_1": attrs.get("CULTURE_D1"),
                "crop_name_2": attrs.get("CULTURE_D2"),
                "region": attrs.get("REGION") or attrs.get("REGION_ext"),
                "parcel_ref_id": attrs.get("PARCEL_ID"),
                "source": attrs.get("source"),
                "parcel_area_m2": attrs.get("area_m2"),
                "parcel_area_ha": attrs.get("area_ha"),
                "parcel_surface_declared": attrs.get("SURF_PARC"),
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return df

    for column in selected_df.columns:
        if column not in df.columns:
            df[column] = selected_df[column].values
        else:
            df[column] = df[column].where(df[column].notna(), selected_df[column])

    return df


def drop_enrichment_columns_with_nan(df: pd.DataFrame, original_columns: Iterable[str]) -> pd.DataFrame:
    protected_columns = set(original_columns)
    new_columns = [column for column in df.columns if column not in protected_columns]
    columns_to_drop = [column for column in new_columns if df[column].isna().any()]
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop)


def add_static_date_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    date_source = None
    if "s2_acquisition_date" in result.columns:
        date_source = "s2_acquisition_date"
    elif "s2_date" in result.columns:
        date_source = "s2_date"

    if date_source is None:
        return result

    result["s2_acquisition_date"] = pd.to_datetime(result[date_source], errors="coerce")
    result["year"] = result.get("year", result["s2_acquisition_date"].dt.year)
    result["month"] = result["s2_acquisition_date"].dt.month
    result["day_of_year"] = result["s2_acquisition_date"].dt.dayofyear

    if "observation_date" in result.columns:
        obs_date = pd.to_datetime(result["observation_date"], errors="coerce")
        result["s2_date_difference_days"] = (result["s2_acquisition_date"] - obs_date).dt.days.abs()
    else:
        result["s2_date_difference_days"] = np.nan

    return result


def drop_temporal_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    temporal_suffixes = (
        "ndvi_slope_30d",
        "ndre_slope_30d",
        "ndvi_anomaly_30d",
        "ndre_anomaly_30d",
        "ndvi_min_30d",
        "ndre_min_30d",
    )
    temporal_columns = [
        column for column in df.columns if any(column.endswith(suffix) for suffix in temporal_suffixes)
    ]
    if not temporal_columns:
        return df
    return df.drop(columns=temporal_columns)


def _is_feature_column(column: str) -> bool:
    if column in NON_FEATURE_COLUMNS:
        return False
    if column.startswith(("calc_", "delta_", "abs_delta_", "lr_compare_", "sr_compare_")):
        return True
    if column.startswith(("lr_", "sr_")) and column not in NON_FEATURE_COLUMNS:
        return True
    return False


def drop_preexisting_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [column for column in df.columns if _is_feature_column(column)]
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def prepare_geometry_for_raster(
    geometry,
    geometry_crs: Any,
    raster_crs: Any,
    buffer_meters: float,
):
    prepared = geometry
    if geometry_crs is not None and raster_crs is not None and str(geometry_crs) != str(raster_crs):
        prepared = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(raster_crs).iloc[0]
    if buffer_meters != 0:
        prepared = prepared.buffer(buffer_meters)
    return prepared


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
        out[f"ratio_{left}_{right}"] = left_band / (right_band + EPS)
    return out


def maybe_texture_features(cube: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    try:
        from skimage.feature import graycomatrix, graycoprops
    except Exception:
        return {}

    def _quantize(img: np.ndarray, img_min: float, img_max: float, levels: int = 32) -> np.ndarray:
        scaled = (img - img_min) / (img_max - img_min + EPS)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * (levels - 1)).astype(np.uint8)

    def _crop_to_mask(img: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return img[y0:y1, x0:x1], m[y0:y1, x0:x1]

    def _glcm_props(img: np.ndarray, m: np.ndarray, prefix: str, lo: float, hi: float) -> Dict[str, float]:
        if not np.any(m):
            return {f"{prefix}_{p}": np.nan for p in TEXTURE_PROPS}

        img_crop, m_crop = _crop_to_mask(img, m)
        if float(np.mean(m_crop)) < 0.60:
            return {f"{prefix}_{p}": np.nan for p in TEXTURE_PROPS}

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
    ndvi = (b(cube, "B08") - b(cube, "B04")) / (b(cube, "B08") + b(cube, "B04") + EPS)

    out: Dict[str, float] = {}
    out.update(_glcm_props(nir, mask, "tex_nir", 0.0, 1.0))
    out.update(_glcm_props(ndvi, mask, "tex_ndvi", -1.0, 1.0))
    return out


def compute_feature_block(cube: np.ndarray, mask: np.ndarray, include_texture: bool) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for band_name in BAND_NAMES:
        features.update(compute_summary_stats(f"band_{band_name}", b(cube, band_name), mask))

    for idx_name, fn in BASE_INDEX_DEFS.items():
        features.update(compute_summary_stats(idx_name, fn(cube), mask))

    for name, arr in red_edge_features(cube).items():
        features.update(compute_summary_stats(name, arr, mask))

    for name, arr in pair_features(cube).items():
        features.update(compute_summary_stats(name, arr, mask))

    if include_texture:
        features.update(maybe_texture_features(cube, mask))

    return features


def resolve_patch_column(prefix: str, columns: Iterable[str]) -> Optional[str]:
    options = (f"{prefix}_patch", f"{prefix}_image_path", f"{prefix}_image")
    return next((c for c in options if c in columns), None)


def resolve_mask_column(prefix: str, columns: Iterable[str]) -> Optional[str]:
    options = (f"{prefix}_mask", f"{prefix}_mask_path", "mask_path")
    return next((c for c in options if c in columns), None)


def build_mask_for_row(
    raster: rasterio.io.DatasetReader,
    geometry,
    mask_path: Optional[Path],
) -> np.ndarray:
    geom_mask = geometry_mask(
        [geometry],
        out_shape=(raster.height, raster.width),
        transform=raster.transform,
        invert=True,
        all_touched=False,
    )

    if mask_path is None or not mask_path.exists():
        return geom_mask

    with rasterio.open(mask_path) as msrc:
        m = msrc.read(1) > 0
    if m.shape != geom_mask.shape:
        return geom_mask
    return geom_mask & m


def compute_row_features(
    row: pd.Series,
    prefix: str,
    root_dir: Path,
    geom_lookup: Dict[str, object],
    geometry_crs: Any,
    include_texture: bool,
    buffer_meters: float,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    patch_col = resolve_patch_column(prefix, row.index)
    if patch_col is None or pd.isna(row[patch_col]):
        return result

    key = canonical_id(row.get("id_plot"))
    geometry = geom_lookup.get(key)

    patch_path = root_dir / str(row[patch_col])
    if not patch_path.exists():
        return result

    mask_col = resolve_mask_column(prefix, row.index)
    mask_path = None
    if mask_col is not None and not pd.isna(row[mask_col]):
        mask_path = root_dir / str(row[mask_col])

    with rasterio.open(patch_path) as src:
        cube = normalize_reflectance(src.read())
        if geometry is not None:
            prepared_geometry = prepare_geometry_for_raster(
                geometry=geometry,
                geometry_crs=geometry_crs,
                raster_crs=src.crs,
                buffer_meters=buffer_meters,
            )
            parcel_mask = build_mask_for_row(src, prepared_geometry, mask_path)
        elif mask_path is not None and mask_path.exists():
            with rasterio.open(mask_path) as msrc:
                m = msrc.read(1) > 0
            if m.shape != (src.height, src.width):
                return result
            prepared_geometry = None
            parcel_mask = m
        else:
            prepared_geometry = None
            parcel_mask = np.ones((src.height, src.width), dtype=bool)

    valid_pixels = np.isfinite(cube).all(axis=0) & (cube > 0).any(axis=0)
    final_mask = parcel_mask & valid_pixels

    pixel_area = abs(src.transform.a * src.transform.e) if hasattr(src.transform, "a") else np.nan
    mask_pixel_count = int(np.sum(parcel_mask))
    valid_pixel_count = int(np.sum(final_mask))
    estimated_mask_area = float(mask_pixel_count * pixel_area) if np.isfinite(pixel_area) else np.nan
    buffered_geom_area = float(prepared_geometry.area) if prepared_geometry is not None else np.nan

    result[f"{prefix}_mask_pixels"] = mask_pixel_count
    result[f"{prefix}_valid_pixels"] = valid_pixel_count
    result[f"{prefix}_mask_fill_ratio"] = float(np.mean(parcel_mask))
    result[f"{prefix}_valid_fill_ratio"] = float(np.mean(final_mask))
    result[f"{prefix}_geometry_area"] = buffered_geom_area
    result[f"{prefix}_mask_area"] = estimated_mask_area
    result[f"{prefix}_mask_area_ratio"] = (
        float(estimated_mask_area / buffered_geom_area)
        if buffered_geom_area and buffered_geom_area > 0 and np.isfinite(estimated_mask_area)
        else np.nan
    )

    block = pipeline_compute_feature_block(cube, final_mask, include_texture=include_texture)
    for name, value in block.items():
        result[f"{prefix}_{name}"] = value

    spatial_block = pipeline_compute_spatial_extras(cube, final_mask, f"{prefix}_")
    for name, value in spatial_block.items():
        result[name] = value

    return result


def enrich_inventory(
    csv_path: Path,
    shapefile_path: Optional[Path],
    root_dir: Optional[Path],
    include_texture: bool,
    buffer_meters: float,
    recompute_all_from_scratch: bool,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if recompute_all_from_scratch:
        df = drop_preexisting_feature_columns(df)
    df = drop_temporal_feature_columns(df)

    data_root = root_dir if root_dir is not None else csv_path.parent
    geom_lookup: Dict[str, object] = {}
    attributes_by_id: Dict[str, Dict[str, object]] = {}
    geometry_crs = None

    if shapefile_path is not None:
        if "id_plot" not in df.columns:
            raise ValueError("Input CSV must include an 'id_plot' column when --shapefile is used.")
        geom_lookup, attributes_by_id, geometry_crs = load_parcel_geometries(shapefile_path, df)
        matched_ids = {canonical_id(parcel_id) for parcel_id in geom_lookup.keys()}
        before_filter = len(df)
        df = df[df["id_plot"].map(canonical_id).isin(matched_ids)].copy()
        if len(df) != before_filter:
            print(f"[INFO] Filtered inventory to {len(df)}/{before_filter} rows with matching parcel geometry")
        df = add_selected_parcel_metadata(df, attributes_by_id)
    else:
        print("[INFO] No shapefile provided: feature calculation will use raster masks only.")

    df = add_static_date_features(df)

    if "crop_code" not in df.columns:
        df["crop_code"] = df.apply(lambda row: pipeline_extract_crop_metadata(row).get("crop_code"), axis=1)
    if "crop_group" not in df.columns:
        df["crop_group"] = df.apply(lambda row: pipeline_extract_crop_metadata(row).get("crop_group"), axis=1)
    if "region" not in df.columns:
        df["region"] = df.apply(lambda row: pipeline_extract_crop_metadata(row).get("region"), axis=1)

    has_lr = resolve_patch_column("lr", df.columns) is not None
    has_sr = resolve_patch_column("sr", df.columns) is not None

    if not has_lr and not has_sr:
        raise ValueError("Could not find LR/SR patch columns in CSV. Expected lr_patch/sr_patch or lr_image_path/sr_image_path.")

    feature_rows = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 25 == 0 or i == total - 1:
            print(f"[INFO] Processing row {i + 1}/{total}")

        new_values: Dict[str, float] = {}
        if has_lr:
            new_values.update(
                compute_row_features(
                    row=row,
                    prefix="lr",
                    root_dir=data_root,
                    geom_lookup=geom_lookup,
                    geometry_crs=geometry_crs,
                    include_texture=include_texture,
                    buffer_meters=buffer_meters,
                )
            )
        if has_sr:
            new_values.update(
                compute_row_features(
                    row=row,
                    prefix="sr",
                    root_dir=data_root,
                    geom_lookup=geom_lookup,
                    geometry_crs=geometry_crs,
                    include_texture=include_texture,
                    buffer_meters=buffer_meters,
                )
            )

        feature_rows.append(new_values)

    features_df = pd.DataFrame(feature_rows)
    if not features_df.empty:
        for column in features_df.columns:
            df[column] = features_df[column].values

    df = df.loc[:, ~df.columns.duplicated()]

    if "s2_acquisition_date" not in df.columns and "s2_date" in df.columns:
        df["s2_acquisition_date"] = df["s2_date"]

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich LR/SR observation inventory with advanced parcel-level features.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("yellowness_dataset") / "observations_inventory.csv",
        help="Path to input observations CSV.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help="Optional parcel shapefile to intersect exact geometry by id_plot.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory to resolve patch and mask relative paths from CSV. Defaults to CSV parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input>_enriched.csv.",
    )
    parser.add_argument(
        "--include-texture",
        action="store_true",
        default=True,
        help="Compute optional GLCM texture features when scikit-image is installed.",
    )
    parser.add_argument(
        "--buffer-meters",
        type=float,
        default=0.0,
        help="Parcel buffer in CRS units applied before raster masking. Use 0 to keep the original parcel geometry.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV instead of writing a new file.",
    )
    parser.add_argument(
        "--from-scratch",
        dest="from_scratch",
        action="store_true",
        help="Drop existing feature columns and recompute all LR/SR features from raster patches and masks.",
    )
    parser.add_argument(
        "--keep-existing-features",
        dest="from_scratch",
        action="store_false",
        help="Keep existing feature columns and append/overwrite computed values.",
    )
    parser.set_defaults(from_scratch=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    shapefile_path = args.shapefile
    if shapefile_path is not None and not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    original_columns = list(pd.read_csv(csv_path, nrows=1).columns)

    enriched = enrich_inventory(
        csv_path=csv_path,
        shapefile_path=shapefile_path,
        root_dir=args.data_root,
        include_texture=args.include_texture,
        buffer_meters=args.buffer_meters,
        recompute_all_from_scratch=args.from_scratch,
    )

    if args.in_place:
        output_path = csv_path
    else:
        output_path = args.output
        if output_path is None:
            output_path = csv_path.with_name(f"{csv_path.stem}_enriched.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)

    print(f"[DONE] Enriched CSV written to: {output_path}")
    print(f"[DONE] Added {len(enriched.columns) - len(original_columns)} columns")


if __name__ == "__main__":
    main()
