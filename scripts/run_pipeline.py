#!/usr/bin/env python3
"""Convenience entry point for the modular Sentinel-2 parcel pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import rasterio
import torch
from shapely.geometry import box

from src.s2_pipeline import ObservationDrivenS2Pipeline, SuperResolutionProcessor, TemporalDateMatcher


def configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> None:
    configure_logging()
    repo_root = Path(__file__).resolve().parent

    stack_path = Path(r"C:/Users/ahtrabelsi/Desktop/stage/S2 data/Sentinel2_T31UEQ_TSG.tif")
    parcels_path = repo_root / "data" / "2020_buf20" / "2020_buf20.shp"
    calendar_path = repo_root / "data" / "Sentinel2_T31UEQ_interpolation_dates.txt"
    output_dir = repo_root / "torchgeo_yellowness_dataset"

    if not stack_path.is_file():
        raise FileNotFoundError(f"Sentinel-2 stack not found: {stack_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_matcher = TemporalDateMatcher(calendar_path)
    gdf = gpd.read_file(parcels_path)

    with rasterio.open(stack_path) as src:
        image_extent = box(*src.bounds)
        parcels = gdf.to_crs(src.crs)
        parcels = parcels[parcels.intersects(image_extent)].reset_index(drop=True)
        crs = src.crs

    # Placeholder model object: replace with your actual SEN2SRLite model wrapper if needed.
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    sr_processor = SuperResolutionProcessor(model=DummyModel().to(device), device=device)

    pipeline = ObservationDrivenS2Pipeline(
        stack_path=stack_path,
        parcels=parcels,
        date_matcher=date_matcher,
        output_dir=output_dir,
        sr_processor=sr_processor,
        crs=crs,
    )
    lr_inventory, sr_inventory = pipeline.run()
    print(f"Wrote {len(lr_inventory)} LR rows and {len(sr_inventory)} SR rows to {output_dir}")


if __name__ == "__main__":
    main()
