from __future__ import annotations

from typing import Tuple

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_bounds
from shapely.geometry.base import BaseGeometry


class ParcelMaskGenerator:
    """Create binary parcel masks aligned with LR and SR grids."""

    @staticmethod
    def build_transform(zone_bounds: Tuple[float, float, float, float], width: int, height: int) -> Affine:
        xmin, ymin, xmax, ymax = zone_bounds
        return from_bounds(xmin, ymin, xmax, ymax, width=width, height=height)

    @classmethod
    def create_masks(
        cls,
        zone_bounds: Tuple[float, float, float, float],
        parcel_geometry: BaseGeometry,
        lr_shape: Tuple[int, int],
        sr_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Affine, Affine]:
        h_lr, w_lr = lr_shape
        h_sr, w_sr = sr_shape

        lr_transform = cls.build_transform(zone_bounds, w_lr, h_lr)
        sr_transform = cls.build_transform(zone_bounds, w_sr, h_sr)

        shapes = [(parcel_geometry, 1)]
        lr_mask = rasterize(shapes, out_shape=(h_lr, w_lr), transform=lr_transform, fill=0, dtype=np.uint8)
        sr_mask = rasterize(shapes, out_shape=(h_sr, w_sr), transform=sr_transform, fill=0, dtype=np.uint8)
        return lr_mask, sr_mask, lr_transform, sr_transform
