from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import rasterio
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class PatchWindow:
    """Pixel window describing a parcel-centered patch in the raster grid."""

    parcel_index: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    zone: BaseGeometry

    @property
    def shape(self) -> Tuple[int, int]:
        return self.row_max - self.row_min, self.col_max - self.col_min


class ParcelPatchExtractor:
    """Build parcel-centered patch windows from vector geometries."""

    def __init__(self, min_patch_size: int = 256) -> None:
        self.min_patch_size = min_patch_size

    def compute_window(
        self,
        geom: BaseGeometry,
        img_height: int,
        img_width: int,
        raster_transform: Optional[Affine],
        raster_bounds: Optional[Tuple[float, float, float, float]] = None,
        parcel_index: int = -1,
    ) -> Optional[PatchWindow]:
        if raster_transform is None:
            return self._compute_window_from_bounds(
                geom=geom,
                img_height=img_height,
                img_width=img_width,
                raster_bounds=raster_bounds,
                parcel_index=parcel_index,
            )

        p_xmin, p_ymin, p_xmax, p_ymax = geom.bounds
        p_row_min, p_col_min = rasterio.transform.rowcol(raster_transform, p_xmin, p_ymax)
        p_row_max, p_col_max = rasterio.transform.rowcol(raster_transform, p_xmax, p_ymin)

        p_width = max(1, p_col_max - p_col_min)
        p_height = max(1, p_row_max - p_row_min)
        p_center_col = p_col_min + (p_width // 2)
        p_center_row = p_row_min + (p_height // 2)

        target_width = max(self.min_patch_size, p_width)
        target_height = max(self.min_patch_size, p_height)

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
            parcel_index=parcel_index,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
            zone=adjusted_zone,
        )

    def _compute_window_from_bounds(
        self,
        geom: BaseGeometry,
        img_height: int,
        img_width: int,
        raster_bounds: Optional[Tuple[float, float, float, float]],
        parcel_index: int,
    ) -> Optional[PatchWindow]:
        if raster_bounds is None:
            raster_bounds = (0.0, 0.0, float(img_width), float(img_height))

        xmin, ymin, xmax, ymax = raster_bounds
        width = max(1, int(xmax - xmin))
        height = max(1, int(ymax - ymin))

        geom_xmin, geom_ymin, geom_xmax, geom_ymax = geom.bounds
        center_x = (geom_xmin + geom_xmax) / 2.0
        center_y = (geom_ymin + geom_ymax) / 2.0
        p_width = max(1, int(geom_xmax - geom_xmin))
        p_height = max(1, int(geom_ymax - geom_ymin))

        target_width = max(self.min_patch_size, p_width)
        target_height = max(self.min_patch_size, p_height)
        col_min = int(round((center_x - xmin) / max(1.0, width / img_width) - target_width / 2.0))
        col_max = col_min + target_width
        row_min = int(round((ymax - center_y) / max(1.0, height / img_height) - target_height / 2.0))
        row_max = row_min + target_height

        col_min = max(0, min(col_min, img_width - target_width))
        col_max = min(img_width, max(col_min + target_width, col_max))
        row_min = max(0, min(row_min, img_height - target_height))
        row_max = min(img_height, max(row_min + target_height, row_max))

        zone = box(xmin, ymin, xmax, ymax)
        return PatchWindow(
            parcel_index=parcel_index,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
            zone=zone,
        )
