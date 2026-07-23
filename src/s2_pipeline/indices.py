from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np


class SpectralIndexComputer:
    """Compute parcel-only spectral-index statistics from a 10-band cube."""

    BAND_NAMES = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    BAND_INDEX = {name: idx for idx, name in enumerate(BAND_NAMES)}
    EPS = 1e-6

    def __init__(self, indices: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None) -> None:
        self.indices = indices or self._default_indices()

    @classmethod
    def _default_indices(cls) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        return {
            "NDRE": cls._index_ndre,
            "NDYVI": cls._index_ndyvi,
            "GCC": cls._index_gcc,
            "NDYI": cls._index_ndyi,
            "mNDblue": cls._index_mndblue,
        }

    @classmethod
    def _band(cls, cube: np.ndarray, name: str) -> np.ndarray:
        return cube[cls.BAND_INDEX[name]].astype(np.float32)

    @classmethod
    def _index_ndre(cls, cube: np.ndarray) -> np.ndarray:
        b8, b5 = cls._band(cube, "B08"), cls._band(cube, "B05")
        return (b8 - b5) / (b8 + b5 + cls.EPS)

    @classmethod
    def _index_ndyvi(cls, cube: np.ndarray) -> np.ndarray:
        b8, b4, b5 = cls._band(cube, "B08"), cls._band(cube, "B04"), cls._band(cube, "B05")
        summed = b4 + b5
        return (b8 - summed) / (b8 + summed + cls.EPS)

    @classmethod
    def _index_gcc(cls, cube: np.ndarray) -> np.ndarray:
        b3, b4, b2 = cls._band(cube, "B03"), cls._band(cube, "B04"), cls._band(cube, "B02")
        return b3 / (b4 + b3 + b2 + cls.EPS)

    @classmethod
    def _index_ndyi(cls, cube: np.ndarray) -> np.ndarray:
        b3, b2 = cls._band(cube, "B03"), cls._band(cube, "B02")
        return (b3 - b2) / (b3 + b2 + cls.EPS)

    @classmethod
    def _index_mndblue(cls, cube: np.ndarray) -> np.ndarray:
        b2, b6, b8 = cls._band(cube, "B02"), cls._band(cube, "B06"), cls._band(cube, "B08")
        return (b2 - b6) / (b2 + b8 + cls.EPS)

    def compute_stats(self, cube: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
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
            for name in self.indices:
                stats[f"{name}_mean"] = float("nan")
                stats[f"{name}_std"] = float("nan")
            return stats

        for name, fn in self.indices.items():
            values = fn(cube)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            parcel_values = values[valid]
            stats[f"{name}_mean"] = float(np.mean(parcel_values))
            stats[f"{name}_std"] = float(np.std(parcel_values))
        return stats
