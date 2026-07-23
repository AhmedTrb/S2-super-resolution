from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import sen2sr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sen2sr = None


class SuperResolutionProcessor:
    """Wrap SR inference with a robust 0-1 normalization step for Sentinel-2 cubes."""

    def __init__(self, model: Any, device: torch.device, overlap: int = 32, upscale: int = 4) -> None:
        self.model = model
        self.device = device
        self.overlap = overlap
        self.upscale = upscale

    @staticmethod
    def is_model_compatible(height: int, width: int, model_multiple: int = 256) -> bool:
        return (height % model_multiple == 0) and (width % model_multiple == 0)

    @staticmethod
    def normalize_cube(cube: np.ndarray) -> np.ndarray:
        is_dn = np.issubdtype(cube.dtype, np.integer) or (np.issubdtype(cube.dtype, np.floating) and cube.max() > 10.0)
        cube = cube.astype(np.float32)
        cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
        if is_dn:
            cube = cube / 10_000.0
        return np.clip(cube, 0.0, 1.0)

    @classmethod
    def bicubic_upsample(cls, cube: np.ndarray, scale: int = 4) -> np.ndarray:
        tensor = torch.from_numpy(cube).to(torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        upsampled = F.interpolate(tensor, scale_factor=scale, mode="bicubic", align_corners=False)
        if upsampled.ndim == 4 and upsampled.shape[0] == 1:
            return upsampled.squeeze(0).numpy()
        return upsampled.numpy()

    def run(self, lr_patch: np.ndarray) -> np.ndarray:
        cube = self.normalize_cube(lr_patch)
        if cube.ndim == 3:
            height, width = cube.shape[1:]
        else:
            height, width = cube.shape[2:]

        if self.is_model_compatible(height, width):
            tensor = torch.from_numpy(cube).to(self.device)
            with torch.no_grad():
                if sen2sr is not None and hasattr(sen2sr, "predict_large"):
                    # sen2sr.predict_large expects CHW and internally adds the batch dim.
                    if tensor.ndim == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    sr = sen2sr.predict_large(model=self.model, X=tensor, overlap=self.overlap)
                elif hasattr(self.model, "__call__"):
                    if tensor.ndim == 3:
                        tensor = tensor.unsqueeze(0)
                    sr = self.model(tensor)
                else:
                    sr = None

            if sr is None:
                sr = self.bicubic_upsample(cube, self.upscale)
            elif isinstance(sr, torch.Tensor):
                sr = sr.cpu().numpy()
            else:
                sr = np.asarray(sr)

            if isinstance(sr, np.ndarray) and sr.ndim == 4 and sr.shape[0] == 1:
                sr = sr.squeeze(0)
        else:
            sr = self.bicubic_upsample(cube, self.upscale)

        return self.normalize_cube(sr.astype(np.float32))
