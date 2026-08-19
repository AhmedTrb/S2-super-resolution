from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = _group_norm(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = _group_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = y + identity
        return self.act(y)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


@dataclass(frozen=True)
class SmallMaskedCNNConfig:
    image_channels: int = 10
    use_mask_channel: bool = True
    pooling_mode: str = "masked_avg"  # "masked_avg" or "global_avg"
    dropout: float = 0.2
    eps: float = 1e-6


class SmallMaskedCNNRegressor(nn.Module):
    """Small masked CNN regressor for Sentinel-2 patch yellowness prediction.

    Input image: [B, C, H, W]
    Input mask:  [B, 1, H, W]
    """

    def __init__(self, config: SmallMaskedCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallMaskedCNNConfig()
        in_channels = self.config.image_channels + 1 if self.config.use_mask_channel else self.config.image_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            _group_norm(24),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(24)

        self.down1 = DownsampleBlock(24, 48)
        self.res2 = ResidualBlock(48)

        self.down2 = DownsampleBlock(48, 64)
        self.res3 = ResidualBlock(64)

        self.head = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(32, 1),
        )

    def _masked_average_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resized_mask = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
        weighted = features * resized_mask
        numerator = weighted.sum(dim=(-2, -1))
        denominator = resized_mask.sum(dim=(-2, -1)).clamp_min(self.config.eps)
        pooled = numerator / denominator
        return pooled

    def _global_average_pool(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(-2, -1))

    def extract_features(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.use_mask_channel:
            x = torch.cat([image, mask], dim=1)
        else:
            x = image

        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)

        resized_mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        if self.config.pooling_mode == "masked_avg":
            pooled = self._masked_average_pool(x, mask)
        elif self.config.pooling_mode == "global_avg":
            pooled = self._global_average_pool(x)
        else:
            raise ValueError(f"Unsupported pooling_mode={self.config.pooling_mode}")

        area_fraction = resized_mask.mean(dim=(-2, -1))
        return pooled, area_fraction

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled, area_fraction = self.extract_features(image, mask)
        fused = torch.cat([pooled, area_fraction], dim=1)
        return self.head(fused).squeeze(-1)
