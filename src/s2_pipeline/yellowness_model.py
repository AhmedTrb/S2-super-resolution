from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class MaskFusionMode(str, Enum):
    IMAGE_ONLY = "image_only"
    MASKED_IMAGE = "masked_image"
    MASK_AS_CHANNEL = "mask_as_channel"
    IMAGE_AND_MASK = "image_and_mask"
    LATE_MASK = "late_mask"
    FEATURE_MASK_POOL = "feature_mask_pool"
    CROP_MASK_BBOX = "crop_mask_bbox"


class PoolingMode(str, Enum):
    GLOBAL_AVG = "global_avg"
    MASKED_AVG = "masked_avg"
    MASKED_MEAN_STD = "masked_mean_std"
    GLOBAL_MAX = "global_max"
    GEM = "gem"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    in_channels: int
    feature_dim: Optional[int] = None
    pretrained: bool = True


def _infer_torchgeo_model_name_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "resnet50"
    if weight_name.startswith("ResNet152_Weights."):
        return "resnet152"
    if weight_name.startswith("ViTSmall16_Weights."):
        return "vit_small_patch16_224"
    if weight_name.startswith("ViTBase16_Weights."):
        return "vit_base_patch16_224"
    if weight_name.startswith("ViTSmall14_DINOv2_Weights."):
        return "vit_small_patch14_dinov2"
    if weight_name.startswith("ViTBase14_DINOv2_Weights."):
        return "vit_base_patch14_dinov2"
    if weight_name.startswith("Swin_V2_T_Weights."):
        return "swin_v2_t"
    if weight_name.startswith("Swin_V2_B_Weights."):
        return "swin_v2_b"
    if weight_name.startswith("FGMAEEarthLoc_Weights."):
        if "RESNET50" in weight_name.upper():
            return "resnet50"
    raise ValueError(
        "Could not infer TorchGeo model name from weight enum. "
        "Pass an explicit backbone like 'torchgeo:resnet50'."
    )


class SimpleCNNBackbone(nn.Module):
    """Small fallback backbone for local experimentation when TorchGeo is unavailable."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TorchGeoBackboneAdapter(nn.Module):
    """Load a TorchGeo backbone lazily and expose a consistent feature interface."""

    def __init__(self, spec: BackboneSpec, torchgeo_weight: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        self.spec = spec
        self.model = self._build_model(spec, torchgeo_weight=torchgeo_weight, **kwargs)
        inferred_input_channels = self._infer_model_input_channels()
        self.input_channels = int(
            inferred_input_channels
            if inferred_input_channels is not None
            else getattr(self.model, "in_chans", getattr(self.model, "in_channels", spec.in_channels))
        )
        self.input_adapter = self._build_input_adapter(spec.in_channels)
        self.resize_size = self._infer_model_input_size()

    def _build_input_adapter(self, in_channels: int) -> nn.Module:
        expected_channels = self._infer_model_input_channels()
        if expected_channels is None:
            expected_channels = 13
        return nn.Identity()

    def _infer_model_input_channels(self) -> Optional[int]:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None:
            proj = getattr(patch_embed, "proj", None)
            if proj is not None and hasattr(proj, "in_channels"):
                return int(proj.in_channels)
        if hasattr(self.model, "in_channels"):
            return int(self.model.in_channels)
        return None

    def _infer_model_input_size(self) -> Optional[Tuple[int, int]]:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None:
            img_size = getattr(patch_embed, "img_size", None)
            if img_size is not None:
                if isinstance(img_size, (tuple, list)):
                    return (int(img_size[0]), int(img_size[1]))
                return (int(img_size), int(img_size))
        img_size = getattr(self.model, "img_size", None)
        if img_size is not None:
            if isinstance(img_size, (tuple, list)):
                return (int(img_size[0]), int(img_size[1]))
            return (int(img_size), int(img_size))
        return None

    @staticmethod
    def _build_model(spec: BackboneSpec, torchgeo_weight: Optional[str] = None, **kwargs: Any) -> nn.Module:
        try:
            import torchgeo.models as torchgeo_models
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TorchGeo is required to use TorchGeo backbones. Install `torchgeo` first."
            ) from exc

        if torchgeo_weight:
            model_name = spec.name
            if model_name == "auto":
                model_name = _infer_torchgeo_model_name_from_weight(torchgeo_weight)

            try:
                weight_enum = torchgeo_models.get_weight(torchgeo_weight)
            except Exception as exc:
                raise ValueError(f"Unknown TorchGeo weight enum '{torchgeo_weight}'.") from exc

            attempts = [
                {"weights": weight_enum, **kwargs},
                {"weights": weight_enum, "in_chans": spec.in_channels, **kwargs},
                {"weights": weight_enum, "in_chans": 13, **kwargs},
            ]
            last_error: Optional[Exception] = None
            for attempt in attempts:
                try:
                    return torchgeo_models.get_model(model_name, **attempt)
                except Exception as exc:  # pragma: no cover - TorchGeo model signatures differ
                    last_error = exc
            raise RuntimeError(
                f"Could not initialize TorchGeo model '{model_name}' with weights '{torchgeo_weight}'."
            ) from last_error

        constructor = getattr(torchgeo_models, spec.name, None)
        if constructor is None:
            available = sorted(name for name in dir(torchgeo_models) if not name.startswith("_"))
            raise ValueError(f"Unknown TorchGeo model '{spec.name}'. Available examples: {available[:20]}")

        attempts = [
            {"weights": "IMAGENET", **kwargs} if spec.pretrained else kwargs,
            {"pretrained": spec.pretrained, **kwargs},
            kwargs,
        ]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return constructor(**attempt)
            except Exception as exc:  # pragma: no cover - constructor compatibility varies by model
                last_error = exc
        raise RuntimeError(f"Could not initialize TorchGeo model '{spec.name}'.") from last_error

    def forward(self, x: torch.Tensor) -> Any:
        if self.resize_size is not None and x.shape[-2:] != self.resize_size:
            x = torch.nn.functional.interpolate(x, size=self.resize_size, mode="bilinear", align_corners=False)
        if x.shape[1] < self.input_channels:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.input_channels - x.shape[1]))
        elif x.shape[1] > self.input_channels:
            x = x[:, : self.input_channels, ...]
        x = self.input_adapter(x)
        x = x.contiguous()
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        return self.model(x)

    def get_finetune_stages(self) -> list[nn.Module]:
        model = self.model

        blocks = getattr(model, "blocks", None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
            return list(blocks)

        layers = getattr(model, "layers", None)
        if isinstance(layers, (nn.ModuleList, nn.Sequential)) and len(layers) > 0:
            return list(layers)

        resnet_stages: list[nn.Module] = []
        for name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(model, name, None)
            if isinstance(stage, nn.Module):
                resnet_stages.append(stage)
        if resnet_stages:
            return resnet_stages

        children = list(model.children())
        return children if children else [model]


class MaskFeatureEncoder(nn.Module):
    """Encode simple parcel-mask summary statistics for late fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 4

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        spatial = tuple(range(2, mask.ndim))
        coverage = mask.mean(dim=spatial)
        density = (mask > 0.5).float().mean(dim=spatial)
        row_profile = mask.mean(dim=3).std(dim=2)
        col_profile = mask.mean(dim=2).std(dim=2)
        return torch.cat([coverage, density, row_profile, col_profile], dim=1)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 64), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MaskedYellownessRegressor(nn.Module):
    """Predict parcel yellowness percentage from a Sentinel-2 patch and binary parcel mask.

    The model is modular along three axes:
    - backbone choice: TorchGeo backbone or a local fallback CNN
    - mask fusion mode: early fusion, masked image, or late fusion
    - regression head: inferred automatically from backbone output size
    """

    def __init__(
        self,
        backbone_name: str = "simple_cnn",
        image_channels: int = 10,
        mask_fusion: MaskFusionMode = MaskFusionMode.FEATURE_MASK_POOL,
        pooling_mode: PoolingMode = PoolingMode.GLOBAL_AVG,
        torchgeo_weight: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        sample_patch_size: int = 256,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.2,
        output_range: Tuple[float, float] = (0.0, 100.0),
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        backbone_band_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.mask_fusion = MaskFusionMode(mask_fusion)
        self.pooling_mode = PoolingMode(pooling_mode)
        self.output_range = output_range
        self.mask_encoder = MaskFeatureEncoder()
        self.backbone_band_indices = tuple(int(index) for index in backbone_band_indices) if backbone_band_indices is not None else None
        self.gem_p = nn.Parameter(torch.tensor(3.0), requires_grad=False)

        backbone_kwargs = backbone_kwargs or {}
        fused_channels = self._fused_input_channels(image_channels, self.mask_fusion)
        backbone_in_channels = image_channels if backbone_name.startswith("torchgeo:") else fused_channels

        if backbone_name.startswith("torchgeo:"):
            torchgeo_name = backbone_name.split(":", 1)[1]
            spec = BackboneSpec(name=torchgeo_name, in_channels=backbone_in_channels, pretrained=pretrained)
            self.backbone = TorchGeoBackboneAdapter(spec, torchgeo_weight=torchgeo_weight, **backbone_kwargs)
            backbone_in_channels = getattr(self.backbone, "input_channels", backbone_in_channels)
        elif backbone_name == "simple_cnn":
            self.backbone = SimpleCNNBackbone(in_channels=backbone_in_channels)
        else:
            raise ValueError(
                "Unsupported backbone_name. Use 'simple_cnn' or 'torchgeo:<ModelName>'."
            )

        self.input_adapter = (
            nn.Identity()
            if fused_channels == backbone_in_channels
            else nn.Conv2d(fused_channels, backbone_in_channels, kernel_size=1)
        )

        feature_dim = self._infer_feature_dim(sample_patch_size)
        self.regression_head = RegressionHead(feature_dim, hidden_dims=hidden_dims, dropout=dropout)

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def unfreeze_last_backbone_stages(self, num_stages: int) -> None:
        if num_stages <= 0:
            self.set_backbone_trainable(True)
            return

        self.set_backbone_trainable(False)
        if hasattr(self.backbone, "get_finetune_stages"):
            stages = self.backbone.get_finetune_stages()
        else:
            children = list(self.backbone.children())
            stages = children if children else [self.backbone]

        for stage in stages[-num_stages:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _fused_input_channels(image_channels: int, mode: MaskFusionMode) -> int:
        if mode in {
            MaskFusionMode.IMAGE_ONLY,
            MaskFusionMode.MASKED_IMAGE,
            MaskFusionMode.LATE_MASK,
            MaskFusionMode.FEATURE_MASK_POOL,
            MaskFusionMode.CROP_MASK_BBOX,
        }:
            return image_channels
        if mode == MaskFusionMode.MASK_AS_CHANNEL:
            return image_channels + 1
        if mode == MaskFusionMode.IMAGE_AND_MASK:
            return (image_channels * 2) + 1
        raise ValueError(f"Unsupported mask fusion mode: {mode}")

    @staticmethod
    def _crop_to_mask_bbox(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        cropped_images: list[torch.Tensor] = []
        cropped_masks: list[torch.Tensor] = []

        for index in range(image.shape[0]):
            sample_image = image[index : index + 1]
            sample_mask = mask[index : index + 1]
            positive = torch.nonzero(sample_mask[0, 0] > 0.5, as_tuple=False)
            if positive.numel() == 0:
                cropped_images.append(sample_image)
                cropped_masks.append(sample_mask)
                continue

            y_min = int(positive[:, 0].min().item())
            y_max = int(positive[:, 0].max().item()) + 1
            x_min = int(positive[:, 1].min().item())
            x_max = int(positive[:, 1].max().item()) + 1

            sample_image = sample_image[..., y_min:y_max, x_min:x_max]
            sample_mask = sample_mask[..., y_min:y_max, x_min:x_max]
            sample_image = nn.functional.interpolate(sample_image, size=(height, width), mode="bilinear", align_corners=False)
            sample_mask = nn.functional.interpolate(sample_mask, size=(height, width), mode="nearest")
            cropped_images.append(sample_image)
            cropped_masks.append(sample_mask)

        return torch.cat(cropped_images, dim=0), torch.cat(cropped_masks, dim=0)

    def _prepare_backbone_input(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.backbone_band_indices is not None:
            image = image[:, list(self.backbone_band_indices), ...]
        image = image.contiguous()

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=image.dtype).contiguous()
        pooling_mask: Optional[torch.Tensor] = mask

        if self.mask_fusion == MaskFusionMode.IMAGE_ONLY:
            fused = image
            aux = None
            feature_mask = None
        elif self.mask_fusion == MaskFusionMode.MASKED_IMAGE:
            fused = image * mask
            aux = None
            feature_mask = None
        elif self.mask_fusion == MaskFusionMode.MASK_AS_CHANNEL:
            fused = torch.cat([image, mask], dim=1)
            aux = None
            feature_mask = None
        elif self.mask_fusion == MaskFusionMode.IMAGE_AND_MASK:
            fused = torch.cat([image, image * mask, mask], dim=1)
            aux = None
            feature_mask = None
        elif self.mask_fusion == MaskFusionMode.LATE_MASK:
            fused = image
            aux = self.mask_encoder(mask)
            feature_mask = None
        elif self.mask_fusion == MaskFusionMode.FEATURE_MASK_POOL:
            fused = image
            aux = None
            feature_mask = mask
        elif self.mask_fusion == MaskFusionMode.CROP_MASK_BBOX:
            fused, pooling_mask = self._crop_to_mask_bbox(image, mask)
            aux = None
            feature_mask = None
        else:
            raise ValueError(f"Unsupported mask fusion mode: {self.mask_fusion}")

        return self.input_adapter(fused).contiguous(), aux, feature_mask, pooling_mask

    def _apply_feature_mask(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            resized_mask = nn.functional.interpolate(mask, size=features.shape[-2:], mode="nearest")
            return (features * resized_mask).contiguous()

        if features.ndim == 3 and features.shape[1] >= features.shape[2]:
            num_tokens = features.shape[1]
            spatial_tokens = features
            cls_token: Optional[torch.Tensor] = None

            side = int(round((num_tokens - 1) ** 0.5))
            if side * side == (num_tokens - 1):
                cls_token = features[:, :1, :]
                spatial_tokens = features[:, 1:, :]
            else:
                side = int(round(num_tokens ** 0.5))
                if side * side != num_tokens:
                    return features

            resized_mask = nn.functional.interpolate(mask, size=(side, side), mode="nearest")
            flat_mask = resized_mask.flatten(2).transpose(1, 2)
            masked_tokens = spatial_tokens * flat_mask
            if cls_token is not None:
                return torch.cat([cls_token, masked_tokens], dim=1).contiguous()
            return masked_tokens.contiguous()

        return features

    def _extract_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("features", "out", "logits", "last_hidden_state", "encoder_output"):
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        if isinstance(output, (tuple, list)):
            for value in reversed(output):
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError("Backbone output could not be converted to a tensor feature map.")

    def _mask_for_token_features(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = features.shape[1]
        spatial_tokens = features
        cls_token: Optional[torch.Tensor] = None

        side = int(round((num_tokens - 1) ** 0.5))
        if side * side == (num_tokens - 1):
            cls_token = features[:, :1, :]
            spatial_tokens = features[:, 1:, :]
        else:
            side = int(round(num_tokens ** 0.5))
            if side * side != num_tokens:
                return features, None, None

        resized_mask = nn.functional.interpolate(mask, size=(side, side), mode="nearest")
        flat_mask = resized_mask.flatten(2).transpose(1, 2)
        return spatial_tokens, flat_mask, cls_token

    def _masked_reduce_4d(self, features: torch.Tensor, mask: torch.Tensor, include_std: bool = False) -> torch.Tensor:
        resized_mask = nn.functional.interpolate(mask, size=features.shape[-2:], mode="nearest")
        weights = resized_mask.clamp_min(0.0)
        denom = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        mean = (features * weights).sum(dim=(2, 3), keepdim=True) / denom
        mean_flat = mean.flatten(1)
        if not include_std:
            return mean_flat

        variance = ((features - mean) ** 2 * weights).sum(dim=(2, 3), keepdim=True) / denom
        std_flat = torch.sqrt(variance.clamp_min(1e-6)).flatten(1)
        return torch.cat([mean_flat, std_flat], dim=1)

    def _masked_reduce_tokens(self, features: torch.Tensor, mask: torch.Tensor, include_std: bool = False) -> torch.Tensor:
        spatial_tokens, flat_mask, cls_token = self._mask_for_token_features(features, mask)
        if flat_mask is None:
            if include_std:
                mean = features.mean(dim=1)
                std = features.std(dim=1, unbiased=False)
                return torch.cat([mean, std], dim=1)
            return features.mean(dim=1)

        weights = flat_mask.clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean = (spatial_tokens * weights).sum(dim=1) / denom.squeeze(1)
        if cls_token is not None:
            mean = 0.5 * (mean + cls_token.squeeze(1))
        if not include_std:
            return mean

        centered = spatial_tokens - mean.unsqueeze(1)
        variance = (centered.pow(2) * weights).sum(dim=1) / denom.squeeze(1)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)

    def _gem_pool_4d(self, features: torch.Tensor) -> torch.Tensor:
        p = float(self.gem_p.item())
        pooled = nn.functional.adaptive_avg_pool2d(features.clamp_min(1e-6).pow(p), output_size=1)
        return pooled.pow(1.0 / p).flatten(1)

    def _gem_pool_tokens(self, features: torch.Tensor) -> torch.Tensor:
        p = float(self.gem_p.item())
        if features.shape[-1] >= features.shape[1]:
            return features.clamp_min(1e-6).pow(p).mean(dim=1).pow(1.0 / p)
        return features.clamp_min(1e-6).pow(p).mean(dim=-1).pow(1.0 / p)

    def _pool_features(self, features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        features = features.contiguous()
        if self.pooling_mode == PoolingMode.MASKED_AVG and mask is not None:
            if features.ndim == 4:
                return self._masked_reduce_4d(features, mask, include_std=False)
            if features.ndim == 3:
                return self._masked_reduce_tokens(features, mask, include_std=False)

        if self.pooling_mode == PoolingMode.MASKED_MEAN_STD and mask is not None:
            if features.ndim == 4:
                return self._masked_reduce_4d(features, mask, include_std=True)
            if features.ndim == 3:
                return self._masked_reduce_tokens(features, mask, include_std=True)

        if features.ndim == 4:
            if self.pooling_mode == PoolingMode.GLOBAL_MAX:
                return torch.flatten(nn.functional.adaptive_max_pool2d(features, output_size=1), 1)
            if self.pooling_mode == PoolingMode.GEM:
                return self._gem_pool_4d(features)
            return torch.flatten(nn.functional.adaptive_avg_pool2d(features, output_size=1), 1)
        if features.ndim == 3:
            if self.pooling_mode == PoolingMode.GLOBAL_MAX:
                if features.shape[-1] >= features.shape[1]:
                    return features.max(dim=1).values
                return features.max(dim=-1).values
            if self.pooling_mode == PoolingMode.GEM:
                return self._gem_pool_tokens(features)
            if features.shape[-1] >= features.shape[1]:
                return features.mean(dim=1)
            return features.mean(dim=-1)
        if features.ndim == 2:
            return features
        if features.ndim == 1:
            return features.unsqueeze(0)
        return torch.flatten(features, start_dim=1)

    def extract_features(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        backbone_input, aux_features, feature_mask, pooling_mask = self._prepare_backbone_input(image, mask)
        raw_features = self.backbone(backbone_input)
        tensor_features = self._extract_tensor(raw_features)
        if feature_mask is not None:
            tensor_features = self._apply_feature_mask(tensor_features, feature_mask)
        pooled_features = self._pool_features(tensor_features, pooling_mask)
        if aux_features is not None:
            pooled_features = torch.cat([pooled_features, aux_features], dim=1)
        return pooled_features

    def debug_feature_shapes(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        """Return key tensor shapes for debugging feature extraction and reduction."""
        with torch.no_grad():
            backbone_input, aux_features, feature_mask, pooling_mask = self._prepare_backbone_input(image, mask)
            raw_features = self.backbone(backbone_input)
            tensor_features = self._extract_tensor(raw_features)
            masked_features = self._apply_feature_mask(tensor_features, feature_mask) if feature_mask is not None else tensor_features
            pooled_features = self._pool_features(masked_features, pooling_mask)
            final_features = torch.cat([pooled_features, aux_features], dim=1) if aux_features is not None else pooled_features

        return {
            "image_shape": tuple(image.shape),
            "mask_shape": tuple(mask.shape),
            "backbone_input_shape": tuple(backbone_input.shape),
            "raw_backbone_shape": tuple(tensor_features.shape),
            "masked_backbone_shape": tuple(masked_features.shape),
            "pooled_shape": tuple(pooled_features.shape),
            "final_feature_shape": tuple(final_features.shape),
        }

    def _infer_feature_dim(self, sample_patch_size: int) -> int:
        with torch.no_grad():
            dummy_image = torch.zeros(1, self.image_channels, sample_patch_size, sample_patch_size)
            dummy_mask = torch.ones(1, 1, sample_patch_size, sample_patch_size)
            features = self.extract_features(dummy_image, dummy_mask)
        return int(features.shape[1])

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(image, mask)
        raw_output = self.regression_head(features)
        lo, hi = self.output_range
        yellowness = torch.sigmoid(raw_output) * (hi - lo) + lo
        return yellowness.squeeze(-1)


def build_yellowness_model(
    backbone_name: str = "simple_cnn",
    mask_fusion: str = "mask_as_channel",
    pooling_mode: str = "global_avg",
    torchgeo_weight: Optional[str] = None,
    freeze_backbone: bool = False,
    **kwargs: Any,
) -> MaskedYellownessRegressor:
    return MaskedYellownessRegressor(
        backbone_name=backbone_name,
        mask_fusion=MaskFusionMode(mask_fusion),
        pooling_mode=PoolingMode(pooling_mode),
        torchgeo_weight=torchgeo_weight,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )