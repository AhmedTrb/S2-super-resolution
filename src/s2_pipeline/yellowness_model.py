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


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    in_channels: int
    feature_dim: Optional[int] = None
    pretrained: bool = True


def _infer_torchgeo_model_name_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "resnet50"
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
        self.input_channels = getattr(self.model, "in_chans", getattr(self.model, "in_channels", spec.in_channels))
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
        x = self.input_adapter(x)
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
        self.output_range = output_range
        self.mask_encoder = MaskFeatureEncoder()
        self.backbone_band_indices = tuple(int(index) for index in backbone_band_indices) if backbone_band_indices is not None else None

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
        if mode in {MaskFusionMode.IMAGE_ONLY, MaskFusionMode.MASKED_IMAGE, MaskFusionMode.LATE_MASK, MaskFusionMode.FEATURE_MASK_POOL}:
            return image_channels
        if mode == MaskFusionMode.MASK_AS_CHANNEL:
            return image_channels + 1
        if mode == MaskFusionMode.IMAGE_AND_MASK:
            return (image_channels * 2) + 1
        raise ValueError(f"Unsupported mask fusion mode: {mode}")

    def _prepare_backbone_input(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.backbone_band_indices is not None:
            image = image[:, list(self.backbone_band_indices), ...]

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=image.dtype)

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
        else:
            raise ValueError(f"Unsupported mask fusion mode: {self.mask_fusion}")

        return self.input_adapter(fused), aux, feature_mask

    def _apply_feature_mask(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            resized_mask = nn.functional.interpolate(mask, size=features.shape[-2:], mode="nearest")
            return features * resized_mask

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
                return torch.cat([cls_token, masked_tokens], dim=1)
            return masked_tokens

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

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return torch.flatten(nn.functional.adaptive_avg_pool2d(features, output_size=1), 1)
        if features.ndim == 3:
            if features.shape[-1] >= features.shape[1]:
                return features.mean(dim=1)
            return features.mean(dim=-1)
        if features.ndim == 2:
            return features
        if features.ndim == 1:
            return features.unsqueeze(0)
        return torch.flatten(features, start_dim=1)

    def extract_features(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        backbone_input, aux_features, feature_mask = self._prepare_backbone_input(image, mask)
        raw_features = self.backbone(backbone_input)
        tensor_features = self._extract_tensor(raw_features)
        if feature_mask is not None:
            tensor_features = self._apply_feature_mask(tensor_features, feature_mask)
        pooled_features = self._pool_features(tensor_features)
        if aux_features is not None:
            pooled_features = torch.cat([pooled_features, aux_features], dim=1)
        return pooled_features

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
    torchgeo_weight: Optional[str] = None,
    freeze_backbone: bool = False,
    **kwargs: Any,
) -> MaskedYellownessRegressor:
    return MaskedYellownessRegressor(
        backbone_name=backbone_name,
        mask_fusion=MaskFusionMode(mask_fusion),
        torchgeo_weight=torchgeo_weight,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )