#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import make_dataloader, make_group_split


ALL_REGRESSORS: tuple[str, ...] = (
    "ridge",
    "pls",
    "random_forest",
    "extra_trees",
    "xgboost",
)

ALL_DR_METHODS: tuple[str, ...] = (
    "none",
    "pca",
    "pls",
)

ALL_POOLING_MODES: tuple[str, ...] = (
    "global_avg",
    "masked_avg",
    "masked_mean_std",
    "global_max",
    "gem",
)


class SafePLSRegressor(BaseEstimator, RegressorMixin):
    """PLS regressor that caps n_components to the valid bound for the current fit data."""

    def __init__(self, preferred_components: int = 8) -> None:
        self.preferred_components = preferred_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafePLSRegressor":
        X_arr = np.asarray(X)
        max_valid_components = int(min(X_arr.shape[0] - 1, X_arr.shape[1]))
        if max_valid_components < 1:
            raise ValueError("PLS regression requires at least two samples and one feature.")

        self.n_components_ = min(int(self.preferred_components), max_valid_components)
        self.model_ = PLSRegression(n_components=self.n_components_)
        self.model_.fit(X_arr, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X).reshape(-1)


class BackboneFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(image, mask)


def _center_crop_last2d(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    h, w = x.shape[-2], x.shape[-1]
    if crop_size is None or crop_size <= 0 or (crop_size == h and crop_size == w):
        return x
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} is larger than sample size {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return x[..., top : top + crop_size, left : left + crop_size]


def _transform_sample(obj: Any, crop_size: int) -> Any:
    if torch.is_tensor(obj):
        return _center_crop_last2d(obj, crop_size)
    if isinstance(obj, dict):
        return {k: _transform_sample(v, crop_size) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_transform_sample(v, crop_size) for v in obj)
    if isinstance(obj, list):
        return [_transform_sample(v, crop_size) for v in obj]
    return obj


class CenterCropDataset(Dataset):
    def __init__(self, base_dataset: Dataset, crop_size: int) -> None:
        self.base_dataset = base_dataset
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.base_dataset[index]
        return _transform_sample(sample, self.crop_size)


def _rebuild_loader(base_loader: DataLoader, dataset: Dataset, shuffle: bool, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=base_loader.pin_memory,
        persistent_workers=num_workers > 0,
    )


def infer_torchgeo_backbone_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "torchgeo:resnet50"
    if weight_name.startswith("ResNet152_Weights."):
        return "torchgeo:resnet152"
    if weight_name.startswith("ViTSmall16_Weights."):
        return "torchgeo:vit_small_patch16_224"
    if weight_name.startswith("ViTBase16_Weights."):
        return "torchgeo:vit_base_patch16_224"
    if weight_name.startswith("ViTSmall14_DINOv2_Weights."):
        return "torchgeo:vit_small_patch14_dinov2"
    if weight_name.startswith("ViTBase14_DINOv2_Weights."):
        return "torchgeo:vit_base_patch14_dinov2"
    if weight_name.startswith("Swin_V2_T_Weights."):
        return "torchgeo:swin_v2_t"
    if weight_name.startswith("Swin_V2_B_Weights."):
        return "torchgeo:swin_v2_b"
    if weight_name.startswith("FGMAEEarthLoc_Weights.") and "RESNET50" in weight_name.upper():
        return "torchgeo:resnet50"
    raise ValueError("Could not infer a TorchGeo backbone from --torchgeo-weight.")


def resolve_backbone_band_indices(weight_name: Optional[str]) -> Optional[list[int]]:
    if not weight_name:
        return None

    satlas_9band_prefixes = (
        "ResNet50_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet152_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet50_Weights.SENTINEL2_MI_MS_SATLAS",
        "Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS",
        "Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_B_Weights.SENTINEL2_MI_MS_SATLAS",
    )
    if weight_name.startswith(satlas_9band_prefixes):
        # Use recipe-based remapping from L2A 10-band order to SATLAS 9-band order.
        return None
    return None


def resolve_band_recipe(weight_name: Optional[str]) -> Optional[str]:
    if not weight_name:
        return None

    satlas_9band_prefixes = (
        "ResNet50_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet152_Weights.SENTINEL2_SI_MS_SATLAS",
        "ResNet50_Weights.SENTINEL2_MI_MS_SATLAS",
        "Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_T_Weights.SENTINEL2_MI_MS_SATLAS",
        "Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS",
        "Swin_V2_B_Weights.SENTINEL2_MI_MS_SATLAS",
    )
    if weight_name.startswith(satlas_9band_prefixes):
        return "satlas_l1c_9_from_l2a"

    if "SENTINEL2_ALL_NDVI_SECO_ECO" in weight_name:
        return "ndvi_seco_eco_9"

    return None


def resolve_backbone(backbone: str, torchgeo_weight: Optional[str]) -> str:
    if backbone != "auto":
        return backbone
    if torchgeo_weight is None:
        return "simple_cnn"
    return infer_torchgeo_backbone_from_weight(torchgeo_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract backbone features and train classic ML regressors for yellowness.")
    parser.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    parser.add_argument("--root-dir", type=Path, default=Path("yellowness_dataset"))
    parser.add_argument("--resolution", choices=("lr", "sr"), default="sr")
    parser.add_argument("--backbone", default="auto")
    parser.add_argument("--torchgeo-weight", default=None)
    parser.add_argument("--mask-fusion", default="feature_mask_pool")
    parser.add_argument("--pooling", default="global_avg", choices=ALL_POOLING_MODES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true")
    parser.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    parser.add_argument("--center-crop-size", type=int, default=224)
    parser.add_argument("--shapefile-path", type=Path, default=None)
    parser.add_argument("--mask-buffer-m", type=float, default=-20.0)
    parser.add_argument(
        "--feature-reduction-method",
        choices=("none", "pca", "pca_var95", "mi_topk", "f_reg_topk", "var_topk"),
        default="none",
    )
    parser.add_argument("--feature-reduction-dim", type=int, default=64)
    parser.add_argument(
        "--feature-reduction-variance",
        type=float,
        default=0.95,
        help="Explained variance target used by --feature-reduction-method pca_var95.",
    )
    parser.add_argument("--log-feature-shapes", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_REGRESSORS],
    )
    parser.add_argument(
        "--dr-methods",
        nargs="+",
        default=["none", "pca"],
        choices=["all", *ALL_DR_METHODS],
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Fraction of variance to preserve with PCA.",
    )
    parser.add_argument(
        "--pls-top-k",
        type=int,
        default=8,
        help="Number of target-correlated features to keep before PLS.",
    )
    parser.add_argument("--save-feature-csv", dest="save_feature_csv", action="store_true")
    parser.add_argument("--no-save-feature-csv", dest="save_feature_csv", action="store_false")
    parser.add_argument("--reuse-cached-features", action="store_true")
    parser.add_argument("--feature-csv-name", default="backbone_embeddings.csv")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--deep-runs-root", type=Path, default=Path("outputs") / "yellowness_regression")
    parser.add_argument("--export-deep-runs-summary", dest="export_deep_runs_summary", action="store_true")
    parser.add_argument("--no-export-deep-runs-summary", dest="export_deep_runs_summary", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "yellowness_backbone_ml")
    parser.set_defaults(save_feature_csv=True, export_deep_runs_summary=True, data_parallel=False)
    return parser.parse_args()


def build_regressor(name: str, random_state: int) -> Any:
    if name == "ridge":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ]
            )
        )

    if name == "pls":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pls", SafePLSRegressor(preferred_components=8)),
                ]
            )
        )

    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )

    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is required for --regressors xgboost.") from exc

        return XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_jobs=4,
        )

    raise ValueError(f"Unsupported regressor '{name}'.")


def build_feature_extractor(model: nn.Module, device: torch.device, use_data_parallel: bool = False) -> nn.Module:
    extractor: nn.Module = BackboneFeatureExtractor(model)
    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        extractor = nn.DataParallel(extractor)
    return extractor.to(device)


def extract_features(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[int]]:
    model.eval()
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    plot_ids: list[str] = []
    observation_dates: list[str] = []
    row_indices: list[int] = []
    logged_shapes = False
    with torch.inference_mode():
        for batch in dataloader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda").contiguous()
            mask = batch["mask"].to(device, non_blocking=device.type == "cuda").contiguous()
            try:
                features = model(image, mask).cpu().numpy()
            except RuntimeError as exc:
                is_cuda_alignment_error = (
                    device.type == "cuda"
                    and (
                        "misaligned address" in str(exc).lower()
                        or "cuda error" in str(exc).lower()
                        or "acceleratorerror" in str(type(exc)).lower()
                    )
                )
                if not is_cuda_alignment_error:
                    raise

                # Fallback for unstable CUDA kernels: run each patch independently.
                single_features: list[np.ndarray] = []
                for i in range(image.shape[0]):
                    feat_i = model(image[i : i + 1], mask[i : i + 1]).cpu().numpy()
                    single_features.append(feat_i)
                features = np.concatenate(single_features, axis=0)

            if not logged_shapes:
                core_model = model.module if isinstance(model, nn.DataParallel) else model
                target_model = getattr(core_model, "model", core_model)
                if hasattr(target_model, "debug_feature_shapes"):
                    try:
                        debug = target_model.debug_feature_shapes(image[:1], mask[:1])
                        print("[shape-log] image:", tuple(debug.get("image_shape", ())))
                        print("[shape-log] raw_backbone:", tuple(debug.get("raw_backbone_shape", ())))
                        print("[shape-log] masked_backbone:", tuple(debug.get("masked_backbone_shape", ())))
                        print("[shape-log] pooled:", tuple(debug.get("pooled_shape", ())))
                        print("[shape-log] final_feature:", tuple(debug.get("final_feature_shape", ())))
                    except Exception:
                        pass
                logged_shapes = True
            targets = batch["target"].cpu().numpy()
            feature_batches.append(features)
            target_batches.append(targets)
            plot_ids.extend(str(value) for value in batch["id_plot"])
            observation_dates.extend(str(value) for value in batch.get("observation_date", [""] * len(targets)))
            row_indices.extend(int(value) for value in batch["row_index"])
    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        plot_ids,
        observation_dates,
        row_indices,
    )


def fit_feature_reduction_on_train(
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    method: str,
    out_dim: int,
    variance_target: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "none":
        return train_features, val_features, {"method": "none", "output_dim": int(train_features.shape[1])}

    if method == "pca":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        n_components = int(min(max(1, out_dim), train_scaled.shape[0] - 1, train_scaled.shape[1]))
        reducer = PCA(n_components=n_components, random_state=0)
        train_reduced = reducer.fit_transform(train_scaled).astype(np.float32)
        val_reduced = reducer.transform(val_scaled).astype(np.float32)
        artifact = {
            "method": "pca",
            "output_dim": int(n_components),
            "scaler": scaler,
            "reducer": reducer,
        }
        return train_reduced, val_reduced, artifact

    if method == "pca_var95":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        target = float(np.clip(variance_target, 0.5, 0.999))
        reducer = PCA(n_components=target, random_state=random_state)
        train_reduced = reducer.fit_transform(train_scaled).astype(np.float32)
        val_reduced = reducer.transform(val_scaled).astype(np.float32)
        artifact = {
            "method": "pca_var95",
            "variance_target": float(target),
            "output_dim": int(train_reduced.shape[1]),
            "scaler": scaler,
            "reducer": reducer,
        }
        return train_reduced, val_reduced, artifact

    if method in {"mi_topk", "f_reg_topk", "var_topk"}:
        feature_count = int(train_features.shape[1])
        top_k = int(min(max(1, out_dim), feature_count))
        target = np.asarray(y_train, dtype=np.float32)

        if method == "mi_topk":
            scores = mutual_info_regression(train_features, y=target, random_state=random_state)
            scores = np.nan_to_num(scores, nan=-np.inf)
        elif method == "f_reg_topk":
            scores, _ = f_regression(train_features, target, center=True)
            scores = np.nan_to_num(scores, nan=-np.inf)
        else:
            scores = np.var(train_features, axis=0)

        selected_indices = np.argsort(-scores, kind="stable")[:top_k]
        train_selected = train_features[:, selected_indices].astype(np.float32)
        val_selected = val_features[:, selected_indices].astype(np.float32)
        artifact = {
            "method": method,
            "output_dim": int(top_k),
            "selected_indices": selected_indices.tolist(),
            "scores": scores[selected_indices].astype(float).tolist(),
        }
        return train_selected, val_selected, artifact

    raise ValueError(f"Unsupported feature reduction method '{method}'.")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _resolve_requested(values: list[str], all_values: tuple[str, ...]) -> list[str]:
    return list(all_values) if "all" in values else list(values)


def _component_candidates(grid: list[int], max_components: int) -> list[int]:
    cleaned = sorted(set(int(v) for v in grid if int(v) > 0 and int(v) <= max_components))
    return cleaned


def _rank_features_by_target_correlation(train_features: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    features = np.asarray(train_features, dtype=np.float64)
    target = np.asarray(y_train, dtype=np.float64)
    target_centered = target - target.mean()
    feature_centered = features - features.mean(axis=0, keepdims=True)

    numerator = np.sum(feature_centered * target_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(feature_centered ** 2, axis=0) * np.sum(target_centered ** 2))
    correlations = np.zeros(features.shape[1], dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = np.abs(numerator[valid] / denominator[valid])
    return np.argsort(-correlations, kind="stable")


def _select_target_correlated_features(
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer for PLS feature selection.")

    ranking = _rank_features_by_target_correlation(train_features, y_train)
    selected_indices = ranking[: min(top_k, train_features.shape[1])]
    return train_features[:, selected_indices], val_features[:, selected_indices], selected_indices


def _save_embeddings_csv(
    output_csv: Path,
    train_features: np.ndarray,
    y_train: np.ndarray,
    train_plot_ids: list[str],
    train_observation_dates: list[str],
    train_row_indices: list[int],
    val_features: np.ndarray,
    y_val: np.ndarray,
    val_plot_ids: list[str],
    val_observation_dates: list[str],
    val_row_indices: list[int],
    backbone_name: str,
    resolution: str,
    mask_fusion: str,
    pooling: str,
    torchgeo_weight: Optional[str],
) -> None:
    feature_columns = [f"feat_{i:04d}" for i in range(train_features.shape[1])]

    train_df = pd.DataFrame(train_features, columns=feature_columns)
    train_df.insert(0, "pooling", pooling)
    train_df.insert(0, "mask_fusion", mask_fusion)
    train_df.insert(0, "resolution", resolution)
    train_df.insert(0, "torchgeo_weight", torchgeo_weight or "")
    train_df.insert(0, "row_index", train_row_indices)
    train_df.insert(0, "observation_date", train_observation_dates)
    train_df.insert(0, "id_plot", train_plot_ids)
    train_df.insert(0, "backbone_name", backbone_name)
    train_df.insert(0, "split", "train")
    train_df["yellowness"] = y_train

    val_df = pd.DataFrame(val_features, columns=feature_columns)
    val_df.insert(0, "pooling", pooling)
    val_df.insert(0, "mask_fusion", mask_fusion)
    val_df.insert(0, "resolution", resolution)
    val_df.insert(0, "torchgeo_weight", torchgeo_weight or "")
    val_df.insert(0, "row_index", val_row_indices)
    val_df.insert(0, "observation_date", val_observation_dates)
    val_df.insert(0, "id_plot", val_plot_ids)
    val_df.insert(0, "backbone_name", backbone_name)
    val_df.insert(0, "split", "val")
    val_df["yellowness"] = y_val

    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)
    np.savez_compressed(
        output_csv.with_suffix(".npz"),
        features=np.concatenate([train_features, val_features], axis=0),
        split=combined["split"].to_numpy(dtype=str),
        id_plot=combined["id_plot"].to_numpy(dtype=str),
        observation_date=combined["observation_date"].to_numpy(dtype=str),
        backbone_name=combined["backbone_name"].to_numpy(dtype=str),
        resolution=combined["resolution"].to_numpy(dtype=str),
        mask_fusion=combined["mask_fusion"].to_numpy(dtype=str),
        pooling=combined["pooling"].to_numpy(dtype=str),
        torchgeo_weight=combined["torchgeo_weight"].to_numpy(dtype=str),
        yellowness=combined["yellowness"].to_numpy(dtype=np.float32),
        row_index=combined["row_index"].to_numpy(dtype=np.int64),
    )


def _load_embeddings_csv(
    input_csv: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[int], np.ndarray, np.ndarray, list[str], list[str], list[int]]:
    df = pd.read_csv(input_csv)
    feature_columns = [column for column in df.columns if column.startswith("feat_")]
    if not feature_columns:
        raise ValueError(f"No feature columns found in cached embedding file: {input_csv}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Cached embedding CSV must include both train and val splits.")

    train_features = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_df["yellowness"].to_numpy(dtype=np.float32)
    train_plot_ids = train_df["id_plot"].astype(str).tolist()
    train_observation_dates = train_df.get("observation_date", pd.Series("", index=train_df.index)).astype(str).tolist()
    train_row_indices = train_df["row_index"].astype(int).tolist()

    val_features = val_df[feature_columns].to_numpy(dtype=np.float32)
    y_val = val_df["yellowness"].to_numpy(dtype=np.float32)
    val_plot_ids = val_df["id_plot"].astype(str).tolist()
    val_observation_dates = val_df.get("observation_date", pd.Series("", index=val_df.index)).astype(str).tolist()
    val_row_indices = val_df["row_index"].astype(int).tolist()

    return (
        train_features,
        y_train,
        train_plot_ids,
        train_observation_dates,
        train_row_indices,
        val_features,
        y_val,
        val_plot_ids,
        val_observation_dates,
        val_row_indices,
    )


def _fit_dimensionality_reduction(
    method: str,
    n_components: Optional[int],
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    pca_variance: float,
    pls_top_k: int,
) -> tuple[np.ndarray, np.ndarray, Any, str]:
    if method == "none":
        return train_features, val_features, None, "none"

    if method == "pca":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        pca = PCA(n_components=pca_variance, random_state=0)
        train_reduced = pca.fit_transform(train_scaled)
        val_reduced = pca.transform(val_scaled)
        artifact = {"scaler": scaler, "pca": pca, "variance_threshold": pca_variance}
        return train_reduced, val_reduced, artifact, "pca_095"

    if method == "pls":
        selected_train, selected_val, selected_indices = _select_target_correlated_features(
            train_features,
            y_train,
            val_features,
            pls_top_k,
        )
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(selected_train)
        val_scaled = scaler.transform(selected_val)
        max_valid_components = int(min(train_scaled.shape[0] - 1, train_scaled.shape[1]))
        if max_valid_components < 1:
            raise ValueError("PLS requires at least two training samples and one selected feature.")
        n_components = min(int(pls_top_k), max_valid_components)
        pls = PLSRegression(n_components=n_components)
        pls.fit(train_scaled, y_train)
        train_reduced = pls.transform(train_scaled)
        val_reduced = pls.transform(val_scaled)
        artifact = {
            "scaler": scaler,
            "pls": pls,
            "selected_feature_indices": selected_indices.tolist(),
        }
        return train_reduced, val_reduced, artifact, f"pls_top{len(selected_indices):03d}_nc{n_components:03d}"

    raise ValueError(f"Unsupported dimensionality reduction method '{method}'.")


def _export_deep_run_summaries(deep_runs_root: Path, output_dir: Path) -> None:
    if not deep_runs_root.exists():
        return

    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for run_config_path in deep_runs_root.glob("**/run_config.json"):
        run_dir = run_config_path.parent
        run_name = run_dir.name
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        best_metrics = run_config.get("best_metrics", {})
        summary_rows.append(
            {
                "run": run_name,
                "resolution": run_config.get("resolution"),
                "backbone": run_config.get("backbone"),
                "torchgeo_weight": run_config.get("torchgeo_weight"),
                "mask_fusion": run_config.get("mask_fusion"),
                "val_loss": best_metrics.get("val_loss"),
                "val_mae": best_metrics.get("mae"),
                "val_rmse": best_metrics.get("rmse"),
                "val_r2": best_metrics.get("r2"),
                "run_dir": str(run_dir),
            }
        )

        history_path = run_dir / "training_history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                for row in history:
                    history_rows.append({"run": run_name, **row})
            except Exception:
                pass

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("val_rmse").to_csv(output_dir / "deep_runs_summary.csv", index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(output_dir / "deep_runs_history.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regressor_names = _resolve_requested(args.regressors, ALL_REGRESSORS)
    dr_methods = _resolve_requested(args.dr_methods, ALL_DR_METHODS)
    backbone_band_indices = resolve_backbone_band_indices(args.torchgeo_weight)
    band_recipe = resolve_band_recipe(args.torchgeo_weight)

    feature_csv_path = args.output_dir / args.feature_csv_name

    if args.reuse_cached_features and feature_csv_path.exists():
        (
            train_features,
            y_train,
            train_plot_ids,
            train_observation_dates,
            train_row_indices,
            val_features,
            y_val,
            val_plot_ids,
            val_observation_dates,
            val_row_indices,
        ) = _load_embeddings_csv(feature_csv_path)
        backbone_name = resolve_backbone(args.backbone, args.torchgeo_weight)
    else:
        split = make_group_split(
            inventory_csv=args.inventory,
            root_dir=args.root_dir,
            resolution=args.resolution,
            test_size=args.test_size,
            random_state=args.random_state,
            band_indices=backbone_band_indices,
            band_recipe=band_recipe,
            shapefile_path=args.shapefile_path,
            shapefile_buffer_m=args.mask_buffer_m,
        )

        train_base_loader = make_dataloader(split.train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_base_loader = make_dataloader(split.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_dataset = CenterCropDataset(train_base_loader.dataset, crop_size=args.center_crop_size)
        val_dataset = CenterCropDataset(val_base_loader.dataset, crop_size=args.center_crop_size)

        train_loader = _rebuild_loader(
            train_base_loader,
            train_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        val_loader = _rebuild_loader(
            val_base_loader,
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            # Favor stability over autotuned kernels for feature extraction.
            torch.backends.cudnn.benchmark = False
        backbone_name = resolve_backbone(args.backbone, args.torchgeo_weight)
        if band_recipe in {"satlas_l1c_9_from_l2a", "ndvi_seco_eco_9"}:
            backbone_image_channels = 9
        else:
            backbone_image_channels = len(backbone_band_indices) if backbone_band_indices is not None else 10
        print(
            f"[band-audit] backbone={backbone_name} | weight={args.torchgeo_weight} | "
            f"band_recipe={band_recipe} | band_indices={backbone_band_indices} | "
            f"image_channels={backbone_image_channels}"
        )
        model = build_yellowness_model(
            backbone_name=backbone_name,
            mask_fusion=args.mask_fusion,
            pooling_mode=args.pooling,
            torchgeo_weight=args.torchgeo_weight,
            freeze_backbone=True,
            image_channels=backbone_image_channels,
            sample_patch_size=args.center_crop_size,
            backbone_band_indices=backbone_band_indices,
        )
        feature_extractor = build_feature_extractor(model, device, use_data_parallel=args.data_parallel)

        train_features, y_train, train_plot_ids, train_observation_dates, train_row_indices = extract_features(feature_extractor, train_loader, device)
        val_features, y_val, val_plot_ids, val_observation_dates, val_row_indices = extract_features(feature_extractor, val_loader, device)

        if args.save_feature_csv:
            _save_embeddings_csv(
                output_csv=feature_csv_path,
                train_features=train_features,
                y_train=y_train,
                train_plot_ids=train_plot_ids,
                train_observation_dates=train_observation_dates,
                train_row_indices=train_row_indices,
                val_features=val_features,
                y_val=y_val,
                val_plot_ids=val_plot_ids,
                val_observation_dates=val_observation_dates,
                val_row_indices=val_row_indices,
                backbone_name=backbone_name,
                resolution=args.resolution,
                mask_fusion=args.mask_fusion,
                pooling=args.pooling,
                torchgeo_weight=args.torchgeo_weight,
            )

    feature_config = {
        "backbone": backbone_name,
        "torchgeo_weight": args.torchgeo_weight,
        "band_recipe": band_recipe,
        "backbone_band_indices": backbone_band_indices,
        "expected_image_channels": int(9 if band_recipe in {"satlas_l1c_9_from_l2a", "ndvi_seco_eco_9"} else (len(backbone_band_indices) if backbone_band_indices is not None else 10)),
        "mask_fusion": args.mask_fusion,
        "pooling": args.pooling,
        "resolution": args.resolution,
        "raw_feature_dim": int(train_features.shape[1]),
        "num_train": int(train_features.shape[0]),
        "num_val": int(val_features.shape[0]),
        "feature_csv": str(feature_csv_path if feature_csv_path.exists() else ""),
    }
    with (args.output_dir / "feature_config.json").open("w", encoding="utf-8") as fh:
        json.dump(feature_config, fh, indent=2)

    np.savez_compressed(
        args.output_dir / "backbone_features.npz",
        X_train=train_features,
        y_train=y_train,
        X_val=val_features,
        y_val=y_val,
    )

    reduced_train_features, reduced_val_features, reduction_artifact = fit_feature_reduction_on_train(
        train_features=train_features,
        y_train=y_train,
        val_features=val_features,
        method=args.feature_reduction_method,
        out_dim=args.feature_reduction_dim,
        variance_target=args.feature_reduction_variance,
        random_state=args.random_state,
    )

    if args.log_feature_shapes:
        print(f"[shape-log] raw feature matrix train={train_features.shape}, val={val_features.shape}")
        print(
            f"[shape-log] reduced feature matrix train={reduced_train_features.shape}, "
            f"val={reduced_val_features.shape}"
        )

    if args.extract_only:
        print(f"Saved features only: {feature_csv_path}")
        return

    dr_sweep_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dr_configs: list[tuple[str, Optional[int]]] = []
    for method in dr_methods:
        if method == "none":
            dr_configs.append(("none", None))
        elif method == "pca":
            dr_configs.append(("pca", None))
        elif method == "pls":
            dr_configs.append(("pls", None))

    for dr_method, n_components in dr_configs:
        transformed_train, transformed_val, dr_artifact, dr_label = _fit_dimensionality_reduction(
            dr_method,
            n_components,
            reduced_train_features,
            y_train,
            reduced_val_features,
            args.pca_variance,
            args.pls_top_k,
        )

        dr_sweep_rows.append(
            {
                "dr_method": dr_method,
                "n_components": n_components,
                "input_dim": int(reduced_train_features.shape[1]),
                "output_dim": int(transformed_train.shape[1]),
                "pca_variance": args.pca_variance if dr_method == "pca" else None,
                "pls_top_k": args.pls_top_k if dr_method == "pls" else None,
            }
        )

        for regressor_name in regressor_names:
            run_dir = args.output_dir / dr_label / regressor_name
            run_dir.mkdir(parents=True, exist_ok=True)

            regressor = build_regressor(regressor_name, args.random_state)
            fit_kwargs: dict[str, Any] = {}
            if regressor_name == "xgboost":
                fit_kwargs = {"eval_set": [(transformed_val, y_val)], "verbose": False}
            regressor.fit(transformed_train, y_train, **fit_kwargs)

            train_pred = regressor.predict(transformed_train)
            val_pred = regressor.predict(transformed_val)

            metrics = {
                "train": evaluate_predictions(y_train, train_pred),
                "val": evaluate_predictions(y_val, val_pred),
            }
            metrics.update(
                {
                    "backbone": backbone_name,
                    "torchgeo_weight": args.torchgeo_weight,
                    "mask_fusion": args.mask_fusion,
                    "resolution": args.resolution,
                    "regressor": regressor_name,
                    "dr_method": dr_method,
                    "dr_label": dr_label,
                    "n_components": n_components,
                    "feature_reduction_method": args.feature_reduction_method,
                    "feature_reduction_dim": args.feature_reduction_dim,
                    "feature_dim": int(transformed_train.shape[1]),
                    "raw_feature_dim": int(train_features.shape[1]),
                    "reduced_backbone_feature_dim": int(reduced_train_features.shape[1]),
                    "feature_csv": str(feature_csv_path if feature_csv_path.exists() else ""),
                }
            )

            with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            with (run_dir / "model.pkl").open("wb") as fh:
                pickle.dump(
                    {
                        "regressor": regressor,
                        "feature_reduction_artifact": reduction_artifact,
                        "dr_method": dr_method,
                        "n_components": n_components,
                        "dr_artifact": dr_artifact,
                    },
                    fh,
                )

            pd.DataFrame(
                {
                    "split": "train",
                    "id_plot": train_plot_ids,
                    "row_index": train_row_indices,
                    "y_true": y_train,
                    "y_pred": train_pred,
                }
            ).to_csv(run_dir / "train_predictions.csv", index=False)
            pd.DataFrame(
                {
                    "split": "val",
                    "id_plot": val_plot_ids,
                    "row_index": val_row_indices,
                    "y_true": y_val,
                    "y_pred": val_pred,
                }
            ).to_csv(run_dir / "val_predictions.csv", index=False)

            summary_rows.append(
                {
                    "regressor": regressor_name,
                    "dr_method": dr_method,
                    "dr_label": dr_label,
                    "n_components": n_components,
                    "train_mae": metrics["train"]["mae"],
                    "train_rmse": metrics["train"]["rmse"],
                    "train_r2": metrics["train"]["r2"],
                    "val_mae": metrics["val"]["mae"],
                    "val_rmse": metrics["val"]["rmse"],
                    "val_r2": metrics["val"]["r2"],
                    "feature_dim": metrics["feature_dim"],
                    "raw_feature_dim": metrics["raw_feature_dim"],
                    "reduced_backbone_feature_dim": metrics["reduced_backbone_feature_dim"],
                }
            )
            print(json.dumps(metrics, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values("val_rmse").reset_index(drop=True)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)
    pd.DataFrame(dr_sweep_rows).to_csv(args.output_dir / "dimensionality_reduction_sweep.csv", index=False)

    if args.export_deep_runs_summary:
        _export_deep_run_summaries(args.deep_runs_root, args.output_dir)

    print(f"Saved summary: {args.output_dir / 'summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()