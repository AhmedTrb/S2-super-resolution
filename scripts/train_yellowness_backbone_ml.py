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
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import make_dataloader, make_group_split


ALL_REGRESSORS: tuple[str, ...] = (
    "ridge",
    "elastic_net",
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


def infer_torchgeo_backbone_from_weight(weight_name: str) -> str:
    if weight_name.startswith("ResNet50_Weights."):
        return "torchgeo:resnet50"
    if weight_name.startswith("ViTSmall16_Weights."):
        return "torchgeo:vit_small_patch16_224"
    if weight_name.startswith("ViTBase16_Weights."):
        return "torchgeo:vit_base_patch16_224"
    if weight_name.startswith("FGMAEEarthLoc_Weights.") and "RESNET50" in weight_name.upper():
        return "torchgeo:resnet50"
    raise ValueError("Could not infer a TorchGeo backbone from --torchgeo-weight.")


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--center-crop-size", type=int, default=224)
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
        default=["all"],
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
    parser.add_argument("--deep-runs-root", type=Path, default=Path("outputs") / "yellowness_regression")
    parser.add_argument("--export-deep-runs-summary", dest="export_deep_runs_summary", action="store_true")
    parser.add_argument("--no-export-deep-runs-summary", dest="export_deep_runs_summary", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "yellowness_backbone_ml")
    parser.set_defaults(save_feature_csv=True, export_deep_runs_summary=True)
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

    if name == "elastic_net":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "elastic_net",
                        ElasticNetCV(
                            alphas=np.logspace(-4, 1, 16),
                            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                            cv=5,
                            random_state=random_state,
                            max_iter=20000,
                        ),
                    ),
                ]
            )
        )

    if name == "pls":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pls", PLSRegression(n_components=8)),
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
            n_jobs=4,
        )

    raise ValueError(f"Unsupported regressor '{name}'.")


def extract_features(model: Any, dataloader: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    model.eval()
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    plot_ids: list[str] = []
    row_indices: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            features = model.extract_features(image, mask).cpu().numpy()
            targets = batch["target"].cpu().numpy()
            feature_batches.append(features)
            target_batches.append(targets)
            plot_ids.extend(str(value) for value in batch["id_plot"])
            row_indices.extend(int(value) for value in batch["row_index"])
    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        plot_ids,
        row_indices,
    )


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
    train_row_indices: list[int],
    val_features: np.ndarray,
    y_val: np.ndarray,
    val_plot_ids: list[str],
    val_row_indices: list[int],
) -> None:
    feature_columns = [f"feat_{i:04d}" for i in range(train_features.shape[1])]

    train_df = pd.DataFrame(train_features, columns=feature_columns)
    train_df.insert(0, "row_index", train_row_indices)
    train_df.insert(0, "id_plot", train_plot_ids)
    train_df.insert(0, "split", "train")
    train_df["yellowness"] = y_train

    val_df = pd.DataFrame(val_features, columns=feature_columns)
    val_df.insert(0, "row_index", val_row_indices)
    val_df.insert(0, "id_plot", val_plot_ids)
    val_df.insert(0, "split", "val")
    val_df["yellowness"] = y_val

    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)


def _load_embeddings_csv(input_csv: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[int], np.ndarray, np.ndarray, list[str], list[int]]:
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
    train_row_indices = train_df["row_index"].astype(int).tolist()

    val_features = val_df[feature_columns].to_numpy(dtype=np.float32)
    y_val = val_df["yellowness"].to_numpy(dtype=np.float32)
    val_plot_ids = val_df["id_plot"].astype(str).tolist()
    val_row_indices = val_df["row_index"].astype(int).tolist()

    return (
        train_features,
        y_train,
        train_plot_ids,
        train_row_indices,
        val_features,
        y_val,
        val_plot_ids,
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

    feature_csv_path = args.output_dir / args.feature_csv_name

    if args.reuse_cached_features and feature_csv_path.exists():
        (
            train_features,
            y_train,
            train_plot_ids,
            train_row_indices,
            val_features,
            y_val,
            val_plot_ids,
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
        )

        train_loader = make_dataloader(split.train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_loader = make_dataloader(split.val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_name = resolve_backbone(args.backbone, args.torchgeo_weight)
        model = build_yellowness_model(
            backbone_name=backbone_name,
            mask_fusion=args.mask_fusion,
            torchgeo_weight=args.torchgeo_weight,
            freeze_backbone=True,
            image_channels=10,
            sample_patch_size=args.center_crop_size,
        ).to(device)

        train_features, y_train, train_plot_ids, train_row_indices = extract_features(model, train_loader, device)
        val_features, y_val, val_plot_ids, val_row_indices = extract_features(model, val_loader, device)

        if args.save_feature_csv:
            _save_embeddings_csv(
                output_csv=feature_csv_path,
                train_features=train_features,
                y_train=y_train,
                train_plot_ids=train_plot_ids,
                train_row_indices=train_row_indices,
                val_features=val_features,
                y_val=y_val,
                val_plot_ids=val_plot_ids,
                val_row_indices=val_row_indices,
            )

    np.savez_compressed(
        args.output_dir / "backbone_features.npz",
        X_train=train_features,
        y_train=y_train,
        X_val=val_features,
        y_val=y_val,
    )

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
            train_features,
            y_train,
            val_features,
            args.pca_variance,
            args.pls_top_k,
        )

        dr_sweep_rows.append(
            {
                "dr_method": dr_method,
                "n_components": n_components,
                "input_dim": int(train_features.shape[1]),
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
                    "feature_dim": int(transformed_train.shape[1]),
                    "raw_feature_dim": int(train_features.shape[1]),
                    "feature_csv": str(feature_csv_path if feature_csv_path.exists() else ""),
                }
            )

            with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            with (run_dir / "model.pkl").open("wb") as fh:
                pickle.dump(
                    {
                        "regressor": regressor,
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