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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.s2_pipeline import build_yellowness_model
from src.s2_pipeline.yellowness_training import make_dataloader, make_group_split


ALL_REGRESSORS: tuple[str, ...] = (
    "ridge",
    "lasso",
    "elastic_net",
    "svm_linear",
    "svm_rbf",
    "knn",
    "pls",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "yellowness_backbone_ml")
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

    if name == "lasso":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("lasso", LassoCV(alphas=np.logspace(-4, 1, 16), cv=5, random_state=random_state, max_iter=20000)),
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

    if name == "svm_linear":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("svm", LinearSVR(C=1.0, epsilon=0.1, random_state=random_state, max_iter=20000)),
                ]
            )
        )

    if name == "svm_rbf":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("svm", SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")),
                ]
            )
        )

    if name == "knn":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("knn", KNeighborsRegressor(n_neighbors=9, weights="distance")),
                ]
            )
        )

    if name == "pls":
        return TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pls", PLSRegression(n_components=16)),
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

    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            l2_regularization=1e-3,
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regressor_names = list(ALL_REGRESSORS) if "all" in args.regressors else list(args.regressors)

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

    np.savez_compressed(
        args.output_dir / "backbone_features.npz",
        X_train=train_features,
        y_train=y_train,
        X_val=val_features,
        y_val=y_val,
    )

    summary_rows: list[dict[str, Any]] = []
    for regressor_name in regressor_names:
        run_dir = args.output_dir / regressor_name
        run_dir.mkdir(parents=True, exist_ok=True)

        regressor = build_regressor(regressor_name, args.random_state)
        fit_kwargs: dict[str, Any] = {}
        if regressor_name == "xgboost":
            fit_kwargs = {"eval_set": [(val_features, y_val)], "verbose": False}
        regressor.fit(train_features, y_train, **fit_kwargs)

        train_pred = regressor.predict(train_features)
        val_pred = regressor.predict(val_features)

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
                "feature_dim": int(train_features.shape[1]),
            }
        )

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        with (run_dir / "model.pkl").open("wb") as fh:
            pickle.dump(regressor, fh)

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
                "train_mae": metrics["train"]["mae"],
                "train_rmse": metrics["train"]["rmse"],
                "train_r2": metrics["train"]["r2"],
                "val_mae": metrics["val"]["mae"],
                "val_rmse": metrics["val"]["rmse"],
                "val_r2": metrics["val"]["r2"],
                "feature_dim": metrics["feature_dim"],
            }
        )
        print(json.dumps(metrics, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values("val_rmse").reset_index(drop=True)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)
    print(f"Saved summary: {args.output_dir / 'summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()