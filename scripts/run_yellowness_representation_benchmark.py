#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import IncrementalPCA, PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALL_REGRESSORS: tuple[str, ...] = (
    "bayesian_ridge",
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)

ALL_DR_METHODS: tuple[str, ...] = (
    "none",
    "pca",
    "incremental_pca",
    "truncated_svd",
    "pls",
)


@dataclass(frozen=True)
class EmbeddingRun:
    path: Path
    run_name: str
    resolution: str
    backbone_name: str
    mask_fusion: str
    pooling: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark representation strategies from cached yellowness features.")
    parser.add_argument("--inventory", type=Path, default=Path("yellowness_dataset") / "observations_inventory.csv")
    parser.add_argument(
        "--embedding-glob",
        default="outputs/yellowness_backbone_ml/*/backbone_embeddings.csv",
        help="Glob pattern used to discover cached embedding CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "yellowness_backbone_ml" / "representation_benchmark",
    )
    parser.add_argument(
        "--component-grid",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 192, 256],
    )
    parser.add_argument("--max-ml-dim", type=int, default=300)
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
    return parser.parse_args()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def resolve_requested(values: list[str], all_values: tuple[str, ...]) -> list[str]:
    return list(all_values) if "all" in values else list(values)


def load_embedding_runs(glob_pattern: str) -> list[EmbeddingRun]:
    runs: list[EmbeddingRun] = []
    for raw_path in sorted(glob.glob(glob_pattern)):
        path = Path(raw_path)
        df = pd.read_csv(path)
        feature_columns = [column for column in df.columns if column.startswith("feat_")]
        if not feature_columns or "split" not in df.columns:
            continue
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            continue
        runs.append(
            EmbeddingRun(
                path=path,
                run_name=path.parent.name,
                resolution=str(df.get("resolution", pd.Series([path.parent.name.split("_", 1)[0]])).iloc[0]),
                backbone_name=str(df.get("backbone_name", pd.Series([""])).iloc[0]),
                mask_fusion=str(df.get("mask_fusion", pd.Series(["unknown"])).iloc[0]),
                pooling=str(df.get("pooling", pd.Series(["global_avg"])).iloc[0]),
                train_df=train_df,
                val_df=val_df,
                feature_columns=feature_columns,
            )
        )
    return runs


def validate_split_consistency(runs: list[EmbeddingRun]) -> tuple[dict[str, tuple[tuple[int, ...], tuple[int, ...]]], list[dict[str, str]]]:
    reference_by_resolution: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    issues: list[dict[str, str]] = []

    for run in runs:
        signature = (
            tuple(run.train_df["row_index"].astype(int).tolist()),
            tuple(run.val_df["row_index"].astype(int).tolist()),
        )
        existing = reference_by_resolution.get(run.resolution)
        if existing is None:
            reference_by_resolution[run.resolution] = signature
            continue
        if existing != signature:
            issues.append(
                {
                    "run": run.run_name,
                    "resolution": run.resolution,
                    "issue": "split_mismatch",
                }
            )

    return reference_by_resolution, issues


def build_handcrafted_frame(inventory_path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path).reset_index(drop=True)
    numeric = inventory.select_dtypes(include=[np.number]).copy()
    drop_columns = [
        "yellowness",
        "row_index",
        "split",
    ]
    numeric = numeric.drop(columns=[column for column in drop_columns if column in numeric.columns], errors="ignore")
    if numeric.empty:
        raise ValueError("No numeric handcrafted features were found in the inventory CSV.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(axis=0, numeric_only=True)
    numeric = numeric.fillna(medians)
    return numeric.astype(np.float32)


def lookup_handcrafted_features(
    handcrafted: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_idx = train_df["row_index"].astype(int).to_numpy()
    val_idx = val_df["row_index"].astype(int).to_numpy()
    x_train = handcrafted.iloc[train_idx].to_numpy(dtype=np.float32)
    x_val = handcrafted.iloc[val_idx].to_numpy(dtype=np.float32)
    return x_train, x_val, handcrafted.columns.tolist()


def build_regressor(name: str, random_state: int, n_train: int) -> Optional[Any]:
    effective_cv = min(5, max(2, n_train // 8))

    if name == "bayesian_ridge":
        return Pipeline([("scale", StandardScaler()), ("model", BayesianRidge())])
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    if name == "elastic_net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        alphas=np.logspace(-4, 1, 16),
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        cv=effective_cv,
                        random_state=random_state,
                        max_iter=100000,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features=0.7,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=6,
            max_iter=400,
            min_samples_leaf=8,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if name == "xgboost":
        try:
            XGBRegressor = importlib.import_module("xgboost").XGBRegressor
        except Exception:
            return None
        return XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=4,
        )
    if name == "lightgbm":
        try:
            LGBMRegressor = importlib.import_module("lightgbm").LGBMRegressor
        except Exception:
            return None
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
        )
    if name == "catboost":
        try:
            CatBoostRegressor = importlib.import_module("catboost").CatBoostRegressor
        except Exception:
            return None
        return CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    return None


def component_candidates(grid: list[int], train_size: int, feature_dim: int, max_ml_dim: int) -> list[int]:
    max_valid = min(train_size - 1, feature_dim, max_ml_dim)
    return sorted({value for value in grid if 0 < int(value) <= max_valid})


def apply_dr(
    method: str,
    n_components: Optional[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any, str]:
    if method == "none":
        return x_train, x_val, None, "none"

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=0)
        return reducer.fit_transform(x_train_scaled), reducer.transform(x_val_scaled), {"scaler": scaler, "reducer": reducer}, f"pca_{n_components:03d}"

    if method == "incremental_pca":
        reducer = IncrementalPCA(n_components=n_components, batch_size=min(256, len(x_train_scaled)))
        return reducer.fit_transform(x_train_scaled), reducer.transform(x_val_scaled), {"scaler": scaler, "reducer": reducer}, f"incremental_pca_{n_components:03d}"

    if method == "truncated_svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=0)
        return reducer.fit_transform(x_train_scaled), reducer.transform(x_val_scaled), {"scaler": scaler, "reducer": reducer}, f"truncated_svd_{n_components:03d}"

    if method == "pls":
        reducer = PLSRegression(n_components=n_components)
        reducer.fit(x_train_scaled, y_train)
        return reducer.transform(x_train_scaled), reducer.transform(x_val_scaled), {"scaler": scaler, "reducer": reducer}, f"pls_{n_components:03d}"

    raise ValueError(f"Unsupported DR method: {method}")


def plot_predicted_vs_true(summary: pd.DataFrame, output_dir: Path) -> None:
    top = summary.head(8)
    if top.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()
    for ax, (_, row) in zip(axes_flat, top.iterrows()):
        pred_path = Path(row["artifact_dir"]) / "val_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        sns.scatterplot(data=pred_df, x="y_true", y="y_pred", ax=ax, s=40)
        bounds = [pred_df[["y_true", "y_pred"]].min().min(), pred_df[["y_true", "y_pred"]].max().max()]
        ax.plot(bounds, bounds, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{row['run_name']}\n{row['representation']} | {row['regressor']}")
    for ax in axes_flat[len(top) :]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true_top_configs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(summary: pd.DataFrame, output_dir: Path) -> None:
    top = summary.head(8)
    if top.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()
    for ax, (_, row) in zip(axes_flat, top.iterrows()):
        pred_path = Path(row["artifact_dir"]) / "val_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
        sns.scatterplot(data=pred_df, x="y_pred", y="residual", ax=ax, s=40)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{row['run_name']}\n{row['dr_label']} | {row['regressor']}")
    for ax in axes_flat[len(top) :]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "residuals_top_configs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_comparison(summary: pd.DataFrame, output_dir: Path, group_col: str, filename: str, title: str) -> None:
    if summary.empty or group_col not in summary.columns:
        return
    grouped = summary.groupby(group_col, as_index=False)[["val_rmse", "val_r2"]].mean().sort_values("val_rmse")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.barplot(data=grouped, x=group_col, y="val_rmse", ax=axes[0], errorbar=None)
    sns.barplot(data=grouped, x=group_col, y="val_r2", ax=axes[1], errorbar=None)
    axes[0].set_title(f"{title}: mean validation RMSE")
    axes[1].set_title(f"{title}: mean validation R2")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_dim(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=summary, x="feature_dim", y="val_rmse", hue="representation", style="dr_method", ax=ax, s=80)
    ax.set_title("Feature dimension vs validation RMSE")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_dim_vs_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    regressors = resolve_requested(args.regressors, ALL_REGRESSORS)
    dr_methods = resolve_requested(args.dr_methods, ALL_DR_METHODS)

    embedding_runs = load_embedding_runs(args.embedding_glob)
    if not embedding_runs:
        raise FileNotFoundError(f"No cached embedding CSV files matched: {args.embedding_glob}")

    split_signatures, split_issues = validate_split_consistency(embedding_runs)
    handcrafted = build_handcrafted_frame(args.inventory)

    summary_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = list(split_issues)

    for run in embedding_runs:
        train_idx_signature, val_idx_signature = split_signatures[run.resolution]
        run_signature = (
            tuple(run.train_df["row_index"].astype(int).tolist()),
            tuple(run.val_df["row_index"].astype(int).tolist()),
        )
        if run_signature != (train_idx_signature, val_idx_signature):
            continue

        y_train = run.train_df["yellowness"].to_numpy(dtype=np.float32)
        y_val = run.val_df["yellowness"].to_numpy(dtype=np.float32)
        deep_train = run.train_df[run.feature_columns].to_numpy(dtype=np.float32)
        deep_val = run.val_df[run.feature_columns].to_numpy(dtype=np.float32)
        handcrafted_train, handcrafted_val, handcrafted_columns = lookup_handcrafted_features(handcrafted, run.train_df, run.val_df)

        representation_map: list[tuple[str, np.ndarray, np.ndarray, list[str], bool]] = [
            ("deep_only", deep_train, deep_val, run.feature_columns, False),
            ("handcrafted_only", handcrafted_train, handcrafted_val, handcrafted_columns, False),
            (
                "deep_plus_handcrafted",
                np.concatenate([deep_train, handcrafted_train], axis=1),
                np.concatenate([deep_val, handcrafted_val], axis=1),
                run.feature_columns + handcrafted_columns,
                True,
            ),
        ]

        for representation_name, x_train_base, x_val_base, feature_names, force_dr in representation_map:
            raw_dim = int(x_train_base.shape[1])
            max_valid_dim = min(args.max_ml_dim, int(x_train_base.shape[0] - 1))

            dr_configs: list[tuple[str, Optional[int], str]] = []
            if "none" in dr_methods and raw_dim <= max_valid_dim and not force_dr:
                dr_configs.append(("none", None, "none"))
            elif "none" in dr_methods and (raw_dim > max_valid_dim or force_dr):
                skip_rows.append(
                    {
                        "run": run.run_name,
                        "representation": representation_name,
                        "dr_method": "none",
                        "issue": "raw_dim_exceeds_limit",
                        "raw_feature_dim": raw_dim,
                        "max_valid_dim": max_valid_dim,
                    }
                )

            component_grid = component_candidates(args.component_grid, len(x_train_base), raw_dim, args.max_ml_dim)
            for method in dr_methods:
                if method == "none":
                    continue
                for n_components in component_grid:
                    dr_configs.append((method, n_components, f"{method}_{n_components:03d}"))

            for dr_method, n_components, dr_label in dr_configs:
                try:
                    x_train, x_val, dr_artifact, resolved_label = apply_dr(dr_method, n_components, x_train_base, y_train, x_val_base)
                except Exception as exc:
                    skip_rows.append(
                        {
                            "run": run.run_name,
                            "representation": representation_name,
                            "dr_method": dr_method,
                            "n_components": n_components,
                            "issue": f"dr_failed: {exc}",
                        }
                    )
                    continue

                feature_dim = int(x_train.shape[1])
                if feature_dim > max_valid_dim:
                    skip_rows.append(
                        {
                            "run": run.run_name,
                            "representation": representation_name,
                            "dr_method": dr_method,
                            "n_components": n_components,
                            "issue": "reduced_dim_exceeds_limit",
                            "feature_dim": feature_dim,
                            "max_valid_dim": max_valid_dim,
                        }
                    )
                    continue

                for regressor_name in regressors:
                    regressor = build_regressor(regressor_name, args.random_state, len(x_train))
                    if regressor is None:
                        skip_rows.append(
                            {
                                "run": run.run_name,
                                "representation": representation_name,
                                "regressor": regressor_name,
                                "issue": "missing_optional_dependency",
                            }
                        )
                        continue

                    artifact_dir = args.output_dir / run.run_name / representation_name / resolved_label / regressor_name
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                    fit_kwargs: dict[str, Any] = {}
                    if regressor_name == "xgboost":
                        fit_kwargs = {"eval_set": [(x_val, y_val)], "verbose": False}
                    regressor.fit(x_train, y_train, **fit_kwargs)

                    train_pred = np.asarray(regressor.predict(x_train)).reshape(-1)
                    val_pred = np.asarray(regressor.predict(x_val)).reshape(-1)
                    train_metrics = metric_dict(y_train, train_pred)
                    val_metrics = metric_dict(y_val, val_pred)

                    config_payload = {
                        "run_name": run.run_name,
                        "resolution": run.resolution,
                        "backbone_name": run.backbone_name,
                        "mask_fusion": run.mask_fusion,
                        "pooling": run.pooling,
                        "embedding_csv": str(run.path),
                        "representation": representation_name,
                        "dr_method": dr_method,
                        "dr_label": resolved_label,
                        "n_components": n_components,
                        "regressor": regressor_name,
                        "raw_feature_dim": raw_dim,
                        "feature_dim": feature_dim,
                        "max_valid_dim": max_valid_dim,
                        "feature_count": len(feature_names),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    }

                    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                        json.dump(config_payload, fh, indent=2)
                    with (artifact_dir / "model.pkl").open("wb") as fh:
                        pickle.dump({"regressor": regressor, "dr_artifact": dr_artifact, "feature_names": feature_names}, fh)

                    pd.DataFrame(
                        {
                            "split": "train",
                            "row_index": run.train_df["row_index"].astype(int).tolist(),
                            "y_true": y_train,
                            "y_pred": train_pred,
                        }
                    ).to_csv(artifact_dir / "train_predictions.csv", index=False)
                    pd.DataFrame(
                        {
                            "split": "val",
                            "row_index": run.val_df["row_index"].astype(int).tolist(),
                            "y_true": y_val,
                            "y_pred": val_pred,
                        }
                    ).to_csv(artifact_dir / "val_predictions.csv", index=False)

                    summary_rows.append(
                        {
                            "run_name": run.run_name,
                            "resolution": run.resolution,
                            "backbone_name": run.backbone_name,
                            "mask_fusion": run.mask_fusion,
                            "pooling": run.pooling,
                            "representation": representation_name,
                            "dr_method": dr_method,
                            "dr_label": resolved_label,
                            "n_components": n_components,
                            "regressor": regressor_name,
                            "raw_feature_dim": raw_dim,
                            "feature_dim": feature_dim,
                            "train_mae": train_metrics["mae"],
                            "train_rmse": train_metrics["rmse"],
                            "train_r2": train_metrics["r2"],
                            "val_mae": val_metrics["mae"],
                            "val_rmse": val_metrics["rmse"],
                            "val_r2": val_metrics["r2"],
                            "artifact_dir": str(artifact_dir),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No valid benchmark runs were produced.")

    summary = summary.sort_values(["val_rmse", "val_r2", "val_mae"], ascending=[True, False, True]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "representation_benchmark_summary.csv", index=False)
    summary.to_json(args.output_dir / "representation_benchmark_summary.json", orient="records", indent=2)
    pd.DataFrame(skip_rows).to_csv(args.output_dir / "skipped_configurations.csv", index=False)

    plot_predicted_vs_true(summary, args.output_dir)
    plot_residuals(summary, args.output_dir)
    plot_feature_dim(summary, args.output_dir)
    plot_group_comparison(summary, args.output_dir, "dr_method", "dr_method_comparison.png", "DR method comparison")
    plot_group_comparison(summary, args.output_dir, "regressor", "regressor_comparison.png", "Regressor comparison")
    plot_group_comparison(summary, args.output_dir, "mask_fusion", "mask_strategy_comparison.png", "Mask strategy comparison")
    plot_group_comparison(summary, args.output_dir, "pooling", "pooling_comparison.png", "Pooling comparison")
    plot_group_comparison(summary, args.output_dir, "representation", "representation_comparison.png", "Representation comparison")

    print(summary.head(30).to_string(index=False))
    print(f"Saved summary: {args.output_dir / 'representation_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()