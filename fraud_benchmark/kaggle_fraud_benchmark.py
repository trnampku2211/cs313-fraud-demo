#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark density-based fraud detection methods on the ULB dataset.

Designed for Kaggle. The default dataset path is:
    /kaggle/input/datasets/phtrnnam/cs313-namphu/creditcard.csv

Run in a Kaggle notebook cell:
    !python /kaggle/working/kaggle_fraud_benchmark.py

Main outputs are written to:
    /kaggle/working/fraud_density_outputs

The target column (Class) is never passed to preprocessing, PCA, parameter
selection, or model fitting. It is used only after fitting for evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler


DEFAULT_DATA_PATH = Path(
    "/kaggle/input/datasets/phtrnnam/cs313-namphu/creditcard.csv"
)
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/fraud_density_outputs")
EXPECTED_COLUMNS = {
    "Time",
    "Amount",
    "Class",
    *(f"V{i}" for i in range(1, 29)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DBSCAN, HDBSCAN, Isolation Forest, and LOF on ULB fraud data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60_000,
        help="Random sample size after deduplication. Use 0 for all rows.",
    )
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument(
        "--alert-rate",
        type=float,
        default=0.01,
        help="Fraction of transactions sent for review in the fair-ranking comparison.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=None,
        help="Override automatic k-distance elbow selection.",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--lof-neighbors", type=int, default=35)
    parser.add_argument("--iforest-trees", type=int, default=300)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Attach the Kaggle dataset or pass --data /path/to/creditcard.csv"
        )
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 or a positive integer")
    if not 2 <= args.pca_components <= 28:
        raise ValueError("--pca-components must be between 2 and 28")
    if not 0 < args.alert_rate < 1:
        raise ValueError("--alert-rate must be between 0 and 1")
    if args.dbscan_eps is not None and args.dbscan_eps <= 0:
        raise ValueError("--dbscan-eps must be positive")


def load_and_sample(
    path: Path, sample_size: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[1/8] Reading {path}")
    raw = pd.read_csv(path)

    missing_columns = sorted(EXPECTED_COLUMNS - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    raw_rows = len(raw)
    missing_values = int(raw.isna().sum().sum())
    duplicate_rows = int(raw.duplicated().sum())
    raw_fraud = int(raw["Class"].sum())

    clean = raw.drop_duplicates().reset_index(drop=False).rename(
        columns={"index": "source_row"}
    )
    clean_rows = len(clean)

    # Random sampling does not inspect Class and approximately preserves prevalence.
    if sample_size and sample_size < clean_rows:
        sample = clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        sample = clean.reset_index(drop=True)

    sample_fraud = int(sample["Class"].sum())
    if sample_fraud == 0:
        raise RuntimeError(
            "The random sample contains no fraud cases. Increase --sample-size."
        )

    summary = {
        "raw_rows": raw_rows,
        "raw_columns": int(raw.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows_removed": duplicate_rows,
        "raw_fraud": raw_fraud,
        "raw_fraud_rate": raw_fraud / raw_rows,
        "clean_rows": clean_rows,
        "sample_rows": len(sample),
        "sample_fraud": sample_fraud,
        "sample_fraud_rate": sample_fraud / len(sample),
        "sample_was_random_without_target_stratification": True,
    }

    print(
        "      "
        f"raw={raw_rows:,}, duplicates={duplicate_rows:,}, "
        f"sample={len(sample):,}, fraud_in_sample={sample_fraud:,} "
        f"({summary['sample_fraud_rate']:.4%})"
    )
    return sample, summary


def build_features(
    sample: pd.DataFrame, pca_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    print("[2/8] Feature engineering, RobustScaler, and PCA")
    y = sample["Class"].to_numpy(dtype=np.int8)

    features = sample.drop(columns=["Class", "source_row"]).copy()
    seconds_in_day = 86_400.0
    angle = 2.0 * math.pi * (features["Time"] % seconds_in_day) / seconds_in_day
    features["Time_sin"] = np.sin(angle)
    features["Time_cos"] = np.cos(angle)
    features["Amount"] = np.log1p(features["Amount"].clip(lower=0))
    features = features.drop(columns=["Time"])

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(features).astype(np.float32, copy=False)

    pca_model = PCA(n_components=pca_components, random_state=seed)
    x_model = pca_model.fit_transform(x_scaled).astype(np.float32, copy=False)

    pca_plot = PCA(n_components=2, random_state=seed)
    x_plot = pca_plot.fit_transform(x_scaled).astype(np.float32, copy=False)

    preprocessing = {
        "input_feature_count": int(features.shape[1]),
        "input_features": list(features.columns),
        "amount_transform": "log1p then RobustScaler",
        "time_transform": "seconds since first transaction to cyclic Time_sin/Time_cos",
        "clustering_pca_components": pca_components,
        "clustering_pca_explained_variance": float(
            pca_model.explained_variance_ratio_.sum()
        ),
        "visualization_pca_components": 2,
        "visualization_pca_explained_variance": float(
            pca_plot.explained_variance_ratio_.sum()
        ),
    }

    print(
        "      "
        f"PCA-{pca_components} explained variance: "
        f"{preprocessing['clustering_pca_explained_variance']:.2%}"
    )
    return x_model, x_plot, y, preprocessing


def compute_k_distances(
    x_model: np.ndarray, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    distances, _ = neighbors.fit(x_model).kneighbors(x_model)
    unsorted = distances[:, -1].astype(np.float64, copy=False)
    return unsorted, np.sort(unsorted)


def estimate_eps(k_distance: np.ndarray) -> tuple[float, int]:
    """Estimate the elbow of a sorted increasing k-distance curve.

    The curve is trimmed at the 99.5th percentile to prevent a handful of extreme
    points from moving the elbow to the final observations. For a convex increasing
    curve, the elbow is the point with the largest vertical gap below the diagonal.
    """
    finite = np.asarray(k_distance[np.isfinite(k_distance)], dtype=np.float64)
    if len(finite) < 20:
        raise ValueError("Not enough finite k-distance values to estimate eps")

    trim_size = max(20, int(len(finite) * 0.995))
    trimmed = finite[:trim_size]
    y_min = float(trimmed[0])
    y_max = float(trimmed[-1])

    if math.isclose(y_min, y_max):
        index = int(0.90 * (len(trimmed) - 1))
        return float(trimmed[index]), index

    x_norm = np.linspace(0.0, 1.0, len(trimmed))
    y_norm = (trimmed - y_min) / (y_max - y_min)
    index = int(np.argmax(x_norm - y_norm))
    eps = float(trimmed[index])

    if eps <= 0:
        index = int(0.90 * (len(trimmed) - 1))
        eps = float(trimmed[index])
    return eps, index


def plot_k_distance(
    k_distance: np.ndarray,
    eps: float,
    elbow_index: int | None,
    min_samples: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(k_distance, color="#2563eb", linewidth=1.4)
    ax.axhline(eps, color="#dc2626", linestyle="--", linewidth=1.4, label=f"eps = {eps:.4f}")
    if elbow_index is not None:
        ax.scatter(
            [elbow_index],
            [eps],
            color="#dc2626",
            s=55,
            zorder=3,
            label="estimated elbow",
        )
    ax.set_title(f"k-distance graph in PCA space (k = {min_samples})")
    ax.set_xlabel("Transactions sorted by k-distance")
    ax.set_ylabel(f"Distance to neighbor {min_samples}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Anomaly score contains no finite values")
    finite_values = values[finite]
    replacement = float(np.median(finite_values))
    upper = float(np.max(finite_values))
    lower = float(np.min(finite_values))
    return np.nan_to_num(values, nan=replacement, posinf=upper, neginf=lower)


def top_budget_prediction(scores: np.ndarray, alert_rate: float) -> np.ndarray:
    count = max(1, int(round(len(scores) * alert_rate)))
    top_indices = np.argpartition(scores, -count)[-count:]
    prediction = np.zeros(len(scores), dtype=np.int8)
    prediction[top_indices] = 1
    return prediction


def metric_block(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "alerts": int(prediction.sum()),
        "alert_rate": float(prediction.mean()),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    name: str,
    y: np.ndarray,
    native_prediction: np.ndarray,
    scores: np.ndarray,
    alert_rate: float,
    runtime_seconds: float,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    clean_scores = sanitize_scores(scores)
    budget_prediction = top_budget_prediction(clean_scores, alert_rate)
    native = metric_block(y, native_prediction)
    budget = metric_block(y, budget_prediction)

    result = {
        "model": name,
        "runtime_seconds": float(runtime_seconds),
        "auprc": float(average_precision_score(y, clean_scores)),
        "native": native,
        "fixed_budget": budget,
        "parameters": parameters,
    }
    return result, clean_scores, budget_prediction


def create_hdbscan(
    min_cluster_size: int, min_samples: int
) -> tuple[Any, str]:
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        return (
            SklearnHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                n_jobs=-1,
            ),
            "sklearn.cluster.HDBSCAN",
        )
    except ImportError:
        try:
            from hdbscan import HDBSCAN as ExternalHDBSCAN

            return (
                ExternalHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    core_dist_n_jobs=-1,
                    prediction_data=False,
                ),
                "hdbscan.HDBSCAN",
            )
        except ImportError as error:
            raise ImportError(
                "HDBSCAN is unavailable. In Kaggle run `%pip install hdbscan`, "
                "restart the session, and rerun this script."
            ) from error


def hdbscan_scores(model: Any) -> np.ndarray:
    if hasattr(model, "outlier_scores_"):
        return np.asarray(model.outlier_scores_, dtype=np.float64)
    if hasattr(model, "probabilities_"):
        return 1.0 - np.asarray(model.probabilities_, dtype=np.float64)
    raise AttributeError("The HDBSCAN implementation provides no usable anomaly score")


def flatten_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "model": result["model"],
            "runtime_seconds": result["runtime_seconds"],
            "auprc": result["auprc"],
        }
        row.update({f"native_{key}": value for key, value in result["native"].items()})
        row.update(
            {
                f"budget_{key}": value
                for key, value in result["fixed_budget"].items()
            }
        )
        row["parameters"] = json.dumps(
            result["parameters"], ensure_ascii=False, default=json_default
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_model_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["budget_precision", "budget_recall", "budget_f1", "auprc"]
    labels = ["Precision@budget", "Recall@budget", "F1@budget", "AUPRC"]
    x = np.arange(len(results_df))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"]
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            results_df[metric],
            width,
            label=label,
            color=colors[index],
        )

    ax.set_title("Fraud detection comparison at a fixed alert budget")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    y: np.ndarray,
    native_predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, (name, prediction) in zip(axes.ravel(), native_predictions.items()):
        matrix = confusion_matrix(y, prediction, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Normal", "Fraud"]
        )
        display.plot(ax=ax, colorbar=False, values_format=",d")
        ax.set_title(f"{name} native output")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dbscan_projection(
    x_plot: np.ndarray,
    y: np.ndarray,
    dbscan_prediction: np.ndarray,
    seed: int,
    output_path: Path,
    max_normal_points: int = 20_000,
) -> None:
    rng = np.random.default_rng(seed)
    normal_indices = np.flatnonzero(y == 0)
    fraud_indices = np.flatnonzero(y == 1)
    if len(normal_indices) > max_normal_points:
        normal_indices = rng.choice(
            normal_indices, size=max_normal_points, replace=False
        )
    display_indices = np.concatenate([normal_indices, fraud_indices])

    x_display = x_plot[display_indices]
    y_display = y[display_indices]
    pred_display = dbscan_prediction[display_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    axes[0].scatter(
        x_display[y_display == 0, 0],
        x_display[y_display == 0, 1],
        s=6,
        alpha=0.25,
        color="#2563eb",
        label="Normal",
    )
    axes[0].scatter(
        x_display[y_display == 1, 0],
        x_display[y_display == 1, 1],
        s=28,
        marker="x",
        color="#dc2626",
        label="Fraud",
    )
    axes[0].set_title("Ground-truth labels (evaluation only)")
    axes[0].legend()

    axes[1].scatter(
        x_display[pred_display == 0, 0],
        x_display[pred_display == 0, 1],
        s=6,
        alpha=0.25,
        color="#2563eb",
        label="Clustered",
    )
    axes[1].scatter(
        x_display[pred_display == 1, 0],
        x_display[pred_display == 1, 1],
        s=28,
        marker="x",
        color="#dc2626",
        label="DBSCAN noise",
    )
    axes[1].set_title("DBSCAN result fitted without Class")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("Visualization PC1")
        ax.set_ylabel("Visualization PC2")
        ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_rankings(
    sample: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    budget_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    ranking = sample[["source_row", "Class", "Time", "Amount"]].copy()
    for name, scores in model_scores.items():
        slug = name.lower().replace(" ", "_")
        ranking[f"{slug}_score"] = scores
        ranking[f"{slug}_rank"] = (
            pd.Series(scores).rank(method="first", ascending=False).astype(int)
        )
        ranking[f"{slug}_top_budget"] = budget_predictions[name]
    return ranking.sort_values(
        by="isolation_forest_rank", ascending=True
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    sample, data_summary = load_and_sample(
        args.data, args.sample_size, args.seed
    )
    write_json(args.output / "data_summary.json", data_summary)

    x_model, x_plot, y, preprocessing = build_features(
        sample, args.pca_components, args.seed
    )

    min_samples = max(10, 2 * args.pca_components)
    print(f"[3/8] k-distance graph with min_samples={min_samples}")
    neighbor_started = time.perf_counter()
    dbscan_score, k_distance = compute_k_distances(x_model, min_samples)
    neighbor_search_seconds = time.perf_counter() - neighbor_started
    if args.dbscan_eps is None:
        eps, elbow_index = estimate_eps(k_distance)
        eps_source = "automatic elbow on trimmed k-distance curve"
    else:
        eps = args.dbscan_eps
        elbow_index = None
        eps_source = "command-line override"
    print(f"      DBSCAN eps={eps:.6f} ({eps_source})")
    plot_k_distance(
        k_distance,
        eps,
        elbow_index,
        min_samples,
        args.output / "k_distance.png",
    )

    results: list[dict[str, Any]] = []
    model_scores: dict[str, np.ndarray] = {}
    native_predictions: dict[str, np.ndarray] = {}
    budget_predictions: dict[str, np.ndarray] = {}

    print("[4/8] Fitting DBSCAN")
    started = time.perf_counter()
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1,
    )
    dbscan_labels = dbscan.fit_predict(x_model)
    runtime = time.perf_counter() - started
    dbscan_native = (dbscan_labels == -1).astype(np.int8)
    result, scores, budget_pred = evaluate_model(
        "DBSCAN",
        y,
        dbscan_native,
        dbscan_score,
        args.alert_rate,
        runtime,
        {
            "eps": eps,
            "eps_source": eps_source,
            "min_samples": min_samples,
            "anomaly_score": "distance to the k-th nearest neighbor",
        },
    )
    results.append(result)
    model_scores["DBSCAN"] = scores
    native_predictions["DBSCAN"] = dbscan_native
    budget_predictions["DBSCAN"] = budget_pred
    del dbscan, dbscan_labels
    gc.collect()

    print("[5/8] Fitting HDBSCAN")
    hdbscan, hdbscan_backend = create_hdbscan(
        args.hdbscan_min_cluster_size, min_samples
    )
    started = time.perf_counter()
    hdbscan_labels = hdbscan.fit_predict(x_model)
    runtime = time.perf_counter() - started
    hdbscan_native = (hdbscan_labels == -1).astype(np.int8)
    result, scores, budget_pred = evaluate_model(
        "HDBSCAN",
        y,
        hdbscan_native,
        hdbscan_scores(hdbscan),
        args.alert_rate,
        runtime,
        {
            "backend": hdbscan_backend,
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": min_samples,
        },
    )
    results.append(result)
    model_scores["HDBSCAN"] = scores
    native_predictions["HDBSCAN"] = hdbscan_native
    budget_predictions["HDBSCAN"] = budget_pred
    del hdbscan, hdbscan_labels
    gc.collect()

    print("[6/8] Fitting Isolation Forest")
    iforest = IsolationForest(
        n_estimators=args.iforest_trees,
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    )
    started = time.perf_counter()
    iforest.fit(x_model)
    iforest_native = (iforest.predict(x_model) == -1).astype(np.int8)
    iforest_score = -iforest.decision_function(x_model)
    runtime = time.perf_counter() - started
    result, scores, budget_pred = evaluate_model(
        "Isolation Forest",
        y,
        iforest_native,
        iforest_score,
        args.alert_rate,
        runtime,
        {
            "n_estimators": args.iforest_trees,
            "contamination": "auto",
            "random_state": args.seed,
        },
    )
    results.append(result)
    model_scores["Isolation Forest"] = scores
    native_predictions["Isolation Forest"] = iforest_native
    budget_predictions["Isolation Forest"] = budget_pred
    del iforest, iforest_score
    gc.collect()

    print("[7/8] Fitting Local Outlier Factor")
    lof = LocalOutlierFactor(
        n_neighbors=args.lof_neighbors,
        contamination="auto",
        n_jobs=-1,
    )
    started = time.perf_counter()
    lof_labels = lof.fit_predict(x_model)
    lof_score = -lof.negative_outlier_factor_
    runtime = time.perf_counter() - started
    lof_native = (lof_labels == -1).astype(np.int8)
    result, scores, budget_pred = evaluate_model(
        "LOF",
        y,
        lof_native,
        lof_score,
        args.alert_rate,
        runtime,
        {
            "n_neighbors": args.lof_neighbors,
            "contamination": "auto",
        },
    )
    results.append(result)
    model_scores["LOF"] = scores
    native_predictions["LOF"] = lof_native
    budget_predictions["LOF"] = budget_pred
    del lof, lof_labels, lof_score
    gc.collect()

    print("[8/8] Writing metrics, rankings, and figures")
    results_df = flatten_results(results)
    results_df.to_csv(args.output / "model_comparison.csv", index=False)
    write_json(args.output / "model_results.json", {"models": results})

    rankings = build_rankings(sample, model_scores, budget_predictions)
    rankings.to_csv(args.output / "candidate_rankings.csv", index=False)

    plot_model_comparison(
        results_df, args.output / "model_comparison.png"
    )
    plot_confusion_matrices(
        y, native_predictions, args.output / "confusion_matrices.png"
    )
    plot_dbscan_projection(
        x_plot,
        y,
        native_predictions["DBSCAN"],
        args.seed,
        args.output / "dbscan_projection.png",
    )

    experiment_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "dataset_path": str(args.data),
        "output_directory": str(args.output),
        "random_seed": args.seed,
        "requested_sample_size": args.sample_size,
        "alert_rate": args.alert_rate,
        "data_summary": data_summary,
        "preprocessing": preprocessing,
        "dbscan_min_samples": min_samples,
        "dbscan_eps": eps,
        "dbscan_eps_source": eps_source,
        "dbscan_neighbor_search_seconds": neighbor_search_seconds,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "lof_neighbors": args.lof_neighbors,
        "iforest_trees": args.iforest_trees,
    }
    write_json(args.output / "experiment_config.json", experiment_config)

    columns = [
        "model",
        "budget_precision",
        "budget_recall",
        "budget_f1",
        "auprc",
        "runtime_seconds",
    ]
    print("\nFixed-budget comparison")
    print(results_df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
