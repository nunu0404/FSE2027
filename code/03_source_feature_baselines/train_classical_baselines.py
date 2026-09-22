#!/usr/bin/env python3
"""Train supervised classical readability baselines on RQ0 splits."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_train_classical")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def regression_models(seed: int) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2)),
            ]
        ),
        "gradient_boosting_regressor": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(random_state=seed))]
        ),
        "svr": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.1))]
        ),
    }


def classification_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")),
            ]
        ),
        "random_forest_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2, class_weight="balanced")),
            ]
        ),
        "gradient_boosting_classifier": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingClassifier(random_state=seed))]
        ),
        "svc": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(C=1.0, probability=False, class_weight="balanced", random_state=seed)),
            ]
        ),
    }


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return math.nan, math.nan
    try:
        stat, pvalue = fn(y_true, y_pred)
        return float(stat), float(pvalue)
    except Exception:
        return math.nan, math.nan


def split_frames(dataset: pd.DataFrame, assignments: pd.DataFrame, split_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = assignments[assignments["split_id"] == split_id]
    train_ids = split[split["role"] == "train"]["rq0_id"]
    test_ids = split[split["role"] == "test"]["rq0_id"]
    train = dataset[dataset["rq0_id"].isin(set(train_ids))].copy()
    test = dataset[dataset["rq0_id"].isin(set(test_ids))].copy()
    return train, test


def positive_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return np.asarray(raw, dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--features", default="experiments/rq0_viability/data/processed/features_313.csv")
    parser.add_argument("--splits", default="experiments/rq0_viability/data/splits/split_assignments.csv")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    root = repo_root / "experiments/rq0_viability"
    logger = setup_logger(root / "outputs/logs/train_classical_baselines.log")
    dataset = pd.read_csv(repo_root / args.dataset)
    features = pd.read_csv(repo_root / args.features)
    assignments = pd.read_csv(repo_root / args.splits)
    df = dataset.merge(features, on=["rq0_id", "dataset_name"], how="inner")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if len(df) != len(dataset):
        raise RuntimeError(f"Feature merge changed row count: dataset={len(dataset)} merged={len(df)}")
    logger.info("loaded rows=%d features=%d splits=%d", len(df), len(feature_cols), assignments["split_id"].nunique())

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_meta = assignments.drop_duplicates("split_id").set_index("split_id")

    for split_idx, split_id in enumerate(sorted(assignments["split_id"].unique())):
        train, test = split_frames(df, assignments, split_id)
        meta = split_meta.loc[split_id]
        split_type = str(meta["split_type"])
        seed = int(meta["seed"])
        fold = int(meta["fold"])
        model_seed = seed if seed >= 0 else 0
        x_train = train[feature_cols]
        x_test = test[feature_cols]

        for target_name, target_col in [
            ("z_normalized", "human_score_z_within_dataset"),
            ("raw_score", "human_mean_score"),
        ]:
            y_train = train[target_col].astype(float).to_numpy()
            y_test = test[target_col].astype(float).to_numpy()
            for model_name, model in regression_models(model_seed).items():
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), dtype=float)
                pearson_r, pearson_p = safe_corr(pearsonr, y_test, pred)
                spearman_r, spearman_p = safe_corr(spearmanr, y_test, pred)
                kendall_tau, kendall_p = safe_corr(kendalltau, y_test, pred)
                metric_rows.append(
                    {
                        "split_id": split_id,
                        "split_type": split_type,
                        "seed": seed,
                        "fold": fold,
                        "task": "regression",
                        "target": target_name,
                        "model_group": "classical_feature",
                        "model": model_name,
                        "input": "features",
                        "training_setting": "supervised",
                        "n_train": len(train),
                        "n_test": len(test),
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "mae": float(mean_absolute_error(y_test, pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    }
                )
                if target_name == "z_normalized":
                    for rq0_id, truth, score in zip(test["rq0_id"], y_test, pred):
                        prediction_rows.append(
                            {
                                "split_id": split_id,
                                "split_type": split_type,
                                "seed": seed,
                                "fold": fold,
                                "task": "regression",
                                "target": target_name,
                                "model_group": "classical_feature",
                                "model": model_name,
                                "input": "features",
                                "training_setting": "supervised",
                                "rq0_id": rq0_id,
                                "y_true": truth,
                                "predicted_score": score,
                            }
                        )

        for label_name, label_col in [
            ("median", "binary_label_median"),
            ("strict_tertile", "binary_label_tertile_strict"),
        ]:
            train_c = train.copy()
            test_c = test.copy()
            if label_name == "strict_tertile":
                train_c = train_c[train_c[label_col] != "discard"].copy()
                test_c = test_c[test_c[label_col] != "discard"].copy()
            if train_c[label_col].nunique() < 2 or test_c[label_col].nunique() < 2:
                logger.warning("skip classification split=%s label=%s due to single class", split_id, label_name)
                continue
            y_train = (train_c[label_col] == "readable").astype(int).to_numpy()
            y_test = (test_c[label_col] == "readable").astype(int).to_numpy()
            x_train_c = train_c[feature_cols]
            x_test_c = test_c[feature_cols]
            for model_name, model in classification_models(model_seed).items():
                model.fit(x_train_c, y_train)
                pred = np.asarray(model.predict(x_test_c), dtype=int)
                score = positive_scores(model, x_test_c)
                try:
                    auc = float(roc_auc_score(y_test, score))
                except Exception:
                    auc = math.nan
                metric_rows.append(
                    {
                        "split_id": split_id,
                        "split_type": split_type,
                        "seed": seed,
                        "fold": fold,
                        "task": "classification",
                        "target": label_name,
                        "model_group": "classical_feature",
                        "model": model_name,
                        "input": "features",
                        "training_setting": "supervised",
                        "n_train": len(train_c),
                        "n_test": len(test_c),
                        "accuracy": float(accuracy_score(y_test, pred)),
                        "macro_f1": float(f1_score(y_test, pred, average="macro")),
                        "auc": auc,
                    }
                )

        if (split_idx + 1) % 10 == 0:
            logger.info("processed splits=%d/%d", split_idx + 1, assignments["split_id"].nunique())

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    out_dir = root / "outputs/classical_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "classical_metrics.csv"
    predictions_path = out_dir / "classical_regression_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    main_summary = (
        metrics[(metrics["task"] == "regression") & (metrics["target"] == "z_normalized")]
        .groupby(["split_type", "model"], dropna=False)
        [["spearman_r", "kendall_tau", "pearson_r", "mae", "rmse"]]
        .agg(["mean", "std", "count"])
    )
    summary_path = root / "outputs/tables/classical_regression_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    main_summary.to_csv(summary_path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str((repo_root / args.dataset).resolve()),
        "features": str((repo_root / args.features).resolve()),
        "splits": str((repo_root / args.splits).resolve()),
        "feature_count": len(feature_cols),
        "preprocessing_rule": "All imputers and scalers are inside sklearn Pipelines and fit on train split only.",
        "hyperparameter_rule": "Fixed defaults/configured constants; no test-set tuning.",
        "gradient_boosting_note": "sklearn GradientBoosting is used as the tree boosting baseline unless XGBoost/LightGBM is added later.",
        "outputs": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
        },
    }
    write_json(out_dir / "classical_manifest.json", manifest)
    logger.info("wrote metrics=%s rows=%d", metrics_path, len(metrics))
    logger.info("wrote predictions=%s rows=%d", predictions_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
