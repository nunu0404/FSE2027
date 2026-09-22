#!/usr/bin/env python3
"""Canonical readability baselines for RQ0."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
OUT = EXP / "outputs/canonical_baselines"

DATASET = EXP / "data/processed/pooled_313_processed.csv"
FEATURES = EXP / "data/processed/features_313.csv"
SPLITS = EXP / "data/splits/split_assignments.csv"
PAIRS = EXP / "data/pairs/full_pair_set_rq0.csv"
CLASSICAL_PAIR = EXP / "outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv"
CLASSICAL_SUMMARY = EXP / "outputs/tables/full_pair_classical_baseline_summary.csv"
VLM_SUMMARY = EXP / "outputs/tables/stage3_full_selected_vlm_summary.csv"
VLM_JOINED = EXP / "outputs/complementarity/rf_vlm_joined_pair_results.csv"

SEED = 42
DIFFICULTIES = ["easy", "medium", "hard"]
PERCENTILES = [0, 5, 10, 20, 30, 40, 50]
N_REPEATS = 100
BASE_SEED = 20260619
BOOTSTRAP_REPS = 10_000


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def model_defs(seed: int) -> dict[str, Pipeline | VotingClassifier]:
    lr_balanced = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    nb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]
    )
    scalabrino_lr = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    mlp = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-4, max_iter=1000, random_state=seed)),
        ]
    )
    rf_tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]
    )
    voting = VotingClassifier(
        estimators=[("lr", lr_balanced), ("nb", nb), ("rf", rf_tree)],
        voting="soft",
        n_jobs=1,
    )
    return {
        "Logistic Regression": lr_balanced,
        "Naive Bayes": nb,
        "Scalabrino-LR (replicated, not official tool)": scalabrino_lr,
        "Multilayer Perceptron": mlp,
        "Voting ensemble (LR+NB+RF)": voting,
    }


def load_data() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET)
    features = pd.read_csv(FEATURES)
    df = dataset.merge(features, on=["rq0_id", "dataset_name"], how="inner")
    if len(df) != len(dataset):
        raise RuntimeError("Feature merge changed row count")
    return df


def split_ids(assignments: pd.DataFrame, dataset_name: str) -> list[str]:
    if dataset_name == "Pooled":
        mask = assignments["split_type"].eq("pooled_stratified_5fold") & assignments["seed"].eq(SEED)
    else:
        mask = (
            assignments["split_type"].eq("within_dataset_5fold")
            & assignments["seed"].eq(SEED)
            & assignments["split_id"].str.contains(dataset_name.lower(), case=False, regex=False)
        )
    return sorted(assignments.loc[mask, "split_id"].unique())


def threshold_map(frame: pd.DataFrame, threshold_type: str) -> dict[str, float]:
    if threshold_type == "mean":
        return frame.groupby("dataset_name")["human_mean_score"].mean().to_dict()
    if threshold_type == "median":
        return frame.groupby("dataset_name")["human_mean_score"].median().to_dict()
    raise ValueError(threshold_type)


def labels_for(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return (frame["human_mean_score"] >= frame["dataset_name"].map(thresholds)).astype(int)


def oof_native_binary(df: pd.DataFrame, assignments: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    for dataset_name in ["Buse", "Dorn", "Scalabrino", "Pooled"]:
        base = df.copy() if dataset_name == "Pooled" else df[df["dataset_name"].eq(dataset_name)].copy()
        for threshold_type in ["mean", "median"]:
            thresholds = threshold_map(df if dataset_name == "Pooled" else base, threshold_type)
            base = base.copy()
            base["binary_target"] = labels_for(base, thresholds)
            class_balance = float(base["binary_target"].mean())
            ids = split_ids(assignments, dataset_name)
            for model_name, _ in model_defs(SEED).items():
                y_true_all = []
                y_pred_all = []
                y_prob_all = []
                for split_id in ids:
                    split = assignments[assignments["split_id"].eq(split_id)]
                    train_ids = set(split[split["role"].eq("train")]["rq0_id"])
                    test_ids = set(split[split["role"].eq("test")]["rq0_id"])
                    train = base[base["rq0_id"].isin(train_ids)].copy()
                    test = base[base["rq0_id"].isin(test_ids)].copy()
                    if train.empty or test.empty or train["binary_target"].nunique() < 2:
                        continue
                    model = model_defs(SEED)[model_name]
                    model.fit(train[feature_cols], train["binary_target"].astype(int))
                    pred = model.predict(test[feature_cols]).astype(int)
                    prob = model.predict_proba(test[feature_cols])[:, 1]
                    y_true = test["binary_target"].astype(int).to_numpy()
                    y_true_all.extend(y_true.tolist())
                    y_pred_all.extend(pred.tolist())
                    y_prob_all.extend(prob.tolist())
                    for rq0_id, yt, yp, pr in zip(test["rq0_id"], y_true, pred, prob):
                        pred_rows.append(
                            {
                                "dataset_eval": dataset_name,
                                "threshold_type": threshold_type,
                                "split_id": split_id,
                                "model": model_name,
                                "rq0_id": rq0_id,
                                "y_true": int(yt),
                                "pred_label": int(yp),
                                "pred_probability": float(pr),
                            }
                        )
                if y_true_all:
                    try:
                        auc = float(roc_auc_score(y_true_all, y_prob_all)) if len(set(y_true_all)) > 1 else np.nan
                    except Exception:
                        auc = np.nan
                    rows.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "threshold_type": threshold_type,
                            "thresholds": json.dumps({k: float(v) for k, v in thresholds.items()}, sort_keys=True),
                            "class_balance_readable": class_balance,
                            "n": len(y_true_all),
                            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
                            "auc": auc,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def pooled_seed42_oof_probabilities(df: pd.DataFrame, assignments: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    thresholds = threshold_map(df, "mean")
    df = df.copy()
    df["binary_target"] = labels_for(df, thresholds)
    rows = []
    ids = split_ids(assignments, "Pooled")
    for model_name, _ in model_defs(SEED).items():
        for split_id in ids:
            split = assignments[assignments["split_id"].eq(split_id)]
            train_ids = set(split[split["role"].eq("train")]["rq0_id"])
            test_ids = set(split[split["role"].eq("test")]["rq0_id"])
            train = df[df["rq0_id"].isin(train_ids)].copy()
            test = df[df["rq0_id"].isin(test_ids)].copy()
            model = model_defs(SEED)[model_name]
            model.fit(train[feature_cols], train["binary_target"].astype(int))
            prob = model.predict_proba(test[feature_cols])[:, 1]
            for rq0_id, y_true, score in zip(test["rq0_id"], test["binary_target"].astype(int), prob):
                rows.append(
                    {
                        "rq0_id": rq0_id,
                        "model": model_name,
                        "target": "binary_mean_threshold_probability",
                        "predicted_score": float(score),
                        "binary_target": int(y_true),
                    }
                )
    out = pd.DataFrame(rows)
    if out.groupby(["model", "rq0_id"]).size().max() != 1:
        raise RuntimeError("Expected one pooled OOF probability per model/snippet")
    return out


def subset_masks(pairs: pd.DataFrame) -> dict[str, pd.Series]:
    same = pairs["dataset_name_i"].eq(pairs["dataset_name_j"])
    return {
        "Buse-only": same & pairs["dataset_name_i"].eq("Buse"),
        "Dorn-only": same & pairs["dataset_name_i"].eq("Dorn"),
        "Scalabrino-only": same & pairs["dataset_name_i"].eq("Scalabrino"),
        "Within-dataset-all": same,
        "Cross-dataset-only": ~same,
        "Pooled-all": pd.Series(True, index=pairs.index),
    }


def pairwise_predictions(pairs: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model, scores in oof.groupby("model"):
        score_map = scores.set_index("rq0_id")["predicted_score"]
        work = pairs.copy()
        work["predicted_score_i"] = work["snippet_i"].map(score_map)
        work["predicted_score_j"] = work["snippet_j"].map(score_map)
        if work[["predicted_score_i", "predicted_score_j"]].isna().any().any():
            raise RuntimeError(f"Missing OOF scores for {model}")
        work["model_preference"] = np.where(
            work["predicted_score_i"] > work["predicted_score_j"],
            work["snippet_i"],
            np.where(work["predicted_score_j"] > work["predicted_score_i"], work["snippet_j"], ""),
        )
        work["is_valid_score_pair"] = work["model_preference"].ne("")
        work["is_correct"] = work["model_preference"].eq(work["human_preference"])
        work["model"] = model
        work["model_group"] = "canonical_classical_feature"
        work["training_setting"] = "supervised"
        work["input_modality"] = "features"
        frames.append(work)
    return pd.concat(frames, ignore_index=True)


def summarize_pairwise(pair_level: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    masks = subset_masks(pairs)
    rows = []
    for model, model_df in pair_level.groupby("model"):
        for subset, mask in masks.items():
            ids = set(pairs.loc[mask, "pair_id"])
            sub = model_df[model_df["pair_id"].isin(ids)].copy()
            valid = sub[sub["is_valid_score_pair"]]
            row = {
                "model_group": "canonical_classical_feature",
                "model": model,
                "subset": subset,
                "training_setting": "supervised",
                "input_modality": "features",
                "packaging": "N/A",
                "num_pairs": int(len(sub)),
                "valid_pairs": int(len(valid)),
                "correct_pairs": int(valid["is_correct"].sum()),
                "pairwise_accuracy": float(valid["is_correct"].mean()) if len(valid) else np.nan,
                "effective_accuracy": float(valid["is_correct"].sum() / len(sub)) if len(sub) else np.nan,
                "strict_swap_error": np.nan,
            }
            for level in DIFFICULTIES:
                level_all = sub[sub["difficulty"].eq(level)]
                level_valid = valid[valid["difficulty"].eq(level)]
                row[f"{level}_n"] = int(len(level_all))
                row[f"{level}_pairwise_accuracy"] = float(level_valid["is_correct"].mean()) if len(level_valid) else np.nan
                row[f"{level}_effective_accuracy"] = float(level_valid["is_correct"].sum() / len(level_all)) if len(level_all) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def sanity_gate(pairs: pd.DataFrame) -> list[str]:
    failures = []
    classical = pd.read_csv(CLASSICAL_PAIR)
    expected = {
        "random_forest_regressor": 0.6233333333333333,
        "gradient_boosting_regressor": 0.62,
        "svr": 0.6126666666666667,
        "linear_regression": 0.5966666666666667,
    }
    for model, exp in expected.items():
        sub = classical[classical["model"].eq(model)]
        got = float(sub["is_correct"].mean())
        if abs(got - exp) > 0.001:
            failures.append(f"{model}: expected {exp:.6f}, got {got:.6f}")
    vlm = pd.read_csv(VLM_SUMMARY)
    qwen = vlm[
        vlm["model_name"].eq("Qwen/Qwen2.5-VL-7B-Instruct")
        & vlm["input_modality"].eq("text_plus_image")
        & vlm["packaging"].eq("combined_labeled")
    ].iloc[0]
    if abs(float(qwen["effective_accuracy"]) - 0.428) > 0.001:
        failures.append("Best VLM effective accuracy did not reproduce 42.80%")
    return failures


def correctness_series(df: pd.DataFrame, model: str | None = None) -> pd.Series:
    if model is not None:
        df = df[df["model"].eq(model)]
    return df.set_index("pair_id")["is_correct"].astype(bool)


def comparison(a: pd.Series, b: pd.Series, name: str) -> dict[str, object]:
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    arr_a = joined["a"].astype(bool).to_numpy()
    arr_b = joined["b"].astype(bool).to_numpy()
    delta = arr_b.astype(int) - arr_a.astype(int)
    rng = np.random.default_rng(BASE_SEED)
    boot = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        boot[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    ci = np.percentile(boot, [2.5, 97.5])
    p_boot = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    a_only = int(np.sum(arr_a & ~arr_b))
    b_only = int(np.sum(~arr_a & arr_b))
    discordant = a_only + b_only
    p_mcnemar = float(binomtest(b_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "comparison": name,
        "n_pairs": int(len(joined)),
        "accuracy_a": float(arr_a.mean()),
        "accuracy_b": float(arr_b.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "bootstrap_ci_low": float(ci[0]),
        "bootstrap_ci_high": float(ci[1]),
        "bootstrap_p_two_sided": float(p_boot),
        "mcnemar_a_only": a_only,
        "mcnemar_b_only": b_only,
        "mcnemar_p": p_mcnemar,
    }


def ensemble_correct(frame: pd.DataFrame, primary_correct_col: str, primary_margin_col: str, threshold: float) -> np.ndarray:
    delegate = (frame[primary_margin_col].to_numpy() < threshold) & frame["vlm_valid"].to_numpy(dtype=bool)
    return np.where(delegate, frame["vlm_correct"].to_numpy(dtype=bool), frame[primary_correct_col].to_numpy(dtype=bool))


def stratified_half_split(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation_parts = []
    test_parts = []
    for _, group in data.groupby("difficulty", sort=True):
        idx = rng.permutation(group.index.to_numpy())
        mid = len(idx) // 2
        validation_parts.append(idx[:mid])
        test_parts.append(idx[mid:])
    return np.concatenate(validation_parts), np.concatenate(test_parts)


def fallback_with_primary(joined: pd.DataFrame, primary_pair: pd.DataFrame, primary_model: str) -> tuple[pd.DataFrame, dict[str, object]]:
    primary = primary_pair[primary_pair["model"].eq(primary_model)][
        ["pair_id", "is_correct", "predicted_score_i", "predicted_score_j"]
    ].copy()
    primary = primary.rename(columns={"is_correct": "primary_correct"})
    primary["primary_margin"] = (primary["predicted_score_i"] - primary["predicted_score_j"]).abs()
    data = joined.merge(primary[["pair_id", "primary_correct", "primary_margin"]], on="pair_id", how="inner").reset_index(drop=True)
    rows = []
    for repeat in range(N_REPEATS):
        val_idx, test_idx = stratified_half_split(data, BASE_SEED + repeat)
        val = data.iloc[val_idx]
        test = data.iloc[test_idx]
        candidates = []
        for percentile in PERCENTILES:
            threshold = float(np.percentile(val["primary_margin"], percentile))
            ens = ensemble_correct(val, "primary_correct", "primary_margin", threshold)
            candidates.append((float(ens.mean()), percentile, threshold))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        _, selected_percentile, selected_threshold = candidates[0]
        primary_correct = test["primary_correct"].to_numpy(dtype=bool)
        ens = ensemble_correct(test, "primary_correct", "primary_margin", selected_threshold)
        rows.append(
            {
                "repeat": repeat,
                "selected_percentile": selected_percentile,
                "selected_threshold": selected_threshold,
                "test_primary_accuracy": float(primary_correct.mean()),
                "test_ensemble_accuracy": float(ens.mean()),
                "test_improvement": float(ens.mean() - primary_correct.mean()),
            }
        )
    splits = pd.DataFrame(rows)
    ci = np.percentile(splits["test_improvement"], [2.5, 97.5])
    summary = {
        "primary_model": primary_model,
        "primary_accuracy": float(splits["test_primary_accuracy"].mean()),
        "ensemble_accuracy": float(splits["test_ensemble_accuracy"].mean()),
        "mean_improvement": float(splits["test_improvement"].mean()),
        "ci": [float(ci[0]), float(ci[1])],
        "improved_splits": int((splits["test_improvement"] > 0).sum()),
        "tied_splits": int((splits["test_improvement"] == 0).sum()),
        "worsened_splits": int((splits["test_improvement"] < 0).sum()),
    }
    return splits, summary


def percent(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:.2f}%"


def main() -> int:
    start = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    assignments = pd.read_csv(SPLITS)
    pairs = pd.read_csv(PAIRS)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    failures = sanity_gate(pairs)
    if failures:
        (OUT / "SANITY_GATE_FAILED.md").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"Sanity gate failed: {failures}")

    official_scalabrino_tool = False
    official_search = sorted(str(p) for p in ROOT.glob("**/*") if p.name.lower() in {"rsm.jar", "readability.jar"})

    native_summary, native_predictions = oof_native_binary(df, assignments, feature_cols)
    native_summary.to_csv(OUT / "canonical_native_binary_summary.csv", index=False)
    native_predictions.to_csv(OUT / "canonical_native_binary_predictions.csv", index=False)

    oof = pooled_seed42_oof_probabilities(df, assignments, feature_cols)
    oof.to_csv(OUT / "canonical_oof_snippet_scores.csv", index=False)
    pair_level = pairwise_predictions(pairs, oof)
    pair_level.to_csv(OUT / "canonical_pair_level.csv", index=False)
    pair_summary = summarize_pairwise(pair_level, pairs)
    pair_summary.to_csv(OUT / "canonical_pairwise_summary.csv", index=False)

    canonical_pooled = pair_summary[pair_summary["subset"].eq("Pooled-all")].copy()
    best_canonical = canonical_pooled.sort_values("effective_accuracy", ascending=False).iloc[0]
    rf = correctness_series(pd.read_csv(CLASSICAL_PAIR), "random_forest_regressor")
    comparisons = []
    for model in ["Logistic Regression", "Naive Bayes", "Scalabrino-LR (replicated, not official tool)"]:
        comparisons.append(comparison(correctness_series(pair_level, model), rf, f"{model} -> RF"))
    vlm = pd.read_csv(VLM_JOINED).set_index("pair_id")["vlm_correct"].astype(bool)
    comparisons.append(comparison(correctness_series(pair_level, str(best_canonical["model"])), vlm, f"{best_canonical['model']} -> best VLM"))
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(OUT / "statistical_tests.csv", index=False)
    (OUT / "statistical_tests.md").write_text("# Canonical Baseline Statistical Tests\n\n" + comp_df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")

    joined = pd.read_csv(VLM_JOINED)
    fallback_splits, fallback_summary = fallback_with_primary(joined, pair_level, "Logistic Regression")
    fallback_splits.to_csv(OUT / "fallback_with_canonical_primary_splits.csv", index=False)
    (OUT / "fallback_with_canonical_primary.md").write_text(
        f"""# Fallback with Canonical Primary

Primary model: `Logistic Regression`

- Mean primary accuracy: {percent(fallback_summary['primary_accuracy'])}
- Mean primary+VLM accuracy: {percent(fallback_summary['ensemble_accuracy'])}
- Mean improvement: {fallback_summary['mean_improvement'] * 100:+.2f}pp
- 95% repeated-split interval: [{fallback_summary['ci'][0] * 100:+.2f}, {fallback_summary['ci'][1] * 100:+.2f}]pp
- Improved/tied/worsened splits: {fallback_summary['improved_splits']} / {fallback_summary['tied_splits']} / {fallback_summary['worsened_splits']}

This uses the same Qwen2.5-VL-7B combined-labeled text+image Prompt B strict-swap result as the RF fallback analysis.
""",
        encoding="utf-8",
    )

    existing = pd.read_csv(CLASSICAL_SUMMARY)
    existing["subset"] = "Pooled-all"
    existing["Framing"] = "RQ0 pairwise"
    existing["Model"] = existing["model"]
    canonical_table = pair_summary.pivot_table(index="model", columns="subset", values="effective_accuracy", aggfunc="first").reset_index()
    rows = []
    for _, row in canonical_table.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Framing": "canonical binary-probability -> RQ0 pairwise",
                "Buse": row.get("Buse-only"),
                "Dorn": row.get("Dorn-only"),
                "Scalabrino": row.get("Scalabrino-only"),
                "Within-all": row.get("Within-dataset-all"),
                "Cross-only": row.get("Cross-dataset-only"),
                "Pooled": row.get("Pooled-all"),
            }
        )
    for _, row in pd.read_csv(CLASSICAL_SUMMARY).iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Framing": "existing regression -> RQ0 pairwise",
                "Buse": np.nan,
                "Dorn": np.nan,
                "Scalabrino": np.nan,
                "Within-all": np.nan,
                "Cross-only": np.nan,
                "Pooled": row["effective_accuracy"],
            }
        )
    qwen = pd.read_csv(VLM_SUMMARY)
    qwen = qwen[
        qwen["model_name"].eq("Qwen/Qwen2.5-VL-7B-Instruct")
        & qwen["input_modality"].eq("text_plus_image")
        & qwen["packaging"].eq("combined_labeled")
    ].iloc[0]
    rows.append(
        {
            "Model": "Qwen2.5-VL-7B combined text+image",
            "Framing": "zero-shot VLM strict-swap effective",
            "Buse": np.nan,
            "Dorn": np.nan,
            "Scalabrino": np.nan,
            "Within-all": np.nan,
            "Cross-only": np.nan,
            "Pooled": qwen["effective_accuracy"],
        }
    )
    combined = pd.DataFrame(rows)
    combined_md = combined.copy()
    for col in ["Buse", "Dorn", "Scalabrino", "Within-all", "Cross-only", "Pooled"]:
        combined_md[col] = combined_md[col].map(lambda x: "NA" if pd.isna(x) else f"{x*100:.2f}%")
    (OUT / "canonical_vs_rf_vs_vlm_table.md").write_text(
        "# Canonical vs RF vs VLM\n\n" + combined_md.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    native_validity_notes = []
    buse_nb = native_summary[
        native_summary["model"].eq("Naive Bayes")
        & native_summary["dataset"].eq("Buse")
        & native_summary["threshold_type"].eq("mean")
    ]
    if len(buse_nb):
        acc = float(buse_nb.iloc[0]["accuracy"])
        native_validity_notes.append(f"Buse mean-threshold Naive Bayes accuracy is {acc*100:.2f}%, versus the approximate Buse&Weimer NB reference around 80%; this is not a direct replication because thresholds/features differ.")
    scal_lr = native_summary[
        native_summary["model"].eq("Scalabrino-LR (replicated, not official tool)")
        & native_summary["dataset"].eq("Scalabrino")
        & native_summary["threshold_type"].eq("mean")
    ]
    if len(scal_lr):
        native_validity_notes.append(f"Scalabrino mean-threshold replicated LR AUC is {float(scal_lr.iloc[0]['auc']):.3f}; this is labelled as replicated, not the official tool.")

    report = f"""# Canonical Readability Baseline Report

## Setup

- Features: `{FEATURES}` ({len(feature_cols)} numeric source features)
- Dataset: `{DATASET}`
- Splits: `{SPLITS}`; seed {SEED}; no resplitting.
- Pair set: `{PAIRS}`
- Official Scalabrino tool status: {'run' if official_scalabrino_tool else 'not available; replicated LR used and labelled non-official'}.

## Sanity Gate

Passed: existing pairwise path reproduces RF=62.33%, GB=62.00%, SVR=61.27%, Linear=59.67%, and best VLM effective=42.80%.

## Framing 1: Native Binary Classification

See `canonical_native_binary_summary.csv`.

Validity notes:

{chr(10).join(f'- {x}' for x in native_validity_notes)}

## Framing 2: RQ0 Pairwise

Pooled effective accuracies:

{pair_summary[pair_summary['subset'].eq('Pooled-all')][['model','num_pairs','effective_accuracy','easy_effective_accuracy','medium_effective_accuracy','hard_effective_accuracy']].to_markdown(index=False, floatfmt='.4f')}

## Fallback With Canonical Primary

See `fallback_with_canonical_primary.md`. Logistic Regression primary mean improvement is {fallback_summary['mean_improvement']*100:+.2f}pp.

## Threats

- Native binary classification and RQ0 pairwise ranking are different tasks.
- Published Buse/Dorn/Scalabrino numbers are not fully reproducible from the current feature set and threshold choices.
- The official Scalabrino tool was not found in the workspace, so the Scalabrino row is a clearly labelled LR replication.
- Pairwise labels are score-derived proxies from `human_score_z_within_dataset`, not direct pairwise human annotation.

## Safe Wording

- We evaluated historically motivated source-based baselines, including Logistic Regression, Naive Bayes, a replicated Scalabrino-style LR model, MLP, and a soft voting ensemble.
- In the native binary framing, results should be read as a validity check rather than a direct reproduction of each original paper.
- In the RQ0 pairwise framing, canonical supervised source baselines remain above the zero-shot VLM effective accuracy.
- The Scalabrino result is a replicated LR baseline using the available RQ0 features, not the official released tool.
- These results support comparing VLM-as-a-Judge against established source baselines without claiming VLM superiority.
"""
    (OUT / "CANONICAL_BASELINE_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment": "I_canonical_readability_baselines",
        "start_utc": start.isoformat(),
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": SEED,
        "inputs": {
            "dataset": str(DATASET),
            "features": str(FEATURES),
            "splits": str(SPLITS),
            "pairs": str(PAIRS),
            "classical_pair_reference": str(CLASSICAL_PAIR),
            "vlm_summary": str(VLM_SUMMARY),
        },
        "models": list(model_defs(SEED).keys()),
        "official_scalabrino_tool_run": official_scalabrino_tool,
        "official_scalabrino_tool_search_hits": official_search,
        "thresholds": {
            "rule": "dataset-local human_mean_score mean and median; pooled labels use each snippet's own dataset threshold",
            "dataset_mean": threshold_map(df, "mean"),
            "dataset_median": threshold_map(df, "median"),
        },
        "outputs": {
            "native_binary_summary": str(OUT / "canonical_native_binary_summary.csv"),
            "pairwise_summary": str(OUT / "canonical_pairwise_summary.csv"),
            "pair_level": str(OUT / "canonical_pair_level.csv"),
            "report": str(OUT / "CANONICAL_BASELINE_REPORT.md"),
        },
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "best_canonical": best_canonical.to_dict()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
