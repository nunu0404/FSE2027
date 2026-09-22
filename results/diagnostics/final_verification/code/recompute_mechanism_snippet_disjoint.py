#!/usr/bin/env python3
"""Snippet-disjoint sensitivity analysis for the mechanism classifiers.

The historical analysis used pair-level StratifiedKFold. This script assigns
snippets to five folds. For fold k, test pairs have both endpoints in k and
training pairs have neither endpoint in k; crossing edges are excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_final_verification/mechanism_snippet_disjoint_auc.csv"
JAVA = ROOT / "results/complementarity_mechanism_clean_qwen_text_plus_image_20260706/final_mechanism_dataset.csv"
PYC = ROOT / "results/python_cuda_missing_experiments_20260716/mechanism_qwen_text_plus_image/final_mechanism_dataset.csv"
ALL_COND = ROOT / "results/python_cuda_missing_experiments_20260716/data/rf_vlm_all_conditions.csv"
SEED = 42
SUFFIXES = ("_diff_abs", "_gold_minus_loser", "_mean", "_max")


def feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    source = [c for c in data if c.startswith("source_") and c.endswith(SUFFIXES)]
    visual = [c for c in data if c.startswith("visual_") and c.endswith(SUFFIXES)]
    return source, visual


def language_of(snippet: str) -> str:
    if str(snippet).startswith("dorn_python_"):
        return "python"
    if str(snippet).startswith("dorn_cuda_"):
        return "cuda"
    return "java"


def assign_folds(data: pd.DataFrame) -> dict[str, int]:
    snippets = sorted(set(data["snippet_id_a"]) | set(data["snippet_id_b"]))
    rng = np.random.default_rng(SEED)
    mapping: dict[str, int] = {}
    by_language: dict[str, list[str]] = {}
    for snippet in snippets:
        by_language.setdefault(language_of(snippet), []).append(snippet)
    for language in sorted(by_language):
        values = np.asarray(by_language[language], dtype=object)
        rng.shuffle(values)
        for index, snippet in enumerate(values):
            mapping[str(snippet)] = index % 5
    return mapping


def evaluate(
    scope: str,
    target_name: str,
    data: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    folds: dict[str, int],
    feature_set: str,
    model: str = "gradient_boosting",
    setting: str = "qwen_text_plus_image",
) -> dict:
    work = data.copy()
    work["target"] = target.astype(int).to_numpy()
    work["fold_a"] = work["snippet_id_a"].map(folds)
    work["fold_b"] = work["snippet_id_b"].map(folds)
    if work[["fold_a", "fold_b"]].isna().any().any():
        raise RuntimeError("Missing snippet fold assignment")
    pred_rows = []
    fold_sizes = []
    for fold in range(5):
        train = work[(work["fold_a"] != fold) & (work["fold_b"] != fold)]
        test = work[(work["fold_a"] == fold) & (work["fold_b"] == fold)]
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            continue
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["difficulty"]),
            ],
            remainder="drop",
        )
        pipe = Pipeline([("pre", pre), ("model", GradientBoostingClassifier(random_state=SEED))])
        pipe.fit(train[numeric + ["difficulty"]], train["target"])
        proba = pipe.predict_proba(test[numeric + ["difficulty"]])[:, 1]
        pred_rows.append(pd.DataFrame({"y": test["target"].to_numpy(), "proba": proba}))
        fold_sizes.append((fold, len(train), len(test)))
    if not pred_rows:
        raise RuntimeError(f"No evaluable folds for {scope}/{target_name}/{feature_set}")
    pred = pd.concat(pred_rows, ignore_index=True)
    labels = (pred["proba"].to_numpy() >= 0.5).astype(int)
    return {
        "scope": scope,
        "model": model,
        "setting": setting,
        "target": target_name,
        "feature_set": feature_set,
        "source_feature_count": sum(c.startswith("source_") for c in numeric),
        "visual_feature_count": sum(c.startswith("visual_") for c in numeric),
        "control_feature_count": 3,
        "raw_input_columns_excluding_target": len(numeric) + 1,
        "total_target_rows": len(work),
        "evaluated_rows": len(pred),
        "excluded_cross_fold_rows": int(len(work) - len(pred)),
        "unique_snippets": len(folds),
        "evaluable_folds": len(pred_rows),
        "fold_train_test_n": ";".join(f"{f}:{tr}/{te}" for f, tr, te in fold_sizes),
        "auc": float(roc_auc_score(pred["y"], pred["proba"])),
        "balanced_accuracy": float(balanced_accuracy_score(pred["y"], labels)),
        "seed": SEED,
        "split_rule": "test: both endpoints in fold; train: neither endpoint in fold; crossing edges excluded",
    }


def main_scope(scope: str, path: Path) -> list[dict]:
    data = pd.read_csv(path)
    source, visual = feature_columns(data)
    controls = ["abs_human_score_gap", "rf_margin_abs", "rf_margin_percentile"]
    numeric = controls + source + visual
    folds = assign_folds(data)
    rf_wrong = ~data["rf_correct"].astype(bool)
    valid_wrong = rf_wrong & data["vlm_strict_swap_valid"].astype(bool)
    targets = [
        ("rf_failure", data, rf_wrong.astype(int)),
        (
            "vlm_rescue_given_rf_wrong",
            data.loc[rf_wrong].copy(),
            data.loc[rf_wrong, "vlm_correct"].astype(bool).astype(int),
        ),
        (
            "vlm_correct_given_rf_wrong_and_valid",
            data.loc[valid_wrong].copy(),
            data.loc[valid_wrong, "vlm_correct"].astype(bool).astype(int),
        ),
    ]
    return [evaluate(scope, name, frame, target, numeric, folds, "source_plus_visual") for name, frame, target in targets]


def all_condition_scope() -> list[dict]:
    base = pd.read_csv(PYC)
    source, visual = feature_columns(base)
    controls = ["abs_human_score_gap", "rf_margin_abs", "rf_margin_percentile"]
    folds = assign_folds(base)
    condition = pd.read_csv(ALL_COND)
    rows = []
    for (model, setting), block in condition.groupby(["model", "setting"], sort=True):
        merged = base.drop(columns=[c for c in ["vlm_strict_swap_valid", "vlm_correct"] if c in base]).merge(
            block[["pair_id", "vlm_valid", "vlm_correct"]], on="pair_id", how="inner", validate="one_to_one"
        )
        wrong = ~merged["rf_correct"].astype(bool)
        frame = merged.loc[wrong].copy()
        target = frame["vlm_valid"].astype(bool) & frame["vlm_correct"].astype(bool)
        for name, cols in (
            ("source_only", controls + source),
            ("source_plus_visual", controls + source + visual),
        ):
            rows.append(evaluate(
                "python_cuda_all_8_conditions",
                "effective_rescue_given_rf_wrong",
                frame,
                target.astype(int),
                cols,
                folds,
                name,
                model=model,
                setting=setting,
            ))
    return rows


def main() -> None:
    rows = main_scope("java", JAVA) + main_scope("python_cuda_pooled", PYC) + all_condition_scope()
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
