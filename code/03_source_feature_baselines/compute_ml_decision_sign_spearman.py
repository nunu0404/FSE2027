#!/usr/bin/env python3
"""Compute language-specific ML decision-sign Spearman correlations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "data/rq1_pairs_9000.csv"
PREDICTIONS_PATH = ROOT / "data/ml_predictions_9models_9000.csv"
OUTPUT_DIR = ROOT / "analysis/ml_decision_sign"

MODEL_ORDER = [
    "Multilayer Perceptron",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "svr",
    "Voting ensemble (LR+NB+RF)",
    "linear_regression",
    "Scalabrino-LR (replicated, not official tool)",
    "Logistic Regression",
    "Naive Bayes",
]

DISPLAY_NAMES = {
    "Multilayer Perceptron": "Multilayer Perceptron",
    "random_forest_regressor": "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "svr": "SVR",
    "Voting ensemble (LR+NB+RF)": "Voting (LR+NB+RF)",
    "linear_regression": "Linear Regression",
    "Scalabrino-LR (replicated, not official tool)": "Scalabrino-LR (replicated)",
    "Logistic Regression": "Logistic Regression",
    "Naive Bayes": "Naive Bayes",
}

EXPECTED_JAVA_REGRESSION_RHO = {
    "random_forest_regressor": 0.2926554333667295,
    "gradient_boosting_regressor": 0.27415175795386776,
    "svr": 0.26606329392823835,
    "linear_regression": 0.22364896923641495,
}


def correlation(model_sign: pd.Series, human_delta: pd.Series) -> tuple[float, float]:
    result = spearmanr(model_sign.astype(float), human_delta.astype(float))
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    pairs = pd.read_csv(PAIRS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if len(pairs) != 9000 or pairs["pair_id"].nunique() != 9000:
        raise ValueError("Expected exactly 9,000 unique pairs")
    if len(predictions) != 81000:
        raise ValueError("Expected exactly 81,000 ML pair predictions")
    if predictions[["language", "pair_id", "model"]].duplicated().any():
        raise ValueError("Duplicate language/pair/model prediction")
    if set(predictions["model"]) != set(MODEL_ORDER):
        raise ValueError("ML model set drift")
    if not np.isfinite(
        predictions[["predicted_score_i", "predicted_score_j"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite ML prediction score")

    pair_truth = pairs[
        ["pair_id", "language", "human_score_i_z", "human_score_j_z"]
    ].copy()
    pair_truth["human_delta_z"] = (
        pair_truth["human_score_i_z"] - pair_truth["human_score_j_z"]
    )
    merged = predictions.merge(
        pair_truth,
        on=["pair_id", "language"],
        how="left",
        validate="many_to_one",
    )
    if merged["human_delta_z"].isna().any():
        raise ValueError("Prediction-to-pair truth mapping failure")

    merged["predicted_score_diff"] = (
        merged["predicted_score_i"] - merged["predicted_score_j"]
    )
    merged["decision_sign"] = np.sign(merged["predicted_score_diff"]).astype(int)
    if not (
        merged["prediction_tie"].astype(bool) == merged["decision_sign"].eq(0)
    ).all():
        raise ValueError("Stored tie flag disagrees with score difference")

    rows: list[dict[str, object]] = []
    for (language, model), group in merged.groupby(["language", "model"], sort=False):
        non_tie = group[group["decision_sign"].ne(0)]
        rho_all, p_all = correlation(group["decision_sign"], group["human_delta_z"])
        rho_non_tie, p_non_tie = correlation(
            non_tie["decision_sign"], non_tie["human_delta_z"]
        )
        continuous_rho, continuous_p = correlation(
            group["predicted_score_diff"], group["human_delta_z"]
        )
        rows.append(
            {
                "language": language,
                "model": model,
                "display_model": DISPLAY_NAMES[model],
                "pairs": len(group),
                "ties": int(group["decision_sign"].eq(0).sum()),
                "decision_sign_spearman_all_zero_ties": rho_all,
                "decision_sign_p_all_zero_ties": p_all,
                "decision_sign_spearman_non_tie": rho_non_tie,
                "decision_sign_p_non_tie": p_non_tie,
                "continuous_diff_spearman": continuous_rho,
                "continuous_diff_p": continuous_p,
            }
        )

    results = pd.DataFrame(rows)
    results["model_order"] = results["model"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    results["language_order"] = results["language"].map(
        {"java": 0, "python": 1, "cuda": 2}
    )
    results = results.sort_values(["model_order", "language_order"]).drop(
        columns=["model_order", "language_order"]
    )

    for model, expected in EXPECTED_JAVA_REGRESSION_RHO.items():
        actual = results.loc[
            results["language"].eq("java") & results["model"].eq(model),
            "decision_sign_spearman_all_zero_ties",
        ].item()
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(
                f"Java regression rho mismatch for {model}: {actual} != {expected}"
            )

    wide = results.pivot(
        index="display_model",
        columns="language",
        values="decision_sign_spearman_all_zero_ties",
    ).reindex([DISPLAY_NAMES[model] for model in MODEL_ORDER])
    wide = wide[["java", "python", "cuda"]].reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "ml_decision_sign_spearman_by_language.csv", index=False)
    wide.to_csv(OUTPUT_DIR / "ml_decision_sign_spearman_wide.csv", index=False)
    metadata = {
        "status": "passed",
        "pairs": 9000,
        "models": 9,
        "prediction_rows": 81000,
        "primary_metric": (
            "Spearman(sign(predicted_score_i - predicted_score_j), "
            "human_score_i_z - human_score_j_z), all pairs; exact score ties use sign 0"
        ),
        "sensitivity_metric": "The same Spearman correlation after excluding exact score ties",
        "java_regression_reference_check": "passed at absolute tolerance 1e-12",
        "inputs": {
            "pairs": str(PAIRS_PATH),
            "predictions": str(PREDICTIONS_PATH),
        },
    }
    (OUTPUT_DIR / "METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print(wide.to_string(index=False))
    print("\nTie counts:")
    print(results.loc[results["ties"].gt(0), ["language", "display_model", "ties"]].to_string(index=False))


if __name__ == "__main__":
    main()
