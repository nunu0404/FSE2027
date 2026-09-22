#!/usr/bin/env python3
"""E1: language-specific debiased VLM versus paired ML baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("/ANON/experiment_root")
PACKAGE = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
BOOTSTRAP_REPS = 10_000
SEED = 20260730
RF_NAME = "random_forest_regressor"


def mcnemar_exact(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    p = (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return left_only, right_only, p


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_differences(
    pairs: pd.DataFrame,
    outcomes: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    snippets = sorted(set(pairs.snippet_i) | set(pairs.snippet_j))
    index = {snippet: i for i, snippet in enumerate(snippets)}
    left = pairs.snippet_i.map(index).to_numpy()
    right = pairs.snippet_j.map(index).to_numpy()
    values = np.column_stack([outcomes[key] for key in outcomes])
    rng = np.random.default_rng(SEED)
    estimates = np.empty((BOOTSTRAP_REPS, len(outcomes)), dtype=float)
    batch_size = 250
    probabilities = np.full(len(snippets), 1 / len(snippets))
    for start in range(0, BOOTSTRAP_REPS, batch_size):
        stop = min(BOOTSTRAP_REPS, start + batch_size)
        multiplicity = rng.multinomial(
            len(snippets), probabilities, size=stop - start
        )
        weights = multiplicity[:, left] * multiplicity[:, right]
        denominators = weights.sum(axis=1)
        if np.any(denominators == 0):
            raise RuntimeError("snippet bootstrap produced an empty pair sample")
        estimates[start:stop] = weights @ values / denominators[:, None]
    return {
        key: (
            float(np.quantile(estimates[:, column], 0.025)),
            float(np.quantile(estimates[:, column], 0.975)),
        )
        for column, key in enumerate(outcomes)
    }


def main() -> None:
    pairs = pd.read_csv(BATTERY / "data/rq1_pairs_9000.csv")
    vlm = pd.read_csv(BATTERY / "analysis/full/pair_level_results.csv")
    ml = pd.read_csv(BATTERY / "data/ml_predictions_9models_9000.csv")
    pair_meta = pairs[
        [
            "protocol_pair_id",
            "pair_id",
            "snippet_i",
            "snippet_j",
            "language",
        ]
    ].rename(columns={"protocol_pair_id": "vlm_pair_id", "pair_id": "source_pair_id"})
    vlm = vlm.merge(
        pair_meta,
        left_on=["pair_id", "language"],
        right_on=["vlm_pair_id", "language"],
        how="left",
        validate="many_to_one",
    )
    if vlm.source_pair_id.isna().any():
        raise RuntimeError("VLM pair IDs do not map to source pair IDs")

    accuracy = (
        ml.groupby(["language", "model"], as_index=False)
        .is_correct.mean()
        .sort_values(["language", "is_correct", "model"], ascending=[True, False, True])
    )
    best_names = accuracy.groupby("language", sort=False).first()["model"].to_dict()
    expected_best = {
        "cuda": "svr",
        "java": "Multilayer Perceptron",
        "python": "Voting ensemble (LR+NB+RF)",
    }
    if best_names != expected_best:
        raise RuntimeError(f"language-best drift: {best_names}")

    comparison_records: list[dict] = []
    result_rows: list[dict] = []
    for language, language_pairs in pairs.groupby("language", sort=True):
        language_pairs = language_pairs.reset_index(drop=True)
        ml_language = ml[ml.language.eq(language)]
        baseline_arrays: dict[str, np.ndarray] = {}
        for label, model_name in (
            ("rf", RF_NAME),
            ("best", best_names[language]),
        ):
            selected = ml_language[ml_language.model.eq(model_name)].set_index("pair_id")
            aligned = selected.reindex(language_pairs.pair_id)
            if aligned.is_correct.isna().any() or len(aligned) != 3000:
                raise RuntimeError(f"{language}/{model_name}: incomplete paired baseline")
            baseline_arrays[label] = aligned.is_correct.astype(bool).to_numpy()

        bootstrap_outcomes: dict[str, np.ndarray] = {}
        model_frames: dict[str, pd.DataFrame] = {}
        for model_key, frame in vlm[vlm.language.eq(language)].groupby(
            "model_key", sort=True
        ):
            aligned = frame.set_index("source_pair_id").reindex(language_pairs.pair_id)
            if len(aligned) != 3000 or aligned.debiased_correct.isna().any():
                raise RuntimeError(f"{model_key}/{language}: incomplete paired VLM data")
            model_frames[model_key] = aligned
            debiased = aligned.debiased_correct.astype(bool).to_numpy()
            for baseline_label, baseline in baseline_arrays.items():
                bootstrap_outcomes[f"{model_key}__{baseline_label}"] = (
                    debiased.astype(float) - baseline.astype(float)
                )
        intervals = bootstrap_differences(language_pairs, bootstrap_outcomes)

        for model_key, aligned in model_frames.items():
            total = len(aligned)
            valid = aligned.valid.astype(bool)
            correct = aligned.correct.astype(bool)
            ties = aligned.content_tie.astype(bool)
            non_tie = ~ties
            debiased = aligned.debiased_correct.astype(bool).to_numpy()
            boundary = np.isclose(
                aligned.content_margin.abs().to_numpy(),
                aligned.position_margin.abs().to_numpy(),
                rtol=0,
                atol=0,
            )
            row = {
                "run_id": "rq1_model_battery_3lang_20260723",
                "model_key": model_key,
                "model": aligned.iloc[0].model,
                "language": language,
                "n_pairs": total,
                "valid_pairs": int(valid.sum()),
                "valid_accuracy": float(correct[valid].mean()) if valid.any() else math.nan,
                "effective_accuracy": float(correct.mean()),
                "strict_swap_error": float((~valid).mean()),
                "content_ties": int(ties.sum()),
                "boundary_pairs_abs_c_eq_abs_b": int(boundary.sum()),
                "debiased_main_tie_incorrect": float(debiased.mean()),
                "debiased_excluding_ties": (
                    float(debiased[non_tie].mean()) if non_tie.any() else math.nan
                ),
                "debiased_excluding_ties_n": int(non_tie.sum()),
                "debiased_tie_half_credit": float(
                    (debiased.sum() + 0.5 * ties.sum()) / total
                ),
                "ab_only_accuracy": float(aligned.ab_correct.astype(bool).mean()),
                "rf_name": RF_NAME,
                "rf_accuracy": float(baseline_arrays["rf"].mean()),
                "language_best_name": best_names[language],
                "language_best_accuracy": float(baseline_arrays["best"].mean()),
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_cluster": "two-way snippet multiplicity",
                "baseline_predictions": "out-of-fold, same source pair IDs",
            }
            for label, baseline in baseline_arrays.items():
                key = f"{model_key}__{label}"
                left_only, right_only, p = mcnemar_exact(debiased, baseline)
                low, high = intervals[key]
                prefix = "rf" if label == "rf" else "language_best"
                row[f"diff_vs_{prefix}"] = float(debiased.mean() - baseline.mean())
                row[f"diff_vs_{prefix}_ci_low"] = low
                row[f"diff_vs_{prefix}_ci_high"] = high
                row[f"mcnemar_vlm_only_correct_vs_{prefix}"] = left_only
                row[f"mcnemar_{prefix}_only_correct"] = right_only
                row[f"mcnemar_exact_p_vs_{prefix}"] = p
                comparison_records.append(
                    {
                        "row_index": len(result_rows),
                        "prefix": prefix,
                        "p": p,
                    }
                )
            result_rows.append(row)

    adjusted = holm_adjust([record["p"] for record in comparison_records])
    for record, value in zip(comparison_records, adjusted):
        row = result_rows[record["row_index"]]
        prefix = record["prefix"]
        row[f"mcnemar_holm_p_vs_{prefix}"] = value
        row[f"mcnemar_holm_reject_vs_{prefix}"] = value < 0.05

    result = pd.DataFrame(result_rows).sort_values(["model_key", "language"])
    output_dir = PACKAGE / "analysis/E1"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "E1_debiased_vs_baseline.csv", index=False)

    exceeds = []
    for _, row in result.iterrows():
        for prefix in ("rf", "language_best"):
            diff = row[f"diff_vs_{prefix}"]
            low = row[f"diff_vs_{prefix}_ci_low"]
            high = row[f"diff_vs_{prefix}_ci_high"]
            if diff > 0:
                exceeds.append(
                    {
                        "model": row.model_key,
                        "language": row.language,
                        "baseline": prefix,
                        "diff": diff,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "holm_p": row[f"mcnemar_holm_p_vs_{prefix}"],
                    }
                )
    if any(item["diff"] > 0 and item["ci_low"] > 0 for item in exceeds):
        conclusion = "c"
    elif exceeds:
        conclusion = "b"
    else:
        conclusion = "a"
    summary = {
        "conclusion": conclusion,
        "rule": {
            "a": "no debiased VLM cell exceeds either paired baseline",
            "b": "at least one exceeds a baseline, but no positive difference excludes zero",
            "c": "at least one exceeds a baseline with a positive CI excluding zero",
        }[conclusion],
        "comparisons": 30,
        "holm_family": "15 model-language cells x 2 baselines",
        "exceeding_comparisons": exceeds,
        "language_best": best_names,
    }
    (output_dir / "E1_CONCLUSION.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    draft_columns = [
        "model", "language", "n_pairs", "valid_accuracy", "effective_accuracy",
        "strict_swap_error", "debiased_main_tie_incorrect", "ab_only_accuracy",
        "rf_accuracy", "language_best_name", "language_best_accuracy",
        "diff_vs_rf", "diff_vs_rf_ci_low", "diff_vs_rf_ci_high",
        "diff_vs_language_best", "diff_vs_language_best_ci_low",
        "diff_vs_language_best_ci_high",
    ]
    python_qwen = result[
        result.model_key.eq("qwen") & result.language.eq("python")
    ].iloc[0]
    draft = f"""# E1 manuscript draft

{result[draft_columns].to_markdown(index=False, floatfmt=".4f")}

Two-order logit averaging was compared with deterministic source-feature
baselines on the same out-of-fold pair predictions. Across 15 model-language
cells, Qwen/Python was the only cell whose point estimate exceeded either
baseline: {python_qwen.debiased_main_tie_incorrect:.2%} versus
{python_qwen.rf_accuracy:.2%} for random forest and
{python_qwen.language_best_accuracy:.2%} for the language-best
{python_qwen.language_best_name}. The paired differences were
{python_qwen.diff_vs_rf:+.2%} (two-way snippet-cluster bootstrap 95% CI
[{python_qwen.diff_vs_rf_ci_low:+.2%}, {python_qwen.diff_vs_rf_ci_high:+.2%}])
and {python_qwen.diff_vs_language_best:+.2%}
([{python_qwen.diff_vs_language_best_ci_low:+.2%},
{python_qwen.diff_vs_language_best_ci_high:+.2%}]), respectively. Neither
cluster interval excluded zero, although the pair-level exact McNemar test
against RF remained significant after Holm adjustment; this difference reflects
the stronger dependence correction in the snippet-cluster interval. We
therefore classify the result as conclusion (b): a VLM point estimate exceeds a
baseline in one language, but the excess is not distinguishable under the
cluster uncertainty analysis. The claim that no VLM exceeds a baseline “in any
language” should be weakened accordingly.
"""
    (output_dir / "E1_MANUSCRIPT_DRAFT.md").write_text(draft, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(
        result[
            [
                "model_key",
                "language",
                "debiased_main_tie_incorrect",
                "rf_accuracy",
                "diff_vs_rf",
                "diff_vs_rf_ci_low",
                "diff_vs_rf_ci_high",
                "language_best_name",
                "language_best_accuracy",
                "diff_vs_language_best",
                "diff_vs_language_best_ci_low",
                "diff_vs_language_best_ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
