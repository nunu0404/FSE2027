#!/usr/bin/env python3
"""Verify the four numerical corrections requested in E3."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
OUT = PKG / "analysis/E3"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    corrections: list[dict[str, object]] = []

    # E3(a): exact deployment numerators and denominators.
    java = pd.read_csv(
        ROOT / "results/screenshot_only_ocr_clean_20260707/clean_multi_ocr_final_comparison_table.csv"
    )
    java_rf = java[
        (java.kind == "oracle_source_classical")
        & (java.model == "random_forest_regressor")
    ].iloc[0]
    java_rapid = java[
        (java.kind == "ocr_classical") & (java.ocr_engine == "rapidocr")
    ].sort_values("correct_pairs", ascending=False).iloc[0]
    java_loss = (java_rf.correct_pairs - java_rapid.correct_pairs) / java_rf.num_pairs

    pycu_source = pd.read_csv(
        ROOT / "results/python_cuda_missing_experiments_20260716/tables/clean_classical_results_overall.csv"
    )
    pycu_ocr = pd.read_csv(
        ROOT / "results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_results_overall.csv"
    )
    pycu_rf = pycu_source[pycu_source.model == "random_forest_regressor"].iloc[0]
    pycu_rapid = pycu_ocr[pycu_ocr.ocr_engine == "rapidocr"].sort_values(
        "correct_pairs" if "correct_pairs" in pycu_ocr else "pair_accuracy",
        ascending=False,
    ).iloc[0]
    pycu_loss = pycu_rf.pair_accuracy - pycu_rapid.pair_accuracy
    corrections.append({
        "item": "E3a Java OCR cost",
        "manuscript_value": "10.5 percentage points",
        "verified_value": java_loss * 100,
        "numerator": f"{int(java_rf.correct_pairs)}-{int(java_rapid.correct_pairs)}",
        "denominator": int(java_rf.num_pairs),
        "definition": (
            f"source RF ({java_rf.effective_accuracy:.6f}) minus best RapidOCR+ML "
            f"({java_rapid.model}, {java_rapid.effective_accuracy:.6f})"
        ),
        "code_path": "code/analyze_e3.py",
    })
    corrections.append({
        "item": "E3a Python+CUDA OCR cost",
        "manuscript_value": "11.90 percentage points",
        "verified_value": pycu_loss * 100,
        "numerator": f"{int(pycu_rf.correct_pairs)}-{int(round(pycu_rapid.pair_accuracy * pycu_rapid.pairs))}",
        "denominator": int(pycu_rf.pairs),
        "definition": "source RF minus best RapidOCR+ML (gradient boosting), pooled Python+CUDA",
        "code_path": "code/analyze_e3.py",
    })

    # E3(b): battery recovery.
    battery = pd.read_csv(
        ROOT / "results/rq1_model_battery_3lang_20260723/analysis/full/metrics_by_scope.csv"
    )
    recovery = battery[battery.scope.isin(["overall", "language"])].copy()
    recovery["recovery"] = recovery.debiased_accuracy - recovery.effective_accuracy
    recovery["aligned_descriptive"] = (
        recovery.spearman_valid_only.gt(0) & recovery.debiased_accuracy.gt(0.5)
    )
    recovery.to_csv(OUT / "E3_battery_recovery.csv", index=False)
    overall = recovery[recovery.scope == "overall"]
    aligned = overall[overall.aligned_descriptive]
    corrections.append({
        "item": "E3b five-model battery recovery range",
        "manuscript_value": "11.7-25.6 percentage points",
        "verified_value": f"{100*overall.recovery.min():.2f}-{100*overall.recovery.max():.2f} pp",
        "numerator": "debiased correct - strict-valid-and-correct, per model",
        "denominator": "9,000 pairs/model",
        "definition": "all five preregistered battery models",
        "code_path": "code/analyze_e3.py",
    })
    corrections.append({
        "item": "E3b descriptive aligned-model range",
        "manuscript_value": "not operationally defined",
        "verified_value": f"{100*aligned.recovery.min():.2f}-{100*aligned.recovery.max():.2f} pp",
        "numerator": "same recovery",
        "denominator": "9,000 pairs/model",
        "definition": (
            "descriptive only: pooled rho>0 and debiased accuracy>0.5; "
            "this rule was not preregistered and must not be called preregistered"
        ),
        "code_path": "code/analyze_e3.py",
    })

    # E3(c): tie and boundary are distinct events.
    decomp = pd.read_csv(
        ROOT / "results/grounded_protocol_3lang_20260721/analysis/"
        "logit_decomposition/logit_decomposition.csv"
    )
    decomp["tie_c0"] = decomp.c_content.eq(0)
    decomp["boundary_abs_c_eq_abs_b"] = decomp.c_content.abs().eq(decomp.b_position.abs())
    cell = (
        decomp.groupby(["experiment", "model", "language", "condition"])
        .agg(
            n=("pair_id", "size"),
            ties=("tie_c0", "sum"),
            tie_rate=("tie_c0", "mean"),
            boundaries=("boundary_abs_c_eq_abs_b", "sum"),
            boundary_rate=("boundary_abs_c_eq_abs_b", "mean"),
        )
        .reset_index()
    )
    cell.to_csv(OUT / "E3_tie_boundary_by_cell.csv", index=False)
    for scope, part in [
        ("all", decomp),
        ("grid", decomp[decomp.experiment == "grid"]),
        ("perturbation", decomp[decomp.experiment == "perturbation"]),
    ]:
        scoped_cell = cell if scope == "all" else cell[cell.experiment == scope]
        corrections.append({
            "item": f"E3c {scope} tie/boundary",
            "manuscript_value": "99,832 non-tied; 8,168 ties; cell ties 1.6%-5.4%",
            "verified_value": (
                f"n={len(part)}; ties={int(part.tie_c0.sum())} "
                f"({100*part.tie_c0.mean():.2f}%); boundaries="
                f"{int(part.boundary_abs_c_eq_abs_b.sum())} "
                f"({100*part.boundary_abs_c_eq_abs_b.mean():.2f}%); "
                f"cell tie range={100*scoped_cell.tie_rate.min():.1f}-"
                f"{100*scoped_cell.tie_rate.max():.1f}%"
            ),
            "numerator": "count(c==0); count(|c|==|b|)",
            "denominator": len(part),
            "definition": "tie and strict-validity boundary reported separately",
            "code_path": "code/analyze_e3.py",
        })

    # E3(d): the quoted ORs came from invalidity, not correctness.
    odds = pd.read_csv(
        ROOT / "results/strict_swap_difficulty_analysis_20260710/primary_pooled_gap_models.csv"
    )
    within = odds[
        odds.analysis == "within_difficulty_continuous_condition_FE_pair_clustered"
    ].copy()
    within["odds_ratio_per_plus_0_1_abs_z"] = within.coefficient.mul(0.1).map(math.exp)
    within.to_csv(OUT / "E3_odds_rescaled.csv", index=False)
    medium = within[within.contrast.str.contains("medium")].iloc[0]
    easy = within[within.contrast.str.contains("easy")].iloc[0]
    corrections.append({
        "item": "E3d odds-ratio sign and response",
        "manuscript_value": "OR 0.55/0.90 for a correct verdict",
        "verified_value": (
            f"OR +1: medium={medium.odds_ratio:.6f}, easy={easy.odds_ratio:.6f}; "
            f"OR +0.1: medium={medium.odds_ratio_per_plus_0_1_abs_z:.6f}, "
            f"easy={easy.odds_ratio_per_plus_0_1_abs_z:.6f}"
        ),
        "numerator": "strict-swap invalid pairs",
        "denominator": "8 conditions x 1,000 pairs within each stratum",
        "definition": (
            "cluster-robust condition-fixed-effect logistic regression; response=invalid "
            "(strict-swap error), not correct verdict; cluster=pair_id"
        ),
        "code_path": "experiments/rq0_viability/scripts/analyze_strict_swap_by_score_gap.py",
    })

    pd.DataFrame(corrections).to_csv(OUT / "E3_corrections.csv", index=False)
    report = f"""# E3 Numerical corrections

## (a) OCR cost

The Java source RF has {int(java_rf.correct_pairs)}/{int(java_rf.num_pairs)}
correct pairs ({100*java_rf.effective_accuracy:.2f}%). The best RapidOCR+ML row
is {java_rapid.model}, with {int(java_rapid.correct_pairs)}/{int(java_rapid.num_pairs)}
({100*java_rapid.effective_accuracy:.2f}%). The exact loss is
{100*java_loss:.2f} percentage points, not 10.5. The 10.5 value is the loss from
source RF to RapidOCR+RF ({100*(java_rf.effective_accuracy - java[java.system.str.startswith('OCR(rapidocr)+random_forest')].iloc[0].effective_accuracy):.2f} pp)
after rounding; it mixes a fixed RF comparison with a table row labeled as the
best OCR+ML system. Python+CUDA remains {100*pycu_loss:.2f} pp.

## (b) Battery recovery

Across all five preregistered models, debiasing recovers
{100*overall.recovery.min():.2f}-{100*overall.recovery.max():.2f} pp. The published
11.7-25.6 pp range excludes both Phi (the lower endpoint) and InternVL (the upper
endpoint). It exactly corresponds, up to rounding, to the post-hoc subset with
positive pooled rho and debiased accuracy above chance: Qwen, Gemma, and
Ministral. Because that alignment rule was not preregistered, the defensible main
statement is the five-model range; the restricted range may only be labeled
descriptive.

## (c) Ties and boundaries

The 108,000 observations comprise 72,000 grid observations (2 models x 3
languages x 12 conditions x 1,000 pairs) plus 36,000 perturbation observations
(2 x 3 x 6 x 1,000). There are 2,946 ties (`c=0`) and 8,168 boundaries
(`|c|=|b|`). Therefore 99,832 is the non-boundary count, not the non-tie count.
The reported 1.6-5.4% range is correctly the cell-level tie range over all 108
cells. The manuscript should use the same two explicit definitions everywhere.

## (d) Odds ratios

The source code defines the response as `invalid = ~is_valid_strict_swap`; it
does not model correct verdicts. OR<1 therefore means that larger score gaps
reduce strict-swap invalidity, which is directionally consistent with improved
reliability. The quoted medium/easy ORs are per +1.0 z, an extrapolation wider
than their respective bins. Per +0.1 z, the ORs are
{medium.odds_ratio_per_plus_0_1_abs_z:.3f} and
{easy.odds_ratio_per_plus_0_1_abs_z:.3f}. Replace “for a correct verdict” with
“for a strict-swap failure” and report the +0.1 scale.
"""
    (OUT / "E3_REPORT.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(corrections).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
