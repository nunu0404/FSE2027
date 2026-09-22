#!/usr/bin/env python3
"""Build human- and paper-facing reports plus a SHA-256 artifact inventory."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.{digits}f}%"


def number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ga0 = json.loads((OUT / "audit/GA0_REPORT.json").read_text())
    execution = json.loads((OUT / "audit/ANALYSIS_EXECUTION.json").read_text())
    if ga0["status"] != "passed" or execution["new_model_calls"] != 0:
        raise RuntimeError("Cannot build reports before passed zero-call analysis")

    a = pd.read_csv(OUT / "A_preference_flip/A_paper_summary_table.csv")
    a_summary = pd.read_csv(OUT / "A_preference_flip/A_flip_summary.csv")
    a_factor = pd.read_csv(OUT / "A_preference_flip/A_flip_by_factor.csv")
    a_matrix = pd.read_csv(OUT / "A_preference_flip/A_flip_matrix.csv")
    b = pd.read_csv(OUT / "B_content_calibration/B_calibration_summary.csv")
    c = pd.read_csv(OUT / "C_bc_joint/C_bc_summary.csv")
    d = pd.read_csv(OUT / "D_first_position/D_first_position_by_language.csv")

    a_display = a.copy()
    a_display["model"] = a_display["model"].replace(
        {
            "OpenGVLab/InternVL3-8B": "InternVL3-8B",
            "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        }
    )
    a_display = a_display[
        [
            "model",
            "language",
            "median_flip_rate_fixed_order_AB",
            "max_flip_rate_fixed_order_AB",
            "median_flip_rate_fixed_order_BA",
            "max_flip_rate_fixed_order_BA",
            "median_flip_rate_strict_valid_both",
            "max_flip_rate_strict_valid_both",
            "median_strict_swap_error",
        ]
    ]
    a_display.columns = [
        "Model",
        "Lang.",
        "AB med.",
        "AB max",
        "BA med.",
        "BA max",
        "Both-valid med.",
        "Both-valid max",
        "Order baseline med.",
    ]
    for column in a_display.columns[2:]:
        a_display[column] = a_display[column].map(pct)
    (OUT / "A_preference_flip/A_paper_table.md").write_text(
        a_display.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    b_overall = b[b["scope"].eq("overall")].copy()
    b_overall["Model"] = b_overall["model_key"].map(
        {
            "qwen": "Qwen2.5-VL-7B",
            "internvl": "InternVL3-8B",
            "gemma": "Gemma-3-12B",
            "ministral": "Ministral-3-8B",
            "phi": "Phi-4-MM",
        }
    )
    b_display = pd.DataFrame(
        {
            "Model": b_overall["Model"],
            "rho(valid)": b_overall["rho_decile_valid_accuracy"].map(number),
            "rho(effective)": b_overall["rho_decile_effective_accuracy"].map(number),
            "rho(debiased)": b_overall["rho_decile_debiased_accuracy"].map(number),
            "D10-D1 effective [95% CI]": [
                f"{pct(row.top_minus_bottom_effective_accuracy)} "
                f"[{pct(row.effective_difference_ci_low)}, {pct(row.effective_difference_ci_high)}]"
                for row in b_overall.itertuples()
            ],
            "ECE": b_overall["isotonic_ece_10_equal_frequency_bins"].map(pct),
            "AUROC": b_overall["isotonic_auroc"].map(number),
        }
    )

    c_overall = c[c["scope"].eq("overall")].copy()
    d_language = d[d["scope"].eq("language")].copy()

    max_fixed = max(
        a["max_flip_rate_fixed_order_AB"].max(),
        a["max_flip_rate_fixed_order_BA"].max(),
    )
    min_fixed_median = min(
        a["median_flip_rate_fixed_order_AB"].min(),
        a["median_flip_rate_fixed_order_BA"].min(),
    )
    max_fixed_median = max(
        a["median_flip_rate_fixed_order_AB"].max(),
        a["median_flip_rate_fixed_order_BA"].max(),
    )
    strict_median_min = a["median_flip_rate_strict_valid_both"].min()
    strict_median_max = a["median_flip_rate_strict_valid_both"].max()
    baseline_min = a["median_strict_swap_error"].min()
    baseline_max = a["median_strict_swap_error"].max()
    significant_above_chance = int(a_matrix["holm_reject_greater_than_0_5"].sum())

    max_factor = a_factor.loc[
        a_factor[a_factor["metric"].isin(["fixed_order_AB", "fixed_order_BA"])][
            "mean_flip_rate"
        ].idxmax()
    ]

    report = f"""# FSE 2027 Reinforcement Analyses A-D

## Scope and integrity

- New model calls: **0**.
- Analysis A uses only `grounded_protocol_3lang_20260721` grid full runs.
- Analyses B-D use only `rq1_model_battery_3lang_20260723` complete five-model full runs.
- The runs are never pooled or compared as matched cells.
- GA0 passed on **234,000 raw calls**: 144,000 grid calls and 90,000 battery calls. Duplicate keys, incomplete AB/BA groups, parse failures, verdict/logit-argmax mismatches, non-finite logits, and image-mapping failures were all zero.

## A. Cross-rendering preference flips

### Method

For each model-language cell, the same 1,000 pairs were compared under all 66 unordered pairs of the 12 rendering conditions. AB and BA decisions were separately normalized to snippet identity. The both-valid analysis retained only pairs that were strict-swap valid under both rendering conditions. Exact 95% binomial intervals accompany every estimate. A one-sided exact test against a 0.5 chance flip rate was Holm-corrected within model-language-metric families.

The order baseline is the within-condition strict-swap error. No repeated full run of the same condition exists, so the deterministic repeat-run preference-flip baseline is **not measurable**; no new run was created.

### Main results

{a_display.to_markdown(index=False)}

Across model-language cells, median fixed-order preference-flip rates ranged from {pct(min_fixed_median)} to {pct(max_fixed_median)}, and the largest individual fixed-order flip rate was {pct(max_fixed)}. When both conditions were strict-valid, median flip rates fell to {pct(strict_median_min)}-{pct(strict_median_max)}. The median within-condition order baseline was much larger, {pct(baseline_min)}-{pct(baseline_max)}, showing that order was the larger disturbance in this run. No condition pair exceeded a 0.5 chance flip rate after Holm correction ({significant_above_chance}/1,188 metric rows). The largest factor-level mean was {max_factor.factor_class} for {max_factor.model}/{max_factor.language}/{max_factor.metric} ({pct(max_factor.mean_flip_rate)}).

Interpretation is limited to **preference change**, not correctness. Rendering changes can reverse a fixed-order preference for unchanged code content, but most such reversals disappear when requiring strict validity in both conditions.

## B. |c| decile calibration

### Method

Within each battery model, nonzero-|c| pairs were sorted into ten equal-count deciles with deterministic pair-ID tie breaking. Exact `c=0` pairs were assigned to D1 and counted as failures in primary debiased metrics; all non-tie sensitivity columns are retained. Decile valid accuracy conditions on strict-valid pairs, whereas effective accuracy counts invalid pairs as failures.

Top-minus-bottom differences request 10,000 two-way snippet-cluster bootstrap replicates. Replicates with a zero denominator in either compared decile are omitted; each `*_bootstrap_reps_used` column records the effective count. Calibration partitions snippets, stratified by language, into deterministic SHA-256 50/50 train/test halves. Isotonic regression is fitted only to train-train pairs and evaluated only on test-test pairs; cross-half pairs are excluded from calibration only. Every model-scope split has zero train/test snippet overlap. ECE uses ten equal-frequency probability bins, and AUROC targets debiased correctness.

### Overall results

{b_display.to_markdown(index=False)}

Effective accuracy increased strongly with |c| for every model (decile rho {b_overall.rho_decile_effective_accuracy.min():.2f}-{b_overall.rho_decile_effective_accuracy.max():.2f}), largely because |c| also predicts strict validity. Correctness *within valid pairs* was model-dependent: Qwen and Gemma were strongly positive, Ministral was moderate, InternVL was flat, and Phi was negative. Debiased correctness showed the same caution: Qwen/Gemma were monotone, Ministral was weaker, and InternVL/Phi were non-monotone or inverse. Thus |c| is a useful failure/coverage signal across the battery, but it is not a universally calibrated content-correctness confidence score.

Phi CUDA and Python have no strict-valid observations in D1, so their D10-D1 valid-accuracy differences are undefined; effective and debiased differences remain defined. Their non-tie D1 cells are also empty, making the corresponding non-tie differences undefined. These denominator failures are retained as `NA`.

## C. Joint (|b|, |c|) distribution

The battery contributes 45,000 pair rows. The identity `|c|>|b|` matched strict validity with **0 non-boundary violations**. There were {int(c_overall.boundary_pairs.sum()):,} exact `|c|=|b|` boundaries; parsed validity is not algebraically determined on those boundaries. Under a conservative boundary-as-nonvalid rule, {int(c_overall.conservative_boundary_inclusive_mismatches.sum()):,} boundary cases differ from parsed validity and are reported separately, not counted as non-boundary violations.

The full pair-level coordinates are in `C_bc_joint.csv`; the five-panel density figure draws `|c|=|b|`. For readability, only the plotted axes are clipped at each model's 99.5th percentile; CSVs and summary statistics retain every observation. `C_grid_condition_trajectory.csv` separately records condition-wise median coordinates within the grid run.

## D. First-position choice rate

The battery was decomposed into 15 model-language cells, each with 3,000 pairs and 6,000 calls. Rates and 95% intervals use 10,000 pair-cluster bootstrap replicates. The observed language-specific range was {pct(d_language.first_position_rate.min())}-{pct(d_language.first_position_rate.max())}. Difficulty-specific rows are retained for the appendix.

## Tie and denominator policy

- B: `c=0` is assigned to D1 and is incorrect in primary debiased metrics; non-tie sensitivity estimates are provided.
- C: exact `|c|=|b|` boundaries are reported separately because the strict inequality cannot classify them.
- Every rate includes `n_pairs`, `n_calls`, `valid_pairs`, or the applicable denominator in its CSV.

## Interpretation cautions

1. Preference flip is not an error rate.
2. Both-valid flip estimates condition on a selected subset and must be shown with their variable denominator.
3. The absent repeated-run baseline is not assumed to be zero; it is marked unmeasurable.
4. Isotonic ECE/AUROC use a snippet-disjoint subset, not all 9,000 pairs.
5. The grid and battery generations remain analytically separate.
"""
    (OUT / "ANALYSES_REPORT.md").write_text(report, encoding="utf-8")

    draft = f"""# Paper Insert Draft (English)

## Analysis A: compact table caption

**Table X. Cross-rendering preference flips for fixed code pairs.** Each cell summarizes all 66 unordered pairs of the 12 rendering conditions within one model-language run. AB and BA columns hold presentation order fixed and report snippet-identity preference changes over 1,000 pairs per condition pair; both-valid columns restrict the denominator to pairs that are strict-swap valid under both conditions. The order baseline is the median within-condition strict-swap error across the 12 renderings. Full condition-pair estimates, exact 95% intervals, denominators, and Holm-adjusted tests are in the replication package.

## Analysis A: body text

Holding presentation order fixed, changing only the rendering reversed the selected snippet for a median {pct(min_fixed_median)}-{pct(max_fixed_median)} of pairs across model-language cells, with individual condition pairs reaching {pct(max_fixed)}. Requiring strict-swap validity under both renderings reduced the median reversal rate to {pct(strict_median_min)}-{pct(strict_median_max)}, indicating that many apparent rendering reversals interact with position instability. The within-rendering order baseline was substantially larger ({pct(baseline_min)}-{pct(baseline_max)} median strict-swap error), so order was the dominant disturbance in this run. We call these events *preference flips*, not errors, because this analysis measures whether a preference changes rather than whether either choice is correct.

## Analysis B: figure caption

**Figure X. Verdict content-margin calibration in the five-model battery.** Pairs are grouped by within-model deciles of `|c|`, where `c=(m_AB-m_BA)/2`; exact `c=0` ties are conservatively assigned to D1. Lines show strict-valid accuracy, effective accuracy (invalid swaps count as failures), and logit-debiased accuracy over 9,000 pairs per model. Full language-specific deciles, non-tie sensitivity estimates, 10,000-replicate two-way snippet-cluster bootstrap intervals, and snippet-disjoint isotonic ECE/AUROC are provided in the replication package.

## Analysis B: body text

Effective accuracy increased monotonically with `|c|` for all five models (decile Spearman rho {b_overall.rho_decile_effective_accuracy.min():.2f}-{b_overall.rho_decile_effective_accuracy.max():.2f}), confirming that the stored verdict margin is informative about strict-swap reliability. This pattern did not imply universal content-correctness calibration: valid-only and debiased accuracy rose strongly for Qwen and Gemma, more weakly for Ministral, but were flat or inverse for InternVL and Phi. On snippet-disjoint test pairs, isotonic ECE ranged from {pct(b_overall.isotonic_ece_10_equal_frequency_bins.min())} to {pct(b_overall.isotonic_ece_10_equal_frequency_bins.max())}, while AUROC ranged from {b_overall.isotonic_auroc.min():.2f} to {b_overall.isotonic_auroc.max():.2f}. Verdict logits should therefore be retained as a diagnostic of coverage and position instability, but `|c|` should not be treated as a model-agnostic probability of correctness.

## Analysis C: supplementary figure caption

**Figure Sx. Joint position and content components of verdict margins.** Hexagonal density plots summarize all 9,000 battery pairs per model in (`|c|`, `|b|`) space; axes are clipped at each model's 99.5th percentile for display only, while all observations remain in the data and statistics. The white diagonal marks `|c|=|b|`. Away from exact boundaries, all 39,423 observations obeyed the algebraic partition `|c|>|b|` iff strict-swap valid (zero violations). The 5,577 exact boundaries are reported separately because parsed validity is not determined by the strict inequality.

## Analysis D: table column description

`First-position rate` is the fraction of all AB/BA calls in which the model selected the first displayed snippet. `95% CI` is a percentile interval from 10,000 pair-cluster bootstrap replicates, preserving the dependence between the two orders of each pair. Each language row contains 3,000 pairs and 6,000 calls.
"""
    (OUT / "PAPER_INSERT_DRAFT_EN.md").write_text(draft, encoding="utf-8")

    inventory_rows = []
    inventory_path = OUT / "ARTIFACT_SHA256.csv"
    for path in sorted(OUT.rglob("*")):
        if (
            not path.is_file()
            or path == inventory_path
            or "__pycache__" in path.parts
        ):
            continue
        inventory_rows.append(
            {
                "relative_path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "artifacts_hashed": len(inventory_rows),
                "inventory_excludes_itself": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
