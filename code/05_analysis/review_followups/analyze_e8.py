#!/usr/bin/env python3
"""Prepare compact, multiplicity-aware inputs for manuscript restructuring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
OUT = PKG / "analysis/E8"
GROUND = ROOT / "results/grounded_protocol_3lang_20260721"
AB = GROUND / "analysis/ab"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # E8(a): preserve every preregistered contrast and mark interpretable exemplars.
    grid = pd.read_csv(AB / "A_GRID_MCNEMAR.csv")
    perturb = pd.read_csv(AB / "B_PERTURBATION_MCNEMAR.csv")
    inventory = pd.concat([grid, perturb], ignore_index=True)
    inventory["direction"] = np.select(
        [inventory.accuracy_delta > 0, inventory.accuracy_delta < 0],
        ["increase", "decrease"], default="no_change",
    )
    inventory["effect_size"] = inventory.accuracy_delta
    inventory["representative_family"] = ""
    grid_cond = inventory.condition.fillna("")
    inventory.loc[grid_cond.str.contains("wrap60"), "representative_family"] = "wrap_60_vs_80"
    inventory.loc[grid_cond.str.contains("fs24"), "representative_family"] = "font_20_vs_24"
    inventory.loc[
        (inventory.experiment == "grid")
        & ~grid_cond.str.startswith("monokai_dark"),
        "representative_family",
    ] = inventory.loc[
        (inventory.experiment == "grid")
        & ~grid_cond.str.startswith("monokai_dark"),
        "representative_family",
    ].replace("", "theme")
    inventory.loc[grid_cond.eq("no_indent"), "representative_family"] = "no_indent"
    inventory.loc[grid_cond.str.startswith("gaussian_sigma"), "representative_family"] = "blur"
    inventory.to_csv(OUT / "E8_contrast_inventory.csv", index=False)

    families = pd.DataFrame([
        {"family": "rendering_grid_within_model_language", "number_of_families": 6,
         "tests_per_family": 11, "total_tests": 66, "adjustment": "Holm within each model-language family"},
        {"family": "perturbation_within_model_language", "number_of_families": 6,
         "tests_per_family": 5, "total_tests": 30, "adjustment": "Holm within each model-language family"},
        {"family": "RQ3_direct_contrasts_within_model", "number_of_families": 2,
         "tests_per_family": 6, "total_tests": 12, "adjustment": "Holm within model"},
        {"family": "E1_VLM_vs_baseline", "number_of_families": 1,
         "tests_per_family": 30, "total_tests": 30, "adjustment": "global Holm over 15 cells x 2 baselines"},
    ])
    families.to_csv(OUT / "E8_multiplicity_families.csv", index=False)

    # E8(b): four conceptual RQ3 rows, with both models retained.
    rq3 = pd.read_csv(
        GROUND / "rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv"
    )
    category = {
        "semantic_and_visual_aligned": "aligned semantics+appearance (2 contrasts)",
        "semantic_only_under_ugly": "semantic-only under ugly",
        "semantic_visual_conflict_primary": "semantic/appearance conflict",
        "visual_only_gold": "appearance-only (gold and trash)",
        "visual_only_trash": "appearance-only (gold and trash)",
    }
    rq3["compact_row"] = rq3.contrast_type.map(category)
    compact_rows = []
    for compact_name, group in rq3.groupby("compact_row", sort=False):
        row: dict[str, object] = {"compact_contrast": compact_name}
        for model, model_group in group.groupby("model"):
            key = "qwen" if model.startswith("Qwen") else "internvl"
            for metric in [
                "effective_target_preference", "target_preference_valid",
                "debiased_target_preference", "strict_swap_error",
            ]:
                vals = model_group[metric]
                row[f"{key}_{metric}_min"] = vals.min()
                row[f"{key}_{metric}_max"] = vals.max()
            row[f"{key}_n_pairs"] = int(model_group.n_pairs.sum())
            row[f"{key}_valid_pairs"] = int(model_group.valid_pairs.sum())
        compact_rows.append(row)
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(OUT / "E8_rq3_compact.csv", index=False)

    # Exact decomposition: eff = coverage * valid accuracy.
    result_files = [
        ("grid", AB / "A_GRID_RESULTS.csv", "monokai_dark__fs20__wrap80__lnon"),
        ("perturbation", AB / "B_PERTURBATION_RESULTS.csv", "baseline"),
    ]
    decomposed = []
    for experiment, path, baseline_name in result_files:
        data = pd.read_csv(path)
        keys = ["model", "language"]
        base = data[data.condition == baseline_name][
            keys + ["valid_rate", "valid_accuracy", "effective_accuracy", "strict_swap_error"]
        ].rename(columns={
            "valid_rate": "baseline_valid_rate",
            "valid_accuracy": "baseline_valid_accuracy",
            "effective_accuracy": "baseline_effective_accuracy",
            "strict_swap_error": "baseline_swap_error",
        })
        merged = data.merge(base, on=keys, validate="many_to_one")
        merged = merged[merged.condition != baseline_name].copy()
        merged["experiment"] = experiment
        merged["effective_delta"] = (
            merged.effective_accuracy - merged.baseline_effective_accuracy
        )
        merged["coverage_component"] = (
            (merged.valid_rate - merged.baseline_valid_rate)
            * merged.baseline_valid_accuracy
        )
        merged["valid_accuracy_component"] = (
            merged.valid_rate
            * (merged.valid_accuracy - merged.baseline_valid_accuracy)
        )
        merged["decomposition_residual"] = (
            merged.effective_delta
            - merged.coverage_component
            - merged.valid_accuracy_component
        )
        merged["swap_error_delta"] = (
            merged.strict_swap_error - merged.baseline_swap_error
        )
        merged["dominant_component"] = np.where(
            merged.coverage_component.abs() >= merged.valid_accuracy_component.abs(),
            "valid_coverage/consistency", "valid-only_accuracy",
        )
        decomposed.append(merged)
    decomposition = pd.concat(decomposed, ignore_index=True)
    decomposition.to_csv(OUT / "E8_rendering_decomposition.csv", index=False)

    # E8(c): effective-accuracy robustness slices for all VLMs and all ML baselines.
    pairs = pd.read_csv(
        ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
    )
    vlm = pd.read_csv(
        ROOT / "results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv"
    ).merge(
        pairs[["protocol_pair_id", "pair_id", "dataset_name_i", "dataset_name_j"]],
        left_on="pair_id", right_on="protocol_pair_id", validate="many_to_one",
        suffixes=("", "_source"),
    )
    ml = pd.read_csv(
        ROOT / "results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv"
    ).merge(
        pairs[["pair_id", "difficulty", "dataset_name_i", "dataset_name_j"]],
        on="pair_id", validate="many_to_one",
    )
    robust_rows = []
    slices = {
        "easy_all_languages": (
            vlm.difficulty.eq("easy"),
            ml.difficulty.eq("easy"),
        ),
        "java_within_dataset": (
            vlm.language.eq("java") & vlm.dataset_name_i.eq(vlm.dataset_name_j),
            ml.language.eq("java") & ml.dataset_name_i.eq(ml.dataset_name_j),
        ),
    }
    for slice_name, (vmask, mmask) in slices.items():
        for (model, language), group in vlm[vmask].groupby(["model", "language"]):
            robust_rows.append({
                "slice": slice_name, "method_type": "VLM", "model": model,
                "language": language, "n": len(group),
                "accuracy_main": group.correct.sum() / len(group),
                "definition": "effective accuracy; invalid pairs incorrect",
            })
        for (model, language), group in ml[mmask].groupby(["model", "language"]):
            robust_rows.append({
                "slice": slice_name, "method_type": "ML", "model": model,
                "language": language, "n": len(group),
                "accuracy_main": group.is_correct.mean(),
                "definition": "deterministic pair accuracy",
            })
    robustness = pd.DataFrame(robust_rows)
    robustness.to_csv(OUT / "E8_label_robustness.csv", index=False)

    dominance = decomposition.dominant_component.value_counts()
    easy = robustness[robustness.slice == "easy_all_languages"]
    within = robustness[robustness.slice == "java_within_dataset"]
    report = f"""# E8 Restructuring inputs

## Contrast inventory and multiplicity

All 66 rendering-grid and 30 perturbation contrasts are retained in
`E8_contrast_inventory.csv`, including effect size, cluster-robust 95% CI,
exact McNemar p, Holm-adjusted p, and direction. Holm adjustment was performed
within six model-language families (11 grid or 5 perturbation comparisons per
family), not as one 96-test family.

## RQ3 compact view

The six direct contrasts reduce to four conceptual rows without discarding either
model: aligned semantics+appearance, semantic-only under ugly rendering,
semantic/appearance conflict, and appearance-only. In the appearance-only rows,
Qwen selects golden over ugly_gold effectively on 56.0% and beautiful_trash over
ugly_trash on 8.3% (25/300 valid); InternVL gives 57.3% and 60.3%.

## Rendering decomposition

The exact product decomposition has maximum numerical residual
{decomposition.decomposition_residual.abs().max():.3g}. The valid-coverage /
consistency component is larger in absolute value for
{int(dominance.get('valid_coverage/consistency', 0))}/{len(decomposition)}
condition cells; valid-only accuracy dominates in
{int(dominance.get('valid-only_accuracy', 0))}/{len(decomposition)}. Thus the
consistency hypothesis is evaluated cell by cell rather than asserted globally.

## Label robustness

The easy slice contains {int(easy[easy.method_type == 'VLM'].n.min())} pairs per
model-language cell. The Java within-dataset slice contains
{int(within[within.method_type == 'VLM'].n.iloc[0])} pairs, after excluding 1,996
cross-dataset pairs. `E8_label_robustness.csv` reports every VLM and all nine ML
baselines so model ranks and the baseline comparison can be audited directly.
"""
    (OUT / "E8_REPORT.md").write_text(report, encoding="utf-8")
    print(f"contrast inventory={len(inventory)}; compact rows={len(compact)}")
    print(dominance.to_string())
    print(robustness.groupby(["slice", "method_type"]).accuracy_main.agg(["max", "min"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
