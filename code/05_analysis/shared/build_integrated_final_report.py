#!/usr/bin/env python3
"""Build the final Korean report, figures, and SHA-256 artifact inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "FINAL_GROUNDED_3LANG_REPORT.md"


def read(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def percent_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = result[column].map(lambda x: "-" if pd.isna(x) else f"{100*x:.2f}%")
    return result


def short_model(value: str) -> str:
    return value.split("/")[-1]


def make_figures(grid: pd.DataFrame, perturb: pd.DataFrame, rq3: pd.DataFrame) -> None:
    out = ROOT / "analysis/figures"
    out.mkdir(parents=True, exist_ok=True)
    for data, name, title in ((grid, "a_grid_overall.png", "Rendering grid"),
                              (perturb, "b_perturbation_overall.png", "Visual perturbations")):
        overall = data.groupby(["model", "condition"], as_index=False).agg(
            effective_accuracy=("effective_accuracy", "mean"), strict_swap_error=("strict_swap_error", "mean"))
        conditions = list(dict.fromkeys(overall.condition))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        for model, d in overall.groupby("model"):
            d = d.set_index("condition").reindex(conditions)
            axes[0].plot(range(len(conditions)), d.effective_accuracy, marker="o", label=short_model(model))
            axes[1].plot(range(len(conditions)), d.strict_swap_error, marker="o", label=short_model(model))
        axes[0].set(ylabel="Effective accuracy", ylim=(0, 1), title=title)
        axes[1].set(ylabel="Strict-swap error", ylim=(0, 1), xticks=range(len(conditions)),
                    xticklabels=conditions)
        axes[1].tick_params(axis="x", rotation=35)
        axes[0].legend(frameon=False)
        fig.savefig(out / name, dpi=200)
        plt.close(fig)
    core = rq3[rq3.contrast_type.eq("semantic_visual_conflict_primary")]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for model, d in core.groupby("model"):
        d = d.set_index("language").reindex(["java", "python", "cuda"])
        ax.plot(d.index, d.effective_target_preference, marker="o", label=short_model(model))
    ax.axhline(.5, color="black", lw=.8, linestyle="--")
    ax.set(ylabel="Effective semantics-preserving preference", ylim=(0, 1),
           title="RQ3: ugly_gold vs beautiful_trash")
    ax.legend(frameon=False)
    fig.savefig(out / "c_rq3_primary_conflict.png", dpi=200)
    plt.close(fig)


def inventory() -> pd.DataFrame:
    included = []
    roots = [ROOT / "analysis", ROOT / "embedding", ROOT / "rq3/analysis", ROOT / "code"]
    explicit = [ROOT / "README.md", ROOT / "config/protocol_grounded.json",
                ROOT / "config/prompt_B_template.txt",
                ROOT / "data/pairs_seed42_grounded.csv", ROOT / "data/render_conditions.csv",
                ROOT / "rendered/RENDER_MANIFEST.json", ROOT / "rendered/metadata/grid_render_metadata.csv",
                ROOT / "rendered/metadata/perturbation_render_metadata.csv",
                ROOT / "audit/GROUNDED_RENDER_AUDIT.json", ROOT / "audit/BLUR_OCR_MANIFEST.json",
                ROOT / "rq3/RQ3_PREPARATION_MANIFEST.json", ROOT / "rq3/audit/RQ3_AUTOMATIC_AUDIT.json",
                ROOT / "rq3/review/GC1_APPROVAL.json"]
    for experiment in ("grid", "perturbation"):
        included.extend((ROOT / f"inference/{experiment}/raw").glob("*full_20260721*"))
    included.extend((ROOT / "rq3/inference/raw").glob("*full_20260721*"))
    for base in roots:
        included.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    included.extend(path for path in explicit if path.exists())
    rows = []
    for path in sorted(set(included)):
        if path.name == "ARTIFACT_SHA256.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                     "sha256": digest.hexdigest()})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "ARTIFACT_SHA256.csv", index=False)
    return result


def main() -> None:
    integrity = json.loads((ROOT / "analysis/ab/INTEGRITY.json").read_text())
    d_integrity = json.loads((ROOT / "embedding/analysis/D_INTEGRITY.json").read_text())
    if not integrity["all_ab_ba_complete"] or len(d_integrity["models"]) != 2:
        raise RuntimeError("final integrity gate failed")
    grid = read("analysis/ab/A_GRID_RESULTS.csv")
    perturb = read("analysis/ab/B_PERTURBATION_RESULTS.csv")
    mcnemar_a = read("analysis/ab/A_GRID_MCNEMAR.csv")
    mcnemar_b = read("analysis/ab/B_PERTURBATION_MCNEMAR.csv")
    interactions = read("analysis/ab/B_LANGUAGE_INTERACTIONS.csv")
    pooled_interactions = read("analysis/ab/B_LANGUAGE_INTERACTIONS_POOLED_SECONDARY.csv")
    dorn_perturb = read("analysis/ab/B_DORN_PRIMARY_RESULTS.csv")
    logit = read("analysis/logit_decomposition/LOGIT_SUMMARY.csv")
    d_stability = read("embedding/analysis/d_embedding_stability.csv")
    d_perturb = read("embedding/analysis/d_perturbation_distance.csv")
    rq3 = read("rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv")
    rq3_qwen_text = read("rq3/analysis/qwen_text_plus_image/RQ3_QWEN_TEXT_PLUS_IMAGE_BY_LANGUAGE_CONTRAST.csv")
    historical = read("rq3/analysis/historical_alignment/RQ3_ORDER_ALIGNMENT.csv")
    arm_status = read("rq3/analysis/historical_alignment/RQ3_ARM_STATUS.csv").fillna("")
    make_figures(grid, perturb, rq3)

    base_grid = grid[grid.condition.eq("monokai_dark__fs20__wrap80__lnon")]
    hb = dorn_perturb[dorn_perturb.condition.isin(["baseline", "no_indent"])].copy()
    hb = hb.pivot(index=["model", "language"], columns="condition", values="valid_accuracy").reset_index()
    hb["no_indent_minus_baseline"] = hb.no_indent - hb.baseline
    core = rq3[rq3.contrast_type.eq("semantic_visual_conflict_primary")]
    qtext_core = rq3_qwen_text[rq3_qwen_text.contrast_type.eq("semantic_visual_conflict_primary")]

    a_cols = ["model", "language", "condition", "n_pairs", "n_valid", "valid_accuracy",
              "effective_accuracy", "strict_swap_error", "spearman_valid_only", "debiased_accuracy"]
    b_cols = a_cols
    r_cols = ["model", "language", "contrast_type", "variant_i", "variant_j", "n_pairs", "valid_pairs",
              "target_preference_valid", "effective_target_preference", "strict_swap_error",
              "debiased_target_preference", "holm_p_within_family"]
    d_cols = ["model", "representation", "language", "n_snippets", "intra_mean_cosine_distance",
              "inter_mean_cosine_distance", "ratio_intra_inter", "ratio_boot_ci_low", "ratio_boot_ci_high",
              "same_snippet_top1", "retrieval_chance", "retrieval_permutation_p"]
    report = f"""# Grounded 3-Language VLM Experiment: Final Report

## Scope and status

This report covers experiments A-D and the AB/BA verdict-logit decomposition on Java, Python, and CUDA. A and B contain **{integrity['raw_calls']:,} calls / {integrity['paired_rows']:,} pair-condition rows**, with zero parse failures. RQ3 image-only contains 7,200 calls / 3,600 contrasts. Qwen text+image contributes a completed supplementary 3,600-call arm. InternVL text+image is not reported as a result because its 100-call GA0 failed twice on the same deterministic malformed verdict; no full inference was run.

{arm_status.to_markdown(index=False)}

## Frozen method

- Models: Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; InternVL3-8B revision `853e3a797a661694b1b8ece0cb72dc2b23e3dac9`.
- Inference: two separately labelled rendered PNGs, frozen Prompt B, image-only primary input, BF16, greedy decoding, temperature 0, top-p inactive, max 24 generated tokens, seed 42.
- Pair protocol: every pair is asked in AB and BA order. Strict-valid means both calls choose the same underlying snippet. Effective accuracy counts invalid swaps as failure; valid accuracy conditions on strict-valid pairs; Spearman uses only valid pair signs against continuous human z-score differences.
- Data: 3,000 fixed pairs, exactly 1,000 per language. Absolute within-dataset z-score difference is continuous; balanced rank tertiles are descriptive only.
- A: the same pair under 12 display x font-size x wrap conditions. The SHA-verified baseline is `monokai_dark__fs20__wrap80__lnon`.
- B: baseline, no-indent, no-blank-lines, and Gaussian blur sigma 1/2/4 px. Both sides receive the same manipulation.
- Inference: Wilson 95% intervals for proportions; exact paired McNemar plus Holm; Cochran Q sensitivity; language interactions with two-way snippet-cluster robust covariance. A nonsignificant difference is not treated as equivalence.

## A. Rendering grid

### Baseline by language

{percent_table(base_grid[a_cols], ['valid_accuracy','effective_accuracy','strict_swap_error','debiased_accuracy']).to_markdown(index=False)}

### All 12 conditions x models x languages

{percent_table(grid[a_cols], ['valid_accuracy','effective_accuracy','strict_swap_error','debiased_accuracy']).to_markdown(index=False)}

The exact McNemar table contains {len(mcnemar_a)} model-language contrasts; {int(mcnemar_a.holm_reject_0_05.sum())} remain significant after within-family Holm correction. Cochran Q and every corrected pairwise result are preserved in `analysis/ab/COCHRAN_Q.csv` and `analysis/ab/A_GRID_MCNEMAR.csv`.

## B. Visual perturbations

{percent_table(perturb[b_cols], ['valid_accuracy','effective_accuracy','strict_swap_error','debiased_accuracy']).to_markdown(index=False)}

### H-B1: no-indent

{percent_table(hb, ['baseline','no_indent','no_indent_minus_baseline']).to_markdown(index=False)}

The primary language interaction is Dorn-only (Java 89 pairs; Python and CUDA 1,000 each); pooled Java is secondary. H-B1 is not supported: Python does not show the prespecified larger valid-accuracy decrease. H-B2 is not established as equivalence: no Dorn-only valid-accuracy interaction survives Holm, but failure to reject a difference is not evidence of equality. In the pooled-Java secondary analysis, sigma-4 effective-accuracy interactions are Holm-significant for InternVL (Python and CUDA vs Java) and Qwen (Python vs Java). Primary and secondary estimates are separated in `B_LANGUAGE_INTERACTIONS_DORN_PRIMARY.csv` and `B_LANGUAGE_INTERACTIONS_POOLED_SECONDARY.csv`; {int(mcnemar_b.holm_reject_0_05.sum())}/{len(mcnemar_b)} perturbation McNemar contrasts are Holm-significant.

## C. RQ3 semantic-visual conflict

RQ3 has generated directional targets, not independent human gold; these are preference rates, never accuracy. The primary contrast is `ugly_gold` versus `beautiful_trash`.

### Image-only primary conflict

{percent_table(core[r_cols], ['target_preference_valid','effective_target_preference','strict_swap_error','debiased_target_preference']).to_markdown(index=False)}

### All image-only language x contrast results

{percent_table(rq3[r_cols], ['target_preference_valid','effective_target_preference','strict_swap_error','debiased_target_preference']).to_markdown(index=False)}

### Qwen text+image supplementary primary conflict

{percent_table(qtext_core[r_cols], ['target_preference_valid','effective_target_preference','strict_swap_error','debiased_target_preference']).to_markdown(index=False)}

The historical Swiss/Elo and new direct-pairwise results are not numerically pooled. Their ordinal variant order agrees exactly in all six new model-language cells:

{historical.to_markdown(index=False)}

The historical report says 200 bases, but its available image-only Elo file contains 160 items per variant. This provenance discrepancy is retained in `rq3/analysis/historical_alignment/LIMITATIONS.md`.

## Logit decomposition

For each pair, `b=(m_AB+m_BA)/2` is the first-position component and `c=(m_AB-m_BA)/2` is the content component. Away from exact BF16 logit-tie boundaries, `|c|>|b|` equals opposite AB/BA margin signs with 0 violations across 99,832 rows. There are 8,168 exact boundaries; parsed validity must be reported separately there.

{percent_table(logit[['experiment','model','language','condition','n','strict_valid_rate','strict_effective_accuracy','strict_valid_accuracy','debiased_accuracy','debiased_tie_rate','debiased_minus_effective','median_abs_b','median_abs_c']], ['strict_valid_rate','strict_effective_accuracy','strict_valid_accuracy','debiased_accuracy','debiased_tie_rate','debiased_minus_effective']).to_markdown(index=False)}

Mann-Whitney comparisons and |c| decile calibration are in `analysis/logit_decomposition/`; the figure is `logit_decomposition_and_calibration.png`. This is a two-pass logit estimator requiring verdict logits, not a claim that invalid pairs become observed strict-valid pairs.

## D. Vision representation stability

The primary common representation is the mean of projected final visual tokens. Native patch means are secondary; InternVL native CLS is architecture-specific supplemental evidence. Qwen has no CLS token, so none was invented. All 552 snippets are evaluated under 12 grid and five nonbaseline perturbation conditions: 9,384 image passes per model, 18,768 total. No text generation is involved.

{percent_table(d_stability[d_cols], ['same_snippet_top1','retrieval_chance']).to_markdown(index=False)}

The old unsupported ratio cutoffs 0.5/0.8 are not used. Ratios are continuous with 1,000 snippet-bootstrap replicates; retrieval is compared with a 1,000-permutation null. Blur/layout distances with bootstrap intervals are in `embedding/analysis/d_perturbation_distance.csv` ({len(d_perturb)} rows).

## Completeness and artifacts

Detailed outputs include pair-level rows, all condition/language/difficulty summaries, cue-present sensitivity, exact McNemar/Holm tables, Cochran Q, crossed-cluster interactions, RQ3 stratum tables, logit effects/calibration, D retrieval and perturbation tables, and three figures. Rendered images are not duplicated in the inventory; their individual SHA-256 hashes are already recorded in the render metadata. `ARTIFACT_SHA256.csv` inventories the code, configs, full raw inference files, manifests, metadata, and analysis outputs.
"""
    REPORT.write_text(report + "\n", encoding="utf-8")
    artifacts = inventory()
    print(json.dumps({"report": str(REPORT), "artifact_inventory_rows": len(artifacts),
                      "a_rows": len(grid), "b_rows": len(perturb), "rq3_rows": len(rq3),
                      "d_rows": len(d_stability)}, indent=2))


if __name__ == "__main__":
    main()
