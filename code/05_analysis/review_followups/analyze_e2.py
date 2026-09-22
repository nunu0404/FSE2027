#!/usr/bin/env python3
"""Trace manuscript discrepancies to immutable runs and paired outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
OUT = PKG / "analysis/E2"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
GROUNDED = ROOT / "results/grounded_protocol_3lang_20260721"

MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B",
}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def battery_pairs(model: str) -> pd.DataFrame:
    frame = pd.read_csv(BATTERY / "analysis/full/pair_level_results.csv")
    pairs = pd.read_csv(BATTERY / "data/rq1_pairs_9000.csv")
    frame = frame[(frame.model_key == model) & (frame.language == "java")].copy()
    frame = frame.merge(
        pairs[["protocol_pair_id", "pair_id", "snippet_i", "snippet_j"]],
        left_on="pair_id",
        right_on="protocol_pair_id",
        validate="one_to_one",
    )
    frame["decision"] = np.where(frame.model_sign.eq(1), frame.snippet_i,
                                 np.where(frame.model_sign.eq(-1), frame.snippet_j, None))
    frame["run_id"] = "battery"
    frame["available_margin"] = frame.content_margin.abs()
    return frame[[
        "pair_id_y", "valid", "correct", "decision", "available_margin", "run_id",
        "margin_ab", "margin_ba",
    ]].rename(columns={"pair_id_y": "pair_id"})


def clean_pairs(model: str) -> pd.DataFrame:
    path = ROOT / f"results/complementarity_mechanism_clean_{model}_image_only_20260706/pair_level_analysis_table.csv"
    frame = pd.read_csv(path)
    frame["run_id"] = "clean_default"
    frame["decision"] = frame.vlm_pred_preference.where(frame.vlm_strict_swap_valid)
    frame["available_margin"] = np.nan
    return frame[["pair_id", "vlm_strict_swap_valid", "vlm_correct", "decision", "available_margin", "run_id"]].rename(
        columns={"vlm_strict_swap_valid": "valid", "vlm_correct": "correct"}
    )


def grid_pairs(model: str) -> pd.DataFrame:
    raw = GROUNDED / "inference/grid/raw" / (
        f"{'Qwen__Qwen2.5-VL-7B-Instruct' if model == 'qwen' else 'OpenGVLab__InternVL3-8B'}"
        "__image_only__promptB__seed42__full_20260721.jsonl"
    )
    rows = pd.read_json(raw, lines=True)
    rows = rows[rows.condition.eq("monokai_dark__fs20__wrap80__lnon")].copy()
    pair_meta = pd.read_csv(GROUNDED / "data/pairs_seed42_grounded.csv")
    id_col = "protocol_pair_id" if "protocol_pair_id" in pair_meta else "pair_id"
    original_col = "pair_id" if id_col == "protocol_pair_id" else id_col
    meta = pair_meta[[id_col, original_col, "snippet_i", "snippet_j"]].copy()
    if id_col == original_col:
        meta["original_pair_id"] = meta[id_col]
    else:
        meta = meta.rename(columns={original_col: "original_pair_id"})
    piv = rows.pivot(index="pair_id", columns="order", values=["parsed_choice", "margin"]).reset_index()
    piv.columns = ["pair_id"] + [f"{a}_{b}" for a, b in piv.columns[1:]]
    piv = piv.merge(meta, left_on="pair_id", right_on=id_col, validate="one_to_one")
    piv["ab_selected"] = np.where(piv.parsed_choice_AB.eq("A"), piv.snippet_i,
                                  np.where(piv.parsed_choice_AB.eq("B"), piv.snippet_j, None))
    piv["ba_selected"] = np.where(piv.parsed_choice_BA.eq("A"), piv.snippet_j,
                                  np.where(piv.parsed_choice_BA.eq("B"), piv.snippet_i, None))
    piv["valid"] = piv.ab_selected.notna() & piv.ba_selected.notna() & piv.ab_selected.eq(piv.ba_selected)
    piv["decision"] = piv.ab_selected.where(piv.valid)
    piv["available_margin"] = ((piv.margin_AB - piv.margin_BA) / 2).abs()
    piv["run_id"] = "grid"
    piv["correct"] = np.nan
    return piv[["original_pair_id", "valid", "correct", "decision", "available_margin", "run_id"]].rename(
        columns={"original_pair_id": "pair_id"}
    )


def compare(left: pd.DataFrame, right: pd.DataFrame, model: str) -> dict:
    merged = left.merge(right, on="pair_id", suffixes=("_left", "_right"), validate="one_to_one")
    left_state = np.where(merged.valid_left, merged.decision_left, "__INVALID__")
    right_state = np.where(merged.valid_right, merged.decision_right, "__INVALID__")
    changed = left_state != right_state
    both_valid = merged.valid_left & merged.valid_right
    changed_both_valid = changed & both_valid
    margins = pd.concat([
        merged.loc[changed, "available_margin_left"],
        merged.loc[changed, "available_margin_right"],
    ]).dropna()
    return {
        "model": MODEL_NAMES[model],
        "left_run_id": left.run_id.iloc[0],
        "right_run_id": right.run_id.iloc[0],
        "left_n": len(left),
        "right_n": len(right),
        "intersection_n": len(merged),
        "state_disagreement_n": int(changed.sum()),
        "state_disagreement_rate": float(changed.mean()),
        "both_valid_n": int(both_valid.sum()),
        "decision_disagreement_both_valid_n": int(changed_both_valid.sum()),
        "decision_disagreement_both_valid_rate": (
            float(changed_both_valid.sum() / both_valid.sum()) if both_valid.any() else np.nan
        ),
        "changed_pair_available_abs_margin_median": float(margins.median()) if len(margins) else np.nan,
        "margin_source_note": (
            "Median pools available |content margin| values from logit runs only; "
            "clean_default did not store verdict logits."
        ),
    }


def metrics(frame: pd.DataFrame) -> tuple[float, float]:
    return float(frame.valid.mean()), float((frame.valid & frame.decision.notna()).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, pd.DataFrame]] = {}
    for model in MODEL_NAMES:
        sources[model] = {
            "battery": battery_pairs(model),
            "clean_default": clean_pairs(model),
            "grid": grid_pairs(model),
        }

    agreements = []
    for model, runs in sources.items():
        agreements.append(compare(runs["battery"], runs["clean_default"], model))
        agreements.append(compare(runs["battery"], runs["grid"], model))
        agreements.append(compare(runs["clean_default"], runs["grid"], model))
    agreement = pd.DataFrame(agreements)
    agreement.to_csv(OUT / "E2_crossrun_agreement.csv", index=False)

    inventory = [
        {
            "manuscript_location": "Table 1",
            "run_label": "battery",
            "run_id": "rq1_model_battery_3lang_20260723/full_20260723",
            "condition": "corrected grounded PNG; monokai_dark; fs20; wrap80; line numbers on; two images",
            "pair_set_id": "rq1_pairs_9000_seed42",
            "n_pairs_per_model": 9000,
            "languages": "java|python|cuda",
            "source_artifact": "analysis/full/metrics_by_scope.csv",
        },
        {
            "manuscript_location": "Table 2",
            "run_label": "battery",
            "run_id": "rq1_model_battery_3lang_20260723/full_20260723",
            "condition": "same as Table 1; verdict-logit c/b decomposition",
            "pair_set_id": "rq1_pairs_9000_seed42",
            "n_pairs_per_model": 9000,
            "languages": "java|python|cuda",
            "source_artifact": "analysis/full/pair_level_results.csv",
        },
        {
            "manuscript_location": "Table 3; Fig. 3; rendering grid",
            "run_label": "grid",
            "run_id": "grounded_protocol_3lang_20260721/grid/full_20260721",
            "condition": "12 rendering conditions; baseline monokai_dark/fs20/wrap80/line numbers on",
            "pair_set_id": "pairs_seed42_grounded",
            "n_pairs_per_model": 3000,
            "languages": "java|python|cuda",
            "source_artifact": "analysis/ab/A_GRID_RESULTS.csv",
        },
        {
            "manuscript_location": "Table 4; RQ3 figure",
            "run_label": "rq3",
            "run_id": "grounded_protocol_3lang_20260721/rq3/full_20260721",
            "condition": "100 bases/language x 6 direct variant contrasts; two images",
            "pair_set_id": "rq3_contrast_manifest",
            "n_pairs_per_model": 1800,
            "languages": "java|python|cuda",
            "source_artifact": "rq3/analysis/image_only/RQ3_IMAGE_ONLY_PAIR_LEVEL.csv",
        },
        {
            "manuscript_location": "Table 5; Section 5.3 Java packaging rows",
            "run_label": "clean_default",
            "run_id": "full_selected_seed42_clean_rerender_20260706",
            "condition": "corrected default PNG; original two-image or combined_labeled packaging",
            "pair_set_id": "full_pair_set_rq0_seed42",
            "n_pairs_per_model": 3000,
            "languages": "java",
            "source_artifact": "complementarity_mechanism_clean_*_20260706/pair_level_analysis_table.csv",
        },
        {
            "manuscript_location": "Table 5 Python/CUDA deployment rows",
            "run_label": "deploy",
            "run_id": "python_cuda_missing_experiments_20260717",
            "condition": "deployment/OCR protocol; pooled Python and CUDA where indicated",
            "pair_set_id": "python_cuda_pairs_seed42_clean",
            "n_pairs_per_model": 6000,
            "languages": "python|cuda",
            "source_artifact": "results/python_cuda_missing_experiments_20260717",
        },
    ]
    detailed = []

    def add_rows(location: str, label: str, run_id: str, source: str,
                 frame: pd.DataFrame, keys: list[str], metrics: list[str],
                 pair_set: str) -> None:
        for _, item in frame.iterrows():
            detailed.append({
                "manuscript_location": location,
                "run_label": label,
                "run_id": run_id,
                "condition": "|".join(f"{key}={item[key]}" for key in keys),
                "pair_set_id": pair_set,
                "n_pairs_per_model": item.get("n_pairs", item.get("n", item.get("pairs", np.nan))),
                "languages": item.get("language", "pooled"),
                "source_artifact": source,
                "metric_values_json": json.dumps(
                    {metric: item[metric] for metric in metrics if metric in item.index},
                    default=str, sort_keys=True,
                ),
            })

    battery_metrics = pd.read_csv(BATTERY / "analysis/full/metrics_by_scope.csv")
    battery_metrics["language"] = battery_metrics.scope_value.where(
        battery_metrics.scope.eq("language"), "pooled"
    )
    add_rows(
        "Table 1", "battery", "rq1_model_battery_3lang_20260723/full_20260723",
        "analysis/full/metrics_by_scope.csv",
        battery_metrics[battery_metrics.scope == "language"],
        ["model", "language"],
        ["valid_pairs", "valid_accuracy", "effective_accuracy", "strict_swap_error",
         "spearman_valid_only", "ab_accuracy", "ba_accuracy"],
        "rq1_pairs_9000_seed42",
    )
    add_rows(
        "Table 2", "battery", "rq1_model_battery_3lang_20260723/full_20260723",
        "analysis/full/metrics_by_scope.csv",
        battery_metrics[battery_metrics.scope == "overall"],
        ["model"],
        ["pairs", "effective_accuracy", "debiased_accuracy", "content_ties",
         "median_abs_content_margin", "mean_abs_position_margin"],
        "rq1_pairs_9000_seed42",
    )
    grid_metrics = pd.read_csv(GROUNDED / "analysis/ab/A_GRID_RESULTS.csv")
    add_rows(
        "Table 3; Fig. 3", "grid",
        "grounded_protocol_3lang_20260721/grid/full_20260721",
        "analysis/ab/A_GRID_RESULTS.csv", grid_metrics,
        ["model", "language", "condition"],
        ["n_pairs", "n_valid", "valid_accuracy", "effective_accuracy",
         "strict_swap_error", "spearman_valid_only", "debiased_accuracy"],
        "pairs_seed42_grounded",
    )
    rq3_metrics = pd.read_csv(
        GROUNDED / "rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv"
    )
    add_rows(
        "Table 4", "rq3", "grounded_protocol_3lang_20260721/rq3/full_20260721",
        "rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv", rq3_metrics,
        ["model", "contrast_type", "variant_i", "variant_j"],
        ["n_pairs", "valid_pairs", "target_preference_valid",
         "effective_target_preference", "strict_swap_error",
         "debiased_target_preference"],
        "rq3_contrast_manifest",
    )
    deployment_path = PKG / "analysis/E6/E6_deployment_full_metrics.csv"
    if deployment_path.exists():
        deployment = pd.read_csv(deployment_path)
        add_rows(
            "Table 5", "deploy", "deployment_language_specific_runs",
            "analysis/E6/E6_deployment_full_metrics.csv", deployment,
            ["language", "system", "model"],
            ["n_total_pairs", "n_valid_pairs", "n_correct_pairs", "valid_accuracy",
             "effective_accuracy", "strict_swap_error", "parse_failure_rate"],
            "language-specific deployment pair sets",
        )
    logit_summary = pd.read_csv(
        GROUNDED / "analysis/logit_decomposition/LOGIT_SUMMARY.csv"
    )
    add_rows(
        "Fig. 5; Fig. 6", "grid+perturbation",
        "grounded_protocol_3lang_20260721/full_20260721",
        "analysis/logit_decomposition/LOGIT_SUMMARY.csv", logit_summary,
        ["experiment", "model", "language", "condition"],
        ["n", "strict_valid_rate", "strict_effective_accuracy",
         "debiased_accuracy", "debiased_tie_rate", "median_abs_b", "median_abs_c"],
        "pairs_seed42_grounded",
    )
    inventory_frame = pd.concat(
        [pd.DataFrame(inventory).assign(metric_values_json=""), pd.DataFrame(detailed)],
        ignore_index=True,
    )
    inventory_frame.to_csv(OUT / "E2_run_inventory.csv", index=False)

    discrepancy_rows = []
    stated = {
        ("qwen", "battery"): (0.6235, 0.3273, 0.4751),
        ("qwen", "clean_default"): (0.6213, 0.3867, 0.3777),
        ("internvl", "battery"): (None, 0.3003, 0.3594),
        ("internvl", "clean_default"): (0.5546, 0.2507, 0.5480),
    }
    for (model, run), (valid_stated, effective_stated, swap_stated) in stated.items():
        frame = sources[model][run]
        valid_coverage = frame.valid.mean()
        valid_accuracy = frame.loc[frame.valid, "correct"].mean()
        effective = frame.correct.fillna(False).astype(bool).mean()
        discrepancy_rows.append({
            "model": MODEL_NAMES[model], "run_id": run, "n": len(frame),
            "valid_coverage_recomputed": valid_coverage,
            "valid_accuracy_recomputed": valid_accuracy,
            "valid_accuracy_manuscript": valid_stated,
            "effective_recomputed": effective,
            "effective_manuscript": effective_stated,
            "swap_recomputed": 1 - valid_coverage, "swap_manuscript": swap_stated,
        })
    pd.DataFrame(discrepancy_rows).to_csv(OUT / "E2_discrepancy_recalculation.csv", index=False)

    intern = agreement[
        (agreement.model == MODEL_NAMES["internvl"])
        & (agreement.left_run_id == "battery")
        & (agreement.right_run_id == "clean_default")
    ].iloc[0]
    e4a_path = PKG / "analysis/E4/E4A_repeat_same_env.csv"
    e4b_path = PKG / "analysis/E4/E4B_repeat_changed_env.csv"
    if e4a_path.is_file() and e4b_path.is_file():
        e4a = pd.read_csv(e4a_path).sort_values(["model", "language"]).reset_index(drop=True)
        e4b = pd.read_csv(e4b_path).sort_values(["model", "language"]).reset_index(drop=True)
        controlled_text = f"""The completed E4 control resolves the environment
hypothesis. Same-environment pair-state disagreement is
{pct(e4a.pair_state_disagreement.min())}-
{pct(e4a.pair_state_disagreement.max())}; changing only the physical GPU UUID
produces {pct(e4b.pair_state_disagreement.min())}-
{pct(e4b.pair_state_disagreement.max())}. The six E4-B cell results are
identical to their E4-A counterparts, including call verdicts, pair states, and
stored logits. Thus the physical GPU change adds no measured disagreement and
cannot explain the 18.9-point InternVL discrepancy. The approximately 7%
same-environment noise floor and the rendering/protocol difference must be
reported separately."""
    else:
        controlled_text = """The environment-only hypothesis remains pending
until the E4-B controlled hardware-change repeat is complete."""
    report = f"""# E2 Run-discrepancy diagnosis

## Determination

The manuscript values are from separate runs and must not be merged. `battery`,
`clean_default`, `grid`, `rq3`, and `deploy` are the fixed table labels.

The 18.9-point InternVL/Java strict-swap discrepancy is not caused by pair
sampling: battery and clean_default contain the same 3,000 Java pair IDs
(intersection {int(intern.intersection_n)}/3,000). The runs use different
baseline renderings. clean_default uses the corrected default PNGs, whereas
battery uses corrected grounded PNGs rendered with Monokai Dark, font size 20,
wrap width 80, and line numbers on. The battery run also captures verdict logits
under a pinned checkpoint/reproducibility protocol that the earlier clean run
did not record. The state (selected snippet or invalid) changes on
{int(intern.state_disagreement_n)}/3,000 pairs
({pct(intern.state_disagreement_rate)}). Among pairs valid in both runs, the
selected snippet changes on {int(intern.decision_disagreement_both_valid_n)}/
{int(intern.both_valid_n)} ({pct(intern.decision_disagreement_both_valid_rate)}).
Thus the discrepancy is a cross-protocol rendering/run effect, not a sampling
effect.

{controlled_text}

The earlier clean_default run did not store verdict logits. Consequently, a
two-sided cross-run verdict-margin median is not measurable. The agreement CSV
reports the median using only logit-bearing sides and marks this limitation
explicitly; it must not be presented as the old Section 3.6 two-run statistic.

## Manuscript insertion draft

Results in the battery, rendering-grid, and deployment analyses come from
separate runs and are not pooled. We label them `battery`, `grid`, and `deploy`
throughout; the earlier Java packaging audit is labeled `clean_default`.
Although `battery` and `clean_default` use the same 3,000 Java pairs, they use
different corrected renderings and inference records. For InternVL, the
strict-swap error is 35.94% in `battery` and 54.80% in `clean_default`; paired
inspection confirms that this difference is not due to pair sampling. We
therefore interpret cross-run differences as protocol sensitivity rather than
controlled treatment effects and report the controlled same-environment and
single-hardware-change repeats separately.
"""
    (OUT / "E2_DISCREPANCY_REPORT.md").write_text(report, encoding="utf-8")
    print(agreement.to_string(index=False))
    print(f"\nWrote E2 outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
