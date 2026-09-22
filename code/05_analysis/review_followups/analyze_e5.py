#!/usr/bin/env python3
"""Analyze completed E5 RQ3 and packaging runs without imputing failures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
OUT = PKG / "analysis/E5"
E5A = PKG / "runs/e5_rq3_text_plus_image/rq3/inference/raw"
E5B = PKG / "runs/e5_internvl_combined_text_plus_image"
SHARED_PATH = ROOT / "results/grounded_protocol_3lang_20260721/code/analyze_rq3_image_only.py"


def load_shared():
    spec = importlib.util.spec_from_file_location("e5_rq3_shared", SHARED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rq3_pairs() -> pd.DataFrame:
    shared = load_shared()
    records = []
    files = [
        E5A / (
            "Qwen__Qwen2.5-VL-7B-Instruct__text_plus_image__promptB__seed42__"
            "e5_full_20260730.jsonl"
        ),
        E5A / (
            "OpenGVLab__InternVL3-8B__text_plus_image__promptB__seed42__"
            "e5_retry2_full_20260730.jsonl"
        ),
    ]
    if len(files) != 2:
        raise RuntimeError(f"E5-A requires two complete files, found {len(files)}")
    for path in files:
        calls = pd.read_json(path, lines=True)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        if manifest.get("status") != "complete" or len(calls) != 3600:
            raise RuntimeError(f"incomplete E5-A run: {path}")
        calls["selected_variant"] = calls.apply(shared.selected_variant, axis=1)
        for contrast_id, group in calls.groupby("contrast_id", sort=False):
            if len(group) != 2 or set(group.order) != {"AB", "BA"}:
                raise RuntimeError(f"incomplete AB/BA: {path}:{contrast_id}")
            ab = group[group.order == "AB"].iloc[0]
            ba = group[group.order == "BA"].iloc[0]
            valid = pd.notna(ab.selected_variant) and ab.selected_variant == ba.selected_variant
            c = (float(ab.margin) - float(ba.margin)) / 2
            b = (float(ab.margin) + float(ba.margin)) / 2
            target_c = c if ab.preference_target_variant == ab.variant_i else -c
            records.append({
                "run_id": (
                    "E5A_rq3_text_plus_image_qwen_20260730"
                    if str(ab.model).startswith("Qwen/")
                    else "E5A_rq3_text_plus_image_internvl_retry2_20260730"
                ),
                "model": ab.model, "model_revision": ab.model_revision,
                "language": ab.language, "contrast_id": contrast_id,
                "contrast_type": ab.contrast_type, "variant_i": ab.variant_i,
                "variant_j": ab.variant_j,
                "preference_target_variant": ab.preference_target_variant,
                "strict_valid": valid,
                "target_selected": bool(valid and ab.selected_variant == ab.preference_target_variant),
                "parse_failure": bool(pd.isna(ab.selected_variant) or pd.isna(ba.selected_variant)),
                "content_signal_c": c, "position_bias_b": b,
                "target_content_signal": target_c,
                "debiased_target_selected": target_c > 0,
                "debiased_tie": target_c == 0,
            })
    pairs = pd.DataFrame(records)
    if len(pairs) != 3600:
        raise RuntimeError(f"E5-A pair allocation drift: {len(pairs)}")
    return pairs


def summarize(group: pd.DataFrame) -> dict:
    n = len(group)
    valid = int(group.strict_valid.sum())
    target = int(group.target_selected.sum())
    tie = int(group.debiased_tie.sum())
    debiased_target = int(group.debiased_target_selected.sum())
    return {
        "n": n, "valid_n": valid, "target_correct_n": target,
        "effective_target_preference": target / n,
        "valid_conditional_target_preference": target / valid if valid else np.nan,
        "strict_swap_error": 1 - valid / n,
        "parse_failure_pairs": int(group.parse_failure.sum()),
        "debiased_ties": tie,
        "debiased_target_main_tie_incorrect": debiased_target / n,
        "debiased_target_excluding_ties": debiased_target / (n - tie) if n > tie else np.nan,
        "debiased_target_half_credit": (debiased_target + 0.5 * tie) / n,
        "boundary_abs_c_eq_abs_b": int(
            group.content_signal_c.abs().eq(group.position_bias_b.abs()).sum()
        ),
    }


def grouped(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in pairs.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **summarize(group)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = rq3_pairs()
    pairs.to_csv(OUT / "E5A_RQ3_TEXT_IMAGE_PAIR_LEVEL.csv", index=False)
    overall_diagnostic = grouped(pairs, ["model"])
    language = grouped(pairs, ["model", "language"])
    contrast = grouped(pairs, ["model", "contrast_type", "variant_i", "variant_j"])
    status = {
        "Qwen/Qwen2.5-VL-7B-Instruct": (
            "pass", True, "",
        ),
        "OpenGVLab/InternVL3-8B": (
            "fail", False,
            "one deterministic call emitted FINAL_VERD without an explicit A/B; "
            "full GA0 parse_failures=1",
        ),
    }
    for frame in (overall_diagnostic, language, contrast):
        frame["completeness_status"] = frame.model.map(lambda model: status[model][0])
        frame["reportable"] = frame.model.map(lambda model: status[model][1])
        frame["failure_reason"] = frame.model.map(lambda model: status[model][2])

    overall_diagnostic.to_csv(
        OUT / "E5A_RQ3_TEXT_IMAGE_DIAGNOSTIC.csv", index=False
    )
    overall = overall_diagnostic.copy()
    metric_columns = [
        column for column in overall.columns
        if column not in {
            "model", "n", "completeness_status", "reportable", "failure_reason"
        }
    ]
    overall.loc[~overall.reportable, metric_columns] = np.nan
    for frame, key_columns in (
        (language, {"model", "language", "n"}),
        (contrast, {"model", "contrast_type", "variant_i", "variant_j", "n"}),
    ):
        columns = [
            column for column in frame.columns
            if column not in key_columns | {
                "completeness_status", "reportable", "failure_reason"
            }
        ]
        frame.loc[~frame.reportable, columns] = np.nan
    overall.to_csv(OUT / "E5A_RQ3_TEXT_IMAGE_OVERALL.csv", index=False)
    language.to_csv(OUT / "E5A_RQ3_TEXT_IMAGE_BY_LANGUAGE.csv", index=False)
    contrast.to_csv(OUT / "E5A_RQ3_TEXT_IMAGE_BY_CONTRAST.csv", index=False)

    completeness = json.loads((E5B / "COMPLETENESS_CHECK.json").read_text())
    metrics = completeness["full"]
    raw_path = E5B / "full/e5b_full_20260730.jsonl"
    raw = pd.read_json(raw_path, lines=True)
    if len(raw) != 3000 or raw.pair_id.nunique() != 3000:
        raise RuntimeError("E5-B pair allocation drift")
    qwen_metrics_path = (
        ROOT / "experiments/rq0_viability/outputs/vlm_judges/"
        "full_factorial_clean_20260703/"
        "Qwen__Qwen2.5-VL-7B-Instruct__combined_labeled_text_plus_image__promptB__seed42.metrics.json"
    )
    qwen = json.loads(qwen_metrics_path.read_text())
    packaging = pd.DataFrame([
        {
            "run_id": "clean_full_factorial_20260703", "model": qwen["model"],
            "language": "java", "n": qwen["all_pairs"], "valid_n": qwen["valid_pairs"],
            "correct_n": qwen["correct_pairs"],
            "valid_accuracy": qwen["accuracy_on_valid_pairs"],
            "effective_accuracy": qwen["effective_accuracy"],
            "strict_swap_error": qwen["strict_swap_error_rate"],
            "parse_failure_rate": qwen["parse_failure_rate"],
            "packaging": "combined_labeled", "modality": "text_plus_image",
        },
        {
            "run_id": "E5B_internvl_combined_text_plus_image_20260730",
            "model": metrics["model"], "language": "java", "n": metrics["all_pairs"],
            "valid_n": metrics["valid_pairs"], "correct_n": metrics["correct_pairs"],
            "valid_accuracy": metrics["accuracy_on_valid_pairs"],
            "effective_accuracy": metrics["effective_accuracy"],
            "strict_swap_error": metrics["strict_swap_error_rate"],
            "parse_failure_rate": metrics["parse_failure_rate"],
            "packaging": "combined_labeled", "modality": "text_plus_image",
        },
    ])
    packaging.to_csv(OUT / "E5B_INTERNVL_PACKAGING_CELL.csv", index=False)

    historical = pd.read_json(
        ROOT / "experiments/rq0_viability/outputs/vlm_judges/"
        "full_factorial_clean_20260703/"
        "OpenGVLab__InternVL3-8B__combined_labeled_text_plus_image__promptB__seed42.jsonl",
        lines=True,
    )
    failures = historical[historical.parse_failures > 0][
        ["pair_id", "order_ab_output", "order_ba_output", "parse_failures"]
    ]
    failures.to_csv(OUT / "E5_HISTORICAL_FAILURE_ROWS.csv", index=False)
    report = f"""# E5 Completion report

## Historical failure

The old InternVL RQ3 text+image GA0 failed reproducibly on one output,
`FINAL_VERDDICT: B`. The old combined-packaging full run completed 3,000 pairs
but failed completeness on {int(historical.parse_failures.sum())} calls across
{len(failures)} pairs because InternVL emitted `VERDDICT`, `VERDDIC`, or `VERDD`.
The E5 adapter changes only output-format parsing for these unambiguous prefixes;
it does not modify prompts, generated strings, A/B tokens, or logits.

## Completeness

    E5-A attempted {len(pairs)} pair rows ({len(pairs)//2} per model). Qwen
passes the full gate. InternVL retry 2 still has one unrecoverable call,
`FINAL_VERD`, with no explicit A/B; its cell therefore remains missing and its
diagnostic estimates are not promoted to the main table. E5-B contains
{len(raw)} Java pairs and parse-failure rate {metrics['parse_failure_rate']:.4%}.
No old and new runs were merged.

## RQ3 text+image

{overall.to_markdown(index=False, floatfmt='.4f')}

## Packaging cell

{packaging.to_markdown(index=False, floatfmt='.4f')}
"""
    (OUT / "E5_REPORT.md").write_text(report, encoding="utf-8")
    print(overall.to_string(index=False))
    print(packaging.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
