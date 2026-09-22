#!/usr/bin/env python3
"""Analyze the completed Qwen RQ3 text-plus-image supplementary arm."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "analyze_rq3_image_only.py"
RAW = ROOT / "rq3/inference/raw/Qwen__Qwen2.5-VL-7B-Instruct__text_plus_image__promptB__seed42__full_20260721.jsonl"
OUT = ROOT / "rq3/analysis/qwen_text_plus_image"

spec = importlib.util.spec_from_file_location("rq3_shared", SOURCE)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calls = pd.read_json(RAW, lines=True)
    manifest = json.loads(RAW.with_suffix(".manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("completed_calls") != 3600 or len(calls) != 3600:
        raise RuntimeError("Qwen text-plus-image full run is incomplete")
    calls["selected_variant"] = calls.apply(shared.selected_variant, axis=1)
    rows = []
    keys = ["model", "model_revision", "language", "base_id", "rq0_id", "score_stratum",
            "contrast_type", "variant_i", "variant_j", "preference_target_variant"]
    for contrast_id, group in calls.groupby("contrast_id", sort=False):
        if len(group) != 2 or set(group.order) != {"AB", "BA"}:
            raise RuntimeError(f"incomplete AB/BA: {contrast_id}")
        ab, ba = group[group.order.eq("AB")].iloc[0], group[group.order.eq("BA")].iloc[0]
        strict_valid = pd.notna(ab.selected_variant) and ab.selected_variant == ba.selected_variant
        c = (float(ab.margin) - float(ba.margin)) / 2
        b = (float(ab.margin) + float(ba.margin)) / 2
        target_c = c if ab.preference_target_variant == ab.variant_i else -c
        rows.append({**{key: ab[key] for key in keys}, "contrast_id": contrast_id,
                     "ab_choice": ab.selected_variant, "ba_choice": ba.selected_variant,
                     "strict_valid": strict_valid,
                     "target_selected": bool(strict_valid and ab.selected_variant == ab.preference_target_variant),
                     "parse_failure": bool(pd.isna(ab.selected_variant) or pd.isna(ba.selected_variant)),
                     "margin_ab": float(ab.margin), "margin_ba": float(ba.margin),
                     "content_signal_c": c, "position_bias_b": b, "target_content_signal": target_c,
                     "debiased_target_selected": target_c > 0, "debiased_tie": target_c == 0})
    pairs = pd.DataFrame(rows)
    if len(pairs) != 1800 or pairs.parse_failure.any():
        raise RuntimeError("Qwen text-plus-image pair allocation or parsing failure")
    pairs.to_csv(OUT / "RQ3_QWEN_TEXT_PLUS_IMAGE_PAIR_LEVEL.csv", index=False)
    definitions = {
        "RQ3_QWEN_TEXT_PLUS_IMAGE_OVERALL.csv": ["model"],
        "RQ3_QWEN_TEXT_PLUS_IMAGE_BY_CONTRAST.csv": ["model", "contrast_type", "variant_i", "variant_j"],
        "RQ3_QWEN_TEXT_PLUS_IMAGE_BY_LANGUAGE.csv": ["model", "language"],
        "RQ3_QWEN_TEXT_PLUS_IMAGE_BY_LANGUAGE_CONTRAST.csv":
            ["model", "language", "contrast_type", "variant_i", "variant_j"],
    }
    for name, group_keys in definitions.items():
        shared.grouped_summary(pairs, group_keys).to_csv(OUT / name, index=False)
    audit = {"status": "complete", "modality": "text_plus_image", "model": manifest["model"],
             "raw_calls": len(calls), "pair_rows": len(pairs), "parse_failure_pairs": 0,
             "model_revision": manifest["model_revision"], "prompt_sha256": manifest["prompt_sha256"]}
    (OUT / "RQ3_QWEN_TEXT_PLUS_IMAGE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
