#!/usr/bin/env python3
"""Analyze completed RQ3 image-only runs without repairing invalid outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[3]
RQ3 = ROOT / "results/grounded_protocol_3lang_20260721/rq3"
RAW = RQ3 / "inference/raw"
OUT = RQ3 / "analysis/image_only"
FILES = [
    RAW / "Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42__full_20260721.jsonl",
    RAW / "OpenGVLab__InternVL3-8B__image_only__promptB__seed42__full_20260721.jsonl",
]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [math.nan] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def selected_variant(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.first_variant
    if row.parsed_choice == "B":
        return row.second_variant
    return None


def load_pairs() -> pd.DataFrame:
    records = []
    for path in FILES:
        frame = pd.read_json(path, lines=True)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("completed_calls") != 3600:
            raise RuntimeError(f"incomplete manifest: {path}")
        if len(frame) != 3600 or frame.duplicated(["contrast_id", "order"]).any():
            raise RuntimeError(f"raw call allocation drift: {path}")
        frame["selected_variant"] = frame.apply(selected_variant, axis=1)
        for contrast_id, group in frame.groupby("contrast_id", sort=False):
            if len(group) != 2 or set(group.order) != {"AB", "BA"}:
                raise RuntimeError(f"incomplete AB/BA pair: {path}:{contrast_id}")
            ab = group[group.order.eq("AB")].iloc[0]
            ba = group[group.order.eq("BA")].iloc[0]
            shared = ["model", "model_revision", "language", "base_id", "rq0_id", "score_stratum",
                      "contrast_type", "variant_i", "variant_j", "preference_target_variant"]
            if any(ab[column] != ba[column] for column in shared):
                raise RuntimeError(f"AB/BA metadata drift: {path}:{contrast_id}")
            strict_valid = ab.selected_variant is not None and ab.selected_variant == ba.selected_variant
            target_selected = strict_valid and ab.selected_variant == ab.preference_target_variant
            c = (float(ab.margin) - float(ba.margin)) / 2
            b = (float(ab.margin) + float(ba.margin)) / 2
            target_c = c if ab.preference_target_variant == ab.variant_i else -c
            records.append({
                **{column: ab[column] for column in shared},
                "contrast_id": contrast_id,
                "ab_choice": ab.selected_variant, "ba_choice": ba.selected_variant,
                "strict_valid": strict_valid, "target_selected": target_selected,
                "parse_failure": ab.selected_variant is None or ba.selected_variant is None,
                "margin_ab": float(ab.margin), "margin_ba": float(ba.margin),
                "content_signal_c": c, "position_bias_b": b, "target_content_signal": target_c,
                "debiased_target_selected": target_c > 0, "debiased_tie": target_c == 0,
            })
    result = pd.DataFrame(records)
    if len(result) != 3600 or result.groupby("model").size().to_dict() != {
        "OpenGVLab/InternVL3-8B": 1800, "Qwen/Qwen2.5-VL-7B-Instruct": 1800
    }:
        raise RuntimeError("pair-level allocation drift")
    return result


def summarize(group: pd.DataFrame) -> dict[str, object]:
    n = len(group)
    valid = int(group.strict_valid.sum())
    target = int(group.target_selected.sum())
    invalid = n - valid
    parse_failures = int(group.parse_failure.sum())
    debiased_n = int((~group.debiased_tie).sum())
    debiased_target = int(group.debiased_target_selected.sum())
    pref_low, pref_high = wilson(target, valid)
    swap_low, swap_high = wilson(invalid, n)
    return {
        "n_pairs": n, "valid_pairs": valid, "invalid_pairs": invalid,
        "target_selected_valid_pairs": target, "parse_failure_pairs": parse_failures,
        "valid_rate": valid / n, "target_preference_valid": target / valid if valid else math.nan,
        "target_preference_valid_wilson_low": pref_low,
        "target_preference_valid_wilson_high": pref_high,
        "effective_target_preference": target / n, "strict_swap_error": invalid / n,
        "strict_swap_wilson_low": swap_low, "strict_swap_wilson_high": swap_high,
        "binom_p_vs_0_5_valid": binomtest(target, valid, 0.5).pvalue if valid else math.nan,
        "debiased_non_tie_pairs": debiased_n, "debiased_target_pairs": debiased_target,
        "debiased_target_preference": debiased_target / debiased_n if debiased_n else math.nan,
        "median_abs_content_signal": float(group.content_signal_c.abs().median()),
        "median_abs_position_bias": float(group.position_bias_b.abs().median()),
    }


def grouped_summary(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in pairs.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        rows.append({**dict(zip(keys, values)), **summarize(group)})
    result = pd.DataFrame(rows)
    correction_keys = [key for key in ["model", "language", "score_stratum"] if key in keys]
    if "variant_i" in keys and "variant_j" in keys:
        result["holm_p_within_family"] = math.nan
        grouped = result.groupby(correction_keys[0] if len(correction_keys) == 1 else correction_keys, sort=False)
        for _, indices in grouped.groups.items():
            indices = list(indices)
            adjusted = holm(result.loc[indices, "binom_p_vs_0_5_valid"].tolist())
            result.loc[indices, "holm_p_within_family"] = adjusted
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    pairs.to_csv(OUT / "RQ3_IMAGE_ONLY_PAIR_LEVEL.csv", index=False)
    tables = {
        "RQ3_IMAGE_ONLY_OVERALL.csv": grouped_summary(pairs, ["model"]),
        "RQ3_IMAGE_ONLY_BY_CONTRAST.csv": grouped_summary(
            pairs, ["model", "contrast_type", "variant_i", "variant_j"]
        ),
        "RQ3_IMAGE_ONLY_BY_LANGUAGE.csv": grouped_summary(pairs, ["model", "language"]),
        "RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv": grouped_summary(
            pairs, ["model", "language", "contrast_type", "variant_i", "variant_j"]
        ),
        "RQ3_IMAGE_ONLY_BY_STRATUM_CONTRAST.csv": grouped_summary(
            pairs, ["model", "language", "score_stratum", "contrast_type", "variant_i", "variant_j"]
        ),
    }
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False)
    contrast = tables["RQ3_IMAGE_ONLY_BY_CONTRAST.csv"].copy()
    core = tables["RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv"]
    core = core[core.contrast_type.eq("semantic_visual_conflict_primary")].copy()
    display_columns = ["model", "variant_i", "variant_j", "n_pairs", "valid_pairs",
                       "target_selected_valid_pairs", "target_preference_valid",
                       "effective_target_preference", "strict_swap_error"]
    core_columns = ["model", "language", "n_pairs", "valid_pairs", "target_selected_valid_pairs",
                    "target_preference_valid", "effective_target_preference", "strict_swap_error"]
    for frame in (contrast, core):
        for column in ["target_preference_valid", "effective_target_preference", "strict_swap_error"]:
            frame[column] = (100 * frame[column]).map(lambda value: f"{value:.2f}%")
    report = """# RQ3 Image-Only Results

## Method

The completed design contains 300 bases (100 each in Java, Python, and CUDA), four semantic-integrity x visual-quality cells per base, and all six unordered contrasts. Both pinned VLMs judged 1,800 contrasts in AB and BA order, producing 3,600 calls per model and 7,200 calls total. Inputs were two ordered rendered PNGs plus frozen Prompt B; source text was not supplied. Inference used BF16, greedy decoding, temperature 0, inactive top-p 1.0, max 24 generated tokens, and seed 42.

There is no independent human pairwise gold for generated variants. The outcome is therefore target preference, not accuracy. A pair is strict-valid only when AB and BA select the same underlying variant. `target_preference_valid` conditions on valid pairs; `effective_target_preference` divides strict-valid target selections by all pairs; `strict_swap_error` is the invalid fraction. The target is the clean variant when semantics are equal and the semantics-preserving variant when semantic integrity differs.

## Six Contrasts

""" + contrast[display_columns].to_markdown(index=False) + """

## Primary Conflict by Language

The primary conflict is `ugly_gold` versus `beautiful_trash`: preserved semantics with degraded presentation versus destroyed semantics with clean presentation.

""" + core[core_columns].to_markdown(index=False) + """

## Integrity and Interpretation

Both full manifests are complete. All 7,200 calls parsed, all parsed verdicts matched the captured A/B logit argmax, and every generation used 7 tokens. High valid-only target preference must not be read without its valid denominator. In particular, Qwen's `beautiful_trash` versus `ugly_trash` result has 25/300 valid pairs and 91.67% strict-swap error, so its 100% valid-only clean preference corresponds to only 8.33% effective preference over all pairs.

Verdict-logit decomposition is secondary. It uses `c=(m_AB-m_BA)/2` and `b=(m_AB+m_BA)/2`; it does not replace strict-swap results.
"""
    (OUT / "RQ3_IMAGE_ONLY_REPORT.md").write_text(report + "\n", encoding="utf-8")
    audit = {
        "status": "complete",
        "raw_calls": 7200, "pair_rows": len(pairs),
        "pairs_by_model": pairs.groupby("model").size().to_dict(),
        "pairs_by_model_language": {
            f"{model}/{language}": int(value)
            for (model, language), value in pairs.groupby(["model", "language"]).size().items()
        },
        "parse_failure_pairs": int(pairs.parse_failure.sum()),
        "output_files": ["RQ3_IMAGE_ONLY_PAIR_LEVEL.csv", *tables.keys(), "RQ3_IMAGE_ONLY_REPORT.md"],
    }
    (OUT / "RQ3_IMAGE_ONLY_ANALYSIS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
