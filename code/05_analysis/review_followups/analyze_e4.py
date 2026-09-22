#!/usr/bin/env python3
"""Measure call-, pair-, metric-, and bit-level reproducibility for E4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
PKG = ROOT / "results/fse2027_review_defense_e1_e8_20260730"
GROUND = ROOT / "results/grounded_protocol_3lang_20260721"
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
BASELINE = "monokai_dark__fs20__wrap80__lnon"
MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B",
}


def pair_states(calls: pd.DataFrame) -> pd.DataFrame:
    frame = calls.copy()
    frame["selected"] = np.where(
        frame.parsed_choice.eq("A"), frame.snippet_first,
        np.where(frame.parsed_choice.eq("B"), frame.snippet_second, None),
    )
    pivot = frame.pivot(
        index=["model", "language", "pair_id", "snippet_i", "snippet_j"],
        columns="order", values=["selected", "margin"],
    ).reset_index()
    pivot.columns = [
        "_".join(part for part in col if part) if isinstance(col, tuple) else col
        for col in pivot.columns
    ]
    pivot["valid"] = (
        pivot.selected_AB.notna() & pivot.selected_BA.notna()
        & pivot.selected_AB.eq(pivot.selected_BA)
    )
    pivot["decision"] = pivot.selected_AB.where(pivot.valid)
    # gold_side in raw is order-specific, so recover gold from the AB row.
    ab = frame[frame.order == "AB"][["model", "language", "pair_id", "gold_side"]].copy()
    ab["gold"] = np.where(ab.gold_side.eq("first"), frame[frame.order == "AB"].snippet_first.values,
                          frame[frame.order == "AB"].snippet_second.values)
    pivot = pivot.merge(
        ab[["model", "language", "pair_id", "gold"]],
        on=["model", "language", "pair_id"], validate="one_to_one",
    )
    pivot["correct"] = pivot.valid & pivot.decision.eq(pivot.gold)
    pivot["content_margin"] = (pivot.margin_AB - pivot.margin_BA) / 2
    pivot["state"] = pivot.decision.fillna("__INVALID__")
    return pivot


def compare_run(original: pd.DataFrame, repeat: pd.DataFrame, run_id: str) -> pd.DataFrame:
    keys = ["model", "language", "pair_id", "order"]
    merged = original.merge(repeat, on=keys, suffixes=("_original", "_repeat"), validate="one_to_one")
    merged["call_verdict_match"] = merged.parsed_choice_original.eq(merged.parsed_choice_repeat)
    merged["logit_a_abs_diff"] = (merged.logit_A_original - merged.logit_A_repeat).abs()
    merged["logit_b_abs_diff"] = (merged.logit_B_original - merged.logit_B_repeat).abs()
    merged["logit_pair_bit_exact"] = (
        merged.logit_A_original.eq(merged.logit_A_repeat)
        & merged.logit_B_original.eq(merged.logit_B_repeat)
    )
    left = pair_states(original)
    right = pair_states(repeat)
    pkeys = ["model", "language", "pair_id"]
    pairs = left.merge(right, on=pkeys, suffixes=("_original", "_repeat"), validate="one_to_one")
    pairs["pair_state_match"] = pairs.state_original.eq(pairs.state_repeat)
    pairs["pair_decision_match_both_valid"] = np.where(
        pairs.valid_original & pairs.valid_repeat,
        pairs.decision_original.eq(pairs.decision_repeat), np.nan,
    )
    pairs["changed_margin"] = pairs.content_margin_original.abs()

    rows = []
    for (model, language), group in pairs.groupby(["model", "language"]):
        cg = merged[(merged.model == model) & (merged.language == language)]
        changed = group[~group.pair_state_match]
        both_valid = group.valid_original & group.valid_repeat
        rows.append({
            "run_id": run_id, "model": model, "language": language,
            "n_pairs": len(group), "n_calls": len(cg),
            "call_verdict_agreement": cg.call_verdict_match.mean(),
            "pair_state_agreement": group.pair_state_match.mean(),
            "pair_state_disagreement": 1 - group.pair_state_match.mean(),
            "both_valid_n": int(both_valid.sum()),
            "decision_agreement_both_valid": (
                group.loc[both_valid, "pair_decision_match_both_valid"].mean()
                if both_valid.any() else np.nan
            ),
            "changed_pair_original_abs_content_margin_median": (
                changed.changed_margin.median() if len(changed) else np.nan
            ),
            "logit_pair_bit_exact_rate": cg.logit_pair_bit_exact.mean(),
            "logit_max_absolute_difference": max(
                cg.logit_a_abs_diff.max(), cg.logit_b_abs_diff.max()
            ),
            "original_valid_accuracy": group.loc[group.valid_original, "correct_original"].mean(),
            "repeat_valid_accuracy": group.loc[group.valid_repeat, "correct_repeat"].mean(),
            "original_effective_accuracy": group.correct_original.mean(),
            "repeat_effective_accuracy": group.correct_repeat.mean(),
            "original_swap_error": 1 - group.valid_original.mean(),
            "repeat_swap_error": 1 - group.valid_repeat.mean(),
        })
    return pd.DataFrame(rows)


def load_grid(path: Path) -> pd.DataFrame:
    data = pd.read_json(path, lines=True)
    return data[data.condition == BASELINE].copy() if "condition" in data else data


def grid_analysis() -> None:
    output = PKG / "analysis/E4"
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    for model_key, model in MODELS.items():
        safe = model.replace("/", "__")
        original = load_grid(
            GROUND / "inference/grid/raw" /
            f"{safe}__image_only__promptB__seed42__full_20260721.jsonl"
        )
        for root_name, tag, run_id in [
            ("repeat_same_env", "e4a_same_env_20260730", "E4A_same_environment"),
            ("repeat_changed_env", "e4b_gpu1_20260730", "E4B_GPU0_to_GPU1"),
        ]:
            repeat = load_grid(
                PKG / "runs" / root_name / "inference/grid/raw" /
                f"{safe}__image_only__promptB__seed42__{tag}.jsonl"
            )
            if len(repeat) != 6000:
                raise RuntimeError(f"incomplete {run_id}/{model}: {len(repeat)}/6000 calls")
            tables.append(compare_run(original, repeat, run_id))
    result = pd.concat(tables, ignore_index=True)
    result[result.run_id == "E4A_same_environment"].to_csv(
        output / "E4A_repeat_same_env.csv", index=False
    )
    result[result.run_id == "E4B_GPU0_to_GPU1"].to_csv(
        output / "E4B_repeat_changed_env.csv", index=False
    )


def battery_analysis() -> None:
    output = PKG / "analysis/E4"
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    for model_key, model in {
        "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
        "internvl": "OpenGVLab/InternVL3-8B",
        "gemma": "google/gemma-3-12b-it",
        "ministral": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "phi": "microsoft/Phi-4-multimodal-instruct",
    }.items():
        original = pd.read_json(
            BATTERY / f"inference/full/{model_key}/raw.jsonl", lines=True
        )
        repeat_path = PKG / f"runs/battery_repeat/inference/full/{model_key}/raw.jsonl"
        repeat = pd.read_json(repeat_path, lines=True)
        if len(repeat) != 18000:
            raise RuntimeError(f"incomplete battery repeat {model}: {len(repeat)}/18000")
        tables.append(compare_run(original, repeat, "E4C_battery_repeat"))
    result = pd.concat(tables, ignore_index=True)
    result["reproducibility_threshold_pass"] = result.pair_state_agreement.ge(0.995)
    result.to_csv(output / "E4C_battery_repeat.csv", index=False)


def report() -> None:
    output = PKG / "analysis/E4"
    a = pd.read_csv(output / "E4A_repeat_same_env.csv")
    b = pd.read_csv(output / "E4B_repeat_changed_env.csv")
    c = pd.read_csv(output / "E4C_battery_repeat.csv")
    environment = """# E4 environment comparison

| Component | Original grid/battery | E4-A/E4-C | E4-B |
|---|---|---|---|
| Software environment | frozen Python/package snapshot | same | same |
| Inference engine | Transformers 5.9.0 | same | same |
| dtype/decoding | BF16, greedy, max 24 | same | same |
| prompt/pairs/images | SHA-pinned originals | same | same |
| physical GPU | GPU0 UUID GPU-98933e9f... | GPU0, same UUID | GPU1 UUID GPU-197872ce... |
| driver/CUDA | 580.95.05 / torch cu130 | same | same |

E4-B changes only the physical GPU UUID. Both devices are NVIDIA RTX PRO 6000
Blackwell Server Edition; software, engine, precision, and assets remain fixed.
"""
    (output / "E4_ENV_DIFF.md").write_text(environment, encoding="utf-8")
    report_text = f"""# E4 Reproducibility report

Same-environment pair-state disagreement ranges from
{100*a.pair_state_disagreement.min():.2f}% to
{100*a.pair_state_disagreement.max():.2f}% across six cells. The controlled
GPU-UUID change ranges from {100*b.pair_state_disagreement.min():.2f}% to
{100*b.pair_state_disagreement.max():.2f}%. Logit bit identity is reported
separately because identical verdicts need not have identical BF16 logits.

For the battery repeat, {int(c.reproducibility_threshold_pass.sum())}/{len(c)}
model-language cells meet the preregistered >=99.5% pair-state agreement
threshold. Original Table 1/2 values are not overwritten under either outcome.

## Draft language

We measured the noise floor by repeating all six baseline grid cells under the
same frozen environment and by changing only the physical GPU. We report
pair-level state disagreement, call-level verdict agreement, and bit-level logit
identity separately. The full five-model battery was then repeated under its
original concurrency schedule. Under the preregistered rule, cells below 99.5%
agreement are reported as reproducibility findings and both runs are retained;
the original headline table is not silently replaced.
"""
    (output / "E4_REPORT.md").write_text(report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["grid", "battery", "report"], required=True)
    args = parser.parse_args()
    if args.part == "grid":
        grid_analysis()
    elif args.part == "battery":
        battery_analysis()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
