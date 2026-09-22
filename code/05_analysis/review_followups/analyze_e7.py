#!/usr/bin/env python3
"""Expose all existing closed-model pilot generations without new API calls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/fse2027_review_defense_e1_e8_20260730/analysis/E7"
PAIR_FILE = ROOT / "experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv"


def valid_rho(raw_path: Path) -> dict[str, float]:
    raw = pd.read_json(raw_path, lines=True)
    pairs = pd.read_csv(PAIR_FILE)[
        ["pair_id", "snippet_i", "snippet_j", "human_score_i_z", "human_score_j_z"]
    ]
    raw = raw.merge(pairs, on=["pair_id", "snippet_i", "snippet_j"], validate="many_to_one")
    valid = raw[raw.is_valid_strict_swap.astype(bool)].copy()
    valid["model_sign"] = np.where(valid.model_preference.eq(valid.snippet_i), 1, -1)
    valid["human_delta"] = valid.human_score_i_z - valid.human_score_j_z
    values = {}
    for condition, group in valid.groupby("condition"):
        values[condition] = float(spearmanr(group.model_sign, group.human_delta).statistic)
    return values


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    standard_specs = [
        ("gpt-5.4-mini", ROOT / "results/openai_vlm_pilot_20260707"),
        ("gpt-5.4", ROOT / "results/openai_vlm_pilot_20260708"),
        ("gpt-5.5", ROOT / "results/openai_vlm_pilot_20260708_gpt55_fixed"),
    ]
    for model, directory in standard_specs:
        summary = pd.read_csv(directory / f"{model}__pilot_summary.csv")
        rho = valid_rho(directory / f"{model}__pilot_raw.jsonl")
        for row in summary.itertuples(index=False):
            rows.append({
                "run_id": directory.name, "model_generation": model,
                "condition": row.condition, "reasoning_effort": "API default/not explicitly set",
                "max_output_tokens": 24, "n": row.num_pairs, "valid_n": row.valid_pairs,
                "valid_accuracy": row.valid_accuracy, "effective_accuracy": row.effective_accuracy,
                "strict_swap_error": row.strict_swap_error,
                "parse_failure_rate": row.parse_failure_rate, "rho_valid_only": rho[row.condition],
                "protocol_note": "300-pair Java pilot; no rendering-grid treatment",
            })

    reasoning = pd.read_csv(
        ROOT / "results/gpt54_reasoning_budget_pilot_90_20260709/reasoning_4096_full300_summary.csv"
    )
    for row in reasoning.itertuples(index=False):
        rows.append({
            "run_id": "gpt54_reasoning_4096_full300", "model_generation": "gpt-5.4-2026-03-05",
            "condition": row.condition, "reasoning_effort": row.reasoning_effort,
            "max_output_tokens": 4096, "n": row.pairs, "valid_n": row.valid_pairs,
            "valid_accuracy": row.valid_accuracy, "effective_accuracy": row.effective_accuracy,
            "strict_swap_error": row.strict_swap_error,
            "parse_failure_rate": row.parse_failure_rate,
            "rho_valid_only": row.spearman_valid_only,
            "protocol_note": "300-pair Java reasoning-budget pilot; no rendering-grid treatment",
        })

    qwen = pd.read_csv(
        ROOT / "experiments/rq0_viability/outputs/strong_vlm_sanity/qwen32b_parse_fixed_summary.csv"
    )
    qwen = qwen[(qwen.task == "source") & qwen.stratum.isna()].iloc[0]
    rows.append({
        "run_id": "qwen32b_parse_fixed", "model_generation": "Qwen2.5-VL-32B",
        "condition": "text_plus_image", "reasoning_effort": "not applicable",
        "max_output_tokens": 1, "n": qwen.num_pairs, "valid_n": qwen.valid_pairs,
        "valid_accuracy": qwen.valid_accuracy, "effective_accuracy": qwen.effective_accuracy,
        "strict_swap_error": qwen.strict_swap_error_rate,
        "parse_failure_rate": qwen.strict_parse_failure_rate, "rho_valid_only": np.nan,
        "protocol_note": "300-pair Java protocol check; forced single-character output; no rendering grid",
    })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "E7_closed_model_pilot.csv", index=False)

    # Verify the manuscript's cross-condition GPT-5.4 high/4096 McNemar statement.
    raw = pd.read_json(
        ROOT / "results/gpt54_reasoning_budget_pilot_90_20260709/"
        "gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl",
        lines=True,
    )
    pivot = raw.pivot(index="pair_id", columns="condition", values="is_valid_strict_swap")
    image_invalid = ~pivot.image_only.astype(bool)
    text_invalid = ~pivot.text_plus_image.astype(bool)
    image_only = int((image_invalid & ~text_invalid).sum())
    text_only = int((~image_invalid & text_invalid).sum())
    mcnemar = float(binomtest(min(image_only, text_only), image_only + text_only, 0.5).pvalue)
    check = pd.DataFrame([{
        "model": "gpt-5.4-2026-03-05", "reasoning_effort": "high",
        "max_output_tokens": 4096, "image_only_swap_error": image_invalid.mean(),
        "text_plus_image_swap_error": text_invalid.mean(),
        "image_invalid_text_valid": image_only,
        "image_valid_text_invalid": text_only, "exact_mcnemar_p": mcnemar,
    }])
    check.to_csv(OUT / "E7_gpt54_text_swap_check.csv", index=False)

    compact = result[
        (result.model_generation.str.startswith("gpt"))
        & (result.condition.isin(["image_only", "text_plus_image"]))
    ][[
        "model_generation", "reasoning_effort", "max_output_tokens", "condition",
        "n", "valid_n", "valid_accuracy", "effective_accuracy", "strict_swap_error",
        "parse_failure_rate", "rho_valid_only",
    ]]
    caption = (
        "Protocol-check results on 300 Java pairs (100 per legacy difficulty stratum). "
        "The rendering grid was not applied, so rows are not a controlled model-scale "
        "comparison. Qwen2.5-VL-32B additionally required forced single-character output."
    )
    (OUT / "E7_TABLE_DRAFT.md").write_text(
        "# Closed-model pilot table\n\n"
        + compact.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n**Caption.** {caption}\n\n"
        + f"GPT-5.4 high/4096 cross-modality check: swap error "
        f"{image_invalid.mean():.2%} -> {text_invalid.mean():.2%}; exact McNemar "
        f"p={mcnemar:.3g} (discordant n={image_only + text_only}).\n",
        encoding="utf-8",
    )
    print(compact.to_string(index=False))
    print(check.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
