#!/usr/bin/env python3
"""Analyze paired packaging/modality robustness outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
PAIR_PATH = OUT / "data/robustness_pairs_900_seed42.csv"
MODELS = ("qwen3", "internvl3_5")
REFERENCE = "separate_image_only"
CONDITIONS = (
    REFERENCE,
    "combined_image_only",
    "separate_text_plus_image",
    "combined_text_plus_image",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()


def chosen(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.snippet_first
    if row.parsed_choice == "B":
        return row.snippet_second
    return None


def pair_results(model_key: str, pairs: pd.DataFrame) -> pd.DataFrame:
    run_dir = OUT / "inference/robustness" / model_key
    validation = json.loads((run_dir / "validation.json").read_text())
    if not validation.get("gate_pass"):
        raise RuntimeError(f"robustness validation failed: {model_key}")
    raw = pd.read_json(run_dir / "raw.jsonl", lines=True)
    raw["chosen"] = raw.apply(chosen, axis=1)
    lookup = pairs.set_index("protocol_pair_id")
    rows = []
    for (condition, pair_id), calls in raw.groupby(["condition", "pair_id"]):
        calls = calls.set_index("order")
        ab, ba = calls.loc["AB"], calls.loc["BA"]
        pair = lookup.loc[pair_id]
        valid = bool(
            ab.chosen is not None and ba.chosen is not None and ab.chosen == ba.chosen
        )
        strict_choice = ab.chosen if valid else None
        m_ab = float(ab.margin) if pd.notna(ab.margin) else np.nan
        m_ba = float(ba.margin) if pd.notna(ba.margin) else np.nan
        complete_logits = bool(pd.notna(m_ab) and pd.notna(m_ba))
        b = (m_ab + m_ba) / 2 if complete_logits else np.nan
        c = (m_ab - m_ba) / 2 if complete_logits else np.nan
        debiased_choice = pair.snippet_i if c > 0 else pair.snippet_j if c < 0 else None
        rows.append(
            {
                "model_key": model_key,
                "model": ab.model,
                "condition": condition,
                "pair_id": pair_id,
                "language": pair.language,
                "difficulty": pair.difficulty,
                "abs_z_diff": pair.abs_z_diff,
                "human_delta": pair.human_score_i_z - pair.human_score_j_z,
                "valid": valid,
                "correct": valid and strict_choice == pair.human_preference,
                "model_sign": (
                    1
                    if strict_choice == pair.snippet_i
                    else -1
                    if strict_choice == pair.snippet_j
                    else np.nan
                ),
                "ab_correct": ab.chosen == pair.human_preference,
                "ba_correct": ba.chosen == pair.human_preference,
                "ab_first": ab.parsed_choice == "A",
                "ba_first": ba.parsed_choice == "A",
                "parse_failure_calls": int(pd.isna(ab.parsed_choice))
                + int(pd.isna(ba.parsed_choice)),
                "position_margin_b": b,
                "content_margin_c": c,
                "debiased_correct": debiased_choice == pair.human_preference,
                "complete_effective_logits": complete_logits,
                "content_tie": complete_logits and c == 0,
                "boundary": complete_logits and c != 0 and abs(c) == abs(b),
                "bc_rule_evaluable": complete_logits and c != 0 and abs(c) != abs(b),
                "bc_valid_prediction": complete_logits and abs(c) > abs(b),
                "bc_rule_violation": (
                    complete_logits
                    and c != 0
                    and abs(c) != abs(b)
                    and valid != (abs(c) > abs(b))
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, scope: str, value: str) -> dict:
    valid = frame[frame.valid]
    correct = int(frame.correct.sum())
    rho = float("nan")
    if len(valid) >= 2 and valid.model_sign.nunique() >= 2:
        rho = float(spearmanr(valid.model_sign, valid.human_delta).statistic)
    return {
        "model_key": frame.iloc[0].model_key,
        "model": frame.iloc[0].model,
        "condition": frame.iloc[0].condition,
        "scope": scope,
        "scope_value": value,
        "pairs": len(frame),
        "valid_pairs": len(valid),
        "valid_accuracy": correct / len(valid) if len(valid) else float("nan"),
        "effective_accuracy": correct / len(frame),
        "strict_swap_error": 1 - len(valid) / len(frame),
        "two_order_averaged_accuracy": float(frame.debiased_correct.mean()),
        "ab_accuracy": float(frame.ab_correct.mean()),
        "ba_accuracy": float(frame.ba_correct.mean()),
        "first_position_choice_rate": float(
            pd.concat([frame.ab_first, frame.ba_first]).mean()
        ),
        "spearman_valid_only": rho,
        "parse_failure_calls": int(frame.parse_failure_calls.sum()),
        "content_ties": int(frame.content_tie.sum()),
        "boundaries": int(frame.boundary.sum()),
        "bc_rule_violations": int(frame.bc_rule_violation.sum()),
        "pairs_missing_effective_logits": int((~frame.complete_effective_logits).sum()),
    }


def metric_table(all_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), frame in all_pairs.groupby(["model_key", "condition"], sort=True):
        rows.append(summarize(frame, "pooled", "all"))
        for language, subset in frame.groupby("language", sort=True):
            rows.append(summarize(subset, "language", language))
        for (language, difficulty), subset in frame.groupby(
            ["language", "difficulty"], sort=True
        ):
            rows.append(
                summarize(subset, "language_difficulty", f"{language}|{difficulty}")
            )
    return pd.DataFrame(rows)


def paired_comparisons(all_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        model_frame = all_pairs[all_pairs.model_key.eq(model)]
        for language in ("pooled", "java", "python", "cuda"):
            scope = (
                model_frame
                if language == "pooled"
                else model_frame[model_frame.language.eq(language)]
            )
            reference = scope[scope.condition.eq(REFERENCE)].set_index("pair_id")
            for condition in CONDITIONS[1:]:
                candidate = scope[scope.condition.eq(condition)].set_index("pair_id")
                joined = reference[["correct", "valid", "debiased_correct"]].join(
                    candidate[["correct", "valid", "debiased_correct"]],
                    lsuffix="_reference",
                    rsuffix="_candidate",
                    how="inner",
                    validate="one_to_one",
                )
                ref_only = int(
                    (joined.correct_reference & ~joined.correct_candidate).sum()
                )
                candidate_only = int(
                    (~joined.correct_reference & joined.correct_candidate).sum()
                )
                discordant = ref_only + candidate_only
                p_value = (
                    float(
                        binomtest(
                            min(ref_only, candidate_only), discordant, 0.5
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                )
                rows.append(
                    {
                        "model_key": model,
                        "language": language,
                        "reference_condition": REFERENCE,
                        "candidate_condition": condition,
                        "pairs": len(joined),
                        "delta_valid_coverage": float(
                            joined.valid_candidate.mean()
                            - joined.valid_reference.mean()
                        ),
                        "delta_effective_accuracy": float(
                            joined.correct_candidate.mean()
                            - joined.correct_reference.mean()
                        ),
                        "delta_two_order_averaged_accuracy": float(
                            joined.debiased_correct_candidate.mean()
                            - joined.debiased_correct_reference.mean()
                        ),
                        "reference_only_effective_correct": ref_only,
                        "candidate_only_effective_correct": candidate_only,
                        "exact_mcnemar_p": p_value,
                    }
                )
    return pd.DataFrame(rows)


def primary_repeat_consistency(all_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        primary_path = OUT / "inference/full" / model / "raw.jsonl"
        primary_calls = pd.read_json(primary_path, lines=True)
        repeat_calls = pd.read_json(
            OUT / "inference/robustness" / model / "raw.jsonl", lines=True
        )
        repeat_calls = repeat_calls[
            repeat_calls.condition.eq(REFERENCE)
        ].copy()
        joined = primary_calls.merge(
            repeat_calls,
            on=["pair_id", "order"],
            suffixes=("_primary", "_repeat"),
            validate="one_to_one",
        )
        for language, frame in joined.groupby("language_primary", sort=True):
            rows.append(
                {
                    "model_key": model,
                    "language": language,
                    "calls": len(frame),
                    "parsed_verdict_agreement": float(
                        frame.parsed_choice_primary.eq(frame.parsed_choice_repeat).mean()
                    ),
                    "median_abs_margin_difference": float(
                        (frame.margin_primary - frame.margin_repeat).abs().median()
                    ),
                    "max_abs_margin_difference": float(
                        (frame.margin_primary - frame.margin_repeat).abs().max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    frames = [pair_results(model, pairs) for model in MODELS]
    all_pairs = pd.concat(frames, ignore_index=True)
    if int(all_pairs.bc_rule_violation.sum()) != 0:
        raise RuntimeError(
            f"b/c validity rule failed for {int(all_pairs.bc_rule_violation.sum())} pairs"
        )
    metrics = metric_table(all_pairs)
    comparisons = paired_comparisons(all_pairs)
    repeat = primary_repeat_consistency(all_pairs)
    analysis_dir = OUT / "analysis"
    all_pairs.to_csv(analysis_dir / "robustness_pair_level.csv", index=False)
    metrics.to_csv(analysis_dir / "robustness_metrics.csv", index=False)
    comparisons.to_csv(analysis_dir / "robustness_paired_comparisons.csv", index=False)
    repeat.to_csv(analysis_dir / "robustness_primary_repeat_consistency.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "package_git_commit": package_commit(),
        "models": list(MODELS),
        "pairs_per_model": 900,
        "conditions": list(CONDITIONS),
        "orders": ["AB", "BA"],
        "calls_per_model": 7200,
        "paired_reference": REFERENCE,
        "paired_test": "two-sided exact McNemar on effective-correct indicator",
        "bc_rule": "strict_valid iff abs(c)>abs(b); zero empirical violations required",
        "bc_rule_violations": int(all_pairs.bc_rule_violation.sum()),
        "inference_manifests": {
            model: str(
                (OUT / f"inference/robustness/{model}/manifest.json").relative_to(OUT)
            )
            for model in MODELS
        },
    }
    (analysis_dir / "robustness_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(repeat.to_string(index=False))


if __name__ == "__main__":
    main()
