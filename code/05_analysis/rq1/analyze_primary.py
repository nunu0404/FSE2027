#!/usr/bin/env python3
"""Analyze validated latest-model primary runs under frozen metric definitions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/latest_vlm_extension_20260830"
PAIR_PATH = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
OLD_PAIR_RESULTS = ROOT / "results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv"
MODELS = ["qwen3", "internvl3_5", "gemma4"]
PREDECESSORS = {
    "qwen3": "qwen",
    "internvl3_5": "internvl",
    "gemma4": "gemma",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(OUT), "rev-parse", "HEAD"], text=True
    ).strip()


def chosen_snippet(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.snippet_first
    if row.parsed_choice == "B":
        return row.snippet_second
    return None


def pair_frame(model_key: str, pairs: pd.DataFrame) -> pd.DataFrame:
    run_dir = OUT / "inference/full" / model_key
    validation = json.loads((run_dir / "validation.json").read_text())
    if not validation.get("gate_pass"):
        raise RuntimeError(f"full validation failed for {model_key}")
    raw = pd.read_json(run_dir / "raw.jsonl", lines=True)
    effective = pd.read_json(run_dir / "effective_logits.jsonl", lines=True)[
        ["pair_id", "order", "effective_margin"]
    ]
    raw = raw.merge(
        effective, on=["pair_id", "order"], how="left", validate="one_to_one"
    )
    raw["chosen_snippet"] = raw.apply(chosen_snippet, axis=1)
    pair_lookup = pairs.set_index("protocol_pair_id")
    rows = []
    for pair_id, calls in raw.groupby("pair_id", sort=False):
        calls = calls.set_index("order")
        ab, ba = calls.loc["AB"], calls.loc["BA"]
        pair = pair_lookup.loc[pair_id]
        valid = bool(
            ab.chosen_snippet is not None
            and ba.chosen_snippet is not None
            and ab.chosen_snippet == ba.chosen_snippet
        )
        strict_choice = ab.chosen_snippet if valid else None
        margin_ab = float(ab.effective_margin) if pd.notna(ab.effective_margin) else np.nan
        margin_ba = float(ba.effective_margin) if pd.notna(ba.effective_margin) else np.nan
        complete_logits = bool(pd.notna(margin_ab) and pd.notna(margin_ba))
        b = (margin_ab + margin_ba) / 2.0 if complete_logits else np.nan
        c = (margin_ab - margin_ba) / 2.0 if complete_logits else np.nan
        debiased_choice = (
            pair.snippet_i if c > 0 else pair.snippet_j if c < 0 else None
        )
        rows.append(
            {
                "model_key": model_key,
                "model": ab.model,
                "pair_id": pair_id,
                "language": pair.language,
                "difficulty": pair.difficulty,
                "abs_z_diff": pair.abs_z_diff,
                "human_delta": pair.human_score_i_z - pair.human_score_j_z,
                "human_preference": pair.human_preference,
                "valid": valid,
                "strict_choice": strict_choice,
                "correct": valid and strict_choice == pair.human_preference,
                "model_sign": (
                    1
                    if strict_choice == pair.snippet_i
                    else -1
                    if strict_choice == pair.snippet_j
                    else np.nan
                ),
                "ab_correct": ab.chosen_snippet == pair.human_preference,
                "ba_correct": ba.chosen_snippet == pair.human_preference,
                "ab_first_choice": ab.parsed_choice == "A",
                "ba_first_choice": ba.parsed_choice == "A",
                "parse_failure_calls": int(pd.isna(ab.parsed_choice))
                + int(pd.isna(ba.parsed_choice)),
                "margin_ab": margin_ab,
                "margin_ba": margin_ba,
                "position_margin_b": b,
                "content_margin_c": c,
                "debiased_choice": debiased_choice,
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
    rho = float("nan")
    if len(valid) >= 2 and valid.model_sign.nunique() >= 2:
        rho = float(spearmanr(valid.model_sign, valid.human_delta).statistic)
    correct = int(frame.correct.sum())
    return {
        "model_key": frame.iloc[0].model_key,
        "model": frame.iloc[0].model,
        "scope": scope,
        "scope_value": value,
        "pairs": len(frame),
        "valid_pairs": len(valid),
        "valid_accuracy": correct / len(valid) if len(valid) else float("nan"),
        "effective_accuracy": correct / len(frame),
        "strict_swap_error": 1 - len(valid) / len(frame),
        "ab_accuracy": float(frame.ab_correct.mean()),
        "ba_accuracy": float(frame.ba_correct.mean()),
        "first_position_choice_rate": float(
            pd.concat([frame.ab_first_choice, frame.ba_first_choice]).mean()
        ),
        "spearman_valid_only": rho,
        "two_order_averaged_accuracy": float(frame.debiased_correct.mean()),
        "content_ties": int(frame.content_tie.sum()),
        "boundaries": int(frame.boundary.sum()),
        "bc_rule_violations": int(frame.bc_rule_violation.sum()),
        "parse_failure_calls": int(frame.parse_failure_calls.sum()),
        "pairs_missing_effective_logits": int((~frame.complete_effective_logits).sum()),
    }


def summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize(frame, "pooled", "all")]
    for language, subset in frame.groupby("language", sort=True):
        rows.append(summarize(subset, "language", language))
    for (language, difficulty), subset in frame.groupby(
        ["language", "difficulty"], sort=True
    ):
        rows.append(summarize(subset, "language_difficulty", f"{language}|{difficulty}"))
    return pd.DataFrame(rows)


def predecessor_pair_comparison(new_pairs: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_PAIR_RESULTS)
    rows = []
    for new_key, old_key in PREDECESSORS.items():
        current = new_pairs[new_pairs.model_key.eq(new_key)].copy()
        predecessor = old[old.model_key.eq(old_key)].copy()
        merged = current.merge(
            predecessor[
                [
                    "pair_id",
                    "valid",
                    "correct",
                    "debiased_correct",
                    "content_margin",
                    "position_margin",
                ]
            ],
            on="pair_id",
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        if len(merged) != 9000:
            raise RuntimeError(f"predecessor pair join failed: {new_key}")
        for language, subset in merged.groupby("language", sort=True):
            rows.append(
                {
                    "new_model": new_key,
                    "predecessor": old_key,
                    "language": language,
                    "pairs": len(subset),
                    "delta_valid_coverage": float(
                        subset.valid_new.mean() - subset.valid_old.mean()
                    ),
                    "delta_effective_accuracy": float(
                        subset.correct_new.mean() - subset.correct_old.mean()
                    ),
                    "delta_two_order_averaged_accuracy": float(
                        subset.debiased_correct_new.mean()
                        - subset.debiased_correct_old.mean()
                    ),
                    "new_only_effective_correct": int(
                        (subset.correct_new & ~subset.correct_old).sum()
                    ),
                    "old_only_effective_correct": int(
                        (~subset.correct_new & subset.correct_old).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def stored_summary(frame: pd.DataFrame) -> dict:
    valid = frame[frame.valid]
    rho = float("nan")
    if len(valid) >= 2 and valid.model_sign.nunique() >= 2:
        rho = float(spearmanr(valid.model_sign, valid.human_delta).statistic)
    return {
        "pairs": len(frame),
        "valid_accuracy": float(valid.correct.mean()) if len(valid) else float("nan"),
        "effective_accuracy": float(frame.correct.mean()),
        "strict_swap_error": 1 - float(frame.valid.mean()),
        "two_order_averaged_accuracy": float(frame.debiased_correct.mean()),
        "first_position_choice_rate": float(
            pd.concat([frame.ab_first_choice, frame.ba_first_choice]).mean()
        ),
        "spearman_valid_only": rho,
    }


def generation_metric_comparison(new_pairs: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_PAIR_RESULTS)
    rows = []
    for new_key, old_key in PREDECESSORS.items():
        current = new_pairs[new_pairs.model_key.eq(new_key)]
        predecessor = old[old.model_key.eq(old_key)]
        scopes = [("pooled", "all", current, predecessor)]
        for language in ("java", "python", "cuda"):
            scopes.append(
                (
                    "language",
                    language,
                    current[current.language.eq(language)],
                    predecessor[predecessor.language.eq(language)],
                )
            )
            for difficulty in ("easy", "medium", "hard"):
                scopes.append(
                    (
                        "language_difficulty",
                        f"{language}|{difficulty}",
                        current[
                            current.language.eq(language)
                            & current.difficulty.eq(difficulty)
                        ],
                        predecessor[
                            predecessor.language.eq(language)
                            & predecessor.difficulty.eq(difficulty)
                        ],
                    )
                )
        for scope, value, new_frame, old_frame in scopes:
            new_metrics = stored_summary(new_frame)
            old_metrics = stored_summary(old_frame)
            row = {
                "new_model": new_key,
                "predecessor": old_key,
                "scope": scope,
                "scope_value": value,
            }
            for metric in old_metrics:
                row[f"old_{metric}"] = old_metrics[metric]
                row[f"new_{metric}"] = new_metrics[metric]
                if metric != "pairs":
                    row[f"delta_{metric}"] = new_metrics[metric] - old_metrics[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    frames = [pair_frame(model, pairs) for model in MODELS]
    all_pairs = pd.concat(frames, ignore_index=True)
    all_metrics = pd.concat([summaries(frame) for frame in frames], ignore_index=True)
    if int(all_pairs.bc_rule_violation.sum()) != 0:
        raise RuntimeError(
            f"b/c validity rule failed for {int(all_pairs.bc_rule_violation.sum())} pairs"
        )
    comparisons = predecessor_pair_comparison(all_pairs)
    metric_comparisons = generation_metric_comparison(all_pairs)
    analysis_dir = OUT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_pairs.to_csv(analysis_dir / "primary_pair_level.csv", index=False)
    all_metrics.to_csv(analysis_dir / "primary_metrics.csv", index=False)
    comparisons.to_csv(analysis_dir / "generation_comparison.csv", index=False)
    metric_comparisons.to_csv(
        analysis_dir / "generation_metric_comparison.csv", index=False
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "package_git_commit": package_commit(),
        "models": MODELS,
        "pairs_per_model": 9000,
        "calls_per_model": 18000,
        "historical_outputs_rerun": False,
        "historical_metrics_reproduced": False,
        "predecessor_comparison_source": str(OLD_PAIR_RESULTS.relative_to(ROOT)),
        "tie_rule": "c=0 is incorrect for primary two-order-averaged accuracy",
        "bc_rule": "strict_valid iff abs(c)>abs(b); zero empirical violations required",
        "bc_rule_violations": int(all_pairs.bc_rule_violation.sum()),
        "inference_manifests": {
            model: str((OUT / f"inference/full/{model}/manifest.json").relative_to(OUT))
            for model in MODELS
        },
    }
    (analysis_dir / "primary_analysis_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(all_metrics.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(metric_comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
