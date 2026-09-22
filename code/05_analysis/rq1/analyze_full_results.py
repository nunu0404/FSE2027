#!/usr/bin/env python3
"""Analyze completed RQ1 full runs under the frozen metric definitions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parents[1]
MODELS = ["qwen", "internvl", "gemma", "ministral", "phi"]


def selected_snippet(row: pd.Series) -> str | None:
    if row.parsed_choice == "A":
        return row.snippet_first
    if row.parsed_choice == "B":
        return row.snippet_second
    return None


def load_completed_model(model_key: str, pairs: pd.DataFrame) -> pd.DataFrame | None:
    run_dir = OUT / "inference/full" / model_key
    manifest_path = run_dir / "manifest.json"
    raw_path = run_dir / "raw.jsonl"
    if not manifest_path.exists() or not raw_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("calls_completed") != 18000:
        return None
    raw = pd.read_json(raw_path, lines=True)
    if len(raw) != 18000:
        raise ValueError(f"{model_key}: raw call count drift")
    if raw[["pair_id", "order"]].duplicated().any():
        raise ValueError(f"{model_key}: duplicate pair/order key")
    expected = pd.MultiIndex.from_product(
        [pairs.protocol_pair_id, ["AB", "BA"]], names=["pair_id", "order"]
    )
    actual = pd.MultiIndex.from_frame(raw[["pair_id", "order"]])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(f"{model_key}: pair/order coverage drift")
    raw["selected_snippet"] = raw.apply(selected_snippet, axis=1)
    pair_rows = []
    pair_lookup = pairs.set_index("protocol_pair_id")
    for pair_id, frame in raw.groupby("pair_id", sort=False):
        frame = frame.set_index("order")
        ab, ba = frame.loc["AB"], frame.loc["BA"]
        source = pair_lookup.loc[pair_id]
        valid = (
            ab.selected_snippet is not None
            and ba.selected_snippet is not None
            and ab.selected_snippet == ba.selected_snippet
        )
        selected = ab.selected_snippet if valid else None
        correct = bool(valid and selected == source.human_preference)
        model_sign = (
            1 if selected == source.snippet_i
            else -1 if selected == source.snippet_j
            else np.nan
        )
        human_delta = source.human_score_i_z - source.human_score_j_z
        margin_ab = float(ab.margin)
        margin_ba = float(ba.margin)
        content_margin = (margin_ab - margin_ba) / 2
        position_margin = (margin_ab + margin_ba) / 2
        debiased_selected = (
            source.snippet_i if content_margin > 0
            else source.snippet_j if content_margin < 0
            else None
        )
        pair_rows.append(
            {
                "model_key": model_key,
                "model": ab.model,
                "concurrency_group": manifest.get("concurrency_group"),
                "pair_id": pair_id,
                "language": source.language,
                "difficulty": source.difficulty,
                "abs_z_diff": source.abs_z_diff,
                "human_delta": human_delta,
                "valid": valid,
                "correct": correct,
                "model_sign": model_sign,
                "ab_correct": ab.selected_snippet == source.human_preference,
                "ba_correct": ba.selected_snippet == source.human_preference,
                "margin_ab": margin_ab,
                "margin_ba": margin_ba,
                "content_margin": content_margin,
                "position_margin": position_margin,
                "debiased_correct": debiased_selected == source.human_preference,
                "content_tie": content_margin == 0,
                "ab_first_choice": ab.parsed_choice == "A",
                "ba_first_choice": ba.parsed_choice == "A",
            }
        )
    return pd.DataFrame(pair_rows)


def summarize(frame: pd.DataFrame, scope: str, value: str) -> dict:
    total = len(frame)
    valid = frame[frame.valid]
    correct = int(frame.correct.sum())
    if len(valid) >= 2 and valid.model_sign.nunique() >= 2:
        rho = float(spearmanr(valid.model_sign, valid.human_delta).statistic)
    else:
        rho = float("nan")
    return {
        "model_key": frame.iloc[0].model_key,
        "model": frame.iloc[0].model,
        "concurrency_group": frame.iloc[0].concurrency_group,
        "scope": scope,
        "scope_value": value,
        "pairs": total,
        "valid_pairs": len(valid),
        "valid_coverage": len(valid) / total,
        "correct_valid_pairs": correct,
        "valid_accuracy": correct / len(valid) if len(valid) else float("nan"),
        "effective_accuracy": correct / total,
        "strict_swap_error": 1 - len(valid) / total,
        "spearman_valid_only": rho,
        "spearman_coverage": len(valid) / total,
        "ab_accuracy": float(frame.ab_correct.mean()),
        "ba_accuracy": float(frame.ba_correct.mean()),
        "debiased_accuracy": float(frame.debiased_correct.mean()),
        "content_ties": int(frame.content_tie.sum()),
        "mean_abs_content_margin": float(frame.content_margin.abs().mean()),
        "median_abs_content_margin": float(frame.content_margin.abs().median()),
        "mean_abs_position_margin": float(frame.position_margin.abs().mean()),
        "first_position_choice_rate": float(
            pd.concat([frame.ab_first_choice, frame.ba_first_choice]).mean()
        ),
    }


def main() -> None:
    pairs = pd.read_csv(OUT / "data/rq1_pairs_9000.csv")
    frames = []
    incomplete = []
    for model in MODELS:
        frame = load_completed_model(model, pairs)
        if frame is None:
            incomplete.append(model)
        else:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no complete model runs")
    pair_results = pd.concat(frames, ignore_index=True)
    summaries = []
    for _, model_frame in pair_results.groupby("model_key", sort=False):
        summaries.append(summarize(model_frame, "overall", "all"))
        for language, frame in model_frame.groupby("language", sort=True):
            summaries.append(summarize(frame, "language", language))
        for difficulty, frame in model_frame.groupby("difficulty", sort=True):
            summaries.append(summarize(frame, "difficulty", difficulty))
    summary = pd.DataFrame(summaries)
    result_dir = OUT / "analysis/full"
    result_dir.mkdir(parents=True, exist_ok=True)
    pair_results.to_csv(result_dir / "pair_level_results.csv", index=False)
    summary.to_csv(result_dir / "metrics_by_scope.csv", index=False)
    metadata = {
        "completed_models": pair_results.model_key.unique().tolist(),
        "incomplete_models_excluded": incomplete,
        "pairs_per_completed_model": 9000,
        "calls_per_completed_model": 18000,
        "metric_contract": {
            "valid": "AB and BA choose the same underlying snippet",
            "valid_accuracy": "correct strict-valid pairs / strict-valid pairs",
            "effective_accuracy": "correct strict-valid pairs / all pairs",
            "strict_swap_error": "non-valid pairs / all pairs",
            "spearman_valid_only": "Spearman(model preference sign, human z-score difference) on strict-valid pairs",
            "debiased_accuracy": "sign((AB margin - BA margin)/2) against gold; zero content margin is incorrect",
        },
    }
    (result_dir / "ANALYSIS_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(summary[summary.scope.eq("overall")].to_string(index=False))
    print(summary[summary.scope.eq("language")].to_string(index=False))
    print(summary[summary.scope.eq("difficulty")].to_string(index=False))


if __name__ == "__main__":
    main()
