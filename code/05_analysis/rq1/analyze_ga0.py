#!/usr/bin/env python3
"""Summarize GA0 measurement integrity and frozen qualification classes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]
MODELS = ["qwen", "internvl", "phi", "gemma", "ministral"]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - spread, center + spread


def selected_snippet(row: pd.Series) -> str:
    return row.snippet_first if row.parsed_choice == "A" else row.snippet_second


def summarize_model(model: str) -> tuple[list[dict], dict]:
    raw = pd.read_json(OUT / "inference" / "ga0" / model / "raw.jsonl", lines=True)
    if len(raw) != 100 or raw.pair_id.nunique() != 50:
        raise ValueError(f"{model}: incomplete GA0")
    raw["selected_snippet"] = raw.apply(selected_snippet, axis=1)
    pair_rows = []
    for pair_id, pair in raw.groupby("pair_id", sort=False):
        if set(pair.order) != {"AB", "BA"} or len(pair) != 2:
            raise ValueError(f"{model}: bad orders for {pair_id}")
        selected = pair.selected_snippet.tolist()
        valid = selected[0] == selected[1]
        correct = valid and selected[0] == (
            pair.iloc[0].snippet_first
            if pair.iloc[0].gold_side == "first"
            else pair.iloc[0].snippet_second
        )
        pair_rows.append(
            {
                "model_key": model,
                "model": pair.iloc[0].model,
                "language": pair.iloc[0].language,
                "pair_id": pair_id,
                "valid": valid,
                "correct": correct,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    rows = []
    capability_languages = 0
    operational_languages = 0
    direct_failure_languages = 0
    for language, frame in pairs.groupby("language", sort=True):
        total = len(frame)
        valid_n = int(frame.valid.sum())
        correct_n = int(frame.correct.sum())
        valid_accuracy = correct_n / valid_n if valid_n else float("nan")
        effective_accuracy = correct_n / total
        valid_ci = wilson(correct_n, valid_n)
        effective_ci = wilson(correct_n, total)
        capability = valid_n > 0 and valid_ci[0] > 0.5
        operational = effective_ci[0] > 0.5
        direct_failure = valid_n > 0 and valid_ci[1] <= 0.5
        capability_languages += capability
        operational_languages += operational
        direct_failure_languages += direct_failure
        rows.append(
            {
                "model_key": model,
                "model": frame.iloc[0].model,
                "language": language,
                "pairs": total,
                "valid_pairs": valid_n,
                "correct_valid_pairs": correct_n,
                "valid_accuracy": valid_accuracy,
                "valid_ci_low": valid_ci[0],
                "valid_ci_high": valid_ci[1],
                "effective_accuracy": effective_accuracy,
                "effective_ci_low": effective_ci[0],
                "effective_ci_high": effective_ci[1],
                "strict_swap_error": 1 - valid_n / total,
                "capability_evidence": capability,
                "operational_evidence": operational,
                "direct_failure_evidence": direct_failure,
            }
        )
    if capability_languages >= 2 and operational_languages >= 2:
        qualification = "capable"
    elif capability_languages >= 2:
        qualification = "capable-fragile"
    elif direct_failure_languages >= 2 and capability_languages == 0:
        qualification = "broken"
    else:
        qualification = "intermediate"
    model_summary = {
        "model_key": model,
        "model": raw.iloc[0].model,
        "ga0_gate_pass": True,
        "capability_evidence_languages": capability_languages,
        "operational_evidence_languages": operational_languages,
        "direct_failure_evidence_languages": direct_failure_languages,
        "qualification_class": qualification,
    }
    return rows, model_summary


def main() -> None:
    metric_rows = []
    model_rows = []
    for model in MODELS:
        metrics, summary = summarize_model(model)
        metric_rows.extend(metrics)
        model_rows.append(summary)
    result_dir = OUT / "analysis" / "ga0"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(result_dir / "language_metrics.csv", index=False)
    pd.DataFrame(model_rows).to_csv(result_dir / "qualification_classes.csv", index=False)
    payload = {"models": model_rows, "language_metrics": metric_rows}
    (result_dir / "GA0_QUALIFICATION.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(pd.DataFrame(model_rows).to_string(index=False))
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
