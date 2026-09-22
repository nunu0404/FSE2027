#!/usr/bin/env python3
"""Complete or explicitly not-run the OCR + text-only LLM screenshot baseline."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/screenshot_only_ocr"
EXP = ROOT / "experiments/rq0_viability"

PAIR_SET = EXP / "data/pairs/full_pair_set_rq0.csv"
SOURCE_CLASSICAL = EXP / "outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv"
OCR_CLASSICAL = RESULTS / "classical_pair_predictions.csv"
DIRECT_VLM = RESULTS / "direct_vlm_pair_predictions.csv"
OCR_TEXT = RESULTS / "ocr_outputs/easyocr/snippet_ocr.csv"
FINAL_TABLE = RESULTS / "final_comparison_table.csv"
STAT_TESTS = RESULTS / "statistical_tests.md"
PAIR_PRED = RESULTS / "ocr_llm_pair_predictions.csv"
SUMMARY = RESULTS / "ocr_llm_summary.md"
MANIFEST = RESULTS / "run_manifest.json"

CANDIDATES = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]


def pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "not_run"
    return f"{x * 100:.2f}%"


def hf_cache_models() -> list[str]:
    hub = Path.home() / ".cache/huggingface/hub"
    if not hub.exists():
        return []
    models = []
    for path in sorted(hub.glob("models--*")):
        if path.is_dir():
            models.append(path.name.replace("models--", "").replace("--", "/"))
    return models


def candidate_cache_status(models: list[str]) -> list[dict[str, str]]:
    lower = {m.lower(): m for m in models}
    out = []
    aliases = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": ["qwen/qwen2.5-coder-7b-instruct"],
        "Qwen/Qwen2.5-7B-Instruct": ["qwen/qwen2.5-7b-instruct"],
        "codellama/CodeLlama-7b-Instruct-hf": [
            "codellama/codellama-7b-instruct-hf",
            "codellama/codellama-7b-instruct",
            "codellama/codellama-7b-instruct-hf",
        ],
        "deepseek-ai/deepseek-coder-6.7b-instruct": [
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ],
    }
    for candidate in CANDIDATES:
        found = None
        for alias in aliases[candidate]:
            if alias in lower:
                found = lower[alias]
                break
        out.append(
            {
                "candidate": candidate,
                "status": "cached" if found else "not_run",
                "reason": "" if found else "not found in local Hugging Face cache",
                "local_repo_id": found or "",
            }
        )
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def summarize_classical(path: Path, model: str) -> dict[str, float | int]:
    df = pd.read_csv(path)
    df = df[df["model"].eq(model)].copy()
    if len(df) != 3000:
        raise RuntimeError(f"Expected 3000 rows for {model} in {path}, got {len(df)}")
    return {
        "num_pairs": int(len(df)),
        "valid_pairs": int(df["is_valid_score_pair"].sum()),
        "correct_pairs": int(df["is_correct"].sum()),
        "effective_accuracy": float(df["is_correct"].mean()),
        "valid_accuracy": float(df.loc[df["is_valid_score_pair"], "is_correct"].mean()),
    }


def summarize_vlm(path: Path) -> dict[str, float | int]:
    df = pd.read_csv(path)
    if len(df) != 3000:
        raise RuntimeError(f"Expected 3000 direct VLM rows in {path}, got {len(df)}")
    valid = df[df["is_valid_strict_swap"] == True]
    correct = int(valid["is_correct"].sum())
    return {
        "num_pairs": int(len(df)),
        "valid_pairs": int(len(valid)),
        "correct_pairs": correct,
        "effective_accuracy": float(correct / len(df)),
        "valid_accuracy": float(correct / len(valid)) if len(valid) else np.nan,
        "strict_swap_error": float(1 - len(valid) / len(df)),
        "parse_failure_rate": float((df["parse_failures"].fillna(0) > 0).mean()) if "parse_failures" in df.columns else np.nan,
    }


def sanity_gate() -> dict[str, object]:
    oracle_rf = summarize_classical(SOURCE_CLASSICAL, "random_forest_regressor")
    ocr_rf = summarize_classical(OCR_CLASSICAL, "random_forest_regressor")
    direct = summarize_vlm(DIRECT_VLM)
    checks = [
        ("Oracle source RF effective", oracle_rf["effective_accuracy"], 0.6233333333333333),
        ("OCR+RF effective", ocr_rf["effective_accuracy"], 0.5236666666666666),
        ("Direct VLM effective", direct["effective_accuracy"], 0.2793333333333333),
        ("Direct VLM valid", direct["valid_accuracy"], 0.4872093023255814),
        ("Direct VLM strict_swap_error", direct["strict_swap_error"], 0.4266666666666667),
    ]
    failures = []
    for label, got, expected in checks:
        if abs(float(got) - expected) > 0.001:
            failures.append({"label": label, "got": float(got), "expected": expected})
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "oracle_rf": oracle_rf,
        "ocr_rf": ocr_rf,
        "direct_vlm": direct,
    }


def not_run_pair_predictions() -> None:
    columns = [
        "pair_id",
        "condition",
        "model",
        "ocr_engine",
        "input_text_source",
        "order_ab_output",
        "order_ba_output",
        "parsed_ab",
        "parsed_ba",
        "preference_ab",
        "preference_ba",
        "is_valid_strict_swap",
        "model_preference",
        "is_correct",
        "parse_failures",
        "not_run_reason",
    ]
    pd.DataFrame(columns=columns).to_csv(PAIR_PRED, index=False)


def final_table_with_not_run(reason: str) -> None:
    table = pd.read_csv(FINAL_TABLE)
    cols = list(table.columns)
    rows = []
    common = {c: np.nan for c in cols}
    for system, kind, uses_raw, deploy, notes in [
        (
            "Source text LLM",
            "source_text_llm",
            "yes",
            "source text",
            "not_run: no requested priority text-only LLM checkpoint was cached; source text LLM upper bound was not executed.",
        ),
        (
            "OCR(EasyOCR)+text-only LLM",
            "ocr_text_llm",
            "no",
            "screenshot via OCR text",
            f"not_run: {reason}",
        ),
    ]:
        row = common.copy()
        row.update(
            {
                "system": system,
                "kind": kind,
                "model": "not_run",
                "num_pairs": 0,
                "valid_pairs": 0,
                "correct_pairs": 0,
                "effective_accuracy": np.nan,
                "valid_accuracy": np.nan,
                "strict_swap_error": np.nan,
                "parse_failure_rate": np.nan,
                "notes": notes,
                "input_available_at_deployment": deploy,
                "uses_raw_source": uses_raw,
                "ocr_engine": "easyocr" if kind == "ocr_text_llm" else np.nan,
                "delta_vs_oracle_rf": np.nan,
                "delta_vs_ocr_rf": np.nan,
            }
        )
        rows.append(row)
    table = table[~table["kind"].isin(["ocr_text_llm", "source_text_llm"])].copy()
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(FINAL_TABLE, index=False)
    table.to_markdown(FINAL_TABLE.with_suffix(".md"), index=False, floatfmt=".4f")


def write_stat_tests_append(reason: str) -> None:
    base = STAT_TESTS.read_text(encoding="utf-8").rstrip()
    extra = f"""

## OCR + Text-only LLM Comparisons

The requested OCR/source text-only LLM strict-swap runs were not executed.

Reason: {reason}

| comparison | status | reason |
|:--|:--|:--|
| OCR+RF -> OCR-text LLM | not_run | OCR-text LLM row unavailable because no requested priority text-only LLM checkpoint was cached. |
| OCR-text LLM -> Direct VLM InternVL image_only | not_run | OCR-text LLM row unavailable because no requested priority text-only LLM checkpoint was cached. |
| Source-text LLM -> OCR-text LLM | not_run | Both source-text and OCR-text LLM rows unavailable because no requested priority text-only LLM checkpoint was cached. |
"""
    STAT_TESTS.write_text(base + extra + "\n", encoding="utf-8")


def write_summary(reason: str, cache_status: list[dict[str, str]], gate: dict[str, object]) -> None:
    rows = pd.DataFrame(cache_status)
    text = f"""# OCR + Text-only LLM Summary

Status: **not_run**

## Reason

{reason}

The local Hugging Face cache was scanned first, as required. The available model repos were not any of the requested text-only priority models. Cached VLM checkpoints were not substituted.

## Candidate Model Cache Status

{rows.to_markdown(index=False)}

## Sanity Gate

Passed before stopping:

- Oracle source RF effective accuracy: {pct(gate['oracle_rf']['effective_accuracy'])}
- OCR+RF effective accuracy: {pct(gate['ocr_rf']['effective_accuracy'])}
- Direct VLM InternVL image-only effective accuracy: {pct(gate['direct_vlm']['effective_accuracy'])}
- Direct VLM InternVL image-only valid accuracy: {pct(gate['direct_vlm']['valid_accuracy'])}
- Direct VLM InternVL image-only strict-swap error: {pct(gate['direct_vlm']['strict_swap_error'])}

## Consequence

No source-text LLM or OCR-text LLM strict-swap predictions were generated. `ocr_llm_pair_predictions.csv` is an empty not-run placeholder with the expected columns. The final comparison table contains explicit not-run rows for both requested LLM conditions.
"""
    SUMMARY.write_text(text, encoding="utf-8")


def update_manifest(cache_models: list[str], cache_status: list[dict[str, str]], reason: str, gate: dict[str, object]) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest["ocr_text_llm_addendum"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": "not_run",
        "reason": reason,
        "candidate_priority": CANDIDATES,
        "candidate_cache_status": cache_status,
        "cached_model_repos_seen": cache_models,
        "prompt_variant": "B",
        "n_pairs_target": 3000,
        "conditions_requested": ["source_text_llm", "ocr_text_llm"],
        "input_files": {
            "easyocr_text": str(OCR_TEXT),
            "pair_set": str(PAIR_SET),
            "source_text": str(EXP / "data/processed/pooled_313_processed.csv"),
            "final_comparison_table": str(FINAL_TABLE),
            "ocr_classical_pair_predictions": str(OCR_CLASSICAL),
            "direct_vlm_pair_predictions": str(DIRECT_VLM),
        },
        "sanity_gate": gate,
        "outputs": {
            "ocr_llm_pair_predictions": str(PAIR_PRED),
            "ocr_llm_summary": str(SUMMARY),
            "final_comparison_table": str(FINAL_TABLE),
            "statistical_tests": str(STAT_TESTS),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    gate = sanity_gate()
    if gate["status"] != "passed":
        path = RESULTS / "OCR_TEXT_LLM_SANITY_GATE_FAILED.md"
        path.write_text("# OCR Text LLM Sanity Gate Failed\n\n" + json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Sanity gate failed. See {path}")

    cache_models = hf_cache_models()
    status = candidate_cache_status(cache_models)
    cached = [row for row in status if row["status"] == "cached"]
    if cached:
        raise SystemExit(
            "A requested text-only model is cached, but this script only handles the not_run path. "
            "Use run_judge_pairs.py text_only strict-swap inference for cached model: "
            + json.dumps(cached)
        )

    reason = (
        "No requested priority text-only code-capable LLM checkpoint was found in the local Hugging Face cache "
        "(checked Qwen2.5-Coder-7B-Instruct, Qwen2.5-7B-Instruct, CodeLlama-7B-Instruct, "
        "DeepSeek-Coder-6.7B-Instruct). Download of a new 7B checkpoint was not explicitly permitted for this run."
    )
    not_run_pair_predictions()
    final_table_with_not_run(reason)
    write_stat_tests_append(reason)
    write_summary(reason, status, gate)
    update_manifest(cache_models, status, reason, gate)
    print(json.dumps({"status": "not_run", "reason": reason, "summary": str(SUMMARY)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
