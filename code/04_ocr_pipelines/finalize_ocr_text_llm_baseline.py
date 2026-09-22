#!/usr/bin/env python3
"""Finalize OCR/source text-only LLM screenshot baseline outputs."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
RESULTS = ROOT / "results/screenshot_only_ocr"
RAW = RESULTS / "ocr_llm_raw"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_LABEL = "Qwen2.5-Coder-7B-Instruct"
PAIR_SET = EXP / "data/pairs/full_pair_set_rq0.csv"
SOURCE_RAW = RAW / "source_text_llm.jsonl"
OCR_RAW = RAW / "ocr_text_llm.jsonl"
PAIR_PRED = RESULTS / "ocr_llm_pair_predictions.csv"
SUMMARY = RESULTS / "ocr_llm_summary.md"
FINAL_TABLE = RESULTS / "final_comparison_table.csv"
STAT_TESTS = RESULTS / "statistical_tests.md"
MANIFEST = RESULTS / "run_manifest.json"
BOOTSTRAP_REPS = 10_000


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def summarize(df: pd.DataFrame, condition: str) -> dict[str, object]:
    n = len(df)
    valid = df[df["is_valid_strict_swap"] == True]
    correct = int(valid["is_correct"].sum()) if len(valid) else 0
    out: dict[str, object] = {
        "condition": condition,
        "model": MODEL_LABEL,
        "num_pairs": int(n),
        "valid_pairs": int(len(valid)),
        "correct_pairs": correct,
        "valid_accuracy": float(correct / len(valid)) if len(valid) else np.nan,
        "effective_accuracy": float(correct / n) if n else np.nan,
        "strict_swap_error": float(1 - len(valid) / n) if n else np.nan,
        "parse_failure_rate": float(df["parse_failures"].sum() / (2 * n)) if n else np.nan,
    }
    for difficulty in ["easy", "medium", "hard"]:
        sub = df[df["difficulty"].eq(difficulty)]
        valid_sub = sub[sub["is_valid_strict_swap"] == True]
        correct_sub = int(valid_sub["is_correct"].sum()) if len(valid_sub) else 0
        out[f"{difficulty}_n"] = int(len(sub))
        out[f"{difficulty}_effective_accuracy"] = float(correct_sub / len(sub)) if len(sub) else np.nan
        out[f"{difficulty}_valid_accuracy"] = float(correct_sub / len(valid_sub)) if len(valid_sub) else np.nan
    return out


def correctness_series(df: pd.DataFrame) -> pd.Series:
    return df.set_index("pair_id")["is_correct"].astype(bool)


def comparison(a: pd.Series, b: pd.Series, name: str, label: str) -> dict[str, object]:
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    arr_a = joined["a"].astype(bool).to_numpy()
    arr_b = joined["b"].astype(bool).to_numpy()
    delta = arr_b.astype(int) - arr_a.astype(int)
    rng = np.random.default_rng(20260630)
    boot = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boot[i] = sample.mean()
    ci = np.percentile(boot, [2.5, 97.5])
    p_boot = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    a_only = int(np.sum(arr_a & ~arr_b))
    b_only = int(np.sum(~arr_a & arr_b))
    discordant = a_only + b_only
    p_mcnemar = float(binomtest(b_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "comparison": name,
        "label": label,
        "n_pairs": int(len(joined)),
        "accuracy_a": float(arr_a.mean()),
        "accuracy_b": float(arr_b.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "bootstrap_ci_low": float(ci[0]),
        "bootstrap_ci_high": float(ci[1]),
        "bootstrap_p_two_sided_sign": float(p_boot),
        "mcnemar_discordant_a_correct": a_only,
        "mcnemar_discordant_b_correct": b_only,
        "mcnemar_p": p_mcnemar,
    }


def table_row(summary: dict[str, object], system: str, kind: str, uses_raw: str, deploy: str, notes: str) -> dict[str, object]:
    return {
        "system": system,
        "kind": kind,
        "model": MODEL_LABEL,
        "num_pairs": summary["num_pairs"],
        "valid_pairs": summary["valid_pairs"],
        "correct_pairs": summary["correct_pairs"],
        "effective_accuracy": summary["effective_accuracy"],
        "valid_accuracy": summary["valid_accuracy"],
        "strict_swap_error": summary["strict_swap_error"],
        "parse_failure_rate": summary["parse_failure_rate"],
        "notes": notes,
        "easy_effective_accuracy": summary["easy_effective_accuracy"],
        "easy_valid_accuracy": summary["easy_valid_accuracy"],
        "easy_n": summary["easy_n"],
        "medium_effective_accuracy": summary["medium_effective_accuracy"],
        "medium_valid_accuracy": summary["medium_valid_accuracy"],
        "medium_n": summary["medium_n"],
        "hard_effective_accuracy": summary["hard_effective_accuracy"],
        "hard_valid_accuracy": summary["hard_valid_accuracy"],
        "hard_n": summary["hard_n"],
        "input_available_at_deployment": deploy,
        "uses_raw_source": uses_raw,
        "ocr_engine": "easyocr" if kind == "ocr_text_llm" else None,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    if not SOURCE_RAW.exists() or not OCR_RAW.exists():
        missing = [str(p) for p in [SOURCE_RAW, OCR_RAW] if not p.exists()]
        raise SystemExit(f"Missing raw LLM outputs: {missing}")
    source = read_jsonl(SOURCE_RAW)
    ocr = read_jsonl(OCR_RAW)
    if len(source) != 3000 or len(ocr) != 3000:
        raise SystemExit(f"Expected 3000 rows each, got source={len(source)} ocr={len(ocr)}")

    source["condition"] = "source_text_llm"
    ocr["condition"] = "ocr_text_llm"
    combined = pd.concat([source, ocr], ignore_index=True)
    combined["ocr_engine"] = np.where(combined["condition"].eq("ocr_text_llm"), "easyocr", None)
    combined["input_text_source"] = np.where(combined["condition"].eq("ocr_text_llm"), "easyocr_ocr_text", "original_raw_source")
    combined.to_csv(PAIR_PRED, index=False)

    source_summary = summarize(source, "source_text_llm")
    ocr_summary = summarize(ocr, "ocr_text_llm")

    table = pd.read_csv(FINAL_TABLE)
    table = table[~table["kind"].isin(["source_text_llm", "ocr_text_llm"])].copy()
    oracle_rf = table[table["system"].eq("Oracle source random_forest_regressor")]["effective_accuracy"].iloc[0]
    ocr_rf = table[table["system"].eq("OCR(EasyOCR)+random_forest_regressor")]["effective_accuracy"].iloc[0]
    rows = [
        table_row(
            source_summary,
            "Source text LLM Qwen2.5-Coder-7B-Instruct",
            "source_text_llm",
            "yes",
            "source text",
            "Oracle text-only LLM upper bound; uses original raw source text, not screenshot-deployable.",
        ),
        table_row(
            ocr_summary,
            "OCR(EasyOCR)+text-only LLM Qwen2.5-Coder-7B-Instruct",
            "ocr_text_llm",
            "no",
            "screenshot via OCR text",
            "Deployable screenshot-only text path; Prompt B over EasyOCR text.",
        ),
    ]
    for row in rows:
        row["delta_vs_oracle_rf"] = float(row["effective_accuracy"] - oracle_rf)
        row["delta_vs_ocr_rf"] = float(row["effective_accuracy"] - ocr_rf)
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(FINAL_TABLE, index=False)
    table.to_markdown(FINAL_TABLE.with_suffix(".md"), index=False, floatfmt=".4f")

    ocr_rf_pairs = pd.read_csv(RESULTS / "classical_pair_predictions.csv")
    ocr_rf_pairs = ocr_rf_pairs[ocr_rf_pairs["model"].eq("random_forest_regressor")]
    direct = pd.read_csv(RESULTS / "direct_vlm_pair_predictions.csv")
    direct_correct = correctness_series(direct)
    ocr_rf_correct = ocr_rf_pairs.set_index("pair_id")["is_correct"].astype(bool)
    source_correct = correctness_series(source)
    ocr_correct = correctness_series(ocr)
    comps = [
        comparison(ocr_rf_correct, ocr_correct, "OCR EasyOCR RF -> OCR Text LLM Qwen2.5-Coder", "OCR RF vs OCR text LLM"),
        comparison(ocr_correct, direct_correct, "OCR Text LLM Qwen2.5-Coder -> Direct VLM OpenGVLab/InternVL3-8B / image_only", "OCR text LLM vs direct VLM"),
        comparison(source_correct, ocr_correct, "Source Text LLM Qwen2.5-Coder -> OCR Text LLM Qwen2.5-Coder", "OCR degradation for LLM path"),
    ]
    base = STAT_TESTS.read_text(encoding="utf-8").split("## OCR + Text-only LLM Comparisons")[0].rstrip()
    comp_df = pd.DataFrame(comps)
    STAT_TESTS.write_text(
        base + "\n\n## OCR + Text-only LLM Comparisons\n\n" + comp_df.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )

    summary_df = pd.DataFrame([source_summary, ocr_summary])
    interpretation = []
    if float(ocr_summary["effective_accuracy"]) > float(ocr_rf):
        interpretation.append("OCR-text LLM outperforms OCR+RF in effective accuracy.")
    else:
        interpretation.append("OCR-text LLM does not outperform OCR+RF in effective accuracy.")
    direct_eff = float(table[table["kind"].eq("direct_vlm")]["effective_accuracy"].iloc[0])
    if float(ocr_summary["effective_accuracy"]) > direct_eff:
        interpretation.append("OCR-text LLM outperforms direct InternVL image-only effective accuracy.")
    else:
        interpretation.append("OCR-text LLM does not outperform direct InternVL image-only effective accuracy.")
    SUMMARY.write_text(
        f"""# OCR + Text-only LLM Summary

Model: `{MODEL_ID}`

## Metrics

{summary_df.to_markdown(index=False, floatfmt=".4f")}

## Interpretation

{chr(10).join(f"- {x}" for x in interpretation)}
- The conclusion is conditional on EasyOCR quality and this specific downloaded text-only model.
- Source-text LLM is an oracle text condition and is not deployable from screenshots without source access.

## Sanity Gate

The pre-run sanity gate was checked in `run_ocr_text_llm_baseline.py`: Oracle RF 62.33%, OCR+RF 52.37%, direct VLM effective 27.93%, direct VLM valid 48.72%, direct VLM swap error 42.67%.

## Outputs

- `ocr_llm_pair_predictions.csv`
- `final_comparison_table.csv`
- `statistical_tests.md`
- `run_manifest.json`
""",
        encoding="utf-8",
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest["ocr_text_llm_addendum"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": "completed",
        "model": MODEL_ID,
        "prompt_variant": "B",
        "conditions_completed": ["source_text_llm", "ocr_text_llm"],
        "n_pairs": 3000,
        "raw_outputs": {"source_text_llm": str(SOURCE_RAW), "ocr_text_llm": str(OCR_RAW)},
        "metrics": {"source_text_llm": source_summary, "ocr_text_llm": ocr_summary},
        "statistical_tests": comps,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"source": source_summary, "ocr": ocr_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
