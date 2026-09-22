#!/usr/bin/env python3
"""Summarize clean-render OCR/source text-only LLM strict-swap runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/screenshot_only_ocr_clean_20260707"
RAW = RESULTS / "ocr_text_llm_raw"
FINAL_TABLE = RESULTS / "clean_multi_ocr_final_comparison_table.csv"
CLASSICAL_PAIRS = RESULTS / "ocr_classical_pair_predictions.csv"
DIRECT_VLM_PAIRS = RESULTS / "clean_direct_vlm_pair_predictions.csv"
OUT_SUMMARY = RESULTS / "clean_ocr_text_llm_summary.csv"
OUT_MD = RESULTS / "clean_ocr_text_llm_summary.md"
OUT_PAIRS = RESULTS / "clean_ocr_text_llm_pair_predictions.csv"
OUT_STATS = RESULTS / "clean_ocr_text_llm_statistical_tests.csv"
MODEL_LABEL = "Qwen2.5-Coder-7B-Instruct"


RUNS = [
    ("source_text_llm", "Source text LLM Qwen2.5-Coder", None),
    ("ocr_text_llm_easyocr", "OCR(EasyOCR)+text-only LLM Qwen2.5-Coder", "easyocr"),
    ("ocr_text_llm_rapidocr", "OCR(RapidOCR)+text-only LLM Qwen2.5-Coder", "rapidocr"),
    (
        "ocr_text_llm_easyocr_preprocessed",
        "OCR(EasyOCR-preprocessed)+text-only LLM Qwen2.5-Coder",
        "easyocr_preprocessed",
    ),
]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, run_name: str, system: str, ocr_engine: str | None) -> dict[str, object]:
    valid = df[df["is_valid_strict_swap"] == True]
    correct = int(valid["is_correct"].sum()) if len(valid) else 0
    out: dict[str, object] = {
        "run_name": run_name,
        "system": system,
        "kind": "source_text_llm" if ocr_engine is None else "ocr_text_llm",
        "model": MODEL_LABEL,
        "ocr_engine": ocr_engine,
        "num_pairs": int(len(df)),
        "valid_pairs": int(len(valid)),
        "correct_pairs": correct,
        "effective_accuracy": float(correct / len(df)) if len(df) else np.nan,
        "valid_accuracy": float(correct / len(valid)) if len(valid) else np.nan,
        "strict_swap_error": float(1 - len(valid) / len(df)) if len(df) else np.nan,
        "parse_failure_rate": float(df["parse_failures"].sum() / (2 * len(df))) if len(df) else np.nan,
    }
    for difficulty in ["easy", "medium", "hard"]:
        sub = df[df["difficulty"].eq(difficulty)]
        valid_sub = sub[sub["is_valid_strict_swap"] == True]
        correct_sub = int(valid_sub["is_correct"].sum()) if len(valid_sub) else 0
        out[f"{difficulty}_n"] = int(len(sub))
        out[f"{difficulty}_effective_accuracy"] = float(correct_sub / len(sub)) if len(sub) else np.nan
        out[f"{difficulty}_valid_accuracy"] = float(correct_sub / len(valid_sub)) if len(valid_sub) else np.nan
    return out


def correctness(df: pd.DataFrame) -> pd.Series:
    return df.set_index("pair_id")["is_correct"].astype(bool)


def paired_comparison(a: pd.Series, b: pd.Series, comparison: str) -> dict[str, object]:
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    arr_a = joined["a"].astype(bool).to_numpy()
    arr_b = joined["b"].astype(bool).to_numpy()
    delta = arr_b.astype(int) - arr_a.astype(int)
    rng = np.random.default_rng(20260707)
    boot = np.empty(10_000, dtype=float)
    for i in range(len(boot)):
        boot[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    a_only = int(np.sum(arr_a & ~arr_b))
    b_only = int(np.sum(~arr_a & arr_b))
    discordant = a_only + b_only
    return {
        "comparison": comparison,
        "n_pairs": int(len(joined)),
        "accuracy_a": float(arr_a.mean()),
        "accuracy_b": float(arr_b.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "mcnemar_a_only": a_only,
        "mcnemar_b_only": b_only,
        "mcnemar_p": float(binomtest(b_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def main() -> int:
    all_rows = []
    summaries = []
    raw_by_run: dict[str, pd.DataFrame] = {}
    missing = []
    for run_name, system, engine in RUNS:
        path = RAW / f"{run_name}.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        df = read_jsonl(path)
        if len(df) != 3000:
            raise SystemExit(f"Expected 3000 rows in {path}, got {len(df)}")
        df["condition"] = run_name
        df["system"] = system
        df["ocr_engine"] = engine
        raw_by_run[run_name] = df
        all_rows.append(df)
        summaries.append(summarize(df, run_name, system, engine))
    if missing:
        raise SystemExit("Missing raw outputs: " + ", ".join(missing))

    pair_df = pd.concat(all_rows, ignore_index=True)
    pair_df.to_csv(OUT_PAIRS, index=False)
    summary = pd.DataFrame(summaries)

    base_table = pd.read_csv(FINAL_TABLE)
    oracle_rf = float(base_table[base_table["system"].eq("Oracle source random_forest_regressor")]["effective_accuracy"].iloc[0])
    best_ocr_rf = float(
        base_table[
            base_table["system"].isin(
                [
                    "OCR(easyocr)+random_forest_regressor",
                    "OCR(easyocr_preprocessed)+random_forest_regressor",
                    "OCR(rapidocr)+random_forest_regressor",
                ]
            )
        ]["effective_accuracy"].max()
    )
    summary["delta_vs_oracle_rf"] = summary["effective_accuracy"] - oracle_rf
    summary["delta_vs_best_ocr_rf"] = summary["effective_accuracy"] - best_ocr_rf
    summary.to_csv(OUT_SUMMARY, index=False)

    comps = []
    classical = pd.read_csv(CLASSICAL_PAIRS)
    for engine in ["easyocr", "rapidocr", "easyocr_preprocessed"]:
        rf = classical[(classical["ocr_engine"].eq(engine)) & (classical["model"].eq("random_forest_regressor"))]
        llm_run = f"ocr_text_llm_{engine}"
        comps.append(
            paired_comparison(
                rf.set_index("pair_id")["is_correct"].astype(bool),
                correctness(raw_by_run[llm_run]),
                f"OCR({engine}) RF -> OCR({engine}) text-only LLM",
            )
        )
    qwen = pd.read_csv(DIRECT_VLM_PAIRS)
    qwen = qwen[qwen["model"].str.contains("Qwen", na=False)]
    comps.append(
        paired_comparison(
            correctness(qwen),
            correctness(raw_by_run["ocr_text_llm_rapidocr"]),
            "Qwen image-only clean -> OCR(RapidOCR) text-only LLM",
        )
    )
    comps.append(
        paired_comparison(
            correctness(raw_by_run["source_text_llm"]),
            correctness(raw_by_run["ocr_text_llm_rapidocr"]),
            "Source text LLM -> OCR(RapidOCR) text-only LLM",
        )
    )
    stats = pd.DataFrame(comps)
    stats.to_csv(OUT_STATS, index=False)

    md = [
        "# Clean OCR + Text-only LLM Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Model: `Qwen/Qwen2.5-Coder-7B-Instruct`",
        "",
        "Prompt/decoding: Prompt B, strict swap AB/BA, temperature 0.0, top_p 1.0, max_new_tokens 24, seed 42.",
        "",
        "## Metrics",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired Tests",
        "",
        stats.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        f"- `{OUT_SUMMARY}`",
        f"- `{OUT_PAIRS}`",
        f"- `{OUT_STATS}`",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT_SUMMARY), "stats": str(OUT_STATS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
