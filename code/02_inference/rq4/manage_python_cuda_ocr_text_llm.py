#!/usr/bin/env python3
"""Prepare and finalize Python/CUDA source/OCR text-only LLM strict-swap runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RUNS = {
    "source_text_llm": ("source", "Source text"),
    "ocr_text_llm_rapidocr": ("rapidocr", "RapidOCR text"),
    "ocr_text_llm_easyocr": ("easyocr", "EasyOCR text"),
    "ocr_text_llm_easyocr_preprocessed": ("easyocr_preprocessed", "EasyOCR preprocessed text"),
}


def summarize(group: pd.DataFrame) -> dict[str, object]:
    valid = group["is_valid_strict_swap"].astype(bool)
    correct = group["is_correct"].astype(bool)
    valid_group = group[valid]
    rho = np.nan
    if len(valid_group) >= 3:
        sign = np.where(valid_group["model_preference"].eq(valid_group["snippet_i"]), 1.0, -1.0)
        human = valid_group["human_score_i_z"].astype(float) - valid_group["human_score_j_z"].astype(float)
        if np.unique(sign).size > 1:
            rho = float(spearmanr(sign, human).statistic)
    return {
        "pairs": len(group),
        "valid_pairs": int(valid.sum()),
        "correct_pairs": int(correct.sum()),
        "valid_accuracy": float(correct.sum() / valid.sum()) if valid.sum() else np.nan,
        "effective_accuracy": float(correct.mean()),
        "strict_swap_error": float((~valid).mean()),
        "parse_failure_calls": int(group["parse_failures"].sum()),
        "spearman_valid_only": rho,
    }


def grouped(data: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(dimensions, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(dimensions, keys)), **summarize(group)})
    return pd.DataFrame(rows)


def prepare(root: Path, results: Path) -> None:
    snippets = pd.read_csv(root / "results/python_cuda_vlm_main_20260715/data/snippets.csv")
    ocr = pd.read_csv(results / "ocr/ocr_text_by_snippet_all_engines.csv")
    inputs = results / "ocr_text_llm_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    snippets.to_csv(inputs / "source_text.csv", index=False)
    for engine in ("rapidocr", "easyocr", "easyocr_preprocessed"):
        text = ocr[ocr["ocr_engine"].eq(engine)][["snippet_id", "ocr_text"]].rename(columns={"snippet_id": "rq0_id"})
        merged = snippets.merge(text, on="rq0_id", validate="one_to_one")
        if len(merged) != len(snippets) or merged["ocr_text"].isna().any():
            raise RuntimeError(f"Incomplete OCR text for {engine}")
        merged["raw_code"] = merged["ocr_text"].fillna("").astype(str)
        merged.drop(columns=["ocr_text"]).to_csv(inputs / f"{engine}_text.csv", index=False)
    print(json.dumps({"status": "prepared", "datasets": 4, "snippets_per_dataset": len(snippets)}))


def finalize(root: Path, results: Path) -> None:
    raw_dir = results / "ocr_text_llm_raw"
    frames = []
    integrity = []
    for run_name, (condition, label) in RUNS.items():
        path = raw_dir / f"{run_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_json(path, lines=True)
        frame["condition"] = condition
        frame["condition_label"] = label
        integrity.append(
            {
                "run_name": run_name,
                "rows": len(frame),
                "unique_pairs": frame["pair_id"].nunique(),
                "duplicate_pairs": int(frame["pair_id"].duplicated().sum()),
            }
        )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != 24000 or any(row["rows"] != 6000 or row["unique_pairs"] != 6000 for row in integrity):
        raise RuntimeError(f"OCR text LLM integrity failure: {integrity}")
    data.to_csv(results / "ocr_text_llm_pair_predictions.csv", index=False)
    tables = results / "tables"
    tables.mkdir(exist_ok=True)
    for suffix, dimensions in {
        "overall": ["condition", "condition_label"],
        "by_language": ["condition", "condition_label", "language"],
        "by_difficulty": ["condition", "condition_label", "difficulty"],
        "by_language_difficulty": ["condition", "condition_label", "language", "difficulty"],
    }.items():
        grouped(data, dimensions).to_csv(tables / f"ocr_text_llm_results_{suffix}.csv", index=False)
    pd.DataFrame(integrity).to_csv(tables / "ocr_text_llm_integrity.csv", index=False)
    manifest = {
        "status": "passed",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "conditions": 4,
        "pairs_per_condition": 6000,
        "directional_calls": 48000,
        "prompt": "B",
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 24,
    }
    (results / "ocr_text_llm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "finalize"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results", default="results/python_cuda_missing_experiments_20260716")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    results = (root / args.results).resolve()
    (prepare if args.command == "prepare" else finalize)(root, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
