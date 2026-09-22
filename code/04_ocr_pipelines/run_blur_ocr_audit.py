#!/usr/bin/env python3
"""Two-engine OCR characterization of fixed blur levels on a stratified sample."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
CONDITIONS = ["baseline", "gaussian_sigma_1", "gaussian_sigma_2", "gaussian_sigma_4"]


def sample_snippets(snippets: pd.DataFrame, per_language: int = 10) -> pd.DataFrame:
    selected = []
    for language, frame in snippets.assign(source_chars=snippets.raw_code.fillna("").str.len()).groupby("language"):
        ordered = frame.sort_values(["source_chars", "rq0_id"]).reset_index(drop=True)
        indices = np.linspace(0, len(ordered) - 1, per_language).round().astype(int)
        part = ordered.iloc[indices].copy()
        part["length_sample_rank"] = np.arange(per_language)
        selected.append(part)
    result = pd.concat(selected, ignore_index=True)
    if result.duplicated("rq0_id").any() or result.groupby("language").size().nunique() != 1:
        raise ValueError("invalid deterministic OCR sample")
    return result


def sort_chunks(chunks: list[tuple[float, float, str]]) -> str:
    chunks.sort(key=lambda item: (round(item[0] / 10) * 10, item[1]))
    return "\n".join(text for _, _, text in chunks if text.strip())


def rapid_reader():
    from rapidocr_onnxruntime import RapidOCR
    reader = RapidOCR()

    def run(path: str) -> tuple[str, float]:
        raw, _ = reader(path)
        chunks = []
        for box, text, _confidence in raw or []:
            chunks.append((min(float(point[1]) for point in box), min(float(point[0]) for point in box), str(text)))
        return sort_chunks(chunks), float("nan")

    return run


def easy_reader():
    import easyocr
    import torch
    try:
        torch.set_num_threads(int(os.environ.get("EASYOCR_CPU_THREADS", "8")))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_dir = ROOT / "results/python_cuda_missing_experiments_20260716/ocr/easyocr_model"
    reader = easyocr.Reader(
        ["en"], gpu=False, verbose=False,
        model_storage_directory=str(model_dir), user_network_directory=str(model_dir), download_enabled=False,
    )

    def run(path: str) -> tuple[str, float]:
        raw = reader.readtext(path, detail=1, paragraph=False)
        chunks, confidence = [], []
        for box, text, conf in raw:
            chunks.append((min(float(point[1]) for point in box), min(float(point[0]) for point in box), str(text)))
            confidence.append(float(conf))
        return sort_chunks(chunks), float(np.mean(confidence)) if confidence else float("nan")

    return run


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def token_f1(reference: str, hypothesis: str) -> float:
    pattern = r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\sA-Za-z_0-9]"
    left, right = Counter(re.findall(pattern, reference)), Counter(re.findall(pattern, hypothesis))
    overlap = sum((left & right).values())
    precision = overlap / max(sum(right.values()), 1)
    recall = overlap / max(sum(left.values()), 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows.condition.eq("baseline")][["engine", "rq0_id", "ocr_text"]].rename(columns={"ocr_text": "baseline_ocr"})
    blur = rows[~rows.condition.eq("baseline")].merge(baseline, on=["engine", "rq0_id"], validate="many_to_one")
    blur["cer_vs_clean_ocr"] = blur.apply(
        lambda row: Levenshtein.distance(normalized(row.baseline_ocr), normalized(row.ocr_text)) / max(len(normalized(row.baseline_ocr)), 1), axis=1,
    )
    blur["token_f1_vs_clean_ocr"] = blur.apply(lambda row: token_f1(row.baseline_ocr, row.ocr_text), axis=1)
    blur.to_csv(OUT / "audit/blur_ocr_pairwise.csv", index=False)
    return blur.groupby(["engine", "language", "condition"]).agg(
        snippets=("rq0_id", "count"), ok_rate=("status", lambda values: float(values.eq("ok").mean())),
        cer_median=("cer_vs_clean_ocr", "median"), cer_mean=("cer_vs_clean_ocr", "mean"),
        token_f1_median=("token_f1_vs_clean_ocr", "median"), token_f1_mean=("token_f1_vs_clean_ocr", "mean"),
        runtime_sec_mean=("runtime_sec", "mean"),
    ).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="rapidocr,easyocr")
    parser.add_argument("--per-language", type=int, default=10)
    args = parser.parse_args()
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    snippets = pd.read_csv(OUT / "data/render_snippets.csv")
    sample = sample_snippets(snippets, args.per_language)
    sample.to_csv(audit_dir / "blur_ocr_sample.csv", index=False)
    metadata = pd.read_csv(OUT / "rendered/metadata/perturbation_render_metadata.csv")
    metadata = metadata[metadata.rq0_id.isin(sample.rq0_id) & metadata.condition.isin(CONDITIONS)]
    tasks = metadata.merge(sample[["rq0_id", "language", "length_sample_rank"]], on=["rq0_id", "language"], validate="many_to_one")
    output = audit_dir / "blur_ocr_raw.csv"
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []
    completed = set(zip(existing.engine, existing.rq0_id, existing.condition)) if len(existing) else set()
    constructors = {"rapidocr": rapid_reader, "easyocr": easy_reader}
    engines = [value.strip() for value in args.engines.split(",") if value.strip()]
    for engine in engines:
        run = constructors[engine]()
        for task in tasks.sort_values(["language", "length_sample_rank", "condition"]).itertuples(index=False):
            key = (engine, task.rq0_id, task.condition)
            if key in completed:
                continue
            start = time.time()
            try:
                text, confidence = run(str(ROOT / task.image_path))
                status, error = ("ok" if text.strip() else "empty"), ""
            except Exception as exc:
                text, confidence, status, error = "", float("nan"), "error", repr(exc)
            rows.append({
                "engine": engine, "rq0_id": task.rq0_id, "language": task.language,
                "length_sample_rank": task.length_sample_rank, "condition": task.condition,
                "image_path": task.image_path, "ocr_text": text, "status": status, "error": error,
                "mean_confidence": confidence, "runtime_sec": time.time() - start,
            })
            completed.add(key)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"{engine} {len(completed)}/{len(tasks) * len(engines)} {task.language} {task.rq0_id} {task.condition} {status}", flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    summary.to_csv(audit_dir / "blur_ocr_summary.csv", index=False)
    manifest = {
        "status": "complete" if len(frame) == len(tasks) * len(engines) else "incomplete",
        "sample_rule": f"{args.per_language} deterministic source-length quantiles per language",
        "snippets": sample.rq0_id.nunique(), "conditions": CONDITIONS, "engines": engines,
        "expected_rows": len(tasks) * len(engines), "rows": len(frame),
        "comparison": "each engine's blurred OCR against that engine's clean-baseline OCR",
    }
    (audit_dir / "BLUR_OCR_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if manifest["status"] == "complete" and frame.status.ne("error").all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
