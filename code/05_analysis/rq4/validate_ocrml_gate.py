#!/usr/bin/env python3
"""Recompute deterministic RapidOCR+ML pair decisions for the F5 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/ANON/experiment_root")
OUT = ROOT / "results/deploy_v2_logit_20260731"
PAIRS = OUT / "data/F5_FROZEN_PAIRS.csv"
EXPECTED = {"java": 0.5296666666666666, "python": 0.5953333333333334, "cuda": 0.6073333333333333}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(
    source: Path, language: str, model: str, pair_ids: set[str]
) -> pd.DataFrame:
    data = pd.read_csv(source)
    subset = data[
        data.ocr_engine.eq("rapidocr")
        & data.model.eq(model)
    ].copy()
    if "language" in subset:
        subset = subset[subset.language.eq(language)]
    subset = subset[subset.pair_id.isin(pair_ids)].copy()
    if len(subset) != 3000 or subset.pair_id.nunique() != 3000:
        raise RuntimeError(f"{language}: OCR+ML coverage drift ({len(subset)})")
    subset["recomputed_preference"] = np.where(
        subset.predicted_score_i.gt(subset.predicted_score_j),
        subset.snippet_i,
        np.where(
            subset.predicted_score_i.lt(subset.predicted_score_j),
            subset.snippet_j,
            None,
        ),
    )
    subset["recomputed_correct"] = subset.recomputed_preference.eq(subset.human_preference)
    subset["language"] = language
    subset["selected_model"] = model
    return subset[
        [
            "pair_id", "language", "snippet_i", "snippet_j", "human_preference",
            "predicted_score_i", "predicted_score_j", "recomputed_preference",
            "recomputed_correct", "selected_model", "ocr_engine",
        ]
    ]


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    java_source = ROOT / "results/screenshot_only_ocr_clean_20260707/ocr_classical_pair_predictions.csv"
    ext_source = ROOT / "results/python_cuda_missing_experiments_20260716/ocr/ocr_classical_pair_predictions.csv"
    specs = {
        "java": (java_source, "svr"),
        "python": (ext_source, "gradient_boosting_regressor"),
        "cuda": (ext_source, "linear_regression"),
    }
    frames = []
    summary = []
    for language, (source, model) in specs.items():
        pair_ids = set(pairs.loc[pairs.language.eq(language), "pair_id"])
        frame = recompute(source, language, model, pair_ids)
        accuracy = float(frame.recomputed_correct.mean())
        passed = np.isclose(accuracy, EXPECTED[language], atol=1e-15)
        summary.append(
            {
                "run_id": "deploy_v2_logit_20260731",
                "language": language,
                "ocr_engine": "rapidocr",
                "model": model,
                "n": len(frame),
                "correct": int(frame.recomputed_correct.sum()),
                "effective_accuracy": accuracy,
                "expected_accuracy": EXPECTED[language],
                "gate_pass": passed,
                "source_asset": str(source.relative_to(ROOT)),
                "source_asset_sha256": sha256(source),
                "recomputation": "preference=sign(predicted_score_i-predicted_score_j)",
            }
        )
        frames.append(frame)
    result = pd.DataFrame(summary)
    pair_output = pd.concat(frames, ignore_index=True)
    pair_output.to_csv(OUT / "data/F5_OCRML_PAIR_PREDICTIONS.csv", index=False)
    result.to_csv(OUT / "data/F5_OCRML_GATE.csv", index=False)
    gate = {
        "run_id": "deploy_v2_logit_20260731",
        "gate": "deterministic_rapidocr_ml",
        "gate_pass": bool(result.gate_pass.all()),
        "new_model_inference_calls": 0,
        "summary": result.to_dict(orient="records"),
        "pair_output_sha256": sha256(OUT / "data/F5_OCRML_PAIR_PREDICTIONS.csv"),
    }
    (OUT / "data/F5_OCRML_GATE.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))
    if not gate["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
