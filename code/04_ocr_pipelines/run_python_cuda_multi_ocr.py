#!/usr/bin/env python3
"""Run clean multi-engine OCR and source-trained/OCR-tested baselines for Python/CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
OCR_IMPL_PATH = ROOT / "experiments/rq0_viability/scripts/run_clean_multi_ocr_experiment.py"


def load_ocr_impl():
    spec = importlib.util.spec_from_file_location("clean_multi_ocr", OCR_IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OCR_IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OCR = load_ocr_impl()


def oof_scores(
    snippets: pd.DataFrame,
    source_features: pd.DataFrame,
    ocr_features: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> pd.DataFrame:
    source_features = source_features.rename(columns={"snippet_uid": "rq0_id"})
    feature_cols = [column for column in source_features.columns if column.startswith("feature_")]
    source = snippets.merge(source_features, on=["rq0_id", "dataset_name", "language"], validate="one_to_one")
    rows = []
    for (engine, language), engine_features in ocr_features.groupby(["ocr_engine", "language"], sort=True):
        ocr = snippets[["rq0_id", "dataset_name", "language", "human_score_z_within_dataset"]].merge(
            engine_features.rename(columns={"snippet_id": "rq0_id"}),
            on=["rq0_id", "dataset_name", "language"],
            validate="one_to_one",
        )
        folds = fold_assignments[fold_assignments["language"].eq(language)].set_index("snippet_uid")["fold"]
        if set(folds.index) != set(ocr["rq0_id"]):
            raise RuntimeError(f"Fold assignment mismatch for {language}")
        for fold in sorted(folds.unique()):
            test_ids = set(folds[folds.eq(fold)].index)
            train = source[source["language"].eq(language) & ~source["rq0_id"].isin(test_ids)]
            test = ocr[ocr["rq0_id"].isin(test_ids)]
            for model_name, model in OCR.regression_models(42).items():
                model.fit(train[feature_cols], train["human_score_z_within_dataset"].astype(float))
                predictions = model.predict(test[feature_cols])
                for rq0_id, prediction in zip(test["rq0_id"], predictions):
                    rows.append(
                        {
                            "rq0_id": rq0_id,
                            "language": language,
                            "fold": int(fold),
                            "ocr_engine": engine,
                            "model": model_name,
                            "predicted_score": float(prediction),
                        }
                    )
    result = pd.DataFrame(rows)
    expected = ocr_features["ocr_engine"].nunique() * 4 * len(snippets)
    if len(result) != expected or result.duplicated(["rq0_id", "ocr_engine", "model"]).any():
        raise RuntimeError(f"OCR OOF output integrity failure: expected {expected}, got {len(result)}")
    return result


def pair_scores(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (engine, model), group in scores.groupby(["ocr_engine", "model"], sort=True):
        score = group.set_index("rq0_id")["predicted_score"]
        work = pairs.copy()
        work["ocr_engine"] = engine
        work["model"] = model
        work["predicted_score_i"] = work["snippet_i"].map(score)
        work["predicted_score_j"] = work["snippet_j"].map(score)
        work["score_diff"] = work["predicted_score_i"] - work["predicted_score_j"]
        work["model_preference"] = np.where(
            work["score_diff"].gt(0), work["snippet_i"],
            np.where(work["score_diff"].lt(0), work["snippet_j"], None),
        )
        work["is_valid_score_pair"] = work["model_preference"].notna()
        work["is_correct"] = work["model_preference"].eq(work["human_preference"])
        frames.append(work)
    return pd.concat(frames, ignore_index=True)


def summarize(group: pd.DataFrame) -> dict[str, object]:
    valid = group["is_valid_score_pair"].astype(bool)
    correct = group["is_correct"].astype(bool)
    human_diff = group["human_score_i_z"].astype(float) - group["human_score_j_z"].astype(float)
    rho = np.nan
    if group["score_diff"].nunique() > 1:
        rho = float(spearmanr(group["score_diff"].astype(float), human_diff).statistic)
    return {
        "pairs": int(len(group)),
        "valid_pairs": int(valid.sum()),
        "tie_pairs": int((~valid).sum()),
        "pair_accuracy": float(correct.mean()),
        "spearman_continuous_all": rho,
    }


def grouped(data: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(dimensions, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(dimensions, keys)), **summarize(group)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/python_cuda_missing_experiments_20260716/ocr")
    parser.add_argument("--engines", default="rapidocr,easyocr,easyocr_preprocessed")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--easyocr-gpu", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_results = ROOT / "results/python_cuda_vlm_main_20260715"
    extension = ROOT / "results/dataset_extension_20260713/data"
    snippets = pd.read_csv(main_results / "data/snippets.csv")
    pairs = pd.read_csv(main_results / "data/pairs_seed42_clean.csv")
    render_meta = pd.read_csv(main_results / "render_metadata/default_render_metadata.csv")
    source_features = pd.read_csv(extension / "source_features_27.csv")
    fold_assignments = pd.read_csv(extension / "oof_predictions.csv")
    fold_assignments = fold_assignments[fold_assignments["model"].eq("linear_regression")]

    engines = [value.strip() for value in args.engines.split(",") if value.strip()]
    frames = []
    for engine in engines:
        if engine == "rapidocr":
            frame = OCR.run_rapidocr_engine(snippets, render_meta, output, args.force_ocr)
        elif engine in {"easyocr", "easyocr_preprocessed"}:
            frame = OCR.run_easyocr_engine(snippets, render_meta, output, engine, args.force_ocr, args.easyocr_gpu)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
        frames.append(frame.merge(snippets[["rq0_id", "language"]], left_on="snippet_id", right_on="rq0_id", validate="one_to_one"))
    ocr_all = pd.concat(frames, ignore_index=True).drop(columns=["rq0_id"])
    ocr_all.to_csv(output / "ocr_text_by_snippet_all_engines.csv", index=False)

    quality, features = OCR.compute_quality_and_features(ocr_all, output)
    language_map = snippets.set_index("rq0_id")["language"]
    quality["language"] = quality["snippet_id"].map(language_map)
    quality.groupby(["ocr_engine", "language"]).agg(
        snippets=("snippet_id", "count"),
        ok_rate=("parse_status", lambda values: float(values.eq("ok").mean())),
        cer=("char_error_rate", "mean"),
        token_f1=("token_f1", "mean"),
        line_count_abs_diff=("line_count_abs_diff", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    ).reset_index().to_csv(output / "ocr_quality_by_language.csv", index=False)
    features["language"] = features["snippet_id"].map(language_map)

    scores = oof_scores(snippets, source_features, features, fold_assignments)
    scores.to_csv(output / "ocr_classical_snippet_predictions.csv", index=False)
    predictions = pair_scores(pairs, scores)
    predictions.to_csv(output / "ocr_classical_pair_predictions.csv", index=False)
    for suffix, dimensions in {
        "overall": ["ocr_engine", "model"],
        "by_language": ["ocr_engine", "model", "language"],
        "by_difficulty": ["ocr_engine", "model", "difficulty"],
        "by_language_difficulty": ["ocr_engine", "model", "language", "difficulty"],
    }.items():
        grouped(predictions, dimensions).to_csv(output / f"ocr_classical_results_{suffix}.csv", index=False)

    manifest = {
        "status": "passed",
        "snippets": len(snippets),
        "pairs": len(pairs),
        "engines": engines,
        "ocr_rows": len(ocr_all),
        "oof_score_rows": len(scores),
        "pair_prediction_rows": len(predictions),
        "folds": 5,
        "seed": 42,
        "cer_definition": "Levenshtein character edit distance / max(reference length, hypothesis length, 1)",
        "cer_backend": "rapidfuzz 3.14.5",
        "training_protocol": "within-language source-feature train, OCR-feature held-out test, same grouped OOF folds as main classical experiment",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
