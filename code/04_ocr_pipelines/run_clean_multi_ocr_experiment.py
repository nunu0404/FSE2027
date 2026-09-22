#!/usr/bin/env python3
"""Clean rerun of screenshot/OCR baselines on corrected rendered images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
DEFAULT_OUT = ROOT / "results/screenshot_only_ocr_clean_20260707"
DIFFICULTIES = ["easy", "medium", "hard"]
CLASSICAL_MODELS = ["random_forest_regressor", "gradient_boosting_regressor", "svr", "linear_regression"]


def load_extract_module():
    path = EXP / "scripts/extract_features.py"
    spec = importlib.util.spec_from_file_location("rq0_extract_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FEATURES = load_extract_module()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def pct(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100 * float(x):.2f}%"


def regression_models(seed: int) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2)),
            ]
        ),
        "gradient_boosting_regressor": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(random_state=seed))]
        ),
        "svr": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.1))]
        ),
    }


def levenshtein_ratio(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    denom = max(len(a), len(b), 1)
    try:
        from rapidfuzz.distance import Levenshtein

        return float(Levenshtein.distance(a, b) / denom)
    except ImportError:
        pass
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return float(prev[-1] / denom)


def token_f1(source: str, ocr: str) -> tuple[float, float, float]:
    src = FEATURES.TOKEN_RE.findall(source or "")
    hyp = FEATURES.TOKEN_RE.findall(ocr or "")
    if not src and not hyp:
        return 1.0, 1.0, 1.0
    cs, ch = Counter(src), Counter(hyp)
    overlap = sum((cs & ch).values())
    precision = overlap / len(hyp) if hyp else 0.0
    recall = overlap / len(src) if src else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def indent_depths(text: str) -> list[int]:
    out = []
    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        tabs = len(line) - len(line.lstrip("\t"))
        out.append(leading + 4 * tabs)
    return out


def indentation_distribution_delta(source: str, ocr: str) -> float:
    a, b = indent_depths(source), indent_depths(ocr)
    if not a and not b:
        return 0.0
    return abs((float(np.mean(a)) if a else 0.0) - (float(np.mean(b)) if b else 0.0))


def preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    if float(np.asarray(img).mean()) < 128:
        img = ImageOps.invert(img)
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    return np.asarray(img.convert("RGB"))


def sort_ocr_chunks(chunks: list[tuple[float, float, str]]) -> str:
    chunks.sort(key=lambda x: (round(x[0] / 10) * 10, x[1]))
    return "\n".join(text for _, _, text in chunks if str(text).strip())


def run_easyocr_engine(dataset: pd.DataFrame, render_meta: pd.DataFrame, out: Path, engine: str, force: bool, gpu: bool) -> pd.DataFrame:
    out_dir = out / "ocr_outputs" / engine
    out_path = out_dir / "snippet_ocr.csv"
    existing = pd.read_csv(out_path) if out_path.exists() and not force else pd.DataFrame()
    if len(existing) == len(dataset):
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import easyocr
        if not gpu:
            import torch

            try:
                torch.set_num_threads(int(os.environ.get("EASYOCR_CPU_THREADS", "8")))
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits inter-op configuration only before parallel work starts.
                # The first EasyOCR condition has already configured it for this process.
                pass
        reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=os.environ.get("EASYOCR_MODEL_DIR", str(out / "easyocr_model")),
            user_network_directory=os.environ.get("EASYOCR_MODEL_DIR", str(out / "easyocr_model")),
        )
        import_status = "ok"
        import_error = ""
    except Exception as exc:
        reader = None
        import_status = "import_failed"
        import_error = repr(exc)

    image_map = render_meta.set_index("rq0_id")["image_path"].to_dict()
    rows = existing.to_dict("records") if len(existing) else []
    completed_ids = set(existing["snippet_id"].astype(str)) if len(existing) else set()
    timeout_ids = {value for value in os.environ.get("EASYOCR_TIMEOUT_IDS", "").split(",") if value}
    for idx, row in enumerate(dataset.itertuples(index=False), 1):
        if str(row.rq0_id) in completed_ids:
            continue
        image_path = str(image_map.get(row.rq0_id, ""))
        start = time.time()
        print(f"[{engine}] starting {row.rq0_id} ({len(rows) + 1}/{len(dataset)})", flush=True)
        try:
            if str(row.rq0_id) in timeout_ids:
                raise TimeoutError("Skipped after a prior EasyOCR call exceeded 120 seconds")
            if reader is None:
                raise RuntimeError(import_error)
            image_input = preprocess_image(image_path) if engine == "easyocr_preprocessed" else image_path
            raw = reader.readtext(image_input, detail=1, paragraph=False)
            chunks = []
            confs = []
            for box, text, conf in raw:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                chunks.append((min(ys), min(xs), str(text)))
                confs.append(float(conf))
            ocr_text = sort_ocr_chunks(chunks)
            status = "ok" if ocr_text.strip() else "empty"
            error = ""
            mean_conf = float(np.mean(confs)) if confs else np.nan
        except TimeoutError as exc:
            ocr_text = ""
            status = "timeout"
            error = repr(exc)
            mean_conf = np.nan
        except Exception as exc:
            ocr_text = ""
            status = import_status if import_status != "ok" else "error"
            error = import_error or repr(exc)
            mean_conf = np.nan
        rows.append(
            {
                "snippet_id": row.rq0_id,
                "dataset_name": row.dataset_name,
                "image_path": image_path,
                "original_source_text": row.raw_code,
                "ocr_text": ocr_text,
                "ocr_engine": engine,
                "parse_status": status,
                "error_status": error,
                "runtime_sec": time.time() - start,
                "mean_confidence": mean_conf,
            }
        )
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[{engine}] processed {len(rows)}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


def run_rapidocr_engine(dataset: pd.DataFrame, render_meta: pd.DataFrame, out: Path, force: bool) -> pd.DataFrame:
    engine = "rapidocr"
    out_dir = out / "ocr_outputs" / engine
    out_path = out_dir / "snippet_ocr.csv"
    existing = pd.read_csv(out_path) if out_path.exists() and not force else pd.DataFrame()
    if len(existing) == len(dataset):
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from rapidocr_onnxruntime import RapidOCR
        reader = RapidOCR()
        import_status = "ok"
        import_error = ""
    except Exception as exc:
        reader = None
        import_status = "import_failed"
        import_error = repr(exc)

    image_map = render_meta.set_index("rq0_id")["image_path"].to_dict()
    rows = existing.to_dict("records") if len(existing) else []
    completed_ids = set(existing["snippet_id"].astype(str)) if len(existing) else set()
    for idx, row in enumerate(dataset.itertuples(index=False), 1):
        if str(row.rq0_id) in completed_ids:
            continue
        image_path = str(image_map.get(row.rq0_id, ""))
        start = time.time()
        try:
            if reader is None:
                raise RuntimeError(import_error)
            raw, _elapsed = reader(image_path)
            raw = raw or []
            chunks = []
            confs = []
            for item in raw:
                box, text, conf = item[0], item[1], item[2]
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                chunks.append((min(ys), min(xs), str(text)))
                confs.append(float(conf))
            ocr_text = sort_ocr_chunks(chunks)
            status = "ok" if ocr_text.strip() else "empty"
            error = ""
            mean_conf = float(np.mean(confs)) if confs else np.nan
        except Exception as exc:
            ocr_text = ""
            status = import_status if import_status != "ok" else "error"
            error = import_error or repr(exc)
            mean_conf = np.nan
        rows.append(
            {
                "snippet_id": row.rq0_id,
                "dataset_name": row.dataset_name,
                "image_path": image_path,
                "original_source_text": row.raw_code,
                "ocr_text": ocr_text,
                "ocr_engine": engine,
                "parse_status": status,
                "error_status": error,
                "runtime_sec": time.time() - start,
                "mean_confidence": mean_conf,
            }
        )
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[{engine}] processed {len(rows)}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


def compute_quality_and_features(ocr_all: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_rows = []
    feature_rows = []
    for row in ocr_all.itertuples(index=False):
        src = str(row.original_source_text or "")
        hyp = str(row.ocr_text or "")
        precision, recall, f1 = token_f1(src, hyp)
        quality = {
            "snippet_id": row.snippet_id,
            "dataset_name": row.dataset_name,
            "ocr_engine": row.ocr_engine,
            "parse_status": row.parse_status,
            "char_error_rate": levenshtein_ratio(src, hyp),
            "original_num_lines": len(src.replace("\r\n", "\n").replace("\r", "\n").split("\n")),
            "ocr_num_lines": len(hyp.replace("\r\n", "\n").replace("\r", "\n").split("\n")) if hyp else 0,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "indent_mean_abs_delta": indentation_distribution_delta(src, hyp),
            "runtime_sec": getattr(row, "runtime_sec", np.nan),
            "mean_confidence": getattr(row, "mean_confidence", np.nan),
        }
        quality["line_count_abs_diff"] = abs(quality["original_num_lines"] - quality["ocr_num_lines"])
        quality_rows.append(quality)
        feat = {"snippet_id": row.snippet_id, "dataset_name": row.dataset_name, "ocr_engine": row.ocr_engine}
        feat.update(FEATURES.extract_one(hyp))
        feature_rows.append(feat)

    quality = pd.DataFrame(quality_rows)
    features = pd.DataFrame(feature_rows)
    quality.to_csv(out / "ocr_quality_by_snippet.csv", index=False)
    features.to_csv(out / "ocr_features_all.csv", index=False)
    summary = quality.groupby(["ocr_engine", "dataset_name"], dropna=False)[
        ["char_error_rate", "token_f1", "line_count_abs_diff", "indent_mean_abs_delta", "runtime_sec", "mean_confidence"]
    ].agg(["mean", "median", "std", "count"])
    summary.to_csv(out / "ocr_quality_summary.csv")
    total = quality.groupby("ocr_engine").agg(
        n=("snippet_id", "count"),
        ok_rate=("parse_status", lambda x: float((x == "ok").mean())),
        char_error_rate=("char_error_rate", "mean"),
        token_f1=("token_f1", "mean"),
        line_count_abs_diff=("line_count_abs_diff", "mean"),
        indent_mean_abs_delta=("indent_mean_abs_delta", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        mean_confidence=("mean_confidence", "mean"),
    ).reset_index()
    total.to_csv(out / "ocr_quality_overall.csv", index=False)
    write_text(out / "ocr_quality_summary.md", "# Clean OCR Quality Summary\n\n" + total.to_markdown(index=False, floatfmt=".4f") + "\n")
    return quality, features


def oof_ocr_scores(dataset: pd.DataFrame, source_features: pd.DataFrame, ocr_features: pd.DataFrame, out: Path) -> pd.DataFrame:
    splits = pd.read_csv(EXP / "data/splits/split_assignments.csv")
    splits = splits[(splits["split_type"] == "pooled_stratified_5fold") & (splits["seed"] == 42)].copy()
    src = dataset.merge(source_features, on=["rq0_id", "dataset_name"], how="inner")
    feature_cols = [c for c in source_features.columns if c.startswith("feature_")]
    rows = []
    for engine, engine_features in ocr_features.groupby("ocr_engine", sort=True):
        ocr = dataset[["rq0_id", "dataset_name", "human_score_z_within_dataset"]].merge(
            engine_features.rename(columns={"snippet_id": "rq0_id"}),
            on=["rq0_id", "dataset_name"],
            how="inner",
        )
        for split_id in sorted(splits["split_id"].unique()):
            split = splits[splits["split_id"] == split_id]
            train_ids = set(split[split["role"] == "train"]["rq0_id"])
            test_ids = set(split[split["role"] == "test"]["rq0_id"])
            train = src[src["rq0_id"].isin(train_ids)].copy()
            test = ocr[ocr["rq0_id"].isin(test_ids)].copy()
            seed = int(split["seed"].iloc[0])
            fold = int(split["fold"].iloc[0])
            x_train = train[feature_cols]
            y_train = train["human_score_z_within_dataset"].astype(float)
            x_test = test[feature_cols]
            for model_name, model in regression_models(seed).items():
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), dtype=float)
                for rq0_id, score in zip(test["rq0_id"], pred):
                    rows.append(
                        {
                            "split_id": split_id,
                            "seed": seed,
                            "fold": fold,
                            "model": model_name,
                            "rq0_id": rq0_id,
                            "predicted_score": score,
                            "input_modality": "ocr_text",
                            "ocr_engine": engine,
                            "training_setting": "source_train_ocr_test_oof",
                        }
                    )
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "ocr_classical_snippet_predictions.csv", index=False)
    return scores


def pairwise_from_scores(pairs: pd.DataFrame, scores: pd.DataFrame, out: Path) -> pd.DataFrame:
    frames = []
    for (engine, model_name), group in scores.groupby(["ocr_engine", "model"], sort=True):
        score_map = group.set_index("rq0_id")["predicted_score"]
        work = pairs.copy()
        work["predicted_score_i"] = work["snippet_i"].map(score_map)
        work["predicted_score_j"] = work["snippet_j"].map(score_map)
        work["model_preference"] = np.where(
            work["predicted_score_i"] > work["predicted_score_j"],
            work["snippet_i"],
            np.where(work["predicted_score_j"] > work["predicted_score_i"], work["snippet_j"], ""),
        )
        work["is_valid_score_pair"] = work["model_preference"] != ""
        work["is_correct"] = work["model_preference"] == work["human_preference"]
        work["model"] = model_name
        work["model_group"] = "ocr_classical_feature"
        work["input_modality"] = "ocr_text"
        work["ocr_engine"] = engine
        frames.append(work)
    pairwise = pd.concat(frames, ignore_index=True)
    pairwise.to_csv(out / "ocr_classical_pair_predictions.csv", index=False)
    return pairwise


def load_oracle_pairs() -> pd.DataFrame:
    path = EXP / "outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv"
    return pd.read_csv(path)


def load_clean_direct_vlm(pairs: pd.DataFrame, out: Path) -> pd.DataFrame:
    paths = [
        EXP / "outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__image_only__promptB__seed42.jsonl",
        EXP / "outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
    ]
    frames = []
    pair_ids = set(pairs["pair_id"])
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_json(path, lines=True)
        frame = frame[frame["pair_id"].isin(pair_ids)].copy()
        frame["system"] = frame["model_name"] + " / " + frame["input_setting"] + " / clean"
        frames.append(frame)
    direct = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    direct.to_csv(out / "clean_direct_vlm_pair_predictions.csv", index=False)
    return direct


def summarize_pair_predictions(frame: pd.DataFrame, name: str, kind: str, model: str, notes: str = "") -> dict[str, object]:
    n = len(frame)
    valid_col = "is_valid_strict_swap" if "is_valid_strict_swap" in frame.columns else "is_valid_score_pair"
    valid = frame[frame[valid_col].astype(bool)] if valid_col in frame.columns else frame
    correct = int(valid["is_correct"].sum()) if "is_correct" in valid.columns else 0
    row: dict[str, object] = {
        "system": name,
        "kind": kind,
        "model": model,
        "num_pairs": int(n),
        "valid_pairs": int(len(valid)),
        "correct_pairs": correct,
        "effective_accuracy": float(correct / n) if n else np.nan,
        "valid_accuracy": float(correct / len(valid)) if len(valid) else np.nan,
        "strict_swap_error": float(1 - len(valid) / n) if valid_col == "is_valid_strict_swap" and n else np.nan,
        "parse_failure_rate": float(frame["parse_failures"].astype(float).sum() / (2 * n)) if "parse_failures" in frame.columns and n else np.nan,
        "notes": notes,
    }
    for diff in DIFFICULTIES:
        sub_all = frame[frame["difficulty"] == diff]
        sub_valid = sub_all[sub_all[valid_col].astype(bool)] if valid_col in sub_all.columns else sub_all
        c = int(sub_valid["is_correct"].sum()) if "is_correct" in sub_valid.columns else 0
        row[f"{diff}_effective_accuracy"] = float(c / len(sub_all)) if len(sub_all) else np.nan
        row[f"{diff}_valid_accuracy"] = float(c / len(sub_valid)) if len(sub_valid) else np.nan
        row[f"{diff}_n"] = int(len(sub_all))
    return row


def build_final_table(oracle: pd.DataFrame, ocr_pairs: pd.DataFrame, direct: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for model, group in oracle.groupby("model", sort=True):
        if model not in CLASSICAL_MODELS:
            continue
        rows.append(
            summarize_pair_predictions(
                group,
                f"Oracle source {model}",
                "oracle_source_classical",
                model,
                "Upper bound for screenshot-only deployment; uses raw source features.",
            ) | {
                "input_available_at_deployment": "source text",
                "uses_raw_source": "yes",
                "ocr_engine": "",
            }
        )
    for (engine, model), group in ocr_pairs.groupby(["ocr_engine", "model"], sort=True):
        rows.append(
            summarize_pair_predictions(
                group,
                f"OCR({engine})+{model}",
                "ocr_classical",
                model,
                "Train on source features; predict held-out snippet scores from OCR text features.",
            ) | {
                "input_available_at_deployment": "screenshot image -> OCR text",
                "uses_raw_source": "no for prediction",
                "ocr_engine": engine,
            }
        )
    if len(direct):
        for system, group in direct.groupby("system", sort=True):
            rows.append(
                summarize_pair_predictions(
                    group,
                    f"Direct VLM {system}",
                    "direct_vlm",
                    system,
                    "Clean full-factorial image-only strict-swap output reused.",
                ) | {
                    "input_available_at_deployment": "screenshot image",
                    "uses_raw_source": "no",
                    "ocr_engine": "",
                }
            )
    table = pd.DataFrame(rows)
    oracle_rf = table.loc[table["system"] == "Oracle source random_forest_regressor", "effective_accuracy"]
    table["delta_vs_oracle_rf"] = table["effective_accuracy"] - float(oracle_rf.iloc[0]) if len(oracle_rf) else np.nan
    ocr_rf = table[(table["kind"] == "ocr_classical") & (table["model"] == "random_forest_regressor")].copy()
    best_ocr_rf = float(ocr_rf["effective_accuracy"].max()) if len(ocr_rf) else np.nan
    table["delta_vs_best_ocr_rf"] = table["effective_accuracy"] - best_ocr_rf if not math.isnan(best_ocr_rf) else np.nan
    table.to_csv(out / "clean_multi_ocr_final_comparison_table.csv", index=False)
    write_text(out / "clean_multi_ocr_final_comparison_table.md", table.to_markdown(index=False, floatfmt=".4f") + "\n")
    return table


def paired_stats(a: pd.Series, b: pd.Series, label: str) -> dict[str, object]:
    joined = pd.DataFrame({"a": a.astype(bool), "b": b.astype(bool)}).dropna()
    arr_a = joined["a"].to_numpy(dtype=bool)
    arr_b = joined["b"].to_numpy(dtype=bool)
    delta = arr_b.astype(int) - arr_a.astype(int)
    rng = np.random.default_rng(20260707)
    boot = np.empty(5000)
    for i in range(len(boot)):
        idx = rng.integers(0, len(delta), size=len(delta))
        boot[i] = float(delta[idx].mean())
    a_only = int((arr_a & ~arr_b).sum())
    b_only = int((~arr_a & arr_b).sum())
    discordant = a_only + b_only
    return {
        "comparison": label,
        "n_pairs": int(len(joined)),
        "accuracy_a": float(arr_a.mean()),
        "accuracy_b": float(arr_b.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "mcnemar_a_only": a_only,
        "mcnemar_b_only": b_only,
        "mcnemar_p": float(binomtest(b_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def write_stats(oracle: pd.DataFrame, ocr_pairs: pd.DataFrame, direct: pd.DataFrame, out: Path) -> None:
    rows = []
    oracle_rf = oracle[oracle["model"] == "random_forest_regressor"].set_index("pair_id")["is_correct"].astype(bool)
    best_ocr = None
    best_acc = -1.0
    for engine, group in ocr_pairs[ocr_pairs["model"] == "random_forest_regressor"].groupby("ocr_engine", sort=True):
        corr = group.set_index("pair_id")["is_correct"].astype(bool)
        rows.append(paired_stats(oracle_rf, corr, f"Oracle RF -> OCR({engine}) RF"))
        acc = float(corr.mean())
        if acc > best_acc:
            best_acc = acc
            best_ocr = (engine, corr)
    if best_ocr is not None and len(direct):
        engine, best_corr = best_ocr
        for system, group in direct.groupby("system", sort=True):
            direct_corr = group.set_index("pair_id")["is_correct"].astype(bool)
            rows.append(paired_stats(best_corr, direct_corr, f"OCR({engine}) RF -> Direct VLM {system}"))
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "clean_multi_ocr_statistical_tests.csv", index=False)
    write_text(out / "clean_multi_ocr_statistical_tests.md", "# Clean Multi-OCR Statistical Tests\n\n" + stats.to_markdown(index=False, floatfmt=".4f") + "\n")


def write_report(table: pd.DataFrame, quality: pd.DataFrame, out: Path, engines: list[str], text_llm_status: str) -> None:
    q = quality.groupby("ocr_engine").agg(
        ok_rate=("parse_status", lambda x: float((x == "ok").mean())),
        token_f1=("token_f1", "mean"),
        char_error_rate=("char_error_rate", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    ).reset_index()
    compact = table[["system", "kind", "model", "ocr_engine", "effective_accuracy", "valid_accuracy", "strict_swap_error", "delta_vs_oracle_rf", "delta_vs_best_ocr_rf"]].copy()
    text = f"""# Clean Multi-OCR Screenshot Experiment

Generated at: {datetime.now(timezone.utc).isoformat()}

## Scope

- Corrected render metadata: `{EXP / 'outputs/render_metadata/default_render_metadata.csv'}`
- Pair set: `{EXP / 'data/pairs/full_pair_set_rq0.csv'}`
- OCR engines attempted: {', '.join(engines)}
- Direct VLM baseline: clean full-factorial image-only outputs from `full_factorial_clean_20260703`
- Text-only OCR LLM status: {text_llm_status}

## OCR Quality

{q.to_markdown(index=False, floatfmt=".4f")}

## Final Comparison

{compact.to_markdown(index=False, floatfmt=".4f")}

## Interpretation Guardrails

- OCR rows are screenshot-deployable only through OCR text and do not use raw source at prediction time.
- Oracle source rows are upper bounds for the same pair set, not screenshot-only systems.
- Direct VLM rows reuse clean rerendered image-only strict-swap outputs; no stale June VLM rows are used.
- OCR text LLM is not included unless a local text-only checkpoint is available and rerun on the newly extracted clean OCR text.
"""
    write_text(out / "CLEAN_MULTI_OCR_EXPERIMENT_REPORT.md", text)


def create_mapping(dataset: pd.DataFrame, pairs: pd.DataFrame, render_meta: pd.DataFrame, out: Path) -> None:
    by_img = render_meta.set_index("rq0_id")["image_path"]
    by_src = dataset.set_index("rq0_id")["raw_code"]
    work = pairs.copy()
    work["image_i_path"] = work["snippet_i"].map(by_img)
    work["image_j_path"] = work["snippet_j"].map(by_img)
    work["source_i_text"] = work["snippet_i"].map(by_src)
    work["source_j_text"] = work["snippet_j"].map(by_src)
    work.to_csv(out / "pair_image_mapping.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--engines", default="easyocr,rapidocr,easyocr_preprocessed")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--easyocr-gpu", action="store_true")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(EXP / "data/processed/pooled_313_processed.csv")
    pairs = pd.read_csv(EXP / "data/pairs/full_pair_set_rq0.csv")
    render_meta = pd.read_csv(EXP / "outputs/render_metadata/default_render_metadata.csv")
    source_features = pd.read_csv(EXP / "data/processed/features_313.csv")
    engines = [x.strip() for x in args.engines.split(",") if x.strip()]

    missing_images = [p for p in render_meta["image_path"].astype(str) if not Path(p).exists()]
    if missing_images:
        raise SystemExit(f"Missing rendered images: {missing_images[:5]}")

    create_mapping(dataset, pairs, render_meta, out)
    frames = []
    for engine in engines:
        if engine in {"easyocr", "easyocr_preprocessed"}:
            frames.append(run_easyocr_engine(dataset, render_meta, out, engine, args.force_ocr, args.easyocr_gpu))
        elif engine == "rapidocr":
            frames.append(run_rapidocr_engine(dataset, render_meta, out, args.force_ocr))
        else:
            raise SystemExit(f"Unknown OCR engine: {engine}")
    ocr_all = pd.concat(frames, ignore_index=True)
    ocr_all.to_csv(out / "clean_ocr_text_by_snippet_all_engines.csv", index=False)

    quality, ocr_features = compute_quality_and_features(ocr_all, out)
    scores = oof_ocr_scores(dataset, source_features, ocr_features, out)
    ocr_pairs = pairwise_from_scores(pairs, scores, out)
    oracle = load_oracle_pairs()
    direct = load_clean_direct_vlm(pairs, out)
    table = build_final_table(oracle, ocr_pairs, direct, out)
    write_stats(oracle, ocr_pairs, direct, out)

    text_llm_status = "not_run: no local Qwen/Qwen2.5-Coder-7B-Instruct checkpoint is currently present in Hugging Face cache"
    write_report(table, quality, out, engines, text_llm_status)
    write_json(
        out / "run_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "output_dir": str(out),
            "engines": engines,
            "dataset": str(EXP / "data/processed/pooled_313_processed.csv"),
            "pairs": str(EXP / "data/pairs/full_pair_set_rq0.csv"),
            "render_metadata": str(EXP / "outputs/render_metadata/default_render_metadata.csv"),
            "clean_direct_vlm_inputs": [
                str(EXP / "outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__image_only__promptB__seed42.jsonl"),
                str(EXP / "outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl"),
            ],
            "tesseract_available": bool(shutil.which("tesseract")),
            "text_llm_status": text_llm_status,
            "outputs": {
                "ocr_text": str(out / "clean_ocr_text_by_snippet_all_engines.csv"),
                "quality": str(out / "ocr_quality_summary.md"),
                "pair_predictions": str(out / "ocr_classical_pair_predictions.csv"),
                "final_table": str(out / "clean_multi_ocr_final_comparison_table.csv"),
                "report": str(out / "CLEAN_MULTI_OCR_EXPERIMENT_REPORT.md"),
            },
        },
    )
    print(f"wrote clean multi-OCR outputs to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
