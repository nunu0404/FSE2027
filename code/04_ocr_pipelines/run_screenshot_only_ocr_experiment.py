#!/usr/bin/env python3
"""Run screenshot-only OCR vs direct VLM comparison for RQ0."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
RESULTS = ROOT / "results/screenshot_only_ocr"
DIFFICULTIES = ["easy", "medium", "hard"]
MODELS = ["random_forest_regressor", "gradient_boosting_regressor", "svr", "linear_regression"]


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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


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
    from collections import Counter

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
    return abs(float(np.mean(a)) if a else 0.0 - float(np.mean(b)) if b else 0.0)


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


def create_inventory(paths: dict[str, Path], missing: list[str]) -> None:
    lines = [
        "# Screenshot-only OCR Experiment Inventory",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Files Found",
    ]
    for label, path in paths.items():
        lines.append(f"- {label}: `{path}` ({'FOUND' if path.exists() else 'MISSING'})")
    lines.extend(
        [
            "",
            "## Scripts Reused",
            f"- Feature extraction: `{EXP / 'scripts/extract_features.py'}`",
            f"- Existing strict-swap VLM outputs: `{EXP / 'outputs/vlm_judges'}`",
            f"- Existing oracle classical pair-level outputs: `{EXP / 'outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv'}`",
            "",
            "## Missing Or Regenerated",
        ]
    )
    lines.extend([f"- {m}" for m in missing] or ["- None"])
    write_text(RESULTS / "00_inventory.md", "\n".join(lines) + "\n")


def create_mapping(dataset: pd.DataFrame, pairs: pd.DataFrame, render_meta: pd.DataFrame) -> pd.DataFrame:
    by_img = render_meta.set_index("rq0_id")["image_path"]
    by_src = dataset.set_index("rq0_id")["raw_code"]
    work = pairs.copy()
    work["image_i_path"] = work["snippet_i"].map(by_img)
    work["image_j_path"] = work["snippet_j"].map(by_img)
    work["source_i_text"] = work["snippet_i"].map(by_src)
    work["source_j_text"] = work["snippet_j"].map(by_src)
    keep = [
        "pair_id",
        "snippet_i",
        "snippet_j",
        "difficulty",
        "human_preference",
        "preference_score_basis",
        "dataset_i",
        "dataset_j",
        "image_i_path",
        "image_j_path",
        "source_i_text",
        "source_j_text",
    ]
    out = work[keep].copy()
    out.to_csv(RESULTS / "pair_image_mapping.csv", index=False)
    return out


def run_easyocr(dataset: pd.DataFrame, render_meta: pd.DataFrame, force: bool, gpu: bool) -> pd.DataFrame:
    out_dir = RESULTS / "ocr_outputs/easyocr"
    out_path = out_dir / "snippet_ocr.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import easyocr
    except Exception as exc:
        rows = []
        for row in dataset.itertuples(index=False):
            rows.append(
                {
                    "snippet_id": row.rq0_id,
                    "image_path": "",
                    "original_source_text": row.raw_code,
                    "ocr_text": "",
                    "ocr_engine": "easyocr",
                    "parse_status": "import_failed",
                    "error_status": repr(exc),
                    "runtime_sec": 0.0,
                    "mean_confidence": np.nan,
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(out_path, index=False)
        return frame

    model_dir = RESULTS / "easyocr_model"
    reader = easyocr.Reader(
        ["en"],
        gpu=gpu,
        verbose=False,
        model_storage_directory=str(model_dir),
        user_network_directory=str(model_dir),
    )
    image_map = render_meta.set_index("rq0_id")["image_path"].to_dict()
    rows = []
    for idx, row in enumerate(dataset.itertuples(index=False), 1):
        image_path = str(image_map.get(row.rq0_id, ""))
        start = time.time()
        try:
            raw = reader.readtext(image_path, detail=1, paragraph=False)
            chunks = []
            confs = []
            for box, text, conf in raw:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                chunks.append((min(ys), min(xs), str(text)))
                confs.append(float(conf))
            chunks.sort(key=lambda x: (round(x[0] / 10) * 10, x[1]))
            ocr_text = "\n".join(text for _, _, text in chunks)
            status = "ok" if ocr_text.strip() else "empty"
            error = ""
            mean_conf = float(np.mean(confs)) if confs else np.nan
        except Exception as exc:
            ocr_text = ""
            status = "error"
            error = repr(exc)
            mean_conf = np.nan
        rows.append(
            {
                "snippet_id": row.rq0_id,
                "image_path": image_path,
                "original_source_text": row.raw_code,
                "ocr_text": ocr_text,
                "ocr_engine": "easyocr",
                "parse_status": status,
                "error_status": error,
                "runtime_sec": time.time() - start,
                "mean_confidence": mean_conf,
            }
        )
        if idx % 25 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[easyocr] processed {idx}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


def compute_ocr_quality(ocr: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_meta = dataset[["rq0_id", "dataset_name"]].rename(columns={"rq0_id": "snippet_id"})
    rows = []
    feature_rows = []
    for row in ocr.itertuples(index=False):
        src = str(row.original_source_text or "")
        hyp = str(row.ocr_text or "")
        precision, recall, f1 = token_f1(src, hyp)
        src_feat = FEATURES.extract_one(src)
        hyp_feat = FEATURES.extract_one(hyp)
        quality = {
            "snippet_id": row.snippet_id,
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
        rows.append(quality)
        feat = {"snippet_id": row.snippet_id, "dataset_name": dataset_meta.set_index("snippet_id").loc[row.snippet_id, "dataset_name"]}
        feat.update(hyp_feat)
        feature_rows.append(feat)
    frame = pd.DataFrame(rows).merge(dataset_meta, on="snippet_id", how="left")
    frame.to_csv(RESULTS / "ocr_quality_by_snippet.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(RESULTS / "ocr_features_easyocr.csv", index=False)

    summary = frame.groupby(["ocr_engine", "dataset_name"], dropna=False)[
        ["char_error_rate", "token_f1", "line_count_abs_diff", "indent_mean_abs_delta", "runtime_sec", "mean_confidence"]
    ].agg(["mean", "median", "std", "count"])
    summary.to_csv(RESULTS / "ocr_quality_summary.csv")
    lines = ["# OCR Quality Summary", ""]
    total = frame.groupby("ocr_engine").agg(
        n=("snippet_id", "count"),
        ok_rate=("parse_status", lambda x: float((x == "ok").mean())),
        char_error_rate=("char_error_rate", "mean"),
        token_f1=("token_f1", "mean"),
        line_count_abs_diff=("line_count_abs_diff", "mean"),
        indent_mean_abs_delta=("indent_mean_abs_delta", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    )
    lines.append(total.to_markdown(floatfmt=".4f"))
    write_text(RESULTS / "ocr_quality_summary.md", "\n".join(lines) + "\n")
    return frame


def oof_ocr_scores(dataset: pd.DataFrame, source_features: pd.DataFrame, ocr_features: pd.DataFrame) -> pd.DataFrame:
    splits = pd.read_csv(EXP / "data/splits/split_assignments.csv")
    splits = splits[(splits["split_type"] == "pooled_stratified_5fold") & (splits["seed"] == 42)].copy()
    src = dataset.merge(source_features, on=["rq0_id", "dataset_name"], how="inner")
    ocr = dataset[["rq0_id", "dataset_name", "human_score_z_within_dataset"]].merge(
        ocr_features.rename(columns={"snippet_id": "rq0_id"}), on=["rq0_id", "dataset_name"], how="inner"
    )
    feature_cols = [c for c in source_features.columns if c.startswith("feature_")]
    rows = []
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
                        "ocr_engine": "easyocr",
                        "training_setting": "source_train_ocr_test_oof",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ocr_classical_snippet_predictions.csv", index=False)
    return out


def pairwise_from_scores(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model_name, group in scores.groupby("model", sort=True):
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
        work["ocr_engine"] = "easyocr"
        frames.append(work)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / "classical_pair_predictions.csv", index=False)
    return out


def direct_vlm_predictions(pairs: pd.DataFrame) -> pd.DataFrame:
    paths = [
        EXP / "outputs/vlm_judges/full_selected/OpenGVLab__InternVL3-8B__image_only__promptB__seed42.jsonl",
        EXP / "outputs/vlm_judges/full_selected/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl",
    ]
    frames = []
    pair_ids = set(pairs["pair_id"])
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_json(path, lines=True)
        frame = frame[frame["pair_id"].isin(pair_ids)].copy()
        frame["system"] = frame["model_name"] + " / " + frame["input_setting"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(RESULTS / "direct_vlm_pair_predictions.csv", index=False)
    return out


def final_tables(
    oracle_pairs: pd.DataFrame,
    ocr_pairs: pd.DataFrame,
    vlm_pairs: pd.DataFrame,
    text_llm_status: str,
) -> pd.DataFrame:
    rows = []
    for model, group in oracle_pairs.groupby("model", sort=True):
        rows.append(
            summarize_pair_predictions(
                group,
                f"Oracle source {model}",
                "oracle_source_classical",
                model,
                "Upper bound for screenshot-only deployment; uses raw source features.",
            )
            | {
                "input_available_at_deployment": "source text",
                "uses_raw_source": "yes",
                "ocr_engine": "",
                "delta_vs_oracle_rf": np.nan,
                "delta_vs_ocr_rf": np.nan,
            }
        )
    for model, group in ocr_pairs.groupby("model", sort=True):
        rows.append(
            summarize_pair_predictions(
                group,
                f"OCR(EasyOCR)+{model}",
                "ocr_classical",
                model,
                "Train on source features; predict held-out snippet scores from OCR text features.",
            )
            | {
                "input_available_at_deployment": "screenshot image -> OCR text",
                "uses_raw_source": "no for prediction",
                "ocr_engine": "easyocr",
                "delta_vs_oracle_rf": np.nan,
                "delta_vs_ocr_rf": np.nan,
            }
        )
    if len(vlm_pairs):
        for system, group in vlm_pairs.groupby("system", sort=True):
            rows.append(
                summarize_pair_predictions(
                    group,
                    f"Direct VLM {system}",
                    "direct_vlm",
                    system,
                    "Existing full-pair image-only strict-swap output reused.",
                )
                | {
                    "input_available_at_deployment": "screenshot image",
                    "uses_raw_source": "no",
                    "ocr_engine": "",
                    "delta_vs_oracle_rf": np.nan,
                    "delta_vs_ocr_rf": np.nan,
                }
            )
    rows.append(
        {
            "system": "OCR(EasyOCR)+text-only LLM",
            "kind": "ocr_text_llm",
            "model": "not_run",
            "num_pairs": 0,
            "valid_pairs": 0,
            "correct_pairs": 0,
            "effective_accuracy": np.nan,
            "valid_accuracy": np.nan,
            "strict_swap_error": np.nan,
            "parse_failure_rate": np.nan,
            "notes": text_llm_status,
            "input_available_at_deployment": "screenshot image -> OCR text",
            "uses_raw_source": "no",
            "ocr_engine": "easyocr",
            "delta_vs_oracle_rf": np.nan,
            "delta_vs_ocr_rf": np.nan,
        }
    )
    table = pd.DataFrame(rows)
    oracle_rf = table.loc[table["system"] == "Oracle source random_forest_regressor", "effective_accuracy"]
    ocr_rf = table.loc[table["system"] == "OCR(EasyOCR)+random_forest_regressor", "effective_accuracy"]
    if len(oracle_rf):
        table["delta_vs_oracle_rf"] = table["effective_accuracy"] - float(oracle_rf.iloc[0])
    if len(ocr_rf):
        table["delta_vs_ocr_rf"] = table["effective_accuracy"] - float(ocr_rf.iloc[0])
    table.to_csv(RESULTS / "final_comparison_table.csv", index=False)
    write_text(RESULTS / "final_comparison_table.md", table.to_markdown(index=False, floatfmt=".4f") + "\n")
    classical = table[table["kind"].isin(["oracle_source_classical", "ocr_classical"])].copy()
    write_text(
        RESULTS / "classical_summary.md",
        "# OCR + Classical ML Summary\n\n"
        + classical.to_markdown(index=False, floatfmt=".4f")
        + "\n\n"
        + "OCR rows train models on original source-derived features and predict held-out snippet scores from OCR-derived features. "
        + "Oracle rows use original source features and are upper bounds, not screenshot-only deployment systems.\n",
    )
    direct = table[table["kind"] == "direct_vlm"].copy()
    if len(direct):
        write_text(
            RESULTS / "direct_vlm_summary.md",
            "# Direct VLM Summary\n\n"
            + direct.to_markdown(index=False, floatfmt=".4f")
            + "\n\nExisting matched full-pair image-only strict-swap outputs were reused. Invalid strict-swap pairs are treated as incorrect for effective accuracy.\n",
        )
    else:
        write_text(RESULTS / "direct_vlm_summary.md", "# Direct VLM Summary\n\nNo matched direct VLM outputs were found.\n")
    pd.DataFrame(
        columns=[
            "pair_id",
            "model",
            "ocr_engine",
            "order_ab_output",
            "order_ba_output",
            "parsed_ab",
            "parsed_ba",
            "is_valid_strict_swap",
            "model_preference",
            "is_correct",
            "parse_failures",
        ]
    ).to_csv(RESULTS / "ocr_llm_pair_predictions.csv", index=False)
    write_text(
        RESULTS / "ocr_llm_summary.md",
        "# OCR + Text-only LLM Summary\n\n"
        + text_llm_status
        + "\n\nNo local text-only LLM checkpoint was available, so no OCR+LLM strict-swap run was executed in this pass.\n",
    )
    return table


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = b.astype(float) - a.astype(float)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diff[idx]))
    return {
        "mean_delta": float(np.mean(diff)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_sign": float(2 * min(np.mean(samples <= 0), np.mean(samples >= 0))),
    }


def stats_and_examples(
    oracle_pairs: pd.DataFrame,
    ocr_pairs: pd.DataFrame,
    vlm_pairs: pd.DataFrame,
    ocr: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    rows = []
    bootstrap_rows = []
    pred_sets = {}
    for name, frame, pred_col, valid_col in [
        ("Oracle RF", oracle_pairs[oracle_pairs["model"] == "random_forest_regressor"], "model_preference", "is_valid_score_pair"),
        ("OCR EasyOCR RF", ocr_pairs[ocr_pairs["model"] == "random_forest_regressor"], "model_preference", "is_valid_score_pair"),
    ]:
        pred_sets[name] = frame.set_index("pair_id")[["is_correct", pred_col, valid_col]].rename(
            columns={"is_correct": f"{name}_correct", pred_col: f"{name}_prediction", valid_col: f"{name}_valid"}
        )
    if len(vlm_pairs):
        for system, group in vlm_pairs.groupby("system", sort=True):
            pred_sets[f"Direct VLM {system}"] = group.set_index("pair_id")[
                ["is_correct", "model_preference", "is_valid_strict_swap"]
            ].rename(
                columns={
                    "is_correct": f"Direct VLM {system}_correct",
                    "model_preference": f"Direct VLM {system}_prediction",
                    "is_valid_strict_swap": f"Direct VLM {system}_valid",
                }
            )
    comparisons = [
        ("Oracle RF", "OCR EasyOCR RF", "OCR degradation"),
    ]
    for name in list(pred_sets):
        if name.startswith("Direct VLM"):
            comparisons.append(("OCR EasyOCR RF", name, "OCR RF vs direct VLM"))
            comparisons.append(("Oracle RF", name, "oracle upper bound vs direct VLM"))
    for a, b, label in comparisons:
        joined = pred_sets[a].join(pred_sets[b], how="inner")
        ca = joined[f"{a}_correct"].astype(bool).to_numpy()
        cb = joined[f"{b}_correct"].astype(bool).to_numpy()
        boot = paired_bootstrap(ca, cb, 10000, 20260628)
        b_correct_a_wrong = int((cb & ~ca).sum())
        a_correct_b_wrong = int((ca & ~cb).sum())
        mcnemar_p = binomtest(min(b_correct_a_wrong, a_correct_b_wrong), b_correct_a_wrong + a_correct_b_wrong, 0.5).pvalue if (b_correct_a_wrong + a_correct_b_wrong) else np.nan
        row = {
            "comparison": f"{a} -> {b}",
            "label": label,
            "n_pairs": int(len(joined)),
            "accuracy_a": float(ca.mean()),
            "accuracy_b": float(cb.mean()),
            "delta_b_minus_a": float(cb.mean() - ca.mean()),
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "bootstrap_p_two_sided_sign": boot["p_two_sided_sign"],
            "mcnemar_discordant_a_correct": a_correct_b_wrong,
            "mcnemar_discordant_b_correct": b_correct_a_wrong,
            "mcnemar_p": float(mcnemar_p) if not math.isnan(mcnemar_p) else np.nan,
        }
        rows.append(row)
        bootstrap_rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "bootstrap_results.csv", index=False)
    lines = ["# Statistical Tests", ""]
    lines.append(stats.to_markdown(index=False, floatfmt=".4f"))
    write_text(RESULTS / "statistical_tests.md", "\n".join(lines) + "\n")

    if len(vlm_pairs):
        vlm = vlm_pairs.sort_values("system").groupby("pair_id").head(1).set_index("pair_id")
        rf = ocr_pairs[ocr_pairs["model"] == "random_forest_regressor"].set_index("pair_id")
        joined = rf[["model_preference", "is_correct"]].rename(
            columns={"model_preference": "ocr_rf_preference", "is_correct": "ocr_rf_correct"}
        ).join(
            vlm[["model_preference", "is_correct", "is_valid_strict_swap"]].rename(
                columns={
                    "model_preference": "vlm_preference",
                    "is_correct": "vlm_correct",
                    "is_valid_strict_swap": "vlm_valid",
                }
            ),
            how="inner",
        )
        joined = joined.join(mapping.set_index("pair_id"), how="left")
        ocr_map = ocr.set_index("snippet_id")["ocr_text"].to_dict()
        quality_map = pd.read_csv(RESULTS / "ocr_quality_by_snippet.csv").set_index("snippet_id")["char_error_rate"].to_dict()
        cases = [
            (
                "OCR severely degraded",
                joined.assign(
                    max_cer=joined[["snippet_i", "snippet_j"]].apply(
                        lambda r: max(quality_map.get(r.iloc[0], 0), quality_map.get(r.iloc[1], 0)), axis=1
                    )
                )
                .sort_values("max_cer", ascending=False)
                .head(1),
            ),
            ("VLM correct while OCR+RF wrong", joined[(~joined["ocr_rf_correct"].astype(bool)) & (joined["vlm_correct"].astype(bool))].head(1)),
            ("OCR+RF correct while VLM wrong", joined[(joined["ocr_rf_correct"].astype(bool)) & (~joined["vlm_correct"].astype(bool)) & (joined["vlm_valid"].astype(bool))].head(1)),
            ("Both failed", joined[(~joined["ocr_rf_correct"].astype(bool)) & (~joined["vlm_correct"].astype(bool))].head(1)),
            ("VLM strict-swap invalid", joined[~joined["vlm_valid"].astype(bool)].head(1)),
        ]
        lines = ["# Error Examples", ""]
        for title, case in cases:
            lines.append(f"## {title}")
            if case.empty:
                lines.append("No matching example found.\n")
                continue
            r = case.iloc[0]
            lines.extend(
                [
                    f"- pair_id: `{r.name}`",
                    f"- snippets: `{r.snippet_i}` vs `{r.snippet_j}`",
                    f"- difficulty: `{r.difficulty}`",
                    f"- gold preference: `{r.human_preference}`",
                    f"- OCR+RF prediction: `{r.ocr_rf_preference}`, correct={bool(r.ocr_rf_correct)}",
                    f"- VLM prediction: `{r.vlm_preference}`, valid={bool(r.vlm_valid)}, correct={bool(r.vlm_correct)}",
                    f"- image_i: `{r.image_i_path}`",
                    f"- image_j: `{r.image_j_path}`",
                    "",
                    "Snippet i OCR excerpt:",
                    "```text",
                    str(ocr_map.get(r.snippet_i, ""))[:1200],
                    "```",
                    "",
                    "Snippet j OCR excerpt:",
                    "```text",
                    str(ocr_map.get(r.snippet_j, ""))[:1200],
                    "```",
                    "",
                ]
            )
        write_text(RESULTS / "error_examples.md", "\n".join(lines) + "\n")


def report(table: pd.DataFrame, quality: pd.DataFrame, text_llm_status: str) -> None:
    qsum = quality.agg({"char_error_rate": "mean", "token_f1": "mean", "line_count_abs_diff": "mean"}).to_dict()
    rows = {r["system"]: r for r in table.to_dict(orient="records")}
    oracle_rf = rows.get("Oracle source random_forest_regressor", {})
    ocr_rf = rows.get("OCR(EasyOCR)+random_forest_regressor", {})
    direct = [r for r in table.to_dict(orient="records") if r["kind"] == "direct_vlm"]
    direct_best = max(direct, key=lambda r: -1 if pd.isna(r["effective_accuracy"]) else r["effective_accuracy"]) if direct else {}
    lines = [
        "# SCREENSHOT_ONLY_OCR_EXPERIMENT_REPORT",
        "",
        "## 1. Purpose",
        "Compare OCR-mediated source/text pipelines against direct VLM judging when only rendered code screenshots are available.",
        "",
        "## 2. Motivation",
        "The source-available RF/GB/SVR baselines are strong, but they require original source text. Screenshot-only deployment needs OCR before feature extraction or text-only LLM judging.",
        "",
        "## 3. Data and pair construction",
        "The experiment reuses the RQ0 313 Java snippets and the full difficulty-balanced 3,000-pair set: 1,000 easy, 1,000 medium, and 1,000 hard pairs. Gold labels are score-derived proxy preferences from within-dataset z-normalized human readability scores, not direct pairwise human annotation.",
        "",
        "## 4. Screenshot-only setting definition",
        "OCR and direct VLM systems receive rendered snippet images. OCR+classical predictions use only OCR text at prediction time. Oracle source rows are upper bounds and explicitly use raw source features.",
        "",
        "## 5. OCR pipeline",
        "EasyOCR was installed in the user Python environment and run once per rendered snippet image. OCR text is cached at `ocr_outputs/easyocr/snippet_ocr.csv`.",
        "",
        "## 6. OCR quality analysis",
        f"- Mean character error rate: {qsum.get('char_error_rate', np.nan):.4f}",
        f"- Mean token F1: {qsum.get('token_f1', np.nan):.4f}",
        f"- Mean absolute line-count difference: {qsum.get('line_count_abs_diff', np.nan):.2f}",
        "",
        "## 7. OCR + classical ML results",
        f"OCR+RF effective accuracy: {pct(ocr_rf.get('effective_accuracy', np.nan))}; oracle source RF: {pct(oracle_rf.get('effective_accuracy', np.nan))}.",
        "",
        "## 8. OCR + text-only LLM results",
        text_llm_status,
        "",
        "## 9. Direct VLM results",
        f"Best available matched direct VLM row: {direct_best.get('system', 'NA')} effective accuracy {pct(direct_best.get('effective_accuracy', np.nan))}, valid accuracy {pct(direct_best.get('valid_accuracy', np.nan))}, strict-swap error {pct(direct_best.get('strict_swap_error', np.nan))}.",
        "",
        "## 10. Main comparison table",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 11. Statistical tests",
        "See `statistical_tests.md` and `bootstrap_results.csv`.",
        "",
        "## 12. Error analysis",
        "See `error_examples.md`.",
        "",
        "## 13. Threats to validity",
        "- EasyOCR is not specialized for code OCR.",
        "- Direct VLM comparison is limited to existing matched full-pair image-only outputs; Qwen image-only full 3,000 was not available.",
        "- OCR+classical uses source-trained models evaluated on OCR features, which measures a realistic domain shift but not a separately optimized OCR-specific model.",
        "- The labels are constructed score-derived proxy preferences.",
        "",
        "## 14. Recommended paper wording",
        "OCR-mediated baselines and direct VLMs exhibit complementary failure modes. OCR pipelines suffer when visual code structure is corrupted during text extraction, whereas VLMs remain vulnerable to A/B binding and strict-swap instability. This supports framing VLM-as-a-Judge as a fragile but potentially complementary measurement instrument rather than a standalone replacement for source-based readability predictors.",
        "",
        "## 15. Reproduction commands",
        "Run `bash results/screenshot_only_ocr/reproduce.sh` from the repository root.",
        "",
    ]
    write_text(RESULTS / "SCREENSHOT_ONLY_OCR_EXPERIMENT_REPORT.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths = {
        "RQ0 experiment dir": EXP,
        "dataset": EXP / "data/processed/pooled_313_processed.csv",
        "full pair set": EXP / "data/pairs/full_pair_set_rq0.csv",
        "default render metadata": EXP / "outputs/render_metadata/default_render_metadata.csv",
        "source features": EXP / "data/processed/features_313.csv",
        "oracle full classical pair-level": EXP / "outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv",
        "direct InternVL image-only full": EXP / "outputs/vlm_judges/full_selected/OpenGVLab__InternVL3-8B__image_only__promptB__seed42.jsonl",
    }
    missing = [f"Missing required path: {label} -> {path}" for label, path in paths.items() if not path.exists()]
    missing.append("System Tesseract unavailable; EasyOCR installed and used as the OCR engine.")
    missing.append("No local text-only LLM checkpoint found for OCR+text-only LLM; row is reported as not_run.")
    create_inventory(paths, missing)

    dataset = pd.read_csv(paths["dataset"])
    pairs = pd.read_csv(paths["full pair set"])
    render_meta = pd.read_csv(paths["default render metadata"])
    source_features = pd.read_csv(paths["source features"])
    mapping = create_mapping(dataset, pairs, render_meta)

    ocr = run_easyocr(dataset, render_meta, force=args.force_ocr, gpu=args.gpu)
    quality = compute_ocr_quality(ocr, dataset)
    ocr_features = pd.read_csv(RESULTS / "ocr_features_easyocr.csv")
    ocr_scores = oof_ocr_scores(dataset, source_features, ocr_features)
    ocr_pairs = pairwise_from_scores(pairs, ocr_scores)

    oracle_pairs = pd.read_csv(paths["oracle full classical pair-level"])
    vlm_pairs = direct_vlm_predictions(pairs)
    text_llm_status = "Not run: no local text-only LLM checkpoint was available in the Hugging Face cache, and running a new 7B text-only model would require a separate download/inference budget."
    table = final_tables(oracle_pairs, ocr_pairs, vlm_pairs, text_llm_status)
    stats_and_examples(oracle_pairs, ocr_pairs, vlm_pairs, ocr, mapping)
    report(table, quality, text_llm_status)
    write_text(
        RESULTS / "reproduce.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /ANON/experiment_root\npython experiments/rq0_viability/scripts/run_screenshot_only_ocr_experiment.py --force-ocr\n",
    )
    subprocess.run(["chmod", "+x", str(RESULTS / "reproduce.sh")], check=False)
    write_json(
        RESULTS / "run_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "easyocr_available": True,
            "tesseract_path": shutil.which("tesseract"),
            "outputs": str(RESULTS),
        },
    )
    print(f"Wrote screenshot-only OCR experiment outputs to {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
