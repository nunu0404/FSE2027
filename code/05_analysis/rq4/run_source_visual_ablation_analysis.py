#!/usr/bin/env python3
"""Ablate source vs visual features for RF/VLM mechanism analysis.

This script reuses existing clean mechanism and judge outputs. It does not run
new VLM inference.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MECH = ROOT / "results/complementarity_mechanism_clean_qwen_text_plus_image_20260706/final_mechanism_dataset.csv"
DEFAULT_OUT = ROOT / "results/source_visual_ablation_clean_qwen_text_plus_image_20260708"


PAIR_SUFFIXES = ("_diff_abs", "_gold_minus_loser", "_mean", "_max")
CORE_SOURCE_PATTERNS = (
    "source_loc_",
    "source_comment_line_",
    "source_identifier_count_",
    "source_identifier_unique_count_",
    "source_avg_identifier_length_",
    "source_avg_line_length_",
    "source_avg_nonempty_line_length_",
    "source_max_line_length_",
    "source_line_length_std_",
)


def selected_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    source = [
        c
        for c in df.columns
        if c.startswith("source_") and c.endswith(PAIR_SUFFIXES)
    ]
    visual = [
        c
        for c in df.columns
        if c.startswith("visual_") and c.endswith(PAIR_SUFFIXES)
    ]
    return source, visual


def core_source_cols(source_cols: list[str]) -> list[str]:
    return [c for c in source_cols if any(c.startswith(p) for p in CORE_SOURCE_PATTERNS)]


def cv_predict(
    data: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: pd.Series,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = data[numeric_cols + categorical_cols].copy()
    y = target.astype(int).to_numpy()
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", GradientBoostingClassifier(random_state=seed)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred, float(roc_auc_score(y, proba)), float(balanced_accuracy_score(y, pred))


def bootstrap_auc_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 20260708, n: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    idx_all = np.arange(len(y))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "bootstrap_reps": int(len(arr)),
        "delta_auc_mean": float(arr.mean()) if len(arr) else np.nan,
        "delta_auc_ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
        "delta_auc_ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
        "delta_auc_p_two_sided": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())) if len(arr) else np.nan,
    }


def run_ablation(final: pd.DataFrame, out: Path) -> None:
    source_cols, visual_cols = selected_feature_cols(final)
    core_cols = core_source_cols(source_cols)
    controls_num = ["abs_human_score_gap", "rf_margin_abs", "rf_margin_percentile"]
    controls_cat = ["difficulty"]
    feature_sets = {
        "controls_only": [],
        "source_only": source_cols,
        "visual_only": visual_cols,
        "source_plus_visual": source_cols + visual_cols,
        "core_source": core_cols,
        "core_source_plus_visual": core_cols + visual_cols,
    }
    rf_wrong_count = int((~final["rf_correct"].astype(bool)).sum())
    valid_wrong_count = int(((~final["rf_correct"].astype(bool)) & final["vlm_strict_swap_valid"].astype(bool)).sum())
    targets = {
        f"rf_failure_all_{len(final)}": (final.copy(), (~final["rf_correct"].astype(bool)).astype(int)),
        f"vlm_rescue_rf_wrong_{rf_wrong_count}": (
            final[~final["rf_correct"].astype(bool)].copy(),
            (
                final.loc[~final["rf_correct"].astype(bool), "vlm_strict_swap_valid"].astype(bool)
                & final.loc[~final["rf_correct"].astype(bool), "vlm_correct"].astype(bool)
            ).astype(int),
        ),
        f"vlm_correct_given_valid_{valid_wrong_count}": (
            final[(~final["rf_correct"].astype(bool)) & final["vlm_strict_swap_valid"].astype(bool)].copy(),
            final.loc[(~final["rf_correct"].astype(bool)) & final["vlm_strict_swap_valid"].astype(bool), "vlm_correct"].astype(int),
        ),
    }
    rows = []
    preds = []
    for target_name, (frame, target) in targets.items():
        frame = frame.reset_index(drop=True)
        target = target.reset_index(drop=True)
        probs_by_set: dict[str, np.ndarray] = {}
        for fs_name, fs_cols in feature_sets.items():
            numeric = [c for c in controls_num + fs_cols if c in frame.columns]
            categorical = [c for c in controls_cat if c in frame.columns]
            proba, pred, auc, bal = cv_predict(frame, numeric, categorical, target)
            probs_by_set[fs_name] = proba
            rows.append(
                {
                    "target": target_name,
                    "feature_set": fs_name,
                    "n": int(len(frame)),
                    "positive_n": int(target.sum()),
                    "positive_rate": float(target.mean()),
                    "numeric_feature_n": int(len(numeric)),
                    "source_feature_n": int(sum(c.startswith("source_") for c in numeric)),
                    "visual_feature_n": int(sum(c.startswith("visual_") for c in numeric)),
                    "cv_auc": auc,
                    "cv_balanced_accuracy": bal,
                }
            )
            for i, (pair_id, y, p, pr) in enumerate(zip(frame["pair_id"], target, proba, pred)):
                preds.append(
                    {
                        "target": target_name,
                        "feature_set": fs_name,
                        "row_index": i,
                        "pair_id": pair_id,
                        "target_value": int(y),
                        "pred_proba": float(p),
                        "pred_label": int(pr),
                    }
                )
        y = target.to_numpy(dtype=int)
        for base, added in [
            ("source_only", "source_plus_visual"),
            ("core_source", "core_source_plus_visual"),
            ("controls_only", "visual_only"),
            ("controls_only", "source_only"),
        ]:
            delta = bootstrap_auc_delta(y, probs_by_set[base], probs_by_set[added])
            rows.append(
                {
                    "target": target_name,
                    "feature_set": f"delta:{added}-{base}",
                    "n": int(len(frame)),
                    "positive_n": int(target.sum()),
                    "positive_rate": float(target.mean()),
                    "numeric_feature_n": np.nan,
                    "source_feature_n": np.nan,
                    "visual_feature_n": np.nan,
                    "cv_auc": float(roc_auc_score(y, probs_by_set[added]) - roc_auc_score(y, probs_by_set[base])),
                    "cv_balanced_accuracy": np.nan,
                    **delta,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "source_visual_ablation_summary.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "source_visual_ablation_predictions.csv", index=False)

    table = summary[~summary["feature_set"].str.startswith("delta:")].copy()
    delta = summary[summary["feature_set"].str.startswith("delta:")].copy()
    lines = [
        "# Source vs Visual Ablation",
        "",
        "All rows use the same GradientBoostingClassifier, 5-fold stratified CV, and existing clean mechanism data. No new VLM inference was run.",
        "",
        "## Feature Counts",
        "",
        f"- Source feature columns used: {len(source_cols)}",
        f"- Visual feature columns used: {len(visual_cols)}",
        f"- Core source feature columns used: {len(core_cols)}",
        "- Controls included in every model: difficulty, abs human score gap, RF absolute margin, RF margin percentile.",
        "",
        "## Main Ablation",
        "",
        table[
            [
                "target",
                "feature_set",
                "n",
                "positive_n",
                "positive_rate",
                "source_feature_n",
                "visual_feature_n",
                "cv_auc",
                "cv_balanced_accuracy",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Visual Increment",
        "",
        delta[
            [
                "target",
                "feature_set",
                "cv_auc",
                "delta_auc_ci_low",
                "delta_auc_ci_high",
                "delta_auc_p_two_sided",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out / "source_visual_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_system(df: pd.DataFrame, rf_wrong_ids: set[str], label: str) -> dict:
    work = df[df["pair_id"].isin(rf_wrong_ids)].copy()
    valid = work["is_valid_strict_swap"].astype(bool)
    correct = work["is_correct"].astype(bool)
    rescue = valid & correct
    wrong_valid = valid & ~correct
    invalid = ~valid
    return {
        "system": label,
        "n_rf_wrong": int(len(work)),
        "valid_n": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "rescue_n": int(rescue.sum()),
        "rescue_rate_rf_wrong": float(rescue.mean()),
        "valid_accuracy": float(correct[valid].mean()) if valid.any() else np.nan,
        "wrong_valid_n": int(wrong_valid.sum()),
        "invalid_n": int(invalid.sum()),
        "rescue_pair_ids": set(work.loc[rescue, "pair_id"]),
        "valid_pair_ids": set(work.loc[valid, "pair_id"]),
    }


def run_input_ablation(final: pd.DataFrame, out: Path) -> None:
    rf_wrong_ids = set(final.loc[~final["rf_correct"].astype(bool), "pair_id"])
    is_python_cuda_extension = final["pair_id"].astype(str).str.startswith("ext_dorn_").all()
    systems = []
    # Qwen text+image from the mechanism file.
    qwen_ti = final.rename(
        columns={
            "vlm_strict_swap_valid": "is_valid_strict_swap",
            "vlm_correct": "is_correct",
        }
    )[["pair_id", "is_valid_strict_swap", "is_correct"]].copy()
    systems.append(summarize_system(qwen_ti, rf_wrong_ids, "Qwen-VL text+image"))

    # Qwen image-only from the matching clean experiment, if present.
    if is_python_cuda_extension:
        qwen_img_path = ROOT / (
            "results/python_cuda_vlm_main_20260715/outputs/"
            "Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl"
        )
    else:
        qwen_img_path = ROOT / (
            "experiments/rq0_viability/outputs/complementarity_clean_20260707/"
            "qwen_image_only/rf_vlm_joined_pair_results.csv"
        )
    if qwen_img_path.exists():
        if qwen_img_path.suffix == ".jsonl":
            qwen_img = pd.read_json(qwen_img_path, lines=True)
        else:
            qwen_img = pd.read_csv(qwen_img_path).rename(
                columns={"vlm_valid": "is_valid_strict_swap", "vlm_correct": "is_correct"}
            )
        systems.append(summarize_system(qwen_img, rf_wrong_ids, "Qwen-VL image-only"))

    if is_python_cuda_extension:
        ocr_llm_path = ROOT / (
            "results/python_cuda_missing_experiments_20260716/"
            "ocr_text_llm_pair_predictions.csv"
        )
    else:
        ocr_llm_path = ROOT / (
            "results/screenshot_only_ocr_clean_20260707/"
            "clean_ocr_text_llm_pair_predictions.csv"
        )
    if ocr_llm_path.exists():
        llm = pd.read_csv(ocr_llm_path)
        if is_python_cuda_extension:
            system_col = "condition"
            wanted = {
                "source": "Qwen2.5-Coder source-text-only",
                "rapidocr": "Qwen2.5-Coder RapidOCR-text-only",
                "easyocr": "Qwen2.5-Coder EasyOCR-text-only",
                "easyocr_preprocessed": "Qwen2.5-Coder EasyOCR-preprocessed-text-only",
            }
        else:
            system_col = "system"
            wanted = {
                "Source text LLM Qwen2.5-Coder": "Qwen2.5-Coder source-text-only",
                "OCR(RapidOCR)+text-only LLM Qwen2.5-Coder": "Qwen2.5-Coder RapidOCR-text-only",
                "OCR(EasyOCR)+text-only LLM Qwen2.5-Coder": "Qwen2.5-Coder EasyOCR-text-only",
                "OCR(EasyOCR-preprocessed)+text-only LLM Qwen2.5-Coder": "Qwen2.5-Coder EasyOCR-preprocessed-text-only",
            }
        for sys_name, label in wanted.items():
            sub = llm[llm[system_col] == sys_name].copy()
            if len(sub):
                systems.append(summarize_system(sub, rf_wrong_ids, label))

    clean_rows = []
    rescue_sets = {}
    valid_sets = {}
    for row in systems:
        rescue_ids = row.pop("rescue_pair_ids")
        valid_ids = row.pop("valid_pair_ids")
        if row["n_rf_wrong"] == 0:
            continue
        rescue_sets[row["system"]] = rescue_ids
        valid_sets[row["system"]] = valid_ids
        clean_rows.append(row)
    summary = pd.DataFrame(clean_rows)
    summary.to_csv(out / "input_ablation_rescue_summary.csv", index=False)

    overlap_rows = []
    labels = list(rescue_sets)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            ia = rescue_sets[a]
            ib = rescue_sets[b]
            va = valid_sets[a]
            vb = valid_sets[b]
            inter = len(ia & ib)
            union = len(ia | ib)
            overlap_rows.append(
                {
                    "system_a": a,
                    "system_b": b,
                    "rescue_overlap_n": inter,
                    "rescue_union_n": union,
                    "rescue_jaccard": inter / union if union else np.nan,
                    "valid_overlap_n": len(va & vb),
                    "valid_jaccard": len(va & vb) / len(va | vb) if (va | vb) else np.nan,
                }
            )
    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(out / "input_ablation_rescue_overlap.csv", index=False)

    # A compact feature characterization for each system's rescue cases.
    feature_cols = [
        "visual_aspect_ratio_gold_minus_loser",
        "source_comment_line_count_gold_minus_loser",
        "source_loc_nonempty_gold_minus_loser",
        "source_identifier_count_gold_minus_loser",
        "source_comment_line_ratio_gold_minus_loser",
        "source_loc_total_gold_minus_loser",
    ]
    char_rows = []
    for label, ids in rescue_sets.items():
        sub = final[final["pair_id"].isin(ids)]
        for c in feature_cols:
            char_rows.append(
                {
                    "system": label,
                    "feature": c,
                    "n_rescue": int(len(sub)),
                    "mean": float(sub[c].mean()) if len(sub) else np.nan,
                    "median": float(sub[c].median()) if len(sub) else np.nan,
                }
            )
    pd.DataFrame(char_rows).to_csv(out / "input_ablation_rescue_feature_characterization.csv", index=False)

    lines = [
        "# VLM/Text Input Ablation Rescue Pattern",
        "",
        "This joins existing clean RF-wrong pairs with existing judge outputs. No new inference was run.",
        "",
        "Important caveat: source/OCR text-only rows use Qwen2.5-Coder, while image/text+image rows use Qwen2.5-VL. This is therefore an input-condition comparison, not a same-model modality-only ablation.",
        "",
        "## RF-Wrong Rescue Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Rescue Overlap",
        "",
        overlaps.to_markdown(index=False, floatfmt=".4f") if len(overlaps) else "No overlap rows.",
        "",
    ]
    (out / "input_ablation_rescue_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", default=str(DEFAULT_MECH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--input-ablation-only",
        action="store_true",
        help="Recompute only the input-condition rescue analysis.",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(args.mechanism)
    if not args.input_ablation_only:
        run_ablation(final, out)
    run_input_ablation(final, out)
    manifest = {
        "mechanism": str(Path(args.mechanism).resolve()),
        "output_dir": str(out.resolve()),
        "rows": int(len(final)),
        "columns": int(len(final.columns)),
        "input_ablation_only": bool(args.input_ablation_only),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
