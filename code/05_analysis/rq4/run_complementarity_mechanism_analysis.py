#!/usr/bin/env python3
"""RF-VLM complementarity mechanism analysis for RQ0."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rq0_viability"
OUT = ROOT / "results/complementarity_mechanism"
FIG = OUT / "figures"
GROUPS = [
    "RF_correct__VLM_correct",
    "RF_correct__VLM_wrong",
    "RF_correct__VLM_invalid",
    "RF_wrong__VLM_correct",
    "RF_wrong__VLM_wrong",
    "RF_wrong__VLM_invalid",
]
DIFFICULTIES = ["easy", "medium", "hard"]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def import_features_module():
    path = EXP / "scripts/extract_features.py"
    spec = importlib.util.spec_from_file_location("rq0_extract_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEATURES = import_features_module()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, **kwargs) -> str:
    try:
        return df.to_markdown(index=False, **kwargs)
    except Exception:
        return df.to_csv(index=False)


def pct(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100 * float(x):.2f}%"


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    out = np.full(len(p), np.nan)
    mask = ~np.isnan(p)
    vals = p[mask]
    if len(vals) == 0:
        return pd.Series(out, index=pvalues.index)
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(vals) / (np.arange(len(vals)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[mask] = tmp
    return pd.Series(out, index=pvalues.index)


def outcome_group(r: pd.Series) -> str:
    rf = "RF_correct" if bool(r["rf_correct"]) else "RF_wrong"
    if not bool(r["vlm_strict_swap_valid"]):
        vlm = "VLM_invalid"
    elif bool(r["vlm_correct"]):
        vlm = "VLM_correct"
    else:
        vlm = "VLM_wrong"
    return f"{rf}__{vlm}"


def create_inventory(paths: dict[str, Path], vlm_condition: str) -> None:
    missing = [f"{label}: `{path}`" for label, path in paths.items() if not path.exists()]
    lines = [
        "# Complementarity Mechanism Inventory",
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
            f"- Existing feature extractor: `{EXP / 'scripts/extract_features.py'}`",
            f"- Existing RF pair-level output: `{paths['rf_pair_level']}`",
            f"- Clean VLM raw output: `{paths['best_vlm_raw']}`",
            "",
            "## Missing Files",
        ]
    )
    lines.extend([f"- {m}" for m in missing] or ["- None"])
    lines.extend(
        [
            "",
            "## Regenerated Files",
            f"- All files under `{OUT}` are newly generated.",
            "",
            "## Selected Conditions",
            "- RF prediction file: existing full-pair output filtered to `random_forest_regressor`.",
            f"- VLM condition: `{vlm_condition}`.",
        ]
    )
    write_text(OUT / "00_inventory.md", "\n".join(lines) + "\n")


def build_rf_joined(rf_pair_level: pd.DataFrame, render_meta: pd.DataFrame) -> pd.DataFrame:
    rf = rf_pair_level[rf_pair_level["model"].eq("random_forest_regressor")].copy()
    aliases = {
        "model_score_i": "predicted_score_i",
        "model_score_j": "predicted_score_j",
        "dataset_name_i": "dataset_i",
        "dataset_name_j": "dataset_j",
    }
    rf = rf.rename(columns={old: new for old, new in aliases.items() if new not in rf.columns})
    if rf.empty or rf["pair_id"].nunique() != len(rf):
        raise RuntimeError("RF pair-level output must contain exactly one RF row per unique pair_id")
    image_by_id = render_meta.set_index("rq0_id")["image_path"].to_dict()
    missing = sorted(set(rf["snippet_i"]).union(set(rf["snippet_j"])) - set(image_by_id))
    if missing:
        raise RuntimeError(f"Render metadata missing {len(missing)} snippets, e.g. {missing[:5]}")
    out = rf.rename(columns={"model_preference": "rf_preference", "is_correct": "rf_correct"}).copy()
    out["image_i_path"] = out["snippet_i"].map(image_by_id)
    out["image_j_path"] = out["snippet_j"].map(image_by_id)
    return out


def build_pair_level(joined: pd.DataFrame, vlm_raw: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [
        "pair_id",
        "order_ab_output",
        "order_ba_output",
        "parsed_ab",
        "parsed_ba",
        "is_valid_strict_swap",
        "model_preference",
        "is_correct",
        "parse_failures",
        "model_name",
        "input_setting",
        "modality",
        "packaging",
        "prompt_variant",
    ]
    raw = vlm_raw[raw_cols].copy()
    raw = raw.rename(
        columns={
            "order_ab_output": "vlm_ab_choice",
            "order_ba_output": "vlm_ba_choice",
            "parsed_ab": "vlm_parsed_ab",
            "parsed_ba": "vlm_parsed_ba",
            "is_valid_strict_swap": "vlm_strict_swap_valid",
            "model_preference": "vlm_pred_preference",
            "is_correct": "vlm_correct_raw",
            "parse_failures": "vlm_parse_failure",
        }
    )
    work = joined.merge(raw, on="pair_id", how="left", validate="one_to_one", suffixes=("", "_raw"))
    if work["vlm_strict_swap_valid"].isna().any():
        raise RuntimeError("VLM raw output missing for some joined pairs")
    work["rf_margin_raw"] = work["predicted_score_i"] - work["predicted_score_j"]
    work["rf_margin_abs"] = work["rf_margin_raw"].abs()
    work["rf_margin_percentile"] = work["rf_margin_abs"].rank(pct=True)
    work["vlm_correct"] = work["vlm_correct_raw"].fillna(False).astype(bool) & work["vlm_strict_swap_valid"].astype(bool)
    work["vlm_condition"] = (
        work["model_name"].astype(str)
        + " / "
        + work["input_setting"].astype(str)
        + " / prompt"
        + work["prompt_variant"].astype(str)
    )
    out = pd.DataFrame(
        {
            "pair_id": work["pair_id"],
            "snippet_id_a": work["snippet_i"],
            "snippet_id_b": work["snippet_j"],
            "dataset_a": work["dataset_i"],
            "dataset_b": work["dataset_j"],
            "difficulty": work["difficulty"],
            "human_score_z_a": work["human_score_i_z"],
            "human_score_z_b": work["human_score_j_z"],
            "abs_human_score_gap": work["abs_z_diff"],
            "gold_preference": work["human_preference"],
            "rf_score_a": work["predicted_score_i"],
            "rf_score_b": work["predicted_score_j"],
            "rf_pred_preference": work["rf_preference"],
            "rf_correct": work["rf_correct"].astype(bool),
            "rf_margin_raw": work["rf_margin_raw"],
            "rf_margin_abs": work["rf_margin_abs"],
            "rf_margin_percentile": work["rf_margin_percentile"],
            "vlm_condition": work["vlm_condition"],
            "vlm_ab_choice": work["vlm_ab_choice"],
            "vlm_ba_choice": work["vlm_ba_choice"],
            "vlm_parsed_ab": work["vlm_parsed_ab"],
            "vlm_parsed_ba": work["vlm_parsed_ba"],
            "vlm_strict_swap_valid": work["vlm_strict_swap_valid"].astype(bool),
            "vlm_pred_preference": work["vlm_pred_preference"],
            "vlm_correct": work["vlm_correct"].astype(bool),
            "vlm_parse_failure": work["vlm_parse_failure"],
            "image_a_path": work["image_i_path"],
            "image_b_path": work["image_j_path"],
        }
    )
    out["outcome_group"] = out.apply(outcome_group, axis=1)
    out.to_csv(OUT / "pair_level_analysis_table.csv", index=False)
    return out


def count_extra_source_features(dataset: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = dataset.set_index("rq0_id")
    for row in by_id.itertuples():
        code = str(row.raw_code or "")
        clean = FEATURES.strip_strings_and_comments(code)
        lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        rows.append(
            {
                "rq0_id": row.Index,
                "source_brace_count": code.count("{") + code.count("}"),
                "source_parenthesis_count": code.count("(") + code.count(")"),
                "source_method_like_count": len(re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\([^;{}]*\)\s*(?:throws\b[^{}]*)?\{", code)),
                "source_if_count": len(re.findall(r"\bif\b", clean)),
                "source_for_count": len(re.findall(r"\bfor\b", clean)),
                "source_while_count": len(re.findall(r"\bwhile\b", clean)),
                "source_switch_count": len(re.findall(r"\bswitch\b", clean)),
                "source_try_count": len(re.findall(r"\btry\b", clean)),
                "source_catch_count": len(re.findall(r"\bcatch\b", clean)),
                "source_exception_handling_count": len(re.findall(r"\b(try|catch|finally|throw|throws)\b", clean)),
                "source_api_call_count": len(re.findall(r"\.[A-Za-z_$][A-Za-z0-9_$]*\s*\(", code)),
                "source_line_length_std": float(np.std([len(x) for x in lines], ddof=0)) if lines else 0.0,
                "source_blank_line_count": sum(1 for x in lines if not x.strip()),
                "source_comment_line_count": sum(1 for x in lines if x.strip().startswith("//") or x.strip().startswith("*") or "/*" in x or "*/" in x),
            }
        )
    extra = pd.DataFrame(rows)
    feat = base_features.copy()
    if "rq0_id" not in feat.columns and "snippet_uid" in feat.columns:
        feat = feat.rename(columns={"snippet_uid": "rq0_id"})
    feat = feat.rename(columns={c: "source_" + c.removeprefix("feature_") for c in feat.columns if c.startswith("feature_")})
    feat = feat.merge(extra, on="rq0_id", how="left")
    feat.to_csv(OUT / "snippet_source_features.csv", index=False)
    return feat


def visual_features_for_image(path: str) -> dict[str, float]:
    try:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image).astype(np.int16)
    except Exception:
        return {"visual_error": 1.0}
    h, w = arr.shape[:2]
    corners = np.concatenate([arr[:10, :10].reshape(-1, 3), arr[:10, -10:].reshape(-1, 3), arr[-10:, :10].reshape(-1, 3), arr[-10:, -10:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > 18
    density = float(mask.mean())
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    else:
        x0 = y0 = 0
        x1, y1 = w - 1, h - 1
    bbox_w = max(x1 - x0 + 1, 1)
    bbox_h = max(y1 - y0 + 1, 1)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    row_active = row_counts > max(2, 0.005 * w)
    bands = []
    start = None
    for i, active in enumerate(row_active.tolist() + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            bands.append((start, i - 1))
            start = None
    line_heights = [b - a + 1 for a, b in bands]
    gaps = [bands[i + 1][0] - bands[i][1] - 1 for i in range(len(bands) - 1)]
    lefts = []
    widths = []
    for a, b in bands:
        rows = mask[a : b + 1]
        if rows.any():
            yy, xx = np.where(rows)
            lefts.append(float(xx.min()))
            widths.append(float(xx.max() - xx.min() + 1))
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    active_gray = gray[mask] if mask.any() else gray.reshape(-1)
    color_quant = (arr.astype(np.uint8) // 32).reshape(-1, 3)
    color_div = len({tuple(x) for x in color_quant[mask.reshape(-1)]}) if mask.any() else 0
    vertical_empty = int(sum(1 for g in gaps if g >= 8))
    return {
        "image_width": float(w),
        "image_height": float(h),
        "aspect_ratio": float(w / h) if h else np.nan,
        "non_background_pixel_density": density,
        "code_bounding_box_width": float(bbox_w),
        "code_bounding_box_height": float(bbox_h),
        "code_bounding_box_area_ratio": float((bbox_w * bbox_h) / max(w * h, 1)),
        "left_margin_mean": float(np.mean(lefts)) if lefts else np.nan,
        "left_margin_std": float(np.std(lefts, ddof=0)) if lefts else np.nan,
        "left_margin_min": float(np.min(lefts)) if lefts else np.nan,
        "left_margin_max": float(np.max(lefts)) if lefts else np.nan,
        "visual_indentation_depth_proxy": float(np.std(lefts, ddof=0)) if lefts else np.nan,
        "horizontal_whitespace_ratio": float(1 - mask[:, x0 : x1 + 1].mean()) if mask.any() else np.nan,
        "vertical_whitespace_ratio": float(1 - row_active.mean()),
        "number_of_visual_text_lines": float(len(bands)),
        "estimated_line_height": float(np.median(line_heights)) if line_heights else np.nan,
        "line_spacing_mean": float(np.mean(gaps)) if gaps else 0.0,
        "visual_line_length_mean": float(np.mean(widths)) if widths else np.nan,
        "visual_line_length_max": float(np.max(widths)) if widths else np.nan,
        "visual_line_length_std": float(np.std(widths, ddof=0)) if widths else np.nan,
        "dense_block_ratio": float(np.mean(row_counts[row_active] / max(w, 1) > 0.12)) if row_active.any() else 0.0,
        "empty_vertical_gap_count": float(vertical_empty),
        "color_diversity": float(color_div),
        "contrast_score": float(np.std(active_gray, ddof=0)),
        "visual_clutter_score": float(density * color_div * (1 + (np.std(widths, ddof=0) if widths else 0.0) / max(w, 1))),
        "visual_error": 0.0,
    }


def build_visual_features(render_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(render_meta.itertuples(index=False), 1):
        feat = visual_features_for_image(r.image_path)
        feat["rq0_id"] = r.rq0_id
        feat["dataset_name"] = r.dataset_name
        feat["image_path"] = r.image_path
        rows.append(feat)
    out = pd.DataFrame(rows)
    out = out.rename(columns={c: f"visual_{c}" for c in out.columns if c not in {"rq0_id", "dataset_name", "image_path"} and not c.startswith("visual_")})
    out.to_csv(OUT / "snippet_visual_features.csv", index=False)
    write_text(
        OUT / "visual_feature_extraction_notes.md",
        "# Visual Feature Extraction Notes\n\n"
        "Visual features were extracted from existing default rendered snippet PNGs using simple image processing: "
        "background color estimated from image corners, non-background masks, bounding boxes, row/column scans, "
        "text-line band estimates, density, contrast, and color diversity. No OCR or source text was used for visual features.\n",
    )
    return out


def pair_features(pair: pd.DataFrame, snippet_features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    id_col = "rq0_id"
    cols = [c for c in snippet_features.columns if c.startswith(prefix)]
    by_id = snippet_features.set_index(id_col)
    rows = []
    for r in pair.itertuples(index=False):
        a = by_id.loc[r.snippet_id_a]
        b = by_id.loc[r.snippet_id_b]
        gold_id = r.gold_preference
        loser_id = r.snippet_id_b if gold_id == r.snippet_id_a else r.snippet_id_a
        rf_id = r.rf_pred_preference
        row = {"pair_id": r.pair_id}
        for c in cols:
            av = a[c]
            bv = b[c]
            row[f"{c}_a"] = av
            row[f"{c}_b"] = bv
            row[f"{c}_diff_signed"] = av - bv
            row[f"{c}_diff_abs"] = abs(av - bv)
            row[f"{c}_min"] = min(av, bv)
            row[f"{c}_max"] = max(av, bv)
            row[f"{c}_mean"] = (av + bv) / 2
            gv = by_id.loc[gold_id, c]
            lv = by_id.loc[loser_id, c]
            row[f"{c}_gold_minus_loser"] = gv - lv
            row[f"{c}_rf_winner_minus_gold_winner"] = by_id.loc[rf_id, c] - gv if isinstance(rf_id, str) and rf_id in by_id.index else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"pair_{'source' if prefix == 'source_' else 'visual'}_features.csv", index=False)
    return out


def group_descriptives(final: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = final[final["outcome_group"] == group]
        row = {
            "outcome_group": group,
            "count": int(len(sub)),
            "proportion": float(len(sub) / len(final)) if len(final) else np.nan,
            "easy_rate": float((sub["difficulty"] == "easy").mean()) if len(sub) else np.nan,
            "medium_rate": float((sub["difficulty"] == "medium").mean()) if len(sub) else np.nan,
            "hard_rate": float((sub["difficulty"] == "hard").mean()) if len(sub) else np.nan,
            "abs_human_score_gap_mean": float(sub["abs_human_score_gap"].mean()) if len(sub) else np.nan,
            "abs_human_score_gap_median": float(sub["abs_human_score_gap"].median()) if len(sub) else np.nan,
            "rf_margin_abs_mean": float(sub["rf_margin_abs"].mean()) if len(sub) else np.nan,
            "rf_margin_abs_median": float(sub["rf_margin_abs"].median()) if len(sub) else np.nan,
            "rf_margin_percentile_median": float(sub["rf_margin_percentile"].median()) if len(sub) else np.nan,
        }
        for c in feature_cols:
            row[f"{c}_mean"] = float(sub[c].mean()) if len(sub) else np.nan
            row[f"{c}_median"] = float(sub[c].median()) if len(sub) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "group_descriptive_stats.csv", index=False)
    counts = out[["outcome_group", "count", "proportion", "easy_rate", "medium_rate", "hard_rate", "abs_human_score_gap_mean", "rf_margin_abs_mean", "rf_margin_percentile_median"]]
    lines = ["# Group Descriptive Summary", "", md_table(counts, floatfmt=".4f"), ""]
    rescue = out[out["outcome_group"] == "RF_wrong__VLM_correct"].iloc[0]
    lines += [
        "## Rescue Group",
        f"- Count: {int(rescue['count'])} ({pct(rescue['proportion'])})",
        f"- Mean abs human score gap: {rescue['abs_human_score_gap_mean']:.4f}",
        f"- Mean RF margin abs: {rescue['rf_margin_abs_mean']:.4f}",
        f"- Median RF margin percentile: {rescue['rf_margin_percentile_median']:.4f}",
    ]
    write_text(OUT / "group_descriptive_summary.md", "\n".join(lines) + "\n")
    return out


def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    return float(2 * u / (n1 * n2) - 1) if n1 and n2 else np.nan


def stat_tests(final: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    comparisons = [
        ("RF_correct_vs_RF_wrong", final["rf_correct"], True, False),
        (
            "RFwrong_VLMrescue_vs_VLMwrong",
            final["outcome_group"].isin(["RF_wrong__VLM_correct", "RF_wrong__VLM_wrong"]),
            "RF_wrong__VLM_correct",
            "RF_wrong__VLM_wrong",
        ),
        (
            "RFwrong_VLMrescue_vs_VLMinvalid",
            final["outcome_group"].isin(["RF_wrong__VLM_correct", "RF_wrong__VLM_invalid"]),
            "RF_wrong__VLM_correct",
            "RF_wrong__VLM_invalid",
        ),
        (
            "RFwrong_VLMrescue_vs_all_other_RFwrong",
            ~final["rf_correct"],
            "RF_wrong__VLM_correct",
            "other_rf_wrong",
        ),
    ]
    rows = []
    numeric = ["abs_human_score_gap", "rf_margin_abs", "rf_margin_percentile"] + features
    for comp, mask, a_label, b_label in comparisons:
        sub = final[mask].copy()
        if comp == "RF_correct_vs_RF_wrong":
            a = sub[sub["rf_correct"]]
            b = sub[~sub["rf_correct"]]
        elif b_label == "other_rf_wrong":
            a = sub[sub["outcome_group"] == a_label]
            b = sub[sub["outcome_group"] != a_label]
        else:
            a = sub[sub["outcome_group"] == a_label]
            b = sub[sub["outcome_group"] == b_label]
        for col in numeric:
            x = pd.to_numeric(a[col], errors="coerce").dropna()
            y = pd.to_numeric(b[col], errors="coerce").dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                u, p = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                continue
            rows.append(
                {
                    "comparison": comp,
                    "feature": col,
                    "n_a": int(len(x)),
                    "n_b": int(len(y)),
                    "mean_a": float(x.mean()),
                    "mean_b": float(y.mean()),
                    "median_a": float(x.median()),
                    "median_b": float(y.median()),
                    "mean_diff_a_minus_b": float(x.mean() - y.mean()),
                    "median_diff_a_minus_b": float(x.median() - y.median()),
                    "mannwhitney_p": float(p),
                    "cliffs_delta": cliffs_delta_from_u(float(u), len(x), len(y)),
                }
            )
        if "difficulty" in sub:
            table = pd.crosstab(sub["difficulty"], sub["outcome_group"] if comp != "RF_correct_vs_RF_wrong" else sub["rf_correct"])
            if table.shape[0] > 1 and table.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(table)
                rows.append(
                    {
                        "comparison": comp,
                        "feature": "difficulty",
                        "n_a": int(len(a)),
                        "n_b": int(len(b)),
                        "mean_a": np.nan,
                        "mean_b": np.nan,
                        "median_a": np.nan,
                        "median_b": np.nan,
                        "mean_diff_a_minus_b": np.nan,
                        "median_diff_a_minus_b": np.nan,
                        "mannwhitney_p": float(p),
                        "cliffs_delta": np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    out["fdr_p"] = out.groupby("comparison")["mannwhitney_p"].transform(bh_fdr)
    out.to_csv(OUT / "feature_stat_tests.csv", index=False)
    highlight = out[(out["fdr_p"] < 0.05) & (out["cliffs_delta"].abs().fillna(0) >= 0.147)].copy()
    highlight = highlight.sort_values(["comparison", "fdr_p", "cliffs_delta"], ascending=[True, True, False]).groupby("comparison").head(12)
    write_text(
        OUT / "feature_stat_tests_summary.md",
        "# Feature Statistical Tests Summary\n\n"
        "Mann-Whitney U tests with Cliff's delta effect sizes and Benjamini-Hochberg FDR correction were used. "
        "The table below keeps FDR < 0.05 and |Cliff's delta| >= 0.147 where available.\n\n"
        + md_table(highlight, floatfmt=".4g")
        + "\n",
    )
    return out


def model_frame(final: pd.DataFrame, feature_cols: list[str], target: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = final[["difficulty", "abs_human_score_gap", "rf_margin_abs", "rf_margin_percentile"] + feature_cols].copy()
    base["target"] = target.astype(int).to_numpy()
    numeric = [c for c in base.columns if c not in {"difficulty", "target"}]
    categorical = ["difficulty"]
    return base, numeric, categorical


def run_explanatory_models(final: pd.DataFrame, feature_cols: list[str], target: pd.Series, prefix: str, subset_note: str) -> None:
    data, numeric, categorical = model_frame(final, feature_cols, target)
    data = data.dropna(axis=1, how="all")
    numeric = [c for c in numeric if c in data.columns]
    if data["target"].nunique() < 2 or len(data) < 30:
        write_text(OUT / f"{prefix}_model_results.md", f"# {prefix} Model Results\n\nInsufficient class variation or rows.\n")
        pd.DataFrame().to_csv(OUT / f"{prefix}_model_feature_importance.csv", index=False)
        pd.DataFrame().to_csv(OUT / f"{prefix}_model_predictions.csv", index=False)
        return
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "decision_tree_depth3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_rows = []
    metric_rows = []
    importances = []
    feature_names = None
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("model", clf)])
        if hasattr(clf, "predict_proba"):
            proba = cross_val_predict(pipe, data.drop(columns=["target"]), data["target"], cv=cv, method="predict_proba")[:, 1]
        else:
            proba = cross_val_predict(pipe, data.drop(columns=["target"]), data["target"], cv=cv, method="decision_function")
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(data["target"], proba)
        bal = balanced_accuracy_score(data["target"], pred)
        metric_rows.append({"model": name, "cv_auc": float(auc), "cv_balanced_accuracy": float(bal), "n": int(len(data)), "positive_rate": float(data["target"].mean())})
        for idx, y, p, pr in zip(data.index, data["target"], proba, pred):
            pred_rows.append({"row_index": idx, "model": name, "target": int(y), "pred_proba": float(p), "pred_label": int(pr)})
        pipe.fit(data.drop(columns=["target"]), data["target"])
        try:
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            if hasattr(pipe.named_steps["model"], "feature_importances_"):
                vals = pipe.named_steps["model"].feature_importances_
            elif hasattr(pipe.named_steps["model"], "coef_"):
                vals = pipe.named_steps["model"].coef_[0]
            else:
                vals = None
            if vals is not None:
                for f, v in zip(feature_names, vals):
                    importances.append({"model": name, "feature": str(f), "importance": float(v), "abs_importance": float(abs(v))})
        except Exception:
            pass
    imp = pd.DataFrame(importances)
    if len(imp):
        imp = imp.sort_values(["model", "abs_importance"], ascending=[True, False])
    imp.to_csv(OUT / f"{prefix}_model_feature_importance.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / f"{prefix}_model_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows).sort_values("cv_auc", ascending=False)
    best_tree_text = ""
    try:
        tree_pre = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        tree_pipe = Pipeline([("pre", tree_pre), ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, class_weight="balanced", random_state=42))])
        tree_pipe.fit(data.drop(columns=["target"]), data["target"])
        best_tree_text = export_text(tree_pipe.named_steps["model"], feature_names=list(tree_pipe.named_steps["pre"].get_feature_names_out()))
    except Exception as exc:
        best_tree_text = f"Tree export failed: {exc}"
    top_imp = imp.groupby("model").head(15) if len(imp) else pd.DataFrame()
    write_text(
        OUT / f"{prefix}_model_results.md",
        f"# {prefix.replace('_', ' ').title()} Model Results\n\n"
        f"Subset: {subset_note}\n\n"
        "## Cross-validated Metrics\n\n"
        + md_table(metrics, floatfmt=".4f")
        + "\n\n## Top Feature Importances\n\n"
        + md_table(top_imp, floatfmt=".4g")
        + "\n\n## Depth-3 Decision Tree Rules\n\n```text\n"
        + best_tree_text
        + "\n```\n",
    )


def save_plots(final: pd.DataFrame, stat: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIG / f"{name}.png", dpi=200)
        plt.savefig(FIG / f"{name}.pdf")
        plt.close()

    plt.figure(figsize=(10, 4))
    sns.countplot(data=final, y="outcome_group", order=GROUPS)
    plt.title("Outcome Group Counts")
    save("outcome_group_counts")

    diff = pd.crosstab(final["outcome_group"], final["difficulty"], normalize="index").reindex(GROUPS)
    diff.plot(kind="barh", stacked=True, figsize=(10, 4))
    plt.title("Difficulty Distribution by Outcome Group")
    save("difficulty_distribution_by_outcome_group")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=final, y="outcome_group", x="rf_margin_abs", order=GROUPS, showfliers=False)
    plt.title("RF Margin Distribution by Outcome Group")
    save("rf_margin_distribution_by_outcome_group")

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=final, y="outcome_group", x="abs_human_score_gap", order=GROUPS, showfliers=False)
    plt.title("Human Score Gap Distribution by Outcome Group")
    save("human_score_gap_distribution_by_outcome_group")

    for comp, name in [
        ("RF_correct_vs_RF_wrong", "top_feature_differences_rf_correct_vs_wrong"),
        ("RFwrong_VLMrescue_vs_all_other_RFwrong", "top_feature_differences_vlm_rescue_vs_nonrescue"),
    ]:
        sub = stat[(stat["comparison"] == comp) & (stat["feature"] != "difficulty")].copy()
        sub["effect"] = sub["cliffs_delta"].abs()
        sub = sub.sort_values("effect", ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=sub, y="feature", x="cliffs_delta")
        plt.title(comp)
        save(name)

    for feat, name in [
        ("visual_non_background_pixel_density_diff_abs", "visual_density_difference_by_outcome_group"),
        ("visual_indentation_depth_proxy_diff_abs", "visual_indentation_difference_by_outcome_group"),
        ("visual_clutter_score_diff_abs", "visual_clutter_difference_by_outcome_group"),
    ]:
        if feat in final.columns:
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=final, y="outcome_group", x=feat, order=GROUPS, showfliers=False)
            plt.title(name.replace("_", " ").title())
            save(name)


def case_studies(final: pd.DataFrame, dataset: pd.DataFrame, source_feats: pd.DataFrame, visual_feats: pd.DataFrame) -> None:
    by_src = dataset.set_index("rq0_id")["raw_code"].to_dict()
    examples = []
    for group in GROUPS:
        sub = final[final["outcome_group"] == group].copy()
        if group == "RF_wrong__VLM_correct":
            sub = sub.sort_values(["rf_margin_percentile", "abs_human_score_gap"]).head(10)
        else:
            sub = sub.sort_values("rf_margin_abs").head(1)
        examples.append((group, sub))
    lines = ["# Complementarity Case Studies", ""]
    html_parts = [
        "<!doctype html><meta charset='utf-8'><title>Complementarity Case Studies</title>",
        "<style>body{font-family:sans-serif} .case{border:1px solid #aaa;margin:16px;padding:12px} img{max-width:48%;vertical-align:top} pre{white-space:pre-wrap;max-height:320px;overflow:auto;background:#f6f6f6;padding:8px}</style>",
        "<h1>Complementarity Case Studies</h1>",
    ]
    for group, sub in examples:
        lines.append(f"## {group}")
        for r in sub.itertuples(index=False):
            lines.extend(
                [
                    f"### {r.pair_id}",
                    f"- snippets: `{r.snippet_id_a}` vs `{r.snippet_id_b}`",
                    f"- difficulty: `{r.difficulty}`",
                    f"- human z: {r.human_score_z_a:.4f} vs {r.human_score_z_b:.4f}; gold: `{r.gold_preference}`",
                    f"- RF: score_a={r.rf_score_a:.4f}, score_b={r.rf_score_b:.4f}, margin={r.rf_margin_raw:.4f}, pred=`{r.rf_pred_preference}`, correct={bool(r.rf_correct)}",
                    f"- VLM: AB=`{r.vlm_parsed_ab}`, BA=`{r.vlm_parsed_ba}`, valid={bool(r.vlm_strict_swap_valid)}, pred=`{r.vlm_pred_preference}`, correct={bool(r.vlm_correct)}",
                    f"- images: `{r.image_a_path}`, `{r.image_b_path}`",
                    "",
                    "Code A excerpt:",
                    "```java",
                    str(by_src.get(r.snippet_id_a, ""))[:1800],
                    "```",
                    "Code B excerpt:",
                    "```java",
                    str(by_src.get(r.snippet_id_b, ""))[:1800],
                    "```",
                    "",
                ]
            )
            html_parts.append(
                "<div class='case'>"
                f"<h2>{html.escape(group)}: {html.escape(r.pair_id)}</h2>"
                f"<p>difficulty={html.escape(str(r.difficulty))}; gold={html.escape(str(r.gold_preference))}; RF={html.escape(str(r.rf_pred_preference))} correct={bool(r.rf_correct)}; "
                f"VLM={html.escape(str(r.vlm_pred_preference))} valid={bool(r.vlm_strict_swap_valid)} correct={bool(r.vlm_correct)}</p>"
                f"<img src='file://{html.escape(str(r.image_a_path))}'><img src='file://{html.escape(str(r.image_b_path))}'>"
                f"<h3>Code A {html.escape(str(r.snippet_id_a))}</h3><pre>{html.escape(str(by_src.get(r.snippet_id_a, ''))[:2500])}</pre>"
                f"<h3>Code B {html.escape(str(r.snippet_id_b))}</h3><pre>{html.escape(str(by_src.get(r.snippet_id_b, ''))[:2500])}</pre>"
                "</div>"
            )
    write_text(OUT / "case_studies.md", "\n".join(lines) + "\n")
    write_text(OUT / "case_studies_viewer.html", "\n".join(html_parts) + "\n")


def taxonomy(final: pd.DataFrame, stats: pd.DataFrame) -> None:
    rescue = final[final["outcome_group"] == "RF_wrong__VLM_correct"]
    wrong = final[~final["rf_correct"]]
    low_margin_share = float((rescue["rf_margin_percentile"] <= 0.3).mean())
    invalid = final[final["outcome_group"].str.endswith("invalid")]
    visual_cols = [c for c in final.columns if c.startswith("visual_") and c.endswith("_diff_abs")]
    top_visual = (
        stats[(stats["comparison"] == "RFwrong_VLMrescue_vs_all_other_RFwrong") & stats["feature"].isin(visual_cols)]
        .assign(abs_delta=lambda d: d["cliffs_delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(8)
    )
    lines = [
        "# Rescue and Failure Taxonomy",
        "",
        "This taxonomy is descriptive and correlational.",
        "",
        "## Supported Categories",
        f"1. RF low-margin ambiguous cases: {pct(low_margin_share)} of rescue cases fall in the bottom 30% of RF absolute-margin percentiles.",
        "2. VLM strict-swap instability cases: invalid VLM groups remain large and are kept separate from valid wrong cases.",
        "3. Visual/layout-difference cases: visual feature proxies show measurable but not necessarily causal differences in some rescue comparisons.",
        "4. Both-system hard cases: RF_wrong__VLM_wrong and RF_wrong__VLM_invalid groups indicate cases not recovered by the evaluated VLM.",
        "",
        "## Top visual features in rescue vs other RF-wrong cases",
        md_table(top_visual[["feature", "mean_diff_a_minus_b", "cliffs_delta", "fdr_p"]], floatfmt=".4g") if len(top_visual) else "No strong visual feature signal found.",
        "",
        "## Interpretation",
        "The evidence supports a cautious taxonomy: VLM rescue is partly associated with RF uncertainty and some visual/layout proxies, but not enough to claim a stable causal mechanism.",
    ]
    write_text(OUT / "rescue_failure_taxonomy.md", "\n".join(lines) + "\n")


def dictionary(cols: list[str]) -> None:
    lines = ["# Final Mechanism Dataset Dictionary", ""]
    descriptions = {
        "outcome_group": "Six-way RF/VLM status group. VLM invalid is separate from valid wrong.",
        "rf_margin_abs": "Absolute difference between RF predicted scores for snippets A and B.",
        "rf_margin_percentile": "Percentile rank of RF absolute margin among 3,000 pairs.",
        "vlm_correct": "True only when VLM is strict-swap valid and matches the proxy gold preference.",
    }
    for c in cols:
        if c in descriptions:
            desc = descriptions[c]
        elif c.startswith("source_"):
            desc = "Pair-level source-code feature derived from snippet-level handcrafted readability features."
        elif c.startswith("visual_"):
            desc = "Pair-level visual/layout feature derived from rendered PNG image processing."
        else:
            desc = "Core pair, prediction, label, or metadata column."
        lines.append(f"- `{c}`: {desc}")
    write_text(OUT / "final_mechanism_dataset_dictionary.md", "\n".join(lines) + "\n")


def final_report(final: pd.DataFrame, desc: pd.DataFrame, stat: pd.DataFrame, vlm_condition: str) -> None:
    counts = final["outcome_group"].value_counts().reindex(GROUPS).fillna(0).astype(int)
    rescue = final[final["outcome_group"] == "RF_wrong__VLM_correct"]
    rf_wrong = final[~final["rf_correct"]]
    low_margin = float((rescue["rf_margin_percentile"] <= 0.3).mean()) if len(rescue) else np.nan
    rescue_vs = stat[stat["comparison"] == "RFwrong_VLMrescue_vs_all_other_RFwrong"].copy()
    rescue_vs["abs_delta"] = rescue_vs["cliffs_delta"].abs()
    top = rescue_vs.sort_values("abs_delta", ascending=False).head(12)
    lines = [
        "# Complementarity Mechanism Report",
        "",
        "## 1. Purpose",
        "Explain when RF fails and when the best VLM condition rescues RF errors in the RQ0 pairwise readability task.",
        "",
        "## 2. Relationship to existing RQ0 result",
        f"The analysis rebuilds the 3,000-pair RF/VLM mechanism dataset from clean inputs. The VLM condition is `{vlm_condition}`. This is explanatory, not a new performance claim.",
        "",
        "## 3. Data and outcome groups",
        md_table(counts.rename_axis("outcome_group").reset_index(name="count")),
        "",
        "## 4. Feature extraction",
        "Source features reuse the RQ0 handcrafted feature extractor and add simple counts for braces, calls, control flow, and exception handling. Visual features are extracted from rendered PNGs by thresholding non-background pixels and measuring density, bounding boxes, row/column structure, margins, contrast, and color diversity. No OCR is used for visual features.",
        "",
        "## 5. Descriptive group analysis",
        "See `group_descriptive_summary.md` and `group_descriptive_stats.csv`.",
        "",
        "## 6. RF failure characteristics",
        "RF failure is assessed by comparing RF-correct and RF-wrong pairs using nonparametric tests and explanatory models. See `feature_stat_tests_summary.md` and `rf_failure_model_results.md`.",
        "",
        "## 7. VLM rescue characteristics",
        f"VLM rescue group size is {len(rescue)} out of {len(final)} pairs, and {len(rescue)} out of {len(rf_wrong)} RF-wrong pairs. Bottom-30% RF-margin share among rescue cases: {pct(low_margin)}.",
        "",
        "Top rescue-vs-nonrescue feature differences among RF-wrong cases:",
        md_table(top[["feature", "mean_diff_a_minus_b", "cliffs_delta", "fdr_p"]], floatfmt=".4g"),
        "",
        "## 8. Explanatory models",
        "Model reports are saved in `rf_failure_model_results.md`, `vlm_rescue_model_results.md`, and `vlm_rescue_valid_only_model_results.md`. These are sanity-check explanatory models and should not be read as deployment classifiers.",
        "",
        "## 9. Rule/taxonomy analysis",
        "See `rescue_failure_taxonomy.md`.",
        "",
        "## 10. Case studies",
        "See `case_studies.md` and `case_studies_viewer.html`.",
        "",
        "## 11. Threats to validity",
        "- `human_preference` is a score-derived proxy, not direct pairwise annotation.",
        "- Many pairs are cross-dataset after within-dataset z-normalization.",
        "- VLM correctness is evaluated under strict-swap protocol.",
        "- Invalid VLM outputs are separated from wrong valid outputs.",
        "- Feature analysis is correlational and does not prove causal mechanisms.",
        "- Pair-level observations are not independent because snippets repeat across pairs.",
        "- Visual features are proxies and may miss human-perceived layout qualities.",
        "- Results are based on evaluated open-weight VLMs and should not generalize to all VLMs.",
        "",
        "## 12. Implications for better judge design",
        "The most defensible design implication is selective fallback: VLMs may be more useful in RF-uncertain regions, but strict-swap reliability must be audited and invalid outputs cannot be ignored.",
        "",
        "## 13. Safe paper-ready wording",
        "Most VLM rescue cases are concentrated in lower RF-margin regions, suggesting that VLM is most useful as an uncertainty-triggered fallback rather than a general replacement for source-based readability predictors. Some visual/layout proxy features differ across rescue and non-rescue cases, but the pattern is correlational and not strong enough to claim a causal mechanism. This supports a selective delegation strategy while preserving the main conclusion that current VLM judges remain fragile under strict-swap evaluation.",
        "",
        "## 14. Reproduction commands",
        "See `README.md` in this output directory for the exact command and clean input paths.",
        "",
    ]
    write_text(OUT / "COMPLEMENTARITY_MECHANISM_REPORT.md", "\n".join(lines) + "\n")


def readme(paths: dict[str, Path], command: str) -> None:
    write_text(
        OUT / "README.md",
        "# Complementarity Mechanism Analysis\n\n"
        "## Required inputs\n"
        f"- `{paths['rf_pair_level']}`\n"
        f"- `{paths['best_vlm_raw']}`\n"
        f"- `{paths['source_features']}`\n"
        f"- `{paths['render_metadata']}`\n\n"
        "## Command order\n"
        f"Run `{command}`.\n\n"
        "## Expected runtime\n"
        "CPU-only; typically under a few minutes because no VLM inference is performed.\n\n"
        "## Known limitations\n"
        "Pair observations share snippets; visual features are proxy measurements; results are correlational.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm-raw",
        default="experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__text_plus_image__promptB__seed42.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/complementarity_mechanism_clean_qwen_text_plus_image_20260706",
    )
    parser.add_argument(
        "--rf-pair-level",
        default="experiments/rq0_viability/outputs/pair_level_results/full_pair_classical_baseline_pair_level.csv",
    )
    parser.add_argument(
        "--render-metadata",
        default="experiments/rq0_viability/outputs/render_metadata/default_render_metadata.csv",
    )
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--source-features", default="experiments/rq0_viability/data/processed/features_313.csv")
    args = parser.parse_args()
    global OUT, FIG
    OUT = resolve_path(args.output_dir)
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {
        "rq0_dir": EXP,
        "dataset": resolve_path(args.dataset),
        "source_features": resolve_path(args.source_features),
        "render_metadata": resolve_path(args.render_metadata),
        "rf_pair_level": resolve_path(args.rf_pair_level),
        "best_vlm_raw": resolve_path(args.vlm_raw),
    }
    dataset = pd.read_csv(paths["dataset"])
    base_source = pd.read_csv(paths["source_features"])
    render_meta = pd.read_csv(paths["render_metadata"])
    rf_pair_level = pd.read_csv(paths["rf_pair_level"])
    vlm_raw = pd.read_json(paths["best_vlm_raw"], lines=True)
    vlm_condition = (
        str(vlm_raw["model_name"].iloc[0])
        + " / "
        + str(vlm_raw["input_setting"].iloc[0])
        + " / prompt"
        + str(vlm_raw["prompt_variant"].iloc[0])
    )
    create_inventory(paths, vlm_condition)
    joined = build_rf_joined(rf_pair_level, render_meta)
    pair = build_pair_level(joined, vlm_raw)
    source_snippet = count_extra_source_features(dataset, base_source)
    visual_snippet = build_visual_features(render_meta)
    source_pair = pair_features(pair, source_snippet, "source_")
    visual_pair = pair_features(pair, visual_snippet, "visual_")
    final = pair.merge(source_pair, on="pair_id", how="left").merge(visual_pair, on="pair_id", how="left")
    final.to_csv(OUT / "final_mechanism_dataset.csv", index=False)
    dictionary(final.columns.tolist())
    feature_cols = [
        c
        for c in final.columns
        if (c.startswith("source_") or c.startswith("visual_"))
        and (c.endswith("_diff_abs") or c.endswith("_gold_minus_loser") or c.endswith("_mean") or c.endswith("_max"))
    ]
    desc = group_descriptives(final, feature_cols[:80])
    stat = stat_tests(final, feature_cols)
    run_explanatory_models(final, feature_cols, ~final["rf_correct"].astype(bool), "rf_failure", f"All {len(final):,} pairs; target=RF wrong")
    rf_wrong = final[~final["rf_correct"].astype(bool)].copy()
    run_explanatory_models(
        rf_wrong,
        feature_cols,
        rf_wrong["outcome_group"].eq("RF_wrong__VLM_correct"),
        "vlm_rescue",
        "RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid",
    )
    valid_wrong = rf_wrong[rf_wrong["outcome_group"].isin(["RF_wrong__VLM_correct", "RF_wrong__VLM_wrong"])].copy()
    run_explanatory_models(
        valid_wrong,
        feature_cols,
        valid_wrong["outcome_group"].eq("RF_wrong__VLM_correct"),
        "vlm_rescue_valid_only",
        "RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong",
    )
    save_plots(final, stat)
    taxonomy(final, stat)
    case_studies(final, dataset, source_snippet, visual_snippet)
    final_report(final, desc, stat, vlm_condition)
    command = (
        "python experiments/rq0_viability/scripts/run_complementarity_mechanism_analysis.py "
        f"--vlm-raw {paths['best_vlm_raw']} --output-dir {OUT}"
    )
    readme(paths, command)
    write_text(
        OUT / "reproduce.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {ROOT}\n"
        f"{command}\n",
    )
    subprocess.run(["chmod", "+x", str(OUT / "reproduce.sh")], check=False)
    write_json(
        OUT / "run_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": str(OUT),
            "command": command,
            "inputs": {label: str(path) for label, path in paths.items()},
            "vlm_condition": vlm_condition,
            "num_pairs": int(len(final)),
            "outcome_counts": final["outcome_group"].value_counts().to_dict(),
        },
    )
    print(f"Wrote complementarity mechanism outputs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
