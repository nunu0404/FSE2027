#!/usr/bin/env python3
"""Statistical analysis and artifact generation for the true Gemma4-12B RQ2 reduced grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path("/ANON/experiment_root")
BASE_EXP = ROOT / "results/grounded_protocol_3lang_20260721"
OUT_DIR = ROOT / "rq2_eval_reduced"

BASE_GRID = "monokai_dark__fs20__wrap80__lnon"
RENDERING_CONDITIONS = [
    "monokai_dark__fs20__wrap80__lnon",
    "monokai_dark__fs20__wrap60__lnon",
    "monokai_dark__fs24__wrap80__lnon",
    "mono_light__fs20__wrap80__lnon",
]
PERTURBATION_CONDITIONS = [
    "no_indent",
    "no_blank_lines",
    "gaussian_sigma_4",
]


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def holm_adjust(p_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted < 0.05, adjusted


def cluster_meat(x: np.ndarray, residual: np.ndarray, groups: np.ndarray) -> np.ndarray:
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in pd.unique(groups):
        idx = groups == group
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    return meat


def snippet_cluster_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Any,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_snippets = np.array(sorted(set(df.snippet_i) | set(df.snippet_j)))
    n_snippets = len(unique_snippets)
    
    s_i = df.snippet_i.to_numpy()
    s_j = df.snippet_j.to_numpy()
    
    boot_stats = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_snippets, size=n_snippets, replace=True)
        counts = pd.Series(sampled).value_counts()
        w_i = np.array([counts.get(s, 0) for s in s_i])
        w_j = np.array([counts.get(s, 0) for s in s_j])
        weights = w_i + w_j
        if weights.sum() == 0:
            continue
        stat = stat_fn(df, weights)
        if np.isfinite(stat):
            boot_stats.append(stat)
            
    if len(boot_stats) < 100:
        return math.nan, math.nan
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def pair_calls(calls_df: pd.DataFrame, pairs_meta: pd.DataFrame) -> pd.DataFrame:
    p_meta = pairs_meta[["protocol_pair_id", "language", "snippet_i", "snippet_j", "human_preference"]].drop_duplicates()
    
    ab = calls_df[calls_df.order == "AB"].copy()
    ba = calls_df[calls_df.order == "BA"].copy()

    merged = pd.merge(
        ab,
        ba,
        on=["condition", "pair_id", "language"],
        suffixes=("_ab", "_ba"),
    )
    merged = pd.merge(merged, p_meta, left_on=["pair_id", "language"], right_on=["protocol_pair_id", "language"])

    parsed_both = (merged.parsed_choice_ab.isin(["A", "B"])) & (merged.parsed_choice_ba.isin(["A", "B"]))
    merged["parsed_both"] = parsed_both

    first_wins_ab = merged.parsed_choice_ab == "A"
    first_wins_ba = merged.parsed_choice_ba == "A"

    strict_valid = parsed_both & (first_wins_ab != first_wins_ba)
    merged["strict_valid"] = strict_valid

    winner = np.where(
        ~strict_valid,
        None,
        np.where(
            first_wins_ab,
            merged.snippet_first_ab,
            merged.snippet_second_ab,
        ),
    )
    merged["strict_winner"] = winner
    merged["strict_correct"] = strict_valid & (merged.strict_winner == merged.human_preference)

    merged["b_position"] = (merged.margin_ab + merged.margin_ba) / 2.0
    merged["c_content"] = (merged.margin_ab - merged.margin_ba) / 2.0

    merged["is_boundary"] = (merged.margin_ab == 0.0) | (merged.margin_ba == 0.0)
    merged["is_exact_tie"] = (merged.logit_tie_ab) | (merged.logit_tie_ba)

    c_sign = np.sign(merged.c_content.fillna(0.0))
    gold_sign = np.where(merged.human_preference == merged.snippet_first_ab, 1.0, -1.0)
    merged["debiased_correct"] = (c_sign == gold_sign).astype(float)
    merged.loc[c_sign == 0.0, "debiased_correct"] = 0.5

    merged["first_pos_ab"] = (merged.parsed_choice_ab == "A").astype(float)
    merged["first_pos_ba"] = (merged.parsed_choice_ba == "A").astype(float)

    return merged


def analyze_model(model_name: str, raw_path: Path, out_subdir: Path, pairs_meta: pd.DataFrame) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"[ANALYSIS] Running complete evaluation for {model_name}...")
    print(f"=======================================================")
    
    (out_subdir / "pair_level").mkdir(parents=True, exist_ok=True)
    (out_subdir / "summary").mkdir(parents=True, exist_ok=True)
    (out_subdir / "figures_input").mkdir(parents=True, exist_ok=True)

    calls = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    calls_df = pd.DataFrame(calls)
    print(f"[ANALYSIS] Loaded {len(calls_df)} raw calls from {raw_path}")

    parsed_calls = calls_df[calls_df.parsed_choice.notna()].copy()
    def row_matches(r):
        if pd.isna(r.logit_A) or pd.isna(r.logit_B):
            return False
        expected = "A" if r.logit_A >= r.logit_B else "B"
        return r.parsed_choice == expected
    parsed_calls["matches_expected_argmax"] = parsed_calls.apply(row_matches, axis=1)
    matches = int(parsed_calls.matches_expected_argmax.sum())
    total_parsed = len(parsed_calls)
    pass_rate = matches / total_parsed if total_parsed > 0 else 0.0
    print(f"[ANALYSIS] Agreement check (deterministic argmax including ties): {matches}/{total_parsed} ({pass_rate*100:.2f}%)")

    paired = pair_calls(calls_df, pairs_meta)
    print(f"[ANALYSIS] Built {len(paired)} paired rows across conditions.")

    pair_csv = out_subdir / "pair_level" / f"{model_name}_pair_level.csv"
    paired.to_csv(pair_csv, index=False)
    print(f"[ANALYSIS] Saved pair-level CSV to {pair_csv}")

    non_boundary = paired[paired.parsed_both & (~paired.is_boundary)]
    c_gt_b = non_boundary.c_content.abs() > non_boundary.b_position.abs()
    equiv_violations = int((non_boundary.strict_valid != c_gt_b).sum())
    print(f"[ANALYSIS] Equivalence violations (|c|>|b| != strict_valid on non-boundary): {equiv_violations}")

    summary_rows = []
    for (lang, cond), part in paired.groupby(["language", "condition"]):
        n = len(part)
        n_parsed = int(part.parsed_both.sum())
        n_valid = int(part.strict_valid.sum())
        n_correct = int(part.strict_correct.sum())
        n_boundary = int(part.is_boundary.sum())
        n_ties = int(part.is_exact_tie.sum())

        valid_acc = n_correct / n_valid if n_valid else math.nan
        eff_acc = n_correct / n
        swap_err = 1.0 - (n_valid / n)
        first_pos_rate = float((part.first_pos_ab.sum() + part.first_pos_ba.sum()) / (2 * n))
        debiased_acc = float(part.debiased_correct.mean())

        eff_lo, eff_hi = wilson(n_correct, n)
        val_lo, val_hi = wilson(n_correct, n_valid)
        b_gt_c_share = float((part.b_position.abs() > part.c_content.abs()).mean())

        summary_rows.append({
            "model": model_name,
            "language": lang,
            "condition": cond,
            "n_pairs": n,
            "n_parsed": n_parsed,
            "n_valid": n_valid,
            "n_correct": n_correct,
            "valid_accuracy": valid_acc,
            "valid_ci_low": val_lo,
            "valid_ci_high": val_hi,
            "effective_accuracy": eff_acc,
            "effective_ci_low": eff_lo,
            "effective_ci_high": eff_hi,
            "strict_swap_error": swap_err,
            "debiased_accuracy_D": debiased_acc,
            "first_position_rate": first_pos_rate,
            "n_boundaries": n_boundary,
            "n_exact_ties": n_ties,
            "share_abs_b_gt_abs_c": b_gt_c_share,
            "median_abs_b": float(part.b_position.abs().median()),
            "median_abs_c": float(part.c_content.abs().median()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_subdir / "summary" / f"{model_name}_summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[ANALYSIS] Saved summary metrics to {summary_csv}")

    contrast_rows = []
    base_pairs = paired[paired.condition == BASE_GRID]

    for lang in ["cuda", "java", "python"]:
        sub_base = base_pairs[base_pairs.language == lang].set_index("pair_id")
        val_base = float(summary_df[(summary_df.language == lang) & (summary_df.condition == BASE_GRID)].valid_accuracy.iloc[0])

        for cond in RENDERING_CONDITIONS[1:] + PERTURBATION_CONDITIONS:
            sub_cond = paired[(paired.language == lang) & (paired.condition == cond)].set_index("pair_id")
            common_idx = sub_base.index.intersection(sub_cond.index)
            b_part = sub_base.loc[common_idx]
            c_part = sub_cond.loc[common_idx]

            both = pd.DataFrame({
                "pair_id": common_idx,
                "snippet_i": b_part.snippet_i,
                "snippet_j": b_part.snippet_j,
                "base_correct": b_part.strict_correct.astype(int),
                "cond_correct": c_part.strict_correct.astype(int),
                "base_valid": b_part.strict_valid.astype(int),
                "cond_valid": c_part.strict_valid.astype(int),
            })

            delta_e = float(c_part.strict_correct.mean() - b_part.strict_correct.mean())
            delta_s = float((1.0 - c_part.strict_valid.mean()) - (1.0 - b_part.strict_valid.mean()))

            y_diff = both.cond_correct - both.base_correct
            x_mat = np.ones((len(both), 1))
            res = y_diff - y_diff.mean()
            meat_i = cluster_meat(x_mat, res.to_numpy(), both.snippet_i.to_numpy())
            meat_j = cluster_meat(x_mat, res.to_numpy(), both.snippet_j.to_numpy())
            var_diff = (meat_i[0, 0] + meat_j[0, 0]) / (len(both) ** 2)
            se_e = math.sqrt(max(0.0, var_diff))

            n01 = int(((both.base_correct == 0) & (both.cond_correct == 1)).sum())
            n10 = int(((both.base_correct == 1) & (both.cond_correct == 0)).sum())
            if n01 + n10 > 0:
                p = float(binomtest(n01, n01 + n10, 0.5).pvalue)
            else:
                p = 1.0

            val_cond = float(summary_df[(summary_df.language == lang) & (summary_df.condition == cond)].valid_accuracy.iloc[0])
            delta_val_acc = val_cond - val_base

            contrast_rows.append({
                "model": model_name,
                "language": lang,
                "contrast_category": "rendering" if cond in RENDERING_CONDITIONS else "perturbation",
                "baseline": BASE_GRID,
                "condition": cond,
                "n_pairs": len(both),
                "delta_effective_accuracy": delta_e,
                "delta_e_cluster_se": se_e,
                "delta_e_ci_low": delta_e - 1.959963984540054 * se_e,
                "delta_e_ci_high": delta_e + 1.959963984540054 * se_e,
                "delta_strict_swap_error": delta_s,
                "delta_valid_accuracy": delta_val_acc,
                "mcnemar_exact_p": p,
            })

    contrast_df = pd.DataFrame(contrast_rows)
    if len(contrast_df) > 0:
        reject, adj_p = holm_adjust(contrast_df.mcnemar_exact_p.tolist())
        contrast_df["holm_p"] = adj_p
        contrast_df["holm_reject_0_05"] = reject

    contrast_csv = out_subdir / "summary" / f"{model_name}_matched_contrasts.csv"
    contrast_df.to_csv(contrast_csv, index=False)
    print(f"[ANALYSIS] Saved matched contrasts to {contrast_csv}")

    order_vs_render_rows = []
    base_calls = calls_df[calls_df.condition == BASE_GRID].copy()

    for lang in ["cuda", "java", "python"]:
        sub_summary = summary_df[(summary_df.language == lang) & (summary_df.condition.isin(RENDERING_CONDITIONS))]
        swap_errors = sub_summary.strict_swap_error.tolist()
        median_swap = float(np.median(swap_errors))

        flips_total = []
        flips_valid_both = []

        sub_base_calls = base_calls[base_calls.language == lang].set_index(["pair_id", "order"])
        for cond in RENDERING_CONDITIONS[1:]:
            sub_cond_calls = calls_df[(calls_df.language == lang) & (calls_df.condition == cond)].set_index(["pair_id", "order"])
            common_keys = sub_base_calls.index.intersection(sub_cond_calls.index)
            b_c = sub_base_calls.loc[common_keys]
            c_c = sub_cond_calls.loc[common_keys]

            both_parsed = (b_c.parsed_choice.isin(["A", "B"])) & (c_c.parsed_choice.isin(["A", "B"]))
            b_c_p = b_c[both_parsed]
            c_c_p = c_c[both_parsed]
            flip = (b_c_p.parsed_choice != c_c_p.parsed_choice).mean()
            flips_total.append(float(flip))

            sub_base_pairs = paired[(paired.language == lang) & (paired.condition == BASE_GRID)].set_index("pair_id")
            sub_cond_pairs = paired[(paired.language == lang) & (paired.condition == cond)].set_index("pair_id")
            both_valid_ids = sub_base_pairs[sub_base_pairs.strict_valid].index.intersection(
                sub_cond_pairs[sub_cond_pairs.strict_valid].index
            )
            if len(both_valid_ids) > 0:
                b_v = sub_base_pairs.loc[both_valid_ids]
                c_v = sub_cond_pairs.loc[both_valid_ids]
                winner_flip = (b_v.strict_winner != c_v.strict_winner).mean()
                flips_valid_both.append(float(winner_flip))
            else:
                flips_valid_both.append(math.nan)

        median_flip = float(np.median(flips_total))
        median_valid_flip = float(np.median(flips_valid_both)) if flips_valid_both else math.nan

        order_vs_render_rows.append({
            "model": model_name,
            "language": lang,
            "median_within_condition_swap_error": median_swap,
            "swap_error_range": f"{min(swap_errors)*100:.1f}% - {max(swap_errors)*100:.1f}%",
            "median_rendering_flip_rate": median_flip,
            "flip_rate_range": f"{min(flips_total)*100:.1f}% - {max(flips_total)*100:.1f}%",
            "median_flip_rate_valid_both": median_valid_flip,
            "order_exceeds_rendering": bool(median_swap > median_flip),
        })

    order_vs_render_df = pd.DataFrame(order_vs_render_rows)
    order_vs_render_csv = out_subdir / "summary" / f"{model_name}_order_vs_rendering.csv"
    order_vs_render_df.to_csv(order_vs_render_csv, index=False)
    print(f"[ANALYSIS] Saved Order vs Rendering comparison to {order_vs_render_csv}")

    e1_holds = bool(order_vs_render_df.order_exceeds_rendering.dropna().all()) if len(order_vs_render_df.order_exceeds_rendering.dropna()) > 0 else None
    wrap_rows = contrast_df[contrast_df.condition == "monokai_dark__fs20__wrap60__lnon"] if len(contrast_df) > 0 else []
    e2_holds = bool(((wrap_rows.delta_effective_accuracy < 0) & (wrap_rows.delta_strict_swap_error > 0)).all()) if len(wrap_rows) > 0 else None
    indent_rows = contrast_df[contrast_df.condition == "no_indent"] if len(contrast_df) > 0 else []
    e3_holds = bool((indent_rows.delta_valid_accuracy.abs() < 0.05).all()) if len(indent_rows) > 0 else None

    expectations = {
        "E1_swap_error_exceeds_flip_rate": e1_holds,
        "E2_wrap80_better_than_wrap60": e2_holds,
        "E3_indent_removal_change_under_5pp": e3_holds,
    }
    exp_json = out_subdir / "summary" / f"{model_name}_expectations.json"
    exp_json.write_text(json.dumps(expectations, indent=2), encoding="utf-8")

    fig_input = out_subdir / "figures_input" / f"{model_name}_fig6_input.csv"
    order_vs_render_df.to_csv(fig_input, index=False)

    return {
        "model": model_name,
        "pass_rate": pass_rate,
        "equiv_violations": equiv_violations,
        "summary": summary_df,
        "contrasts": contrast_df,
        "order_vs_render": order_vs_render_df,
        "expectations": expectations,
    }


def create_handoff_bundle():
    print("\n[HANDOFF] Packaging Gemma4-12B results into ~/fse2027_handoff_rq2_gemma4.zip...")
    bundle_dir = OUT_DIR / "handoff_gemma4"
    if bundle_dir.exists():
        import shutil
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    import shutil
    (bundle_dir / "summary").mkdir(parents=True)
    (bundle_dir / "figures_input").mkdir(parents=True)
    (bundle_dir / "manifests").mkdir(parents=True)

    for f in (OUT_DIR / "gemma4/summary").glob("*"):
        shutil.copy2(f, bundle_dir / "summary" / f.name)
    for f in (OUT_DIR / "gemma4/figures_input").glob("*"):
        shutil.copy2(f, bundle_dir / "figures_input" / f.name)
    shutil.copy2(OUT_DIR / "manifests/gemma4_12b_reduced_manifest.json", bundle_dir / "manifests/gemma4_12b_reduced_manifest.json")

    # Generate SOURCES.md
    def get_info(p):
        size = p.stat().st_size
        with open(p, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
        except Exception:
            lines = None
        return size, sha, lines

    files = sorted([p for p in bundle_dir.rglob("*") if p.is_file() and p.name != "SOURCES.md"])
    lines_md = [
        "# SOURCES.md: FSE2027 RQ2 Reduced Grid Gemma4-12B Handoff Bundle",
        "",
        "> **생성 일시**: 2026-09-12 (KST)",
        "> **대상 모델**: `google/gemma-4-12B-it` (리비전: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`)",
        "",
        "## 1. 파일 목록 및 SHA-256 체크섬",
        "",
        "| 파일 경로 | 크기 (Bytes) | 행 수 (Lines) | SHA-256 체크섬 | 설명 |",
        "| :--- | :---: | :---: | :--- | :--- |",
    ]
    for p in files:
        rel_p = p.relative_to(bundle_dir)
        size, sha, lines = get_info(p)
        lines_md.append(f"| `{rel_p}` | {size} | {lines} | `{sha}` | |")

    sources_path = bundle_dir / "SOURCES.md"
    sources_path.write_text("\n".join(lines_md) + "\n", encoding="utf-8")

    # Zip
    zip_path = Path("/ANON/home/fse2027_handoff_rq2_gemma4.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                zipf.write(p, p.relative_to(bundle_dir))
    print(f"[HANDOFF] Successfully created {zip_path} ({zip_path.stat().st_size} bytes)")


def main():
    pairs_meta = pd.read_csv(BASE_EXP / "data/pairs_seed42_grounded.csv")
    gemma4_raw = OUT_DIR / "gemma4/raw_calls/gemma4_12b_reduced_raw.jsonl"
    if gemma4_raw.exists():
        res = analyze_model("Gemma4-12B", gemma4_raw, OUT_DIR / "gemma4", pairs_meta)
        print("\n--- Summary Table ---")
        print(res["summary"].to_string())
        print("\n--- Order vs Rendering Table ---")
        print(res["order_vs_render"].to_string())
        print("\n--- Expectations ---")
        print(json.dumps(res["expectations"], indent=2))
        create_handoff_bundle()
    else:
        print(f"[ERROR] Raw file {gemma4_raw} does not exist yet.")


if __name__ == "__main__":
    main()
