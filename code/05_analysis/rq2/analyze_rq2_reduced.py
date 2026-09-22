#!/usr/bin/env python3
"""Statistical analysis and artifact generation for the RQ2 reduced grid (Gemma, InternVL3.5, and Qwen Anchor)."""

from __future__ import annotations

import argparse
import json
import math
import re
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
    """Cluster bootstrap resampling on snippet endpoints."""
    rng = np.random.default_rng(seed)
    unique_snippets = np.array(sorted(set(df.snippet_i) | set(df.snippet_j)))
    n_snippets = len(unique_snippets)
    
    # Pre-map pairs to snippets
    s_i = df.snippet_i.to_numpy()
    s_j = df.snippet_j.to_numpy()
    
    boot_stats = []
    for _ in range(n_boot):
        sampled_set = set(rng.choice(unique_snippets, size=n_snippets, replace=True))
        # Keep pairs where both endpoints are in sampled set
        mask = np.isin(s_i, list(sampled_set)) & np.isin(s_j, list(sampled_set))
        if mask.sum() < 10:
            continue
        sub_df = df[mask]
        val = stat_fn(sub_df)
        if np.isfinite(val):
            boot_stats.append(val)
            
    if len(boot_stats) < 100:
        return math.nan, math.nan
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def pair_calls(calls: pd.DataFrame, pairs_meta: pd.DataFrame) -> pd.DataFrame:
    meta = pairs_meta.drop(columns=["pair_id"]).rename(columns={"protocol_pair_id": "pair_id"})
    keep = [
        "pair_id", "z_i", "z_j", "human_score_i_z", "human_score_j_z",
        "human_preference", "dataset_name_i", "dataset_name_j", "preference_score_basis",
    ]
    rows = []
    group = ["experiment", "condition", "model", "model_revision", "language", "pair_id"]
    for key, part in calls.groupby(group, sort=False, dropna=False):
        by_order = {r.order: r for r in part.itertuples(index=False)}
        if set(by_order) != {"AB", "BA"}:
            continue
        ab, ba = by_order["AB"], by_order["BA"]

        def chosen_snippet(row):
            if row.parsed_choice == "A":
                return row.snippet_first
            if row.parsed_choice == "B":
                return row.snippet_second
            return None

        selected_ab, selected_ba = chosen_snippet(ab), chosen_snippet(ba)
        parsed = selected_ab is not None and selected_ba is not None
        strict_valid = parsed and (selected_ab == selected_ba)
        gold = ab.snippet_i if ab.gold_side == "first" else ab.snippet_j

        margin_ab = float(ab.margin) if ab.margin is not None else 0.0
        margin_ba = float(ba.margin) if ba.margin is not None else 0.0
        b_val = (margin_ab + margin_ba) / 2.0
        c_val = (margin_ab - margin_ba) / 2.0

        is_boundary = abs(abs(c_val) - abs(b_val)) < 1e-6
        is_exact_tie = (c_val == 0.0)

        rows.append({
            **dict(zip(group, key)),
            "source_pair_id": ab.source_pair_id,
            "difficulty_rank": ab.difficulty_rank,
            "abs_z_diff": float(ab.abs_z_diff),
            "snippet_i": ab.snippet_i,
            "snippet_j": ab.snippet_j,
            "gold_snippet": gold,
            "selected_ab": selected_ab,
            "selected_ba": selected_ba,
            "parsed_both": parsed,
            "strict_valid": strict_valid,
            "strict_correct": bool(strict_valid and selected_ab == gold),
            "margin_ab": margin_ab,
            "margin_ba": margin_ba,
            "b_position": b_val,
            "c_content": c_val,
            "is_boundary": is_boundary,
            "is_exact_tie": is_exact_tie,
            "first_pos_ab": (ab.parsed_choice == "A"),
            "first_pos_ba": (ba.parsed_choice == "A"),
        })

    paired = pd.DataFrame(rows).merge(meta[keep], on="pair_id", how="left", validate="many_to_one")
    paired["effective_correct"] = paired.strict_correct.astype(int)
    paired["debiased_selected"] = np.select(
        [paired.c_content > 0, paired.c_content < 0],
        [paired.snippet_i, paired.snippet_j],
        default=None,
    )
    paired["debiased_correct"] = (~paired.is_exact_tie) & (paired.debiased_selected == paired.human_preference)
    return paired


def analyze_model(model_name: str, raw_path: Path, out_subdir: Path, pairs_meta: pd.DataFrame) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"[ANALYSIS] Running analysis for {model_name}...")
    print(f"=======================================================")

    calls = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    calls_df = pd.DataFrame(calls)
    print(f"[ANALYSIS] Loaded {len(calls_df)} raw calls.")

    # 1. generated-verdict vs logit-sign agreement check
    parsed_calls = calls_df[calls_df.parsed_choice.notna()].copy()
    def row_matches(r):
        if pd.isna(r.logit_A) or pd.isna(r.logit_B):
            return False
        # torch.argmax returns index 0 (A) on exact tie
        expected = "A" if r.logit_A >= r.logit_B else "B"
        return r.parsed_choice == expected
    parsed_calls["matches_expected_argmax"] = parsed_calls.apply(row_matches, axis=1)
    matches = int(parsed_calls.matches_expected_argmax.sum())
    total_parsed = len(parsed_calls)
    pass_rate = matches / total_parsed if total_parsed > 0 else 0.0
    print(f"[ANALYSIS] Agreement check (deterministic argmax including ties): {matches}/{total_parsed} ({pass_rate*100:.2f}%)")

    # 2. Build paired rows
    paired = pair_calls(calls_df, pairs_meta)
    print(f"[ANALYSIS] Built {len(paired)} paired rows across conditions.")

    # Save pair-level CSV
    pair_csv = out_subdir / "pair_level" / f"{model_name}_pair_level.csv"
    paired.to_csv(pair_csv, index=False)
    print(f"[ANALYSIS] Saved pair-level CSV to {pair_csv}")

    # 3. Equivalence check: on parsed non-boundary pairs, strict_valid <=> |c| > |b|
    non_boundary = paired[paired.parsed_both & (~paired.is_boundary)]
    c_gt_b = non_boundary.c_content.abs() > non_boundary.b_position.abs()
    equiv_violations = int((non_boundary.strict_valid != c_gt_b).sum())
    print(f"[ANALYSIS] Equivalence violations (|c|>|b| != strict_valid on non-boundary): {equiv_violations}")

    # 4. Summary metrics table per language and condition
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

        # Share of |b| > |c|
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

    # 5. Section 5.2 / 6a, 6b: Matched Contrast Analysis (McNemar + Cluster Robust SE + Bootstrap)
    contrast_rows = []
    for lang, block in paired.groupby("language"):
        wide_eff = block.pivot(index="pair_id", columns="condition", values="effective_correct")
        wide_valid = block.pivot(index="pair_id", columns="condition", values="strict_valid")
        wide_val_acc = block.pivot(index="pair_id", columns="condition", values="strict_correct")

        # Baseline condition
        if BASE_GRID not in wide_eff:
            continue

        for cond in RENDERING_CONDITIONS[1:] + PERTURBATION_CONDITIONS:
            if cond not in wide_eff:
                continue

            both = wide_eff[[BASE_GRID, cond]].dropna().astype(int)
            base_only = int(((both[BASE_GRID] == 1) & (both[cond] == 0)).sum())
            cond_only = int(((both[BASE_GRID] == 0) & (both[cond] == 1)).sum())
            discordant = base_only + cond_only
            p = binomtest(min(base_only, cond_only), discordant, 0.5).pvalue if discordant else 1.0

            # Delta E
            delta_e = float((both[cond] - both[BASE_GRID]).mean())

            # Cluster robust SE
            pair_meta = block.drop_duplicates("pair_id").set_index("pair_id").loc[both.index]
            diff = (both[cond] - both[BASE_GRID]).to_numpy(dtype=float)
            res = diff - diff.mean()
            x = np.ones((len(diff), 1))
            g1 = pair_meta.snippet_i.astype(str).to_numpy()
            g2 = pair_meta.snippet_j.astype(str).to_numpy()
            g12 = np.char.add(np.char.add(g1, "||"), g2)
            meat = cluster_meat(x, res, g1) + cluster_meat(x, res, g2) - cluster_meat(x, res, g12)
            se_e = math.sqrt(max(0.0, float(meat[0, 0]))) / len(diff)

            # Delta S (Strict Swap Error delta = -(Valid rate delta))
            valid_both = wide_valid[[BASE_GRID, cond]].dropna().astype(int)
            delta_s = float((1 - valid_both[cond]).mean() - (1 - valid_both[BASE_GRID]).mean())

            # Change in valid accuracy (only for valid pairs in both, or separate)
            val_base = float(summary_df[(summary_df.language == lang) & (summary_df.condition == BASE_GRID)].valid_accuracy.iloc[0])
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
        # Apply Holm correction per model x category
        for cat in ["rendering", "perturbation"]:
            mask = contrast_df.contrast_category == cat
            if mask.any():
                reject, adj_p = holm_adjust(contrast_df.loc[mask, "mcnemar_exact_p"].tolist())
                contrast_df.loc[mask, "holm_p"] = adj_p
                contrast_df.loc[mask, "holm_reject_0_05"] = reject
    contrast_csv = out_subdir / "summary" / f"{model_name}_matched_contrasts.csv"
    contrast_df.to_csv(contrast_csv, index=False)
    print(f"[ANALYSIS] Saved matched contrast tests to {contrast_csv}")

    # 6. Section 6.1: Order vs Rendering Comparison
    order_vs_render_rows = []
    calls_rend = calls_df[calls_df.condition.isin(RENDERING_CONDITIONS)]
    
    for lang, lang_calls in calls_rend.groupby("language"):
        # Within-condition swap error (4 conditions)
        swap_errors = []
        for cond in RENDERING_CONDITIONS:
            part = paired[(paired.language == lang) & (paired.condition == cond)]
            if len(part) > 0:
                sw_err = 1.0 - (part.strict_valid.sum() / len(part))
                swap_errors.append(sw_err)

        # 6 pairs of rendering conditions:
        flip_rates_all = []
        flip_rates_valid_both = []
        for i in range(len(RENDERING_CONDITIONS)):
            for j in range(i + 1, len(RENDERING_CONDITIONS)):
                c1, c2 = RENDERING_CONDITIONS[i], RENDERING_CONDITIONS[j]
                for ord_val in ["AB", "BA"]:
                    sub1 = lang_calls[(lang_calls.condition == c1) & (lang_calls.order == ord_val)].set_index("pair_id")
                    sub2 = lang_calls[(lang_calls.condition == c2) & (lang_calls.order == ord_val)].set_index("pair_id")
                    common = sub1.index.intersection(sub2.index)
                    if len(common) > 0:
                        p1 = sub1.loc[common, "parsed_choice"]
                        p2 = sub2.loc[common, "parsed_choice"]
                        valid_mask = p1.notna() & p2.notna()
                        if valid_mask.any():
                            flips = (p1[valid_mask] != p2[valid_mask]).mean()
                            flip_rates_all.append(flips)

                        # Restricted to pairs valid under both renderings
                        part1 = paired[(paired.language == lang) & (paired.condition == c1)].set_index("pair_id")
                        part2 = paired[(paired.language == lang) & (paired.condition == c2)].set_index("pair_id")
                        val_both_pairs = part1.index[part1.strict_valid & part2.strict_valid]
                        sub_val1 = sub1.loc[sub1.index.intersection(val_both_pairs), "parsed_choice"]
                        sub_val2 = sub2.loc[sub2.index.intersection(val_both_pairs), "parsed_choice"]
                        if len(sub_val1) > 0:
                            val_flips = (sub_val1 != sub_val2).mean()
                            flip_rates_valid_both.append(val_flips)

        median_swap_err = float(np.median(swap_errors)) if swap_errors else math.nan
        median_flip_rate = float(np.median(flip_rates_all)) if flip_rates_all else math.nan
        median_flip_valid_both = float(np.nanmedian(flip_rates_valid_both)) if flip_rates_valid_both else math.nan

        order_vs_render_rows.append({
            "model": model_name,
            "language": lang,
            "median_within_condition_swap_error": median_swap_err,
            "swap_error_range": f"{min(swap_errors)*100:.1f}% - {max(swap_errors)*100:.1f}%" if swap_errors else "N/A",
            "median_rendering_flip_rate": median_flip_rate,
            "flip_rate_range": f"{min(flip_rates_all)*100:.1f}% - {max(flip_rates_all)*100:.1f}%" if flip_rates_all else "N/A",
            "median_flip_rate_valid_both": median_flip_valid_both,
            "order_exceeds_rendering": bool(median_swap_err > median_flip_rate) if (np.isfinite(median_swap_err) and np.isfinite(median_flip_rate)) else None,
        })

    order_vs_render_df = pd.DataFrame(order_vs_render_rows)
    order_vs_render_csv = out_subdir / "summary" / f"{model_name}_order_vs_rendering.csv"
    order_vs_render_df.to_csv(order_vs_render_csv, index=False)
    print(f"[ANALYSIS] Saved Order vs Rendering comparison to {order_vs_render_csv}")

    # 7. Evaluate Expectations E1, E2, E3
    print("\n--- Prespecified Expectations Evaluation ---")
    e1_holds = bool(order_vs_render_df.order_exceeds_rendering.dropna().all()) if len(order_vs_render_df.order_exceeds_rendering.dropna()) > 0 else None
    print(f"E1: Swap error > Rendering flip rate in every language -> {e1_holds}")

    wrap_rows = contrast_df[contrast_df.condition == "monokai_dark__fs20__wrap60__lnon"] if len(contrast_df) > 0 else []
    e2_holds = bool(((wrap_rows.delta_effective_accuracy < 0) & (wrap_rows.delta_strict_swap_error > 0)).all()) if len(wrap_rows) > 0 else None
    print(f"E2: Wrap 80 raises E and lowers S relative to Wrap 60 in every language -> {e2_holds}")

    indent_rows = contrast_df[contrast_df.condition == "no_indent"] if len(contrast_df) > 0 else []
    e3_holds = bool((indent_rows.delta_valid_accuracy.abs() < 0.05).all()) if len(indent_rows) > 0 else None
    print(f"E3: Indentation removal changes valid accuracy by < 5 pp in every language -> {e3_holds}")

    expectations = {
        "E1_swap_error_exceeds_flip_rate": e1_holds,
        "E2_wrap80_better_than_wrap60": e2_holds,
        "E3_indent_removal_change_under_5pp": e3_holds,
    }
    exp_json = out_subdir / "summary" / f"{model_name}_expectations.json"
    exp_json.write_text(json.dumps(expectations, indent=2), encoding="utf-8")

    # Figure input CSV
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


def analyze_anchor_rerun(pairs_meta: pd.DataFrame) -> None:
    print(f"\n=======================================================")
    print(f"[ANCHOR-ANALYSIS] Analyzing Qwen2.5-VL-7B Baseline Anchor Re-run...")
    print(f"=======================================================")

    raw_path = OUT_DIR / "anchor_qwen/raw_calls/qwen25_vl_7b_baseline_anchor_raw.jsonl"
    if not raw_path.exists():
        print("[ANCHOR-ANALYSIS] Anchor raw file does not exist yet.")
        return

    calls = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    calls_df = pd.DataFrame(calls)
    if len(calls_df) == 0:
        return

    paired = pair_calls(calls_df, pairs_meta)
    anchor_rows = []
    
    # Original RQ2 numbers from A_GRID_RESULTS.csv
    orig_path = BASE_EXP / "analysis/ab/A_GRID_RESULTS.csv"
    orig_df = pd.read_csv(orig_path) if orig_path.exists() else None

    for lang, part in paired.groupby("language"):
        n = len(part)
        n_valid = int(part.strict_valid.sum())
        n_correct = int(part.strict_correct.sum())

        val_acc = n_correct / n_valid if n_valid else math.nan
        eff_acc = n_correct / n
        swap_err = 1.0 - (n_valid / n)

        # Get original values
        orig_val = math.nan
        orig_eff = math.nan
        orig_swap = math.nan
        if orig_df is not None:
            sub_orig = orig_df[
                (orig_df.model == "Qwen/Qwen2.5-VL-7B-Instruct") &
                (orig_df.language == lang) &
                (orig_df.condition == BASE_GRID)
            ]
            if len(sub_orig) > 0:
                orig_val = float(sub_orig.valid_accuracy.iloc[0])
                orig_eff = float(sub_orig.effective_accuracy.iloc[0])
                orig_swap = float(sub_orig.strict_swap_error.iloc[0])

        anchor_rows.append({
            "model": "Qwen2.5-VL-7B-Instruct",
            "language": lang,
            "condition": BASE_GRID,
            "n_pairs": n,
            "new_valid_accuracy": val_acc,
            "orig_valid_accuracy": orig_val,
            "valid_acc_gap_pp": (val_acc - orig_val) * 100 if np.isfinite(orig_val) else math.nan,
            "new_effective_accuracy": eff_acc,
            "orig_effective_accuracy": orig_eff,
            "effective_acc_gap_pp": (eff_acc - orig_eff) * 100 if np.isfinite(orig_eff) else math.nan,
            "new_strict_swap_error": swap_err,
            "orig_strict_swap_error": orig_swap,
            "swap_err_gap_pp": (swap_err - orig_swap) * 100 if np.isfinite(orig_swap) else math.nan,
        })

    anchor_df = pd.DataFrame(anchor_rows)
    anchor_csv = OUT_DIR / "anchor_qwen" / "qwen25_anchor_vs_original.csv"
    anchor_df.to_csv(anchor_csv, index=False)
    print(f"[ANCHOR-ANALYSIS] Saved Anchor comparison to {anchor_csv}")
    print(anchor_df.to_string(index=False))


def main():
    pairs_meta = pd.read_csv(BASE_EXP / "data/pairs_seed42_grounded.csv")

    # Gemma
    gemma_raw = OUT_DIR / "gemma/raw_calls/gemma4_12b_reduced_raw.jsonl"
    if gemma_raw.exists():
        analyze_model("Gemma4-12B", gemma_raw, OUT_DIR / "gemma", pairs_meta)

    # InternVL3.5
    internvl_raw = OUT_DIR / "internvl3_5/raw_calls/internvl3_5_8b_reduced_raw.jsonl"
    if internvl_raw.exists():
        analyze_model("InternVL3.5-8B", internvl_raw, OUT_DIR / "internvl3_5", pairs_meta)

    # Anchor
    analyze_anchor_rerun(pairs_meta)

    print("\n[ALL ANALYSIS COMPLETE]")


if __name__ == "__main__":
    main()
