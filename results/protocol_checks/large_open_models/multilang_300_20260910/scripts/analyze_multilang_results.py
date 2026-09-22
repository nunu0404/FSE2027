#!/usr/bin/env python3
"""Statistical Analysis for Multi-Language Reliability Benchmark (Java, Python, CUDA)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path("/ANON/experiment_root/results/multilang_cross_family_3model_python_cuda_java_20260910")
PAIR_CSV = ROOT_DIR / "pair_ids_multilang.csv"

MODELS = [
    ("qwen", "Qwen2.5-VL-32B-Instruct", ROOT_DIR / "qwen25_vl_32b_multilang_raw.jsonl"),
    ("gemma", "gemma-3-27b-it", ROOT_DIR / "gemma3_27b_multilang_raw.jsonl"),
    ("mistral", "Mistral-Small-3.1-24B-Instruct-2503", ROOT_DIR / "mistral_small_31_24b_multilang_raw.jsonl"),
]

N_BOOTSTRAP = 10000
RNG_SEED = 42


def load_data():
    pairs_meta = pd.read_csv(PAIR_CSV).set_index("pair_id")
    pair_rows = []

    for m_key, m_name, raw_path in MODELS:
        calls = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    calls[(rec["pair_id"], rec["order"])] = rec

        for pair_id, meta in pairs_meta.iterrows():
            ab = calls.get((pair_id, "AB"))
            ba = calls.get((pair_id, "BA"))

            if ab is None or ba is None:
                continue

            ab_choice = ab.get("actual_choice_snippet")
            ba_choice = ba.get("actual_choice_snippet")
            ab_fail = ab.get("parse_failure", True)
            ba_fail = ba.get("parse_failure", True)
            pref = meta["human_preference"]
            lang = meta["language"]
            diff = meta["difficulty"]

            is_valid = (not ab_fail) and (not ba_fail) and (ab_choice is not None) and (ab_choice == ba_choice)
            is_correct = is_valid and (ab_choice == pref)

            pair_rows.append({
                "model_key": m_key,
                "model_name": m_name,
                "pair_id": pair_id,
                "language": lang,
                "difficulty": diff,
                "snippet_x_id": meta["snippet_x_id"],
                "snippet_y_id": meta["snippet_y_id"],
                "human_preference": pref,
                "ab_verdict": ab.get("parsed_verdict"),
                "ba_verdict": ba.get("parsed_verdict"),
                "ab_choice": ab_choice,
                "ba_choice": ba_choice,
                "ab_parse_fail": ab_fail,
                "ba_parse_fail": ba_fail,
                "is_valid_strict_swap": is_valid,
                "is_correct_and_valid": is_correct,
            })

    return pd.DataFrame(pair_rows)


def run_bootstrap_metrics(df_sub, rng):
    all_snippets = list(set(df_sub["snippet_x_id"]) | set(df_sub["snippet_y_id"]))
    n_snippets = len(all_snippets)
    n_pairs = len(df_sub)

    boot_valid_acc = []
    boot_eff_acc = []
    boot_swap_err = []

    for _ in range(N_BOOTSTRAP):
        sample_snips = set(rng.choice(all_snippets, size=n_snippets, replace=True))
        mask = df_sub["snippet_x_id"].isin(sample_snips) & df_sub["snippet_y_id"].isin(sample_snips)
        sample = df_sub[mask]
        if len(sample) == 0:
            continue

        v_cnt = sample["is_valid_strict_swap"].sum()
        c_cnt = sample["is_correct_and_valid"].sum()
        tot = len(sample)

        boot_swap_err.append(1.0 - (v_cnt / tot))
        boot_eff_acc.append(c_cnt / tot)
        if v_cnt > 0:
            boot_valid_acc.append(c_cnt / v_cnt)

    def ci(arr):
        if not arr:
            return (np.nan, np.nan)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return {
        "swap_err_ci": ci(boot_swap_err),
        "eff_acc_ci": ci(boot_eff_acc),
        "valid_acc_ci": ci(boot_valid_acc),
    }


def main():
    print("=== Analyzing Multi-Language Evaluation Results ===")
    df = load_data()
    pair_csv_out = ROOT_DIR / "pair_level_multilang_results.csv"
    df.to_csv(pair_csv_out, index=False)
    print(f"Saved {len(df)} pair-level rows to {pair_csv_out}")

    rng = np.random.default_rng(RNG_SEED)

    # 1. Aggregate metrics by Model and Language
    records = []
    print("\nCalculating metrics and 10,000 cluster bootstraps...")
    for (m_key, m_name), m_group in df.groupby(["model_key", "model_name"]):
        # Overall
        groups = [("ALL", m_group)] + [(lang, l_group) for lang, l_group in m_group.groupby("language")]
        for lang, sub in groups:
            tot = len(sub)
            v_cnt = int(sub["is_valid_strict_swap"].sum())
            c_cnt = int(sub["is_correct_and_valid"].sum())
            fails = int(sub["ab_parse_fail"].sum() + sub["ba_parse_fail"].sum())

            eff_acc = c_cnt / tot
            swap_err = 1.0 - (v_cnt / tot)
            valid_acc = (c_cnt / v_cnt) if v_cnt > 0 else 0.0

            boot_res = run_bootstrap_metrics(sub, rng)

            records.append({
                "model_key": m_key,
                "model_name": m_name,
                "language": lang,
                "total_pairs": tot,
                "valid_pairs": v_cnt,
                "correct_and_valid": c_cnt,
                "strict_swap_error": swap_err,
                "swap_err_ci_low": boot_res["swap_err_ci"][0],
                "swap_err_ci_high": boot_res["swap_err_ci"][1],
                "valid_accuracy": valid_acc,
                "valid_acc_ci_low": boot_res["valid_acc_ci"][0],
                "valid_acc_ci_high": boot_res["valid_acc_ci"][1],
                "effective_accuracy": eff_acc,
                "eff_acc_ci_low": boot_res["eff_acc_ci"][0],
                "eff_acc_ci_high": boot_res["eff_acc_ci"][1],
                "parse_failures": fails,
            })

    agg_df = pd.DataFrame(records)
    agg_out = ROOT_DIR / "aggregate_metrics_multilang.csv"
    agg_df.to_csv(agg_out, index=False)
    print(f"Saved aggregate metrics to {agg_out}")

    # 2. Paired model comparisons overall and per language
    p_records = []
    comparisons = [
        ("qwen", "gemma", "Qwen2.5-VL-32B vs Gemma 3-27B"),
        ("qwen", "mistral", "Qwen2.5-VL-32B vs Mistral Small 3.1-24B"),
        ("gemma", "mistral", "Gemma 3-27B vs Mistral Small 3.1-24B"),
    ]

    piv = df.pivot(index=["pair_id", "language", "snippet_x_id", "snippet_y_id"], columns="model_key", values=["is_valid_strict_swap", "is_correct_and_valid"]).reset_index()

    for k1, k2, label in comparisons:
        for lang in ["ALL", "java", "python", "cuda"]:
            sub_piv = piv if lang == "ALL" else piv[piv["language"] == lang]
            n = len(sub_piv)

            eff_diff = (sub_piv[("is_correct_and_valid", k1)].mean() - sub_piv[("is_correct_and_valid", k2)].mean()) * 100
            swap_diff = ((1 - sub_piv[("is_valid_strict_swap", k1)].mean()) - (1 - sub_piv[("is_valid_strict_swap", k2)].mean())) * 100

            b = int(((sub_piv[("is_valid_strict_swap", k1)] == True) & (sub_piv[("is_valid_strict_swap", k2)] == False)).sum())
            c = int(((sub_piv[("is_valid_strict_swap", k1)] == False) & (sub_piv[("is_valid_strict_swap", k2)] == True)).sum())
            mcnemar_res = stats.binomtest(min(b, c), b + c, p=0.5) if (b + c) > 0 else None
            p_val = mcnemar_res.pvalue if mcnemar_res else 1.0

            p_records.append({
                "comparison": label,
                "language": lang,
                "pairs": n,
                "eff_acc_diff_pp": round(eff_diff, 2),
                "swap_error_diff_pp": round(swap_diff, 2),
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_raw_p": p_val,
            })

    comp_df = pd.DataFrame(p_records)
    comp_out = ROOT_DIR / "paired_comparisons_multilang.csv"
    comp_df.to_csv(comp_out, index=False)
    print(f"Saved comparisons to {comp_out}")

    print("\n=== Multi-Language Analysis Complete! ===")

if __name__ == "__main__":
    main()
