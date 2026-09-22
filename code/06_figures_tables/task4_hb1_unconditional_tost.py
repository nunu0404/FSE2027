import os
import pandas as pd
import numpy as np
from scipy.stats import t, norm

print("=== Task 4: H-B1 (No-Indent) Power, Unconditional Effect & TOST ===")

# Load data sources
# 1. Gemma-3, InternVL3.5, Gemma4
df_g3 = pd.read_csv("rq2_eval_reduced/gemma/summary/Gemma-3-12B_summary_metrics.csv")
df_i35 = pd.read_csv("rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv")
df_g4 = pd.read_csv("rq2_eval_reduced/gemma4/summary/Gemma4-12B_summary_metrics.csv")

# 2. Qwen2.5 & InternVL3
df_ab = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/A_GRID_RESULTS.csv")
df_b = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/B_PERTURBATION_RESULTS.csv")

# Also load pair-level files to compute paired differences and standard errors:
pair_files = {
    "Gemma-3-12B": "rq2_eval_reduced/gemma/pair_level/Gemma-3-12B_pair_level.csv",
    "InternVL3.5-8B": "rq2_eval_reduced/internvl3_5/pair_level/InternVL3.5-8B_pair_level.csv",
    "Gemma4-12B": "rq2_eval_reduced/gemma4/pair_level/Gemma4-12B_pair_level.csv",
}
# Qwen2.5 & InternVL3 pair-level:
# results/grounded_protocol_3lang_20260721/analysis/ab/A_B_PAIR_LEVEL.csv

# Let's inspect valid pair counts across all model x language x condition cells:
print("\n--- 1. Valid Pair Sample Sizes Across Cells (Looking for the 89-pair cell) ---")
all_summaries = []

# Collect from g3, i35, g4:
for df_m, mname in [(df_g3, "Gemma-3-12B"), (df_i35, "InternVL3.5-8B"), (df_g4, "Gemma4-12B")]:
    for _, r in df_m.iterrows():
        all_summaries.append({
            "model": mname,
            "language": r["language"],
            "condition": r["condition"],
            "n_pairs": r["n_pairs"],
            "n_valid": r["n_valid"],
            "n_correct": r["n_correct"],
            "valid_accuracy": r["valid_accuracy"],
            "effective_accuracy": r["effective_accuracy"],
            "strict_swap_error": r["strict_swap_error"]
        })

# Collect from df_b (Qwen2.5 & InternVL3):
for _, r in df_b.iterrows():
    all_summaries.append({
        "model": r["model"],
        "language": r["language"],
        "condition": r["condition"],
        "n_pairs": r["n_pairs"],
        "n_valid": r["n_valid"],
        "n_correct": r["n_correct"],
        "valid_accuracy": r["valid_accuracy"],
        "effective_accuracy": r["effective_accuracy"],
        "strict_swap_error": r["strict_swap_error"]
    })

# Also from df_ab (for baseline):
for _, r in df_ab[df_ab["condition"] == "monokai_dark__fs20__wrap80__lnon"].iterrows():
    all_summaries.append({
        "model": r["model"],
        "language": r["language"],
        "condition": "baseline_grid",
        "n_pairs": r["n_pairs"],
        "n_valid": r["n_valid"],
        "n_correct": r["n_correct"],
        "valid_accuracy": r["valid_accuracy"],
        "effective_accuracy": r["effective_accuracy"],
        "strict_swap_error": r["strict_swap_error"]
    })

sum_df = pd.DataFrame(all_summaries)
# Check min n_valid:
print("Cells with smallest n_valid:")
print(sum_df.sort_values("n_valid")[["model", "language", "condition", "n_valid", "n_pairs", "valid_accuracy"]].head(10).to_string())

# Check specifically for n_valid == 89:
match_89 = sum_df[sum_df["n_valid"] == 89]
print("\nCells with exactly n_valid == 89:")
print(match_89[["model", "language", "condition", "n_valid", "n_pairs", "valid_accuracy"]].to_string())

# Also check Python/CUDA missing experiments or other perturbation tables if 89 pairs was there:
p_legacy_b = "results/grounded_protocol_3lang_20260721/analysis/ab/B_LANGUAGE_INTERACTIONS.csv"
if os.path.exists(p_legacy_b):
    df_inter = pd.read_csv(p_legacy_b)
    print("\nChecking B_LANGUAGE_INTERACTIONS.csv:")
    if "n_valid" in df_inter.columns:
        print(df_inter[df_inter["n_valid"] == 89][["model", "language", "condition", "n_valid"]].to_string())

# 2. Unconditional Effect of Indentation Removal (no_indent vs baseline)
# Let's compute effective accuracy difference: ΔE = E_no_indent - E_baseline
# for each model and language, and pooled.
print("\n--- 2. Unconditional Indentation Removal Effect (Effective Accuracy) ---")

# Let's pair up baseline and no_indent for each model and language
models_5 = [
    ("InternVL3-8B", "OpenGVLab/InternVL3-8B"),
    ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("InternVL3.5-8B", "InternVL3.5-8B"),
    ("Gemma-3-12B", "Gemma-3-12B"),
    ("Gemma4-12B", "Gemma4-12B"),
]

# Let's load pair-level data directly to get exact paired differences:
df_ab_pairs = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/A_B_PAIR_LEVEL.csv")

pair_data_dict = {}
for short_m, full_m in models_5[:2]:
    pair_data_dict[short_m] = df_ab_pairs[df_ab_pairs["model"] == full_m]

for short_m, path in pair_files.items():
    pair_data_dict[short_m] = pd.read_csv(path)

hb1_results = []
bounds = 0.03 # 3 percentage points equivalence bound for TOST

for short_m, _ in models_5:
    pdf = pair_data_dict[short_m]
    # Identify condition names
    base_c = "baseline" if "baseline" in pdf["condition"].unique() else "monokai_dark__fs20__wrap80__lnon"
    no_ind_c = "no_indent"
    
    for lang in ["cuda", "java", "python", "pooled"]:
        if lang == "pooled":
            sub_base = pdf[pdf["condition"] == base_c].set_index("pair_id")
            sub_nind = pdf[pdf["condition"] == no_ind_c].set_index("pair_id")
        else:
            sub_base = pdf[(pdf["condition"] == base_c) & (pdf["language"] == lang)].set_index("pair_id")
            sub_nind = pdf[(pdf["condition"] == no_ind_c) & (pdf["language"] == lang)].set_index("pair_id")
            
        common_pairs = sub_base.index.intersection(sub_nind.index)
        n = len(common_pairs)
        
        # In pair-level files, check column for effective correctness:
        cor_col_base = "effective_correct" if "effective_correct" in sub_base.columns else "strict_correct"
        cor_col_nind = "effective_correct" if "effective_correct" in sub_nind.columns else "strict_correct"
        
        y_base = sub_base.loc[common_pairs, cor_col_base].astype(int).to_numpy()
        y_nind = sub_nind.loc[common_pairs, cor_col_nind].astype(int).to_numpy()
        
        e_base = y_base.mean()
        e_nind = y_nind.mean()
        diff = y_nind - y_base # paired difference per pair (+1, 0, -1)
        mean_diff = diff.mean()
        se_diff = diff.std(ddof=1) / np.sqrt(n)
        
        # Valid accuracy (conditional)
        val_col_base = "strict_valid" if "strict_valid" in sub_base.columns else "valid"
        val_col_nind = "strict_valid" if "strict_valid" in sub_nind.columns else "valid"
        v_base = sub_base.loc[common_pairs, val_col_base].astype(bool)
        v_nind = sub_nind.loc[common_pairs, val_col_nind].astype(bool)
        n_val_base = v_base.sum()
        n_val_nind = v_nind.sum()
        
        val_acc_base = y_base[v_base].mean() if n_val_base > 0 else np.nan
        val_acc_nind = y_nind[v_nind].mean() if n_val_nind > 0 else np.nan
        
        # TOST for equivalence with bound delta = 0.03 (3%p)
        # H01: mean_diff <= -bounds  vs  H11: mean_diff > -bounds
        # H02: mean_diff >= +bounds  vs  H12: mean_diff < +bounds
        t1 = (mean_diff - (-bounds)) / se_diff
        p1 = 1.0 - t.cdf(t1, df=n-1)
        t2 = (bounds - mean_diff) / se_diff
        p2 = 1.0 - t.cdf(t2, df=n-1)
        p_tost = max(p1, p2)
        
        # Minimum Detectable Effect (MDE) at alpha=0.05, power=0.80:
        # MDE = (z_{1-alpha/2} + z_{1-beta}) * se_diff
        # for two-sided paired t-test: (1.96 + 0.8416) * se_diff = 2.8016 * se_diff
        mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se_diff
        
        hb1_results.append({
            "model": short_m,
            "language": lang,
            "n_pairs": n,
            "n_val_base": n_val_base,
            "n_val_nind": n_val_nind,
            "val_acc_base": val_acc_base * 100,
            "val_acc_nind": val_acc_nind * 100,
            "val_acc_diff": (val_acc_nind - val_acc_base) * 100,
            "eff_acc_base": e_base * 100,
            "eff_acc_nind": e_nind * 100,
            "eff_acc_diff": mean_diff * 100,
            "se_eff_diff": se_diff * 100,
            "ci95_low": (mean_diff - 1.96 * se_diff) * 100,
            "ci95_high": (mean_diff + 1.96 * se_diff) * 100,
            "tost_p": p_tost,
            "is_equivalent_3pp": p_tost < 0.05,
            "mde_pp": mde * 100
        })

df_hb1 = pd.DataFrame(hb1_results)
print(df_hb1[["model", "language", "n_pairs", "n_val_base", "n_val_nind", "eff_acc_diff", "ci95_low", "ci95_high", "tost_p", "mde_pp"]].to_string())

