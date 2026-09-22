import json
import pandas as pd
import numpy as np

print("=== Task 9: Closed-Model Pilot (300 Pairs) Bootstrap 95% Intervals ===")

# 1. Load raw data for the closed models
files = {
    "gpt-5.4-mini": "results/openai_vlm_pilot_20260707/gpt-5.4-mini__pilot_raw.jsonl",
    "gpt-5.4": "results/openai_vlm_pilot_20260708/gpt-5.4__pilot_raw.jsonl",
    "gpt-5.5": "results/openai_vlm_pilot_20260708_gpt55_fixed/gpt-5.5__pilot_raw.jsonl",
}

def load_pilot(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

dfs = {m: load_pilot(p) for m, p in files.items()}

# Also reasoning budget:
p_reason_high = "results/gpt54_reasoning_budget_pilot_90_20260709/gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl"
p_reason_low = "results/gpt54_reasoning_budget_pilot_90_20260709/gpt-5.4-2026-03-05__reasoning_low__mot_4096_raw.jsonl"
dfs["gpt-5.4-high-budget"] = load_pilot(p_reason_high)
dfs["gpt-5.4-low-budget"] = load_pilot(p_reason_low)

# Conditions in standard pilot:
# 'image_only', 'text_plus_image', 'combined_labeled_image_only', 'combined_labeled_text_plus_image'

rng = np.random.default_rng(20260914)
N_BOOT = 10000

# Function to compute snippet-cluster bootstrap or pair bootstrap:
# Since 300 Java pairs have unique snippets, let's check snippet cluster vs pair bootstrap.
# Let's see if snippet_i, snippet_j exist in dfs:
# Yes! 'snippet_i', 'snippet_j' exist.
def bootstrap_metrics(df_sub, n_boot=N_BOOT):
    # df_sub has unique pair_id per row (or one row per pair)
    # df_sub columns: 'is_valid_strict_swap', 'is_correct'
    n = len(df_sub)
    snippets = sorted(set(df_sub["snippet_i"]) | set(df_sub["snippet_j"]))
    s_map = {s: i for i, s in enumerate(snippets)}
    left = df_sub["snippet_i"].map(s_map).to_numpy()
    right = df_sub["snippet_j"].map(s_map).to_numpy()
    
    probs = np.full(len(snippets), 1.0 / len(snippets))
    mult = rng.multinomial(len(snippets), probs, size=n_boot) # n_boot x S
    w = mult[:, left] * mult[:, right] # n_boot x n
    w_tot = w.sum(axis=1) # n_boot
    
    val_arr = df_sub["is_valid_strict_swap"].to_numpy().astype(float)
    cor_arr = df_sub["is_correct"].to_numpy().astype(float)
    
    # Point estimates
    pt_val = cor_arr.sum() / val_arr.sum() if val_arr.sum() > 0 else 0.0
    pt_e = cor_arr.mean()
    pt_s = 1.0 - val_arr.mean()
    
    # Bootstrap replicates
    w_val = (w * val_arr).sum(axis=1)
    w_cor = (w * cor_arr).sum(axis=1)
    w_inval = (w * (1.0 - val_arr)).sum(axis=1)
    
    reps_val = np.where(w_val > 0, w_cor / w_val, 0.0)
    reps_e = w_cor / w_tot
    reps_s = w_inval / w_tot
    
    return {
        "valid": (pt_val, np.percentile(reps_val, 2.5), np.percentile(reps_val, 97.5)),
        "E": (pt_e, np.percentile(reps_e, 2.5), np.percentile(reps_e, 97.5)),
        "S": (pt_s, np.percentile(reps_s, 2.5), np.percentile(reps_s, 97.5)),
        "w": w,
        "w_tot": w_tot,
        "cor_arr": cor_arr,
        "val_arr": val_arr
    }

# Compute all conditions:
results_table = []
cond_boot_dict = {}

for m, mdf in dfs.items():
    conds = mdf["condition"].unique()
    for c in sorted(conds):
        sub = mdf[mdf["condition"] == c].copy()
        res = bootstrap_metrics(sub)
        cond_boot_dict[(m, c)] = res
        
        for met in ["E", "S", "valid"]:
            pt, lo, hi = res[met]
            results_table.append({
                "model": m,
                "condition": c,
                "n_pairs": len(sub),
                "metric": met,
                "point": pt * 100,
                "ci_low": lo * 100,
                "ci_high": hi * 100,
                "ci_str": f"{pt*100:.2f} [{lo*100:.2f}, {hi*100:.2f}]"
            })

df_boot_res = pd.DataFrame(results_table)
print("=== Individual Point Estimates & 95% CIs (Sample) ===")
print(df_boot_res[df_boot_res["metric"] == "E"][["model", "condition", "n_pairs", "ci_str"]].to_string(index=False))

# Now evaluate Contrasts:
# 1. Modality Effect: Text+Image vs Image-Only (within same format)
# ΔE = E(text_plus_image) - E(image_only)
# 2. Layout Effect: Combined vs Separate
# 3. Reasoning Budget: High vs Low
print("\n=== Contrast Tests (Checking which CIs include 0) ===")
contrasts = []

# (A) Text+Image vs Image-Only for gpt-5.4-mini, gpt-5.4, gpt-5.5
for m in ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]:
    for layout in ["separate", "combined"]:
        c_text = "text_plus_image" if layout == "separate" else "combined_labeled_text_plus_image"
        c_img = "image_only" if layout == "separate" else "combined_labeled_image_only"
        
        sub_t = dfs[m][dfs[m]["condition"] == c_text].set_index("pair_id")
        sub_i = dfs[m][dfs[m]["condition"] == c_img].set_index("pair_id")
        common = sub_t.index.intersection(sub_i.index)
        
        # Paired bootstrap
        sub_t = sub_t.loc[common]
        sub_i = sub_i.loc[common]
        
        snippets = sorted(set(sub_t["snippet_i"]) | set(sub_t["snippet_j"]))
        s_map = {s: idx for idx, s in enumerate(snippets)}
        left = sub_t["snippet_i"].map(s_map).to_numpy()
        right = sub_t["snippet_j"].map(s_map).to_numpy()
        probs = np.full(len(snippets), 1.0 / len(snippets))
        mult = rng.multinomial(len(snippets), probs, size=N_BOOT)
        w = mult[:, left] * mult[:, right]
        w_tot = w.sum(axis=1)
        
        # Delta E
        cor_t = sub_t["is_correct"].to_numpy().astype(float)
        cor_i = sub_i["is_correct"].to_numpy().astype(float)
        diff_e = cor_t.mean() - cor_i.mean()
        reps_diff_e = (w * cor_t).sum(axis=1) / w_tot - (w * cor_i).sum(axis=1) / w_tot
        ci_e_lo = np.percentile(reps_diff_e, 2.5)
        ci_e_hi = np.percentile(reps_diff_e, 97.5)
        
        # Delta S
        val_t = sub_t["is_valid_strict_swap"].to_numpy().astype(float)
        val_i = sub_i["is_valid_strict_swap"].to_numpy().astype(float)
        diff_s = (1.0 - val_t.mean()) - (1.0 - val_i.mean())
        reps_diff_s = (w * (1.0 - val_t)).sum(axis=1) / w_tot - (w * (1.0 - val_i)).sum(axis=1) / w_tot
        ci_s_lo = np.percentile(reps_diff_s, 2.5)
        ci_s_hi = np.percentile(reps_diff_s, 97.5)
        
        includes_0_e = (ci_e_lo <= 0 <= ci_e_hi)
        includes_0_s = (ci_s_lo <= 0 <= ci_s_hi)
        
        contrasts.append({
            "model": m,
            "contrast_name": f"{layout}: Text+Image vs Image-Only",
            "metric": "Effective Accuracy E",
            "point_diff_pp": diff_e * 100,
            "ci95_low": ci_e_lo * 100,
            "ci95_high": ci_e_hi * 100,
            "includes_zero": includes_0_e,
            "interpretation": "NOT SIGNIFICANT (includes 0)" if includes_0_e else "SIGNIFICANT"
        })
        contrasts.append({
            "model": m,
            "contrast_name": f"{layout}: Text+Image vs Image-Only",
            "metric": "Strict-Swap Error S",
            "point_diff_pp": diff_s * 100,
            "ci95_low": ci_s_lo * 100,
            "ci95_high": ci_s_hi * 100,
            "includes_zero": includes_0_s,
            "interpretation": "NOT SIGNIFICANT (includes 0)" if includes_0_s else "SIGNIFICANT"
        })

# (B) Reasoning Budget: High vs Low for gpt-5.4
for cond in ["image_only", "text_plus_image"]:
    sub_h = dfs["gpt-5.4-high-budget"][dfs["gpt-5.4-high-budget"]["condition"] == cond].set_index("pair_id")
    sub_l = dfs["gpt-5.4-low-budget"][dfs["gpt-5.4-low-budget"]["condition"] == cond].set_index("pair_id")
    common = sub_h.index.intersection(sub_l.index)
    sub_h = sub_h.loc[common]
    sub_l = sub_l.loc[common]
    
    snippets = sorted(set(sub_h["snippet_i"]) | set(sub_h["snippet_j"]))
    s_map = {s: idx for idx, s in enumerate(snippets)}
    left = sub_h["snippet_i"].map(s_map).to_numpy()
    right = sub_h["snippet_j"].map(s_map).to_numpy()
    probs = np.full(len(snippets), 1.0 / len(snippets))
    mult = rng.multinomial(len(snippets), probs, size=N_BOOT)
    w = mult[:, left] * mult[:, right]
    w_tot = w.sum(axis=1)
    
    cor_h = sub_h["is_correct"].to_numpy().astype(float)
    cor_l = sub_l["is_correct"].to_numpy().astype(float)
    diff_e = cor_h.mean() - cor_l.mean()
    reps_diff_e = (w * cor_h).sum(axis=1) / w_tot - (w * cor_l).sum(axis=1) / w_tot
    ci_e_lo = np.percentile(reps_diff_e, 2.5)
    ci_e_hi = np.percentile(reps_diff_e, 97.5)
    
    val_h = sub_h["is_valid_strict_swap"].to_numpy().astype(float)
    val_l = sub_l["is_valid_strict_swap"].to_numpy().astype(float)
    diff_s = (1.0 - val_h.mean()) - (1.0 - val_l.mean())
    reps_diff_s = (w * (1.0 - val_h)).sum(axis=1) / w_tot - (w * (1.0 - val_l)).sum(axis=1) / w_tot
    ci_s_lo = np.percentile(reps_diff_s, 2.5)
    ci_s_hi = np.percentile(reps_diff_s, 97.5)
    
    includes_0_e = (ci_e_lo <= 0 <= ci_e_hi)
    includes_0_s = (ci_s_lo <= 0 <= ci_s_hi)
    
    contrasts.append({
        "model": "gpt-5.4",
        "contrast_name": f"Reasoning Budget High vs Low ({cond})",
        "metric": "Effective Accuracy E",
        "point_diff_pp": diff_e * 100,
        "ci95_low": ci_e_lo * 100,
        "ci95_high": ci_e_hi * 100,
        "includes_zero": includes_0_e,
        "interpretation": "NOT SIGNIFICANT (includes 0)" if includes_0_e else "SIGNIFICANT"
    })
    contrasts.append({
        "model": "gpt-5.4",
        "contrast_name": f"Reasoning Budget High vs Low ({cond})",
        "metric": "Strict-Swap Error S",
        "point_diff_pp": diff_s * 100,
        "ci95_low": ci_s_lo * 100,
        "ci95_high": ci_s_hi * 100,
        "includes_zero": includes_0_s,
        "interpretation": "NOT SIGNIFICANT (includes 0)" if includes_0_s else "SIGNIFICANT"
    })

df_cont = pd.DataFrame(contrasts)
print(df_cont.to_string(index=False))

# List specifically those that INCLUDE 0:
print("\n=== CONSTRASTS WHERE 95% CI INCLUDES ZERO (NOT STATISTICALLY SIGNIFICANT) ===")
zero_inc = df_cont[df_cont["includes_zero"]]
print(zero_inc[["model", "contrast_name", "metric", "point_diff_pp", "ci95_low", "ci95_high"]].to_string(index=False))

