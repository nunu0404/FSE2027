import pandas as pd
import numpy as np

print("=== Task 5: H-B2 (Gaussian Blur) Full Table & Directional Concordance ===")

# 1. Load data
# Qwen2.5 & InternVL3: results/grounded_protocol_3lang_20260721/analysis/ab/B_PERTURBATION_RESULTS.csv
df_b = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/B_PERTURBATION_RESULTS.csv")

# Gemma-3, InternVL3.5, Gemma4
df_g3 = pd.read_csv("rq2_eval_reduced/gemma/summary/Gemma-3-12B_summary_metrics.csv")
df_i35 = pd.read_csv("rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv")
df_g4 = pd.read_csv("rq2_eval_reduced/gemma4/summary/Gemma4-12B_summary_metrics.csv")

# Collect rows for:
# Model x Language x Blur Radius (1, 2, 4)
# Baseline is also included for reference.

rows = []

# Models 1 & 2: Qwen2.5 and InternVL3
for _, r in df_b[df_b["condition"].isin(["baseline", "gaussian_sigma_1", "gaussian_sigma_2", "gaussian_sigma_4"])].iterrows():
    rows.append({
        "model": "Qwen2.5-VL-7B" if "Qwen" in r["model"] else "InternVL3-8B",
        "language": r["language"],
        "condition": r["condition"],
        "blur_radius": 0 if r["condition"] == "baseline" else int(r["condition"].split("_")[-1]),
        "effective_accuracy": r["effective_accuracy"] * 100,
        "strict_swap_error": r["strict_swap_error"] * 100,
        "valid_accuracy": r["valid_accuracy"] * 100,
        "n_pairs": r["n_pairs"]
    })

# Models 3, 4, 5:
for df_m, mname in [(df_i35, "InternVL3.5-8B"), (df_g3, "Gemma-3-12B"), (df_g4, "Gemma4-12B")]:
    base_cond = "monokai_dark__fs20__wrap80__lnon"
    sub = df_m[df_m["condition"].isin([base_cond, "gaussian_sigma_4"])]
    for _, r in sub.iterrows():
        is_base = (r["condition"] == base_cond)
        rows.append({
            "model": mname,
            "language": r["language"],
            "condition": "baseline" if is_base else r["condition"],
            "blur_radius": 0 if is_base else 4,
            "effective_accuracy": r["effective_accuracy"] * 100,
            "strict_swap_error": r["strict_swap_error"] * 100,
            "valid_accuracy": r["valid_accuracy"] * 100,
            "n_pairs": r["n_pairs"]
        })

df_table = pd.DataFrame(rows)

# Add pooled rows for each model and condition:
pooled_rows = []
for (m, cond), grp in df_table.groupby(["model", "condition"]):
    tot_pairs = grp["n_pairs"].sum()
    pooled_eff = (grp["effective_accuracy"] * grp["n_pairs"]).sum() / tot_pairs
    pooled_swap = (grp["strict_swap_error"] * grp["n_pairs"]).sum() / tot_pairs
    pooled_val = (grp["valid_accuracy"] * grp["n_pairs"]).sum() / tot_pairs
    b_rad = grp["blur_radius"].iloc[0]
    pooled_rows.append({
        "model": m,
        "language": "pooled",
        "condition": cond,
        "blur_radius": b_rad,
        "effective_accuracy": pooled_eff,
        "strict_swap_error": pooled_swap,
        "valid_accuracy": pooled_val,
        "n_pairs": tot_pairs
    })

df_all = pd.concat([df_table, pd.DataFrame(pooled_rows)], ignore_index=True)
df_all = df_all.sort_values(["model", "language", "blur_radius"])

print("\n=== H-B2 Full Table: Model x Language x Blur Radius ===")
print(df_all[["model", "language", "blur_radius", "effective_accuracy", "strict_swap_error", "n_pairs"]].to_string(index=False))

# 2. Directional Concordance Test Across Languages:
# For each model, does the effect of blur (ΔE = E_blur - E_baseline) have the SAME sign across Java, Python, CUDA?
print("\n=== Directional Concordance Test Across Languages ===")
concordance_results = []

for m in ["Qwen2.5-VL-7B", "InternVL3-8B", "InternVL3.5-8B", "Gemma-3-12B", "Gemma4-12B"]:
    sub_m = df_table[df_table["model"] == m]
    
    # Check each available blur radius:
    avail_radii = [r for r in sorted(sub_m["blur_radius"].unique()) if r > 0]
    for rad in avail_radii:
        deltas = {}
        for l in ["cuda", "java", "python"]:
            e_base = sub_m[(sub_m["language"] == l) & (sub_m["blur_radius"] == 0)]["effective_accuracy"].iloc[0]
            e_blur = sub_m[(sub_m["language"] == l) & (sub_m["blur_radius"] == rad)]["effective_accuracy"].iloc[0]
            deltas[l] = e_blur - e_base
            
        signs = [np.sign(deltas[l]) for l in ["cuda", "java", "python"]]
        all_same_sign = (len(set(signs)) == 1)
        # Sign test exact p-value under null of p(+)=0.5:
        # P(all 3 positive or all 3 negative) = 2 * (0.5)^3 = 0.25
        p_sign_test = 0.25 if all_same_sign else 1.0
        
        concordance_results.append({
            "model": m,
            "blur_radius": rad,
            "delta_cuda": deltas["cuda"],
            "delta_java": deltas["java"],
            "delta_python": deltas["python"],
            "all_same_sign": all_same_sign,
            "sign": "+" if signs[0] > 0 else "-" if signs[0] < 0 else "0",
            "p_value": p_sign_test
        })

df_conc = pd.DataFrame(concordance_results)
print(df_conc.to_string(index=False))

