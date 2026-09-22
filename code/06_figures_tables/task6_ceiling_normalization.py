import pandas as pd
import numpy as np

print("=== Task 6: Noise Ceiling Normalization ===")

# 1. Noise Ceilings from User Specification:
# Python: 98.8%
# CUDA: 98.7%
# Scalabrino (Java): 80.5%
ceilings = {
    "python": 0.988,
    "cuda": 0.987,
    "java": 0.805
}

# 2. Load ML Baselines
p_ml = "results/rq1_model_battery_3lang_20260723/analysis/table1_ml_replication/canonical_pairwise_summary.csv"
df_ml = pd.read_csv(p_ml)

# 3. Load VLM Models
# Primary 5 + 3 Diagnostic = 8 models
p_rq1 = "/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv"
df_vlm = pd.read_csv(p_rq1)

# Filter for language in ['cuda', 'java', 'python'] and metric in ['valid', 'E']
vlm_rows = []
for (judge, lang), grp in df_vlm[df_vlm["language"].isin(["cuda", "java", "python"])].groupby(["judge", "language"]):
    valid_acc = grp[grp["metric"] == "valid"]["point"].iloc[0]
    eff_acc = grp[grp["metric"] == "E"]["point"].iloc[0]
    ceil = ceilings[lang]
    vlm_rows.append({
        "type": "VLM",
        "model": judge,
        "language": lang,
        "ceiling": ceil * 100,
        "raw_valid_acc": valid_acc * 100,
        "norm_valid_acc": (valid_acc / ceil) * 100,
        "raw_eff_acc": eff_acc * 100,
        "norm_eff_acc": (eff_acc / ceil) * 100,
    })

df_vlm_norm = pd.DataFrame(vlm_rows)

# Normalize ML Baselines
ml_rows = []
for _, r in df_ml.iterrows():
    lang = r["language"].lower()
    ceil = ceilings[lang]
    eff_acc = r["effective_accuracy"] # For ML, valid_acc == eff_acc (deterministic, no position bias)
    val_acc = r["valid_accuracy"]
    ml_rows.append({
        "type": "ML_Baseline",
        "model": r["model_display_name"] if "model_display_name" in r else r["model"],
        "language": lang,
        "ceiling": ceil * 100,
        "raw_valid_acc": val_acc * 100,
        "norm_valid_acc": (val_acc / ceil) * 100,
        "raw_eff_acc": eff_acc * 100,
        "norm_eff_acc": (eff_acc / ceil) * 100,
    })

df_ml_norm = pd.DataFrame(ml_rows)

# Combine
df_all = pd.concat([df_vlm_norm, df_ml_norm], ignore_index=True)

# Inspect rankings per language before and after normalization:
print("\n=== Model Rankings Per Language (Raw vs Normalized) ===")
for lang in ["cuda", "java", "python"]:
    sub = df_all[df_all["language"] == lang].copy()
    
    # Rank by raw_eff_acc vs norm_eff_acc
    # Note: Within a single language, multiplying/dividing by a constant scalar (the language ceiling)
    # preserves monotonic order exactly! So rank within each language is identical.
    print(f"\n--- Language: {lang.upper()} (Ceiling = {ceilings[lang]*100:.1f}%) ---")
    sub_sorted = sub.sort_values("raw_eff_acc", ascending=False)
    print(sub_sorted[["type", "model", "raw_eff_acc", "norm_eff_acc", "raw_valid_acc", "norm_valid_acc"]].to_string(index=False))

# What about pooled cross-language ranking?
# In pooled evaluation, Java has ceiling 80.5% while Python/CUDA have ~98.8%.
# So raw pooling puts Java accuracy on equal footing with Python/CUDA,
# whereas normalized pooling scales Java by 1/0.805 = 1.242!
print("\n=== Cross-Language Pooled Summary (Raw vs Normalized by Ceiling) ===")
pooled_summary = []
for m, grp in df_all.groupby("model"):
    mtype = grp["type"].iloc[0]
    raw_mean_eff = grp["raw_eff_acc"].mean()
    norm_mean_eff = grp["norm_eff_acc"].mean()
    raw_mean_val = grp["raw_valid_acc"].mean()
    norm_mean_val = grp["norm_valid_acc"].mean()
    pooled_summary.append({
        "type": mtype,
        "model": m,
        "raw_eff_acc_mean": raw_mean_eff,
        "norm_eff_acc_mean": norm_mean_eff,
        "raw_valid_acc_mean": raw_mean_val,
        "norm_valid_acc_mean": norm_mean_val,
    })

df_pool = pd.DataFrame(pooled_summary)
df_pool["raw_eff_rank"] = df_pool["raw_eff_acc_mean"].rank(ascending=False)
df_pool["norm_eff_rank"] = df_pool["norm_eff_acc_mean"].rank(ascending=False)
df_pool["rank_diff"] = df_pool["raw_eff_rank"] - df_pool["norm_eff_rank"]

df_pool = df_pool.sort_values("norm_eff_acc_mean", ascending=False)
print(df_pool[["type", "model", "raw_eff_acc_mean", "norm_eff_acc_mean", "raw_eff_rank", "norm_eff_rank", "rank_diff"]].to_string(index=False))

# Check VLM vs ML Baseline gap conclusions:
print("\n=== VLM vs Best ML Baseline Gap (Raw vs Normalized) ===")
for lang in ["cuda", "java", "python"]:
    sub = df_all[df_all["language"] == lang]
    vlm_sub = sub[sub["type"] == "VLM"]
    ml_sub = sub[sub["type"] == "ML_Baseline"]
    
    best_vlm_raw = vlm_sub["raw_eff_acc"].max()
    best_vlm_norm = vlm_sub["norm_eff_acc"].max()
    best_vlm_m = vlm_sub.loc[vlm_sub["raw_eff_acc"].idxmax(), "model"]
    
    best_ml_raw = ml_sub["raw_eff_acc"].max()
    best_ml_norm = ml_sub["norm_eff_acc"].max()
    best_ml_m = ml_sub.loc[ml_sub["raw_eff_acc"].idxmax(), "model"]
    
    raw_gap = best_vlm_raw - best_ml_raw
    norm_gap = best_vlm_norm - best_ml_norm
    
    print(f"Language: {lang.upper()}")
    print(f"  Best VLM: {best_vlm_m:<15} (Raw E = {best_vlm_raw:.2f}%, Norm E = {best_vlm_norm:.2f}%)")
    print(f"  Best ML:  {best_ml_m:<15} (Raw E = {best_ml_raw:.2f}%, Norm E = {best_ml_norm:.2f}%)")
    print(f"  Raw Gap (VLM - ML):        {raw_gap:+.2f}%p")
    print(f"  Normalized Gap (VLM - ML): {norm_gap:+.2f}%p (Scaled by 1/Ceiling = {1/(ceilings[lang]):.4f}x)")

