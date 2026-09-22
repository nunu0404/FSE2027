import os
import pandas as pd
import numpy as np

print("=== Task 7: Within-Dataset Java (n=1,004) Recalculation ===")

# Load within dataset mapping from rq1_pairs_9000
p_pairs = "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
df_pairs = pd.read_csv(p_pairs)

java_pairs = df_pairs[df_pairs["language"] == "java"].copy()
# Check dataset_name_i and dataset_name_j
java_pairs["within"] = java_pairs["dataset_name_i"] == java_pairs["dataset_name_j"]
within_pids = set(java_pairs[java_pairs["within"]]["protocol_pair_id"])
print(f"Java total pairs: {len(java_pairs)}, Within-dataset pairs: {len(within_pids)}")
print("Within pairs by dataset:")
print(java_pairs[java_pairs["within"]]["dataset_name_i"].value_counts().to_dict())

# Load Primary 5 models:
# A: Qwen2.5, Gemma-3
# B: Qwen3, InternVL3.5, Gemma4
df_a = pd.read_csv("results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv")
df_b = pd.read_csv("results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv")

# Also 3 diagnostic models for complete comparison:
# InternVL3, Ministral, Phi-4
models_all = [
    ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct", "A"),
    ("Qwen3-VL-8B", "Qwen/Qwen3-VL-8B-Instruct", "B"),
    ("Gemma-3-12B", "google/gemma-3-12b-it", "A"),
    ("Gemma4-12B", "google/gemma-4-12B-it", "B"),
    ("InternVL3.5-8B", "OpenGVLab/InternVL3_5-8B-HF", "B"),
    ("InternVL3-8B", "OpenGVLab/InternVL3-8B", "A"),
    ("Ministral-3-8B", "mistralai/Ministral-3-8B-Instruct-2512-BF16", "A"),
    ("Phi-4-multimodal", "microsoft/Phi-4-multimodal-instruct", "A"),
]

# Map pair_id to protocol_pair_id in df_pairs
pid_map = dict(zip(df_pairs["pair_id"], df_pairs["protocol_pair_id"]))

java_comparison_rows = []

for dname, mstr, src in models_all:
    df_src = df_a if src == "A" else df_b
    sub = df_src[(df_src["model"] == mstr) & (df_src["language"] == "java")].copy()
    
    # Check pair id matching
    # In df_b, pair_id is already protocol_pair_id or full?
    # Let check sample pair_id:
    s_pid = sub["pair_id"].iloc[0]
    if s_pid in within_pids:
        sub["is_within"] = sub["pair_id"].isin(within_pids)
    elif s_pid in pid_map:
        sub["is_within"] = sub["pair_id"].map(pid_map).isin(within_pids)
    else:
        # Check matching by index or other
        sub["is_within"] = sub["pair_id"].map(lambda x: pid_map.get(x, x)).isin(within_pids)
        
    w = sub[sub["is_within"]]
    w_n = len(w)
    
    # Compute metrics for Within (n=1004)
    w_val = w["valid"].sum()
    w_cor = w["correct"].sum()
    w_deb = w["debiased_correct"].sum()
    w_val_acc = (w_cor / w_val * 100.0) if w_val > 0 else 0.0
    w_e = w_cor / w_n * 100.0
    w_d = w_deb / w_n * 100.0
    w_s = (w_n - w_val) / w_n * 100.0
    
    # Compute metrics for Mixed (n=3000)
    m_n = len(sub)
    m_val = sub["valid"].sum()
    m_cor = sub["correct"].sum()
    m_deb = sub["debiased_correct"].sum()
    m_val_acc = (m_cor / m_val * 100.0) if m_val > 0 else 0.0
    m_e = m_cor / m_n * 100.0
    m_d = m_deb / m_n * 100.0
    m_s = (m_n - m_val) / m_n * 100.0
    
    java_comparison_rows.append({
        "judge": dname,
        "within_n": w_n,
        "within_val_acc": w_val_acc,
        "mixed_val_acc": m_val_acc,
        "diff_val_acc": w_val_acc - m_val_acc,
        "within_e": w_e,
        "mixed_e": m_e,
        "diff_e": w_e - m_e,
        "within_d": w_d,
        "mixed_d": m_d,
        "diff_d": w_d - m_d,
        "within_s": w_s,
        "mixed_s": m_s,
        "diff_s": w_s - m_s,
    })

# Add ML Baselines for Java within vs mixed:
p_ml = "results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv"
if os.path.exists(p_ml):
    df_ml_preds = pd.read_csv(p_ml)
    df_ml_java = df_ml_preds[df_ml_preds["language"] == "java"].copy()
    df_ml_java["is_within"] = df_ml_java["pair_id"].map(lambda x: pid_map.get(x, x)).isin(within_pids)
    
    for m in df_ml_java["model"].unique():
        sub_ml = df_ml_java[df_ml_java["model"] == m]
        w_ml = sub_ml[sub_ml["is_within"]]
        w_acc = w_ml["is_correct"].mean() * 100.0
        m_acc = sub_ml["is_correct"].mean() * 100.0
        java_comparison_rows.append({
            "judge": f"ML: {m}",
            "within_n": len(w_ml),
            "within_val_acc": w_acc,
            "mixed_val_acc": m_acc,
            "diff_val_acc": w_acc - m_acc,
            "within_e": w_acc,
            "mixed_e": m_acc,
            "diff_e": w_acc - m_acc,
            "within_d": w_acc,
            "mixed_d": m_acc,
            "diff_d": w_acc - m_acc,
            "within_s": 0.0,
            "mixed_s": 0.0,
            "diff_s": 0.0,
        })

df_comp = pd.DataFrame(java_comparison_rows)
print("\n=== Table: Java Within-Dataset (n=1,004) vs Mixed (n=3,000) ===")
print(df_comp[["judge", "within_val_acc", "mixed_val_acc", "diff_val_acc", "within_e", "mixed_e", "diff_e", "within_s", "mixed_s", "diff_s"]].to_string(index=False))

# Now, recalculate POOLED RQ1 metrics if Java is replaced with Within-Dataset (n=1004 + Python 3000 + CUDA 3000 = n=7004):
print("\n=== Recalculated Pooled Metrics (Java Within n=1,004 + Python 3,000 + CUDA 3,000 = n=7,004) ===")
pooled_recalc = []
for dname, mstr, src in models_all[:5]: # Primary 5 models
    df_src = df_a if src == "A" else df_b
    sub_m = df_src[df_src["model"] == mstr].copy()
    
    # Split
    sub_cuda = sub_m[sub_m["language"] == "cuda"]
    sub_py = sub_m[sub_m["language"] == "python"]
    sub_java = sub_m[sub_m["language"] == "java"].copy()
    sub_java["is_within"] = sub_java["pair_id"].map(lambda x: pid_map.get(x, x)).isin(within_pids)
    sub_java_w = sub_java[sub_java["is_within"]]
    
    # Original pooled (n=9,000)
    orig_n = len(sub_m)
    orig_val = sub_m["valid"].sum()
    orig_cor = sub_m["correct"].sum()
    orig_deb = sub_m["debiased_correct"].sum()
    orig_e = orig_cor / orig_n * 100.0
    orig_d = orig_deb / orig_n * 100.0
    orig_s = (orig_n - orig_val) / orig_n * 100.0
    
    # Recalculated pooled (n=7,004)
    sub_recalc = pd.concat([sub_cuda, sub_py, sub_java_w], ignore_index=True)
    rec_n = len(sub_recalc)
    rec_val = sub_recalc["valid"].sum()
    rec_cor = sub_recalc["correct"].sum()
    rec_deb = sub_recalc["debiased_correct"].sum()
    rec_e = rec_cor / rec_n * 100.0
    rec_d = rec_deb / rec_n * 100.0
    rec_s = (rec_n - rec_val) / rec_n * 100.0
    
    pooled_recalc.append({
        "judge": dname,
        "orig_pooled_n": orig_n,
        "recalc_pooled_n": rec_n,
        "orig_E": orig_e,
        "recalc_E": rec_e,
        "diff_E": rec_e - orig_e,
        "orig_D": orig_d,
        "recalc_D": rec_d,
        "diff_D": rec_d - orig_d,
        "orig_S": orig_s,
        "recalc_S": rec_s,
        "diff_S": rec_s - orig_s,
    })

df_pool_rec = pd.DataFrame(pooled_recalc)
print(df_pool_rec.to_string(index=False))

