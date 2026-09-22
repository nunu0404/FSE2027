import os, sys, hashlib, time, zipfile
import numpy as np
import pandas as pd

t_start = time.time()

BUNDLE_DIR = "/ANON/home/fse2027_handoff_rq1_20260912"
os.makedirs(f"{BUNDLE_DIR}/tab", exist_ok=True)
os.makedirs(f"{BUNDLE_DIR}/fig_data", exist_ok=True)
os.makedirs(f"{BUNDLE_DIR}/text", exist_ok=True)

# ----------------------------------------------------
# 1. Load Data
# ----------------------------------------------------
pairs_path = "/ANON/home/fse2027_rq1_recovery/gpusystem/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
p1_path = "/ANON/home/fse2027_rq1_recovery/gpusystem/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv"
p2_path = "/ANON/home/fse2027_rq1_recovery/gpusystem/ANON/experiment_root/results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv"
ml_path = "/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv"

pdf = pd.read_csv(pairs_path)
df1 = pd.read_csv(p1_path)
df2 = pd.read_csv(p2_path)
ml_df = pd.read_csv(ml_path)

cols = ["model_key", "model", "pair_id", "language", "difficulty", "valid", "correct", "debiased_correct"]
combined_vlm = pd.concat([df1[cols], df2[cols]], ignore_index=True)

pair_meta = pdf[["protocol_pair_id", "snippet_i", "snippet_j", "dataset_name_i", "dataset_name_j"]].rename(columns={"protocol_pair_id": "pair_id"})
merged = pd.merge(combined_vlm, pair_meta, on="pair_id", how="left")

# ----------------------------------------------------
# Part 2a: Java Within vs Mixed
# ----------------------------------------------------
java_pairs = pdf[pdf["language"] == "java"].copy()
java_pairs["within"] = java_pairs["dataset_name_i"] == java_pairs["dataset_name_j"]
proto_to_within = dict(zip(java_pairs["protocol_pair_id"], java_pairs["within"]))
pair_to_within = dict(zip(java_pairs["pair_id"], java_pairs["within"]))

buse_cnt = int((java_pairs["within"] & (java_pairs["dataset_name_i"] == "Buse")).sum())
dorn_cnt = int((java_pairs["within"] & (java_pairs["dataset_name_i"] == "Dorn")).sum())
scal_cnt = int((java_pairs["within"] & (java_pairs["dataset_name_i"] == "Scalabrino")).sum())
total_within = buse_cnt + dorn_cnt + scal_cnt

vlm_java = merged[merged["language"] == "java"].copy()
vlm_java["within"] = vlm_java["pair_id"].map(proto_to_within)

primary_judges = [
    ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("Qwen3-VL-8B", "Qwen/Qwen3-VL-8B-Instruct"),
    ("Gemma-3-12B", "google/gemma-3-12b-it"),
    ("Gemma4-12B", "google/gemma-4-12B-it"),
    ("InternVL3.5-8B", "OpenGVLab/InternVL3_5-8B-HF")
]

java_rows = []
for dname, mstr in primary_judges:
    sub = vlm_java[vlm_java["model"] == mstr]
    w = sub[sub["within"]]
    w_n = len(w)
    w_val = w["valid"].sum()
    w_cor = w["correct"].sum()
    w_val_acc = (w_cor / w_val * 100.0) if w_val > 0 else 0.0
    w_e = w_cor / w_n * 100.0
    w_s = (w_n - w_val) / w_n * 100.0
    
    m_n = len(sub)
    m_val = sub["valid"].sum()
    m_cor = sub["correct"].sum()
    m_val_acc = (m_cor / m_val * 100.0) if m_val > 0 else 0.0
    m_e = m_cor / m_n * 100.0
    m_s = (m_n - m_val) / m_n * 100.0
    
    java_rows.append({
        "judge": dname,
        "within_valid_acc": w_val_acc,
        "within_e": w_e,
        "within_s": w_s,
        "mixed_valid_acc": m_val_acc,
        "mixed_e": m_e,
        "mixed_s": m_s
    })

# ML baseline for Java
ml_java = ml_df[ml_df["pair_id"].isin(pair_to_within.keys())].copy()
ml_java["within"] = ml_java["pair_id"].map(pair_to_within)

w_mlp = ml_java[(ml_java["model"] == "Multilayer Perceptron") & (ml_java["within"])]
m_mlp = ml_java[(ml_java["model"] == "Multilayer Perceptron")]
w_mlp_acc = w_mlp["is_correct"].mean() * 100.0
m_mlp_acc = m_mlp["is_correct"].mean() * 100.0

java_rows.append({
    "judge": "Multilayer Perceptron (Best Baseline)",
    "within_valid_acc": w_mlp_acc,
    "within_e": w_mlp_acc,
    "within_s": 0.0,
    "mixed_valid_acc": m_mlp_acc,
    "mixed_e": m_mlp_acc,
    "mixed_s": 0.0
})

java_within_df = pd.DataFrame(java_rows)
java_within_df.to_csv(os.path.join(BUNDLE_DIR, "java_within.csv"), index=False)
java_within_df.to_csv(os.path.join(BUNDLE_DIR, "tab", "vlm_tab_java_within.csv"), index=False)

tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Comparison of VLM judges and the best machine learning baseline on within-dataset Java pairs ($n=1,004$; Buse: %d, Dorn: %d, Scalabrino: %d) versus the mixed Java evaluation set ($n=3,000$). All values are percentages. For the feature-based baseline, order swap error is by definition zero, so valid accuracy equals effective accuracy.}" % (buse_cnt, dorn_cnt, scal_cnt),
    r"\label{tab:vlm_tab_java_within}",
    r"\small",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"& \multicolumn{3}{c}{Within-Dataset ($n=1,004$)} & \multicolumn{3}{c}{Mixed ($n=3,000$)} \\",
    r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
    r"Judge & Valid & $E$ & $S$ & Valid & $E$ & $S$ \\",
    r"\midrule"
]
for r in java_rows:
    jname = r["judge"]
    if "Best Baseline" in jname:
        tex_lines.append(r"\midrule")
    line_str = "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\" % (
        jname, r["within_valid_acc"], r["within_e"], r["within_s"],
        r["mixed_valid_acc"], r["mixed_e"], r["mixed_s"]
    )
    tex_lines.append(line_str)

tex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}"
])

with open(os.path.join(BUNDLE_DIR, "tab", "vlm_tab_java_within.tex"), "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print("Part 2a completed successfully.")

# ----------------------------------------------------
# Part 2b: 10,000-Resample Snippet-Cluster Bootstrap
# ----------------------------------------------------
all_judges = [
    ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("Qwen3-VL-8B", "Qwen/Qwen3-VL-8B-Instruct"),
    ("Gemma-3-12B", "google/gemma-3-12b-it"),
    ("Gemma4-12B", "google/gemma-4-12B-it"),
    ("InternVL3.5-8B", "OpenGVLab/InternVL3_5-8B-HF"),
    ("InternVL3-8B", "OpenGVLab/InternVL3-8B"),
    ("Ministral-3-8B", "mistralai/Ministral-3-8B-Instruct-2512-BF16"),
    ("Phi-4-multimodal", "microsoft/Phi-4-multimodal-instruct")
]

languages = ["java", "python", "cuda", "pooled"]
metrics_list = ["valid", "E", "D", "S"]
BOOTSTRAP_REPS = 10000

scope_weights = {}
for lang in languages:
    if lang == "pooled":
        sub_p = pdf
    else:
        sub_p = pdf[pdf["language"] == lang]
    
    snippets = sorted(set(sub_p.snippet_i) | set(sub_p.snippet_j))
    idx_map = {s: i for i, s in enumerate(snippets)}
    left = sub_p.snippet_i.map(idx_map).to_numpy()
    right = sub_p.snippet_j.map(idx_map).to_numpy()
    
    rng = np.random.default_rng(42)
    probs = np.full(len(snippets), 1.0 / len(snippets))
    mult = rng.multinomial(len(snippets), probs, size=BOOTSTRAP_REPS)
    w = mult[:, left] * mult[:, right]
    w_tot = w.sum(axis=1)
    
    scope_weights[lang] = {
        "pair_ids": sub_p["protocol_pair_id"].tolist(),
        "weights": w,
        "w_tot": w_tot,
        "n_pairs": len(sub_p)
    }

intervals_records = []
fig3_records = []
fig4_records = []
pooled_tex_rows = []

for dname, mstr in all_judges:
    judge_data = merged[merged["model"] == mstr].copy()
    judge_data_by_id = judge_data.set_index("pair_id")
    
    for lang in languages:
        info = scope_weights[lang]
        pids = info["pair_ids"]
        w = info["weights"]
        w_tot = info["w_tot"]
        n_p = info["n_pairs"]
        
        j_sub = judge_data_by_id.loc[pids]
        val_arr = j_sub["valid"].to_numpy().astype(float)
        cor_arr = j_sub["correct"].to_numpy().astype(float)
        deb_arr = j_sub["debiased_correct"].to_numpy().astype(float)
        
        val_point = cor_arr.sum() / val_arr.sum() if val_arr.sum() > 0 else 0.0
        e_point = cor_arr.sum() / n_p
        d_point = deb_arr.sum() / n_p
        s_point = (1.0 - val_arr).sum() / n_p
        
        w_val = (w * val_arr).sum(axis=1)
        w_cor = (w * cor_arr).sum(axis=1)
        w_deb = (w * deb_arr).sum(axis=1)
        w_inval = (w * (1.0 - val_arr)).sum(axis=1)
        
        val_reps = np.where(w_val > 0, w_cor / w_val, 0.0)
        e_reps = w_cor / w_tot
        d_reps = w_deb / w_tot
        s_reps = w_inval / w_tot
        
        ci_dict = {
            "valid": (val_point, float(np.quantile(val_reps, 0.025)), float(np.quantile(val_reps, 0.975))),
            "E": (e_point, float(np.quantile(e_reps, 0.025)), float(np.quantile(e_reps, 0.975))),
            "D": (d_point, float(np.quantile(d_reps, 0.025)), float(np.quantile(d_reps, 0.975))),
            "S": (s_point, float(np.quantile(s_reps, 0.025)), float(np.quantile(s_reps, 0.975)))
        }
        
        for m in metrics_list:
            pt, lo, hi = ci_dict[m]
            intervals_records.append({
                "judge": dname,
                "language": lang,
                "metric": m,
                "point": pt,
                "ci_low": lo,
                "ci_high": hi,
                "n_pairs": n_p
            })
        
        if lang in ["java", "python", "cuda"]:
            fig3_records.append({
                "judge": dname,
                "language": lang,
                "valid": ci_dict["valid"][0],
                "valid_lo": ci_dict["valid"][1],
                "valid_hi": ci_dict["valid"][2],
                "E": ci_dict["E"][0],
                "E_lo": ci_dict["E"][1],
                "E_hi": ci_dict["E"][2]
            })
            
        if lang == "pooled":
            fig4_records.append({
                "judge": dname,
                "E": ci_dict["E"][0],
                "E_lo": ci_dict["E"][1],
                "E_hi": ci_dict["E"][2],
                "D": ci_dict["D"][0],
                "D_lo": ci_dict["D"][1],
                "D_hi": ci_dict["D"][2],
                "S": ci_dict["S"][0],
                "S_lo": ci_dict["S"][1],
                "S_hi": ci_dict["S"][2]
            })
            
            pooled_tex_rows.append({
                "judge": dname,
                "E_str": "%.2f [%.2f, %.2f]" % (ci_dict['E'][0]*100, ci_dict['E'][1]*100, ci_dict['E'][2]*100),
                "D_str": "%.2f [%.2f, %.2f]" % (ci_dict['D'][0]*100, ci_dict['D'][1]*100, ci_dict['D'][2]*100),
                "S_str": "%.2f [%.2f, %.2f]" % (ci_dict['S'][0]*100, ci_dict['S'][1]*100, ci_dict['S'][2]*100),
                "E": ci_dict['E'][0]*100, "E_lo": ci_dict['E'][1]*100, "E_hi": ci_dict['E'][2]*100,
                "D": ci_dict['D'][0]*100, "D_lo": ci_dict['D'][1]*100, "D_hi": ci_dict['D'][2]*100,
                "S": ci_dict['S'][0]*100, "S_lo": ci_dict['S'][1]*100, "S_hi": ci_dict['S'][2]*100,
            })

pd.DataFrame(intervals_records).to_csv(os.path.join(BUNDLE_DIR, "rq1_intervals.csv"), index=False)
pd.DataFrame(fig3_records).to_csv(os.path.join(BUNDLE_DIR, "fig_data", "fig3_data.csv"), index=False)
pd.DataFrame(fig4_records).to_csv(os.path.join(BUNDLE_DIR, "fig_data", "fig4_data.csv"), index=False)

tex_pooled_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Pooled RQ1 evaluation metrics across all three languages ($n=9,000$) with 10,000-resample snippet-cluster bootstrap 95\% confidence intervals. All values are percentages. $E$: strict effective accuracy; $D$: two-order-averaged accuracy; $S$: strict-swap error.}",
    r"\label{tab:vlm_tab_rq1_intervals}",
    r"\small",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Judge & Effective Accuracy ($E$) & Debiased Accuracy ($D$) & Strict-Swap Error ($S$) \\",
    r"\midrule",
    r"\multicolumn{4}{l}{\textit{Primary Judges}} \\"
]

for i, r in enumerate(pooled_tex_rows):
    if i == 5:
        tex_pooled_lines.extend([
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Diagnosis Checkpoints}} \\"
        ])
    tex_pooled_lines.append(f"{r['judge']} & {r['E_str']} & {r['D_str']} & {r['S_str']} \\\\")

tex_pooled_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}"
])

with open(os.path.join(BUNDLE_DIR, "tab", "vlm_tab_rq1_intervals.tex"), "w") as f:
    f.write("\n".join(tex_pooled_lines) + "\n")

pd.DataFrame(pooled_tex_rows).to_csv(os.path.join(BUNDLE_DIR, "tab", "vlm_tab_rq1_intervals.csv"), index=False)
print("Part 2b completed successfully.")

# ----------------------------------------------------
# Part 2c: text/noise_ceiling_sentences.md
# ----------------------------------------------------
noise_ceiling_md = """### Section 3.2

To evaluate the stability of the reference direction, we performed a split-half resampling procedure across raters with 1,000 random splits. In each trial, the relative preference direction determined by one random half of raters was compared against the direction established by the full rater panel. Across the 3,000 evaluation pairs, the direction from one random half matched the all-rater direction in 98.8 percent of Python pairs and 98.7 percent of CUDA pairs. Even when restricted to the most challenging pairs, the agreement remained high, matching in 96.1 to 96.5 percent of hard pairs. By contrast, the Scalabrino pool contains nine raters and yields a half-versus-all agreement of 80.5 percent, indicating that its within-dataset pairs carry a lower reference ceiling.

### Section 4.2

The stability analysis indicates that the observed valid accuracies of 62 to 76 percent cannot be explained by reference noise. Because the reference direction matches the full-panel consensus in 98.8 percent of Python pairs, 98.7 percent of CUDA pairs, and 96.1 to 96.5 percent of hard pairs, the benchmark preference signal remains highly consistent across independent subsets of raters. Consequently, the performance shortfall of vision-language judges reflects model assessment characteristics rather than instability in the underlying readability annotations.

### Section 10

As documented in this manuscript, the reference direction derived from human ratings demonstrates high empirical stability across resampled rater splits, maintaining 98.8 percent agreement in Python and 98.7 percent in CUDA. The lower reference ceiling observed in the Scalabrino within-dataset subset, where nine raters produce 80.5 percent agreement, underscores the necessity of reporting dataset-specific consistency bounds when evaluating visual readability assessment.
"""

with open(os.path.join(BUNDLE_DIR, "text", "noise_ceiling_sentences.md"), "w") as f:
    f.write(noise_ceiling_md.strip() + "\n")

# ----------------------------------------------------
# Part 2d: text/bootstrap_correction.md
# ----------------------------------------------------
bootstrap_correction_md = "Change to 2.2 and [−6.18, +10.23], sourced from ~/experiment_26_v1/fse2027/external_runs/fse2027_review_defense_e1_e8_20260730/analysis/E1/E1_debiased_vs_baseline.csv (the earlier draft values 2.1 and [−6.19, +10.52] originated from ~/experiment_26_v1/results/rq1_model_battery_3lang_20260723/analysis/F7/F7_E1_debiased_vs_language_best.csv).\n"

with open(os.path.join(BUNDLE_DIR, "text", "bootstrap_correction.md"), "w") as f:
    f.write(bootstrap_correction_md)

# ----------------------------------------------------
# Part 2e: text/figure5_counts.md
# ----------------------------------------------------
figure5_counts_md = """# Closed Model Pilot Pair Counts and Metric Verification

This document reports unique pairs, call counts per presentation order (AB and BA), and recomputed metrics across all closed-model pilot result folders under `~/experiment_26_v1/results/`.

### Summary of Conditions and Verification Table

| Folder | File | Model (Condition) | Condition Name | Unique Pairs | Calls (AB / BA) | Total Calls | Valid Pairs | Correct Pairs | Valid Acc (%) | Effective Acc E (%) | Strict Swap Err S (%) | Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | combined_labeled_image_only | 300 | 300 / 300 | 600 | 225 | 142 | 63.11 | 47.33 | 25.00 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 147 | 92 | 62.59 | 30.67 | 51.00 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | image_only | 300 | 300 / 300 | 600 | 229 | 146 | 63.76 | 48.67 | 23.67 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | text_plus_image | 300 | 300 / 300 | 600 | 166 | 112 | 67.47 | 37.33 | 44.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | combined_labeled_image_only | 300 | 300 / 300 | 600 | 259 | 159 | 61.39 | 53.00 | 13.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 234 | 155 | 66.24 | 51.67 | 22.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | image_only | 300 | 300 / 300 | 600 | 249 | 151 | 60.64 | 50.33 | 17.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | text_plus_image | 300 | 300 / 300 | 600 | 220 | 148 | 67.27 | 49.33 | 26.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (buggy prompt) | image_only | 300 | 300 / 300 | 600 | 0 | 0 | NaN | 0.00 | 100.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (buggy prompt) | text_plus_image | 122 | 122 / 122 | 244 | 0 | 0 | NaN | 0.00 | 100.00 | FLAG_NOT_300 (122 pairs) |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | combined_labeled_image_only | 300 | 300 / 300 | 600 | 231 | 155 | 67.10 | 51.67 | 23.00 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 238 | 154 | 64.71 | 51.33 | 20.67 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | image_only | 300 | 300 / 300 | 600 | 244 | 167 | 68.44 | 55.67 | 18.67 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | text_plus_image | 300 | 300 / 300 | 600 | 224 | 150 | 66.96 | 50.00 | 25.33 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_256_raw.jsonl` | gpt-5.4 (high, mot 256) | image_only | 90 | 90 / 90 | 180 | 19 | 14 | 73.68 | 15.56 | 78.89 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_256_raw.jsonl` | gpt-5.4 (high, mot 256) | text_plus_image | 90 | 90 / 90 | 180 | 48 | 35 | 72.92 | 38.89 | 46.67 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl` | gpt-5.4 (high, mot 4096) | image_only | 300 | 300 / 300 | 600 | 266 | 169 | 63.53 | 56.33 | 11.33 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl` | gpt-5.4 (high, mot 4096) | text_plus_image | 300 | 300 / 300 | 600 | 213 | 148 | 69.48 | 49.33 | 29.00 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_256_raw.jsonl` | gpt-5.4 (low, mot 256) | image_only | 90 | 90 / 90 | 180 | 76 | 45 | 59.21 | 50.00 | 15.56 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_256_raw.jsonl` | gpt-5.4 (low, mot 256) | text_plus_image | 90 | 90 / 90 | 180 | 67 | 44 | 65.67 | 48.89 | 25.56 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_4096_raw.jsonl` | gpt-5.4 (low, mot 4096) | image_only | 300 | 300 / 300 | 600 | 256 | 156 | 60.94 | 52.00 | 14.67 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_4096_raw.jsonl` | gpt-5.4 (low, mot 4096) | text_plus_image | 300 | 300 / 300 | 600 | 219 | 145 | 66.21 | 48.33 | 27.00 | OK_300 |

### Meaning of "90" in Folder Name
The "90" in `gpt54_reasoning_budget_pilot_90_20260709` denotes that for the reduced reasoning budget condition (`mot_256`), 30 pairs per difficulty bin (easy, medium, hard) were evaluated, resulting in exactly 90 unique pairs per condition (180 calls each). By contrast, full-budget conditions (`mot_4096`) evaluated all 300 Java pairs (600 calls each).

### Flagged Conditions (FLAG_NOT_300)
1. `openai_vlm_pilot_20260708` -> `gpt-5.5` `text_plus_image`: Aborted at 122 pairs (244 calls).
2. `gpt54_reasoning_budget_pilot_90_20260709` -> All four `mot_256` conditions: Evaluated on 90 pairs each (180 calls each).
"""

with open(os.path.join(BUNDLE_DIR, "text", "figure5_counts.md"), "w") as f:
    f.write(figure5_counts_md.strip() + "\n")

# ----------------------------------------------------
# Part 2f: text/data_availability_clause.md
# ----------------------------------------------------
data_avail_md = "Closed-model runs provide parsed verdicts only, without logits, at 300 Java pairs per condition (with the 256-token reasoning budget exploratory condition evaluated at 90 pairs per condition).\n"

with open(os.path.join(BUNDLE_DIR, "text", "data_availability_clause.md"), "w") as f:
    f.write(data_avail_md)

# ----------------------------------------------------
# Part 2g: text/ai_disclosure.md
# ----------------------------------------------------
ai_disclosure_md = """NOT FOUND

Searched locations:
- `/ANON/experiment_root/fse2027/`
- `/ANON/experiment_root/results/`
- `/ANON/home/fse2027_rq1_recovery/`
- `/ANON/home/fse2027_noise_ceiling/`
- `/data/tmp/`
No code, scripts, or project notes detailing model-assisted coding or image labeling for SWE-bench Multimodal and GitHub images (1,498 images) were found on this server.
"""

with open(os.path.join(BUNDLE_DIR, "text", "ai_disclosure.md"), "w") as f:
    f.write(ai_disclosure_md.strip() + "\n")

# ----------------------------------------------------
# SOURCES.md Generation
# ----------------------------------------------------
def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

bundle_files = []
for root, dirs, files in os.walk(BUNDLE_DIR):
    for f in sorted(files):
        if f == "SOURCES.md": continue
        fp = os.path.join(root, f)
        rel = os.path.relpath(fp, BUNDLE_DIR)
        bundle_files.append((rel, fp, get_sha256(fp)))

sources_lines = [
    "# SOURCES.md: RQ1 Handoff Bundle Provenance and Checksums",
    "",
    f"**Bundle Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} KST",
    "**Generating Script**: `generate_rq1_bundle.py`",
    "**Host**: Linux workstation (CPU-only execution)",
    "",
    "## Input Files and Sources",
    f"1. `rq1_pairs_9000.csv`:\n   - Path: `{pairs_path}`\n   - SHA-256: `{get_sha256(pairs_path)}`",
    f"2. `pair_level_results.csv` (Primary models & diagnosis checkpoints: Qwen2.5, InternVL3, Gemma-3, Ministral, Phi-4):\n   - Path: `{p1_path}`\n   - SHA-256: `{get_sha256(p1_path)}`",
    f"3. `primary_pair_level.csv` (Extension primary models: Qwen3, InternVL3.5, Gemma4):\n   - Path: `{p2_path}`\n   - SHA-256: `{get_sha256(p2_path)}`",
    f"4. `ml_predictions_9models_9000.csv` (Machine learning baselines):\n   - Path: `{ml_path}`\n   - SHA-256: `{get_sha256(ml_path)}`",
    f"5. `ceiling_by_language_bin.csv` (Noise ceiling reference agreement statistics):\n   - Path: `/ANON/home/fse2027_noise_ceiling/ceiling_by_language_bin.csv`\n   - SHA-256: `{get_sha256('/ANON/home/fse2027_noise_ceiling/ceiling_by_language_bin.csv')}`",
    "",
    "## Output Files in this Bundle",
    "",
    "| Output Relative Path | Description | SHA-256 Checksum |",
    "| :--- | :--- | :--- |"
]

for rel, fp, h in bundle_files:
    sources_lines.append(f"| `{rel}` | Generated by `generate_rq1_bundle.py` | `{h}` |")

with open(os.path.join(BUNDLE_DIR, "SOURCES.md"), "w") as f:
    f.write("\n".join(sources_lines) + "\n")
print("Saved SOURCES.md")

# ----------------------------------------------------
# Zip Bundle to ~/fse2027_handoff_rq1_20260912.zip
# ----------------------------------------------------
ZIP_PATH = "/ANON/home/fse2027_handoff_rq1_20260912.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BUNDLE_DIR):
        for f in sorted(files):
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, BUNDLE_DIR)
            zipf.write(fp, arcname)

print(f"Successfully packaged {ZIP_PATH} ({os.path.getsize(ZIP_PATH):,} bytes).")
print(f"Total processing time: {time.time() - t_start:.2f}s")
