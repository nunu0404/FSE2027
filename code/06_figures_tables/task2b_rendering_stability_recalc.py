import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. Load RQ1 S (pooled strict_swap_error)
df_rq1 = pd.read_csv("/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv")
rq1_s = df_rq1[(df_rq1["language"] == "pooled") & (df_rq1["metric"] == "S")].set_index("judge")["point"].to_dict()

grid_models = ["InternVL3-8B", "Qwen2.5-VL-7B", "InternVL3.5-8B", "Gemma-3-12B", "Gemma4-12B"]
models4 = ["Qwen2.5-VL-7B", "InternVL3.5-8B", "Gemma-3-12B", "Gemma4-12B"]

# 2. Conditions to keep: Wrap and Font only
base_cond = "monokai_dark__fs20__wrap80__lnon"
wrap_cond = "monokai_dark__fs20__wrap60__lnon"
font_cond = "monokai_dark__fs24__wrap80__lnon"
target_conds = [base_cond, wrap_cond, font_cond]

df_g3 = pd.read_csv("rq2_eval_reduced/gemma/summary/Gemma-3-12B_summary_metrics.csv")
df_i35 = pd.read_csv("rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv")
df_g4 = pd.read_csv("rq2_eval_reduced/gemma4/summary/Gemma4-12B_summary_metrics.csv")
df_a = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/A_GRID_RESULTS.csv")

def get_per_lang_s(df, model_filter=None):
    if model_filter:
        df = df[df["model"] == model_filter]
    
    # We need to extract S for each language and condition
    # S = (n_pairs - n_valid) / n_pairs or strict_swap_error
    records = []
    for (lang, cond), group in df.groupby(["language", "condition"]):
        if cond not in target_conds:
            continue
        if "n_valid" in group.columns and "n_pairs" in group.columns:
            p = group["n_pairs"].sum()
            v = group["n_valid"].sum()
            s_val = (p - v) / p
        else:
            s_val = group["strict_swap_error"].mean()
        records.append({"language": lang, "condition": cond, "S": s_val})
    return pd.DataFrame(records)

s_dfs = {
    "Gemma-3-12B": get_per_lang_s(df_g3),
    "InternVL3.5-8B": get_per_lang_s(df_i35),
    "Gemma4-12B": get_per_lang_s(df_g4),
    "Qwen2.5-VL-7B": get_per_lang_s(df_a, "Qwen/Qwen2.5-VL-7B-Instruct"),
    "InternVL3-8B": get_per_lang_s(df_a, "OpenGVLab/InternVL3-8B"),
}

print("=== Maximum Absolute S Change (Wrap and Font only, Per Language) ===")
delta_s_max = {}

for m in grid_models:
    df_m = s_dfs[m]
    max_diff = 0.0
    max_info = ""
    for lang in df_m["language"].unique():
        sub = df_m[df_m["language"] == lang].set_index("condition")
        if base_cond not in sub.index:
            continue
        base_s = sub.loc[base_cond, "S"]
        for c in [wrap_cond, font_cond]:
            if c in sub.index:
                diff = abs(sub.loc[c, "S"] - base_s)
                if diff > max_diff:
                    max_diff = diff
                    max_info = f"{lang}, {c} (Base: {base_s*100:.2f}%, Cond: {sub.loc[c, 'S']*100:.2f}%)"
    delta_s_max[m] = max_diff
    print(f"{m:<15}: Max |ΔS| = {max_diff*100:5.2f}%p  --> {max_info}")

# 3. Spearman correlation and bootstrap CI
def boot_spearman(x, y, n_boot=10000):
    rng = np.random.default_rng(20260914)
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if np.std(x[idx]) > 1e-9 and np.std(y[idx]) > 1e-9:
            r, _ = spearmanr(x[idx], y[idx])
            if not np.isnan(r):
                rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

# 4 models (InternVL3 excluded)
x4 = [rq1_s[m] for m in models4]
y4 = [delta_s_max[m] for m in models4]
rho4, p4 = spearmanr(x4, y4)
ci4 = boot_spearman(x4, y4)

print("\n=== Spearman Correlation (4 Models, Wrap/Font Only) ===")
print(f"rho = {rho4:.4f}, p = {p4:.4f}")
print(f"95% CI: [{ci4[0]:.4f}, {ci4[1]:.4f}]")

# 4. Monte Carlo Null Distribution
print("\n=== Monte Carlo Null Distribution (Per Language, Wrap/Font Only) ===")
# 3 languages, 2 contrasts per language. N = 1000 per cell.
# We generate Base, Cond1, Cond2 for each language.
def sim_null_per_lang(models, n_pairs=1000, n_langs=3, n_contrasts=2, n_sim=10000):
    rng = np.random.default_rng(20260915)
    sim_rhos = []
    
    for _ in range(n_sim):
        cur_deltas = []
        cur_rq1_p = []
        for m in models:
            p = rq1_s[m]
            cur_rq1_p.append(p)
            
            m_max_diff = 0.0
            for _lang in range(n_langs):
                # 1 base + 2 contrasts
                s_samples = rng.binomial(n_pairs, p, size=1+n_contrasts) / n_pairs
                base_s = s_samples[0]
                diffs = np.abs(s_samples[1:] - base_s)
                lang_max = np.max(diffs)
                if lang_max > m_max_diff:
                    m_max_diff = lang_max
            cur_deltas.append(m_max_diff)
        
        r, _ = spearmanr(cur_rq1_p, cur_deltas)
        if not np.isnan(r):
            sim_rhos.append(r)
    return np.array(sim_rhos)

sim_rhos_4 = sim_null_per_lang(models4, n_pairs=1000, n_langs=3, n_contrasts=2)
pct_4 = (sim_rhos_4 < rho4).mean() * 100
print(f"Null Rho: Mean = {sim_rhos_4.mean():.4f}, Std = {sim_rhos_4.std():.4f}")
print(f"Observed rho = {rho4:.4f} is at {pct_4:.2f}th percentile of null distribution")
print(f"Null p-value = {(sim_rhos_4 >= rho4).mean():.4f}")
