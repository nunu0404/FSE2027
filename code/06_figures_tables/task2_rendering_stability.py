import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. Load RQ1 S (pooled strict_swap_error) from rq1_intervals.csv
df_rq1 = pd.read_csv("/ANON/home/fse2027_handoff_rq1_20260912/rq1_intervals.csv")
# Filter pooled, metric S
rq1_s = df_rq1[(df_rq1["language"] == "pooled") & (df_rq1["metric"] == "S")].set_index("judge")["point"].to_dict()
print("=== RQ1 Pooled Strict-Swap Error S ===")
for k, v in sorted(rq1_s.items()):
    print(f"  {k}: {v*100:.2f}%")

# Models in reduced-grid set:
# InternVL3-8B, Qwen2.5-VL-7B, InternVL3.5-8B, Gemma-3-12B, Gemma4-12B
grid_models = ["InternVL3-8B", "Qwen2.5-VL-7B", "InternVL3.5-8B", "Gemma-3-12B", "Gemma4-12B"]

# 2. Get Reduced Grid S for each model across the 7 conditions:
# The 7 conditions are:
# - monokai_dark__fs20__wrap80__lnon (baseline)
# - monokai_dark__fs20__wrap60__lnon
# - monokai_dark__fs24__wrap80__lnon
# - mono_light__fs20__wrap80__lnon
# - no_indent
# - no_blank_lines
# - gaussian_sigma_4

# Load Gemma-3-12B
df_g3 = pd.read_csv("rq2_eval_reduced/gemma/summary/Gemma-3-12B_summary_metrics.csv")
# Load InternVL3.5-8B
df_i35 = pd.read_csv("rq2_eval_reduced/internvl3_5/summary/InternVL3.5-8B_summary_metrics.csv")
# Load Gemma4-12B
df_g4 = pd.read_csv("rq2_eval_reduced/gemma4/summary/Gemma4-12B_summary_metrics.csv")

# Load Qwen2.5 & InternVL3 from grounded_protocol_3lang_20260721
df_a = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/A_GRID_RESULTS.csv")
df_b = pd.read_csv("results/grounded_protocol_3lang_20260721/analysis/ab/B_PERTURBATION_RESULTS.csv")

# Let's inspect how S is pooled across languages or per language in reduced grid:
# In reduced grid, there are 1,000 pairs per language = 3,000 pairs per condition.
# Let's compute pooled S per condition for each model.

def get_pooled_s_dict(df, model_filter=None):
    # df has language, condition, n_pairs, n_valid (or strict_swap_error)
    # If df has n_valid and n_pairs:
    s_by_cond = {}
    if model_filter:
        df = df[df["model"] == model_filter]
    for cond, cdf in df.groupby("condition"):
        if "n_valid" in cdf.columns and "n_pairs" in cdf.columns:
            tot_pairs = cdf["n_pairs"].sum()
            tot_valid = cdf["n_valid"].sum()
            s_by_cond[cond] = (tot_pairs - tot_valid) / tot_pairs
        elif "strict_swap_error" in cdf.columns:
            s_by_cond[cond] = cdf["strict_swap_error"].mean()
    return s_by_cond

# For Qwen2.5 & InternVL3, map conditions:
# A_GRID has:
# monokai_dark__fs20__wrap80__lnon (baseline)
# monokai_dark__fs20__wrap60__lnon
# monokai_dark__fs24__wrap80__lnon
# mono_light__fs20__wrap80__lnon
# B_PERTURBATION has:
# no_indent
# no_blank_lines
# gaussian_sigma_4
# baseline (same as monokai_dark__fs20__wrap80__lnon)

def get_legacy_grid(model_name):
    sub_a = df_a[df_a["model"] == model_name]
    sub_b = df_b[df_b["model"] == model_name]
    
    cond_s = {}
    # from sub_a
    for c in ["monokai_dark__fs20__wrap80__lnon", "monokai_dark__fs20__wrap60__lnon", "monokai_dark__fs24__wrap80__lnon", "mono_light__fs20__wrap80__lnon"]:
        part = sub_a[sub_a["condition"] == c]
        tot_p = part["n_pairs"].sum()
        tot_v = part["n_valid"].sum()
        cond_s[c] = (tot_p - tot_v) / tot_p
        
    # from sub_b
    for c in ["no_indent", "no_blank_lines", "gaussian_sigma_4"]:
        part = sub_b[sub_b["condition"] == c]
        tot_p = part["n_pairs"].sum()
        tot_v = part["n_valid"].sum()
        cond_s[c] = (tot_p - tot_v) / tot_p
        
    return cond_s

s_models = {
    "Gemma-3-12B": get_pooled_s_dict(df_g3),
    "InternVL3.5-8B": get_pooled_s_dict(df_i35),
    "Gemma4-12B": get_pooled_s_dict(df_g4),
    "Qwen2.5-VL-7B": get_legacy_grid("Qwen/Qwen2.5-VL-7B-Instruct"),
    "InternVL3-8B": get_legacy_grid("OpenGVLab/InternVL3-8B"),
}

print("\n=== Pooled S by Condition for 5 Models ===")
delta_s_max = {}
base_cond = "monokai_dark__fs20__wrap80__lnon"

for m in grid_models:
    conds = s_models[m]
    s_base = conds[base_cond]
    diffs = {c: abs(s - s_base) for c, s in conds.items() if c != base_cond}
    max_c = max(diffs, key=diffs.get)
    max_val = diffs[max_c]
    delta_s_max[m] = max_val
    print(f"\nModel: {m} (Baseline S = {s_base*100:.2f}%)")
    for c, s in sorted(conds.items()):
        print(f"  {c:<35}: S={s*100:.2f}% (diff={abs(s-s_base)*100:.2f}%p)")
    print(f"  --> Max |ΔS| = {max_val*100:.2f}%p at {max_c}")

# Let's also check if max |ΔS| could be per language or across all (lang, cond) cells:
print("\n=== Check: Max |ΔS| across all 21 (lang, cond) cells per model ===")
delta_s_max_cell = {}
for m in grid_models:
    if m == "Gemma-3-12B":
        df_m = df_g3
    elif m == "InternVL3.5-8B":
        df_m = df_i35
    elif m == "Gemma4-12B":
        df_m = df_g4
    else:
        full_m = "Qwen/Qwen2.5-VL-7B-Instruct" if m == "Qwen2.5-VL-7B" else "OpenGVLab/InternVL3-8B"
        part_a = df_a[df_a["model"] == full_m]
        part_b = df_b[df_b["model"] == full_m]
        # combine
        df_m = pd.concat([part_a, part_b[part_b["condition"] != "baseline"]], ignore_index=True)
    
    # Per language baseline:
    cell_diffs = []
    for lang, ldf in df_m.groupby("language"):
        base_sub = ldf[ldf["condition"].isin([base_cond, "baseline"])]
        if len(base_sub) > 0:
            s_base_l = base_sub["strict_swap_error"].iloc[0]
            for _, r in ldf.iterrows():
                if r["condition"] not in [base_cond, "baseline"]:
                    cell_diffs.append(abs(r["strict_swap_error"] - s_base_l))
    delta_s_max_cell[m] = max(cell_diffs)
    print(f"Model {m}: max cell |ΔS| = {max(cell_diffs)*100:.2f}%p")

# Let's compute Spearman correlations and bootstrap CIs:
# (1) 5 models (pooled S vs pooled max |ΔS|)
x5_rq1 = [rq1_s[m] for m in grid_models]
y5_pooled = [delta_s_max[m] for m in grid_models]
y5_cell = [delta_s_max_cell[m] for m in grid_models]

rho5_p, p5_p = spearmanr(x5_rq1, y5_pooled)
rho5_c, p5_c = spearmanr(x5_rq1, y5_cell)
print(f"\n5 Models (with InternVL3):")
print(f"  Pooled max |ΔS|: Spearman rho = {rho5_p:.4f}, p = {p5_p:.4f}")
print(f"  Cell max |ΔS|:   Spearman rho = {rho5_c:.4f}, p = {p5_c:.4f}")

# (2) 4 models (without InternVL3: Qwen2.5, InternVL3.5, Gemma-3, Gemma4)
models4 = ["Qwen2.5-VL-7B", "InternVL3.5-8B", "Gemma-3-12B", "Gemma4-12B"]
x4_rq1 = [rq1_s[m] for m in models4]
y4_pooled = [delta_s_max[m] for m in models4]
y4_cell = [delta_s_max_cell[m] for m in models4]

rho4_p, p4_p = spearmanr(x4_rq1, y4_pooled)
rho4_c, p4_c = spearmanr(x4_rq1, y4_cell)
print(f"\n4 Models (without InternVL3 - PRIMARY ANALYSIS):")
print(f"  Pooled max |ΔS|: Spearman rho = {rho4_p:.4f}, p = {p4_p:.4f}")
print(f"  Cell max |ΔS|:   Spearman rho = {rho4_c:.4f}, p = {p4_c:.4f}")

# Bootstrap CI for Spearman rho (n=10,000 resamples):
rng = np.random.default_rng(20260914)

def boot_spearman(x, y, n_boot=10000):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # check if variance is > 0
        if np.std(x[idx]) > 1e-9 and np.std(y[idx]) > 1e-9:
            r, _ = spearmanr(x[idx], y[idx])
            if not np.isnan(r):
                rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

ci5_p = boot_spearman(x5_rq1, y5_pooled)
ci4_p = boot_spearman(x4_rq1, y4_pooled)
ci5_c = boot_spearman(x5_rq1, y5_cell)
ci4_c = boot_spearman(x4_rq1, y4_cell)

print(f"\nBootstrap 95% CIs:")
print(f"  5 models (pooled): rho = {rho5_p:.4f} [{ci5_p[0]:.4f}, {ci5_p[1]:.4f}]")
print(f"  4 models (pooled): rho = {rho4_p:.4f} [{ci4_p[0]:.4f}, {ci4_p[1]:.4f}]")
print(f"  5 models (cell):   rho = {rho5_c:.4f} [{ci5_c[0]:.4f}, {ci5_c[1]:.4f}]")
print(f"  4 models (cell):   rho = {rho4_c:.4f} [{ci4_c[0]:.4f}, {ci4_c[1]:.4f}]")

# 3. Null Distribution via Monte Carlo:
# "귀무 분포를 만든다. 각 모델의 RQ1 S 를 모수로 두고 같은 표본 크기에서
#  '최대 절대 S 변화' 를 몬테카를로로 10,000회 생성한다. 관측값의 분위를 보고한다.
#  비율의 분산이 0.5 근처에서 최대라는 점이 대안 설명이므로 이 검정이 필요하다."
print("\n=== Monte Carlo Null Distribution ===")
# Under the null hypothesis:
# For each model, S under all conditions is generated from Binomial(N, S_rq1) / N.
# Sample size N per condition is N=3,000 (pooled) or N=1,000 (per lang).
# Let's test both pooled (N=3,000, 7 conditions, 6 non-baseline) and per-cell (N=1,000, 21 cells, 18 non-baseline):

def sim_null_delta_s(models, n_pairs, n_conds, n_sim=10000):
    # For each model, generate baseline S ~ Binom(N, p)/N, and (n_conds-1) conditions ~ Binom(N, p)/N
    # Then max |S_k - S_baseline|
    # Then compute Spearman rho between true p (or observed baseline) and max |ΔS|
    sim_rhos = []
    sim_max_deltas = {m: [] for m in models}
    
    for _ in range(n_sim):
        cur_deltas = []
        cur_rq1_p = []
        for m in models:
            p = rq1_s[m]
            cur_rq1_p.append(p)
            s_samples = rng.binomial(n_pairs, p, size=n_conds) / n_pairs
            s_base = s_samples[0]
            max_d = np.max(np.abs(s_samples[1:] - s_base))
            cur_deltas.append(max_d)
            sim_max_deltas[m].append(max_d)
        r, _ = spearmanr(cur_rq1_p, cur_deltas)
        if not np.isnan(r):
            sim_rhos.append(r)
    return np.array(sim_rhos), sim_max_deltas

# Simulation for 4 models (pooled: N=3000, 7 conditions)
sim_rhos_4, sim_deltas_4 = sim_null_delta_s(models4, n_pairs=3000, n_conds=7)
# Simulation for 5 models (pooled: N=3000, 7 conditions)
sim_rhos_5, sim_deltas_5 = sim_null_delta_s(grid_models, n_pairs=3000, n_conds=7)

# Percentile of observed rho in null distribution:
pct_4 = (sim_rhos_4 < rho4_p).mean() * 100
pct_5 = (sim_rhos_5 < rho5_p).mean() * 100
print(f"4 Models Null Rho: Mean = {sim_rhos_4.mean():.4f}, Std = {sim_rhos_4.std():.4f}")
print(f"  Observed rho4 = {rho4_p:.4f} is at {pct_4:.2f}th percentile of null distribution (p = {(sim_rhos_4 >= rho4_p).mean():.4f})")

print(f"\n5 Models Null Rho: Mean = {sim_rhos_5.mean():.4f}, Std = {sim_rhos_5.std():.4f}")
print(f"  Observed rho5 = {rho5_p:.4f} is at {pct_5:.2f}th percentile of null distribution (p = {(sim_rhos_5 >= rho5_p).mean():.4f})")

# Also report observed max |ΔS| vs null max |ΔS| for each model:
print("\nModel-level Max |ΔS| vs Null Expected (Pooled):")
for m in grid_models:
    obs = delta_s_max[m]
    null_dist = np.array(sim_deltas_5[m])
    null_mean = null_dist.mean()
    null_p95 = np.percentile(null_dist, 95)
    pct = (null_dist < obs).mean() * 100
    print(f"  {m:<15}: Obs={obs*100:.2f}%p | Null Mean={null_mean*100:.2f}%p (95th={null_p95*100:.2f}%p) | Percentile={pct:.1f}%")
