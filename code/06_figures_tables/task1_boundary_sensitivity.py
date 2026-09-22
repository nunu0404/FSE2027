import pandas as pd
import numpy as np

# Load diagnostic battery (df_A)
path_A = "results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv"
df_A = pd.read_csv(path_A)

# Equivalence check: |c| == |b| vs (margin_ab == 0 or margin_ba == 0)
c = df_A["content_margin"].to_numpy()
b = df_A["position_margin"].to_numpy()
m_ab = df_A["margin_ab"].to_numpy()
m_ba = df_A["margin_ba"].to_numpy()

cond_cb = np.isclose(np.abs(c), np.abs(b), atol=1e-5)
cond_m0 = np.isclose(m_ab, 0, atol=1e-5) | np.isclose(m_ba, 0, atol=1e-5)

equiv = (cond_cb == cond_m0).all()
diff_count = (cond_cb != cond_m0).sum()
print(f"=== Equivalence Check: |c|==|b| vs (m_AB==0 or m_BA==0) ===")
print(f"Exactly equivalent across all 45,000 rows? {equiv} (mismatch count = {diff_count})")
print(f"Count of |c|==|b|: {cond_cb.sum()}")
print(f"Count of (m_AB==0 or m_BA==0): {cond_m0.sum()}")

# Sensitivity analysis under two rules:
# Conservative rule: boundary (|c| == |b|) treated as invalid
# Permissive rule: boundary where model actually chose same snippet (valid==True) is kept valid
# Wait, let's understand existing columns:
# In df_A:
#   'valid': was computed using what rule?
# Let's inspect df_A's existing 'valid' and 'correct':
print("\n=== df_A existing valid on boundary ===")
print("Total boundary rows:", cond_cb.sum())
print("Boundary rows with valid==True:", df_A.loc[cond_cb, "valid"].sum())
print("Boundary rows with correct==True:", df_A.loc[cond_cb, "correct"].sum())

# Let's verify what the Conservative Rule vs Permissive Rule means:
# In standard debiased/effective analysis:
# Permissive rule: If model chose same snippet in AB and BA (ab_first != ba_first), it is valid.
# Conservative rule: If |c| == |b| (which includes one margin == 0), invalidate the pair (valid = False, correct = False).
# Let's compute both rules:

results = []

# Conservative:
# valid_cons = valid & (~cond_cb)
# correct_cons = correct & (~cond_cb)
# Permissive:
# valid_perm = valid (i.e. existing valid, where identical choice means valid, even if on boundary)
# correct_perm = correct (existing correct)

df_A["is_boundary"] = cond_cb
df_A["valid_perm"] = df_A["valid"].astype(bool)
df_A["correct_perm"] = df_A["correct"].astype(bool)

df_A["valid_cons"] = df_A["valid_perm"] & (~df_A["is_boundary"])
df_A["correct_cons"] = df_A["correct_perm"] & (~df_A["is_boundary"])

# Groupings: pooled, per model, per language, per model x language
scopes = []

# 1. Pooled
scopes.append(("Pooled (All 5 models)", df_A))

# 2. Per model
for m, mdf in df_A.groupby("model"):
    scopes.append((f"Model: {m}", mdf))

# 3. Per language
for l, ldf in df_A.groupby("language"):
    scopes.append((f"Language: {l}", ldf))

# 4. Per model x language
for (m, l), mldf in df_A.groupby(["model", "language"]):
    scopes.append((f"{m} - {l}", mldf))

print("\n=== Boundary Sensitivity Metrics ===")
out_rows = []
for name, data in scopes:
    n = len(data)
    n_b = data["is_boundary"].sum()
    
    # Permissive metrics
    val_perm = data["valid_perm"].sum()
    cor_perm = data["correct_perm"].sum()
    valid_acc_perm = cor_perm / val_perm if val_perm > 0 else np.nan
    eff_acc_perm = cor_perm / n
    swap_err_perm = (n - val_perm) / n
    
    # Conservative metrics
    val_cons = data["valid_cons"].sum()
    cor_cons = data["correct_cons"].sum()
    valid_acc_cons = cor_cons / val_cons if val_cons > 0 else np.nan
    eff_acc_cons = cor_cons / n
    swap_err_cons = (n - val_cons) / n
    
    out_rows.append({
        "scope": name,
        "n_pairs": n,
        "n_boundary": n_b,
        "valid_acc_perm": valid_acc_perm * 100,
        "valid_acc_cons": valid_acc_cons * 100,
        "valid_acc_diff": (valid_acc_perm - valid_acc_cons) * 100,
        "eff_acc_perm": eff_acc_perm * 100,
        "eff_acc_cons": eff_acc_cons * 100,
        "eff_acc_diff": (eff_acc_perm - eff_acc_cons) * 100,
        "swap_err_perm": swap_err_perm * 100,
        "swap_err_cons": swap_err_cons * 100,
        "swap_err_diff": (swap_err_cons - swap_err_perm) * 100, # how much swap error increases under conservative
    })

res_df = pd.DataFrame(out_rows)
print(res_df.head(15).to_string())

# Abstract numbers check: Qwen2.5-VL-7B (pooled across all 3 languages)
qwen_res = res_df[res_df["scope"] == "Model: Qwen/Qwen2.5-VL-7B-Instruct"].iloc[0]
print("\n=== Abstract Numbers (Qwen2.5-VL-7B) ===")
print(f"Permissive Valid Acc: {qwen_res['valid_acc_perm']:.2f}% (Abstract says 71.17%)")
print(f"Conservative Valid Acc: {qwen_res['valid_acc_cons']:.2f}% (Diff: {qwen_res['valid_acc_diff']:+.2f}%p)")
print(f"Permissive Strict-Swap Error: {qwen_res['swap_err_perm']:.2f}% (Abstract says 45.78%)")
print(f"Conservative Strict-Swap Error: {qwen_res['swap_err_cons']:.2f}% (Diff: {qwen_res['swap_err_diff']:+.2f}%p)")
