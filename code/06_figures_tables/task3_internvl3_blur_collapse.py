import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import chi2, mannwhitneyu

print("=== Task 3: InternVL3-8B CUDA Blur Swap Drop & Visual Feature Regression ===")

# 1. Load Pair-Level data for InternVL3-8B CUDA
pair_path = "results/grounded_protocol_3lang_20260721/analysis/ab/A_B_PAIR_LEVEL.csv"
df_pairs = pd.read_csv(pair_path)

sub_base = df_pairs[(df_pairs["model"] == "OpenGVLab/InternVL3-8B") & 
                    (df_pairs["language"] == "cuda") & 
                    (df_pairs["condition"] == "baseline")].copy()

sub_blur = df_pairs[(df_pairs["model"] == "OpenGVLab/InternVL3-8B") & 
                    (df_pairs["language"] == "cuda") & 
                    (df_pairs["condition"] == "gaussian_sigma_4")].copy()

print(f"Loaded pairs: Baseline={len(sub_base)}, Gaussian Sigma 4={len(sub_blur)}")
print(f"Baseline Strict-Swap Error: {(1.0 - sub_base['strict_valid'].mean())*100:.2f}%")
print(f"Blur Strict-Swap Error:     {(1.0 - sub_blur['strict_valid'].mean())*100:.2f}%")

# 2. Extract visual features for CUDA snippets
cuda_snippets = sorted(set(sub_base["snippet_i"]) | set(sub_base["snippet_j"]))
print(f"Total unique CUDA snippets: {len(cuda_snippets)}")

img_dir_base = "results/grounded_protocol_3lang_20260721/rendered/perturbation/baseline"

def extract_features(snippet_id, img_dir):
    path = os.path.join(img_dir, f"{snippet_id}.png")
    if not os.path.exists(path):
        path = os.path.join("results/grounded_protocol_3lang_20260721/rendered/grid/monokai_dark__fs20__wrap80__lnon", f"{snippet_id}.png")
    
    img = Image.open(path).convert("RGB")
    arr = np.array(img) # H, W, 3
    h, w, _ = arr.shape
    
    bg_color = arr[0, 0, :]
    diff = np.abs(arr.astype(int) - bg_color.astype(int)).sum(axis=-1)
    fg_mask = diff > 25
    
    total_area = float(h * w)
    fg_pixels = float(fg_mask.sum())
    non_blank_ratio = fg_pixels / total_area if total_area > 0 else 0.0
    
    row_has_fg = fg_mask.any(axis=1)
    line_indices = np.where(row_has_fg)[0]
    
    if len(line_indices) > 0:
        y_min, y_max = line_indices[0], line_indices[-1]
        col_has_fg = fg_mask.any(axis=0)
        col_indices = np.where(col_has_fg)[0]
        x_min, x_max = col_indices[0], col_indices[-1]
        block_width = float(x_max - x_min + 1)
        
        line_count = 0
        in_line = False
        indents = []
        for r in range(h):
            if row_has_fg[r]:
                if not in_line:
                    line_count += 1
                    in_line = True
                    first_c = np.where(fg_mask[r, :])[0][0]
                    indents.append(first_c)
            else:
                in_line = False
        avg_indent = float(np.mean(indents)) if indents else 0.0
    else:
        block_width = 0.0
        line_count = 0
        avg_indent = 0.0
        
    return {
        "line_count": float(line_count),
        "non_blank_pixel_ratio": float(non_blank_ratio),
        "block_width": float(block_width),
        "avg_indent_depth": float(avg_indent),
        "total_area": float(total_area)
    }

feats_base = {s: extract_features(s, img_dir_base) for s in cuda_snippets}

feature_names = ["line_count", "non_blank_pixel_ratio", "block_width", "avg_indent_depth", "total_area"]

def prepare_regression_data(sub_df, feat_dict):
    X_list = []
    y_list = []
    for _, row in sub_df.iterrows():
        si = row["snippet_i"]
        sj = row["snippet_j"]
        if si not in feat_dict or sj not in feat_dict:
            continue
        fi = feat_dict[si]
        fj = feat_dict[sj]
        diff_feat = [fi[k] - fj[k] for k in feature_names]
        X_list.append(diff_feat)
        y = 1 if row["selected_ab"] == si else 0
        y_list.append(y)
    X = pd.DataFrame(X_list, columns=feature_names)
    y = np.array(y_list, dtype=float)
    # Standardize X
    X_std = (X - X.mean()) / (X.std() + 1e-9)
    # Add intercept column
    X_std.insert(0, "intercept", 1.0)
    return X_std.to_numpy(), y

X_base, y_base = prepare_regression_data(sub_base, feats_base)
X_blur, y_blur = prepare_regression_data(sub_blur, feats_base)

def fit_logistic(X, y):
    n, p = X.shape
    # Negative log-likelihood
    def loss(w):
        z = np.clip(X @ w, -30, 30)
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-15
        p_hat = np.clip(p_hat, eps, 1.0 - eps)
        ll = y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat)
        return -np.sum(ll)
    
    init_w = np.zeros(p)
    res = minimize(loss, init_w, method='L-BFGS-B')
    w_opt = res.x
    
    z = np.clip(X @ w_opt, -30, 30)
    p_hat = 1.0 / (1.0 + np.exp(-z))
    ll_model = -res.fun
    
    # Null log likelihood (intercept only)
    y_mean = np.mean(y)
    eps = 1e-15
    y_mean = np.clip(y_mean, eps, 1.0 - eps)
    ll_null = np.sum(y * np.log(y_mean) + (1.0 - y) * np.log(1.0 - y_mean))
    
    mcfadden_r2 = 1.0 - (ll_model / ll_null)
    lr_stat = 2.0 * (ll_model - ll_null)
    pval = 1.0 - chi2.cdf(lr_stat, df=p - 1)
    
    # AUC via Mann-Whitney U
    pos = p_hat[y == 1]
    neg = p_hat[y == 0]
    if len(pos) > 0 and len(neg) > 0:
        u_stat, _ = mannwhitneyu(pos, neg, alternative='greater')
        auc = u_stat / (len(pos) * len(neg))
    else:
        auc = 0.5
        
    return w_opt, p_hat, auc, mcfadden_r2, lr_stat, pval

w_base, p_hat_base, auc_base, r2_base, lr_base, pval_base = fit_logistic(X_base, y_base)
w_blur, p_hat_blur, auc_blur, r2_blur, lr_blur, pval_blur = fit_logistic(X_blur, y_blur)

col_names = ["intercept"] + feature_names
print("\n=== Logistic Regression Results (Outcome = Model picked snippet_i in AB) ===")
print("--- Baseline Condition ---")
print(f"  McFadden Pseudo R2: {r2_base:.4f}")
print(f"  ROC AUC:            {auc_base:.4f}")
print(f"  LR Stat (df=5):     {lr_base:.2f} (p = {pval_base:.4e})")
for cn, w in zip(col_names, w_base):
    print(f"    {cn:<25}: w = {w:+.4f}")

print("\n--- Gaussian Blur (sigma=4) Condition ---")
print(f"  McFadden Pseudo R2: {r2_blur:.4f}")
print(f"  ROC AUC:            {auc_blur:.4f}")
print(f"  LR Stat (df=5):     {lr_blur:.2f} (p = {pval_blur:.4e})")
for cn, w in zip(col_names, w_blur):
    print(f"    {cn:<25}: w = {w:+.4f}")

# 4. Choice / Order Behavior Distribution
print("\n=== Choice / Order Behavior Distribution ===")
for cond_name, df_c in [("Baseline", sub_base), ("Gaussian Blur (sigma=4)", sub_blur)]:
    ab_pick_1st = (df_c["selected_ab"] == df_c["snippet_i"]).mean()
    ba_pick_1st = (df_c["selected_ba"] == df_c["snippet_j"]).mean()
    valid_rate = df_c["strict_valid"].mean()
    both_i = ((df_c["selected_ab"] == df_c["snippet_i"]) & (df_c["selected_ba"] == df_c["snippet_i"])).mean()
    both_j = ((df_c["selected_ab"] == df_c["snippet_j"]) & (df_c["selected_ba"] == df_c["snippet_j"])).mean()
    always_1st = ((df_c["selected_ab"] == df_c["snippet_i"]) & (df_c["selected_ba"] == df_c["snippet_j"])).mean()
    always_2nd = ((df_c["selected_ab"] == df_c["snippet_j"]) & (df_c["selected_ba"] == df_c["snippet_i"])).mean()
    
    print(f"\nCondition: {cond_name}")
    print(f"  AB presentation - Picked 1st (A): {ab_pick_1st*100:.2f}%, Picked 2nd (B): {(1-ab_pick_1st)*100:.2f}%")
    print(f"  BA presentation - Picked 1st (A): {ba_pick_1st*100:.2f}%, Picked 2nd (B): {(1-ba_pick_1st)*100:.2f}%")
    print(f"  Position Strategy: Always 1st = {always_1st*100:.2f}%, Always 2nd = {always_2nd*100:.2f}% (Total Pos Bias: {(always_1st+always_2nd)*100:.2f}%)")
    print(f"  Snippet Tracking:  Always snippet_i = {both_i*100:.2f}%, Always snippet_j = {both_j*100:.2f}% (Total Valid: {(both_i+both_j)*100:.2f}%)")
    print(f"  Strict-Swap Error S: {(1-valid_rate)*100:.2f}%")
