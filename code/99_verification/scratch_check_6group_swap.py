import pandas as pd
df_a = pd.read_csv("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/analysis/ab/A_GRID_RESULTS.csv")
# For each model and language, calculate the swap error across conditions
for model in ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B"]:
    sub = df_a[df_a["model"] == model]
    for lang, ldf in sub.groupby("language"):
        # We need median swap error across the 4 conditions in A_GRID
        s_vals = []
        for c, cdf in ldf.groupby("condition"):
            s = (cdf["n_pairs"].sum() - cdf["n_valid"].sum()) / cdf["n_pairs"].sum()
            s_vals.append(s)
        med_s = pd.Series(s_vals).median()
        print(f"{model} {lang}: median S = {med_s*100:.2f}% (min={min(s_vals)*100:.2f}, max={max(s_vals)*100:.2f})")
