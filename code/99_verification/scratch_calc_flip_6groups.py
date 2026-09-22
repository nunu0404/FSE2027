import pandas as pd
import numpy as np
import json
import math
from pathlib import Path

RAW_DIR = Path("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/inference/grid/raw")

models = [
    ("Qwen2.5-VL-7B", "Qwen__Qwen2.5-VL-7B-Instruct"),
    ("InternVL3-8B", "OpenGVLab__InternVL3-8B")
]

def load_jsonl(path):
    calls = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Map condition if it's missing but we know it's full 20260721
                cond = d.get("condition")
                calls.append(d)
    df = pd.DataFrame(calls)
    if "pair_id" not in df.columns and "protocol_pair_id" in df.columns:
        df["pair_id"] = df["protocol_pair_id"]
    return df

results = []

for m_short, m_prefix in models:
    path = RAW_DIR / f"{m_prefix}__image_only__promptB__seed42__full_20260721.jsonl"
    df = load_jsonl(path)
    
    conds = sorted(df["condition"].dropna().unique().tolist())
    print(f"{m_short}: {len(conds)} unique conditions")
    
    # Pre-calculate paired dataframe (strict_valid)
    paired_rows = []
    for (pair_id, lang, cond), group in df.groupby(["pair_id", "language", "condition"]):
        ab = group[group["order"] == "AB"]
        ba = group[group["order"] == "BA"]
        if len(ab) == 1 and len(ba) == 1:
            ab_row = ab.iloc[0]
            ba_row = ba.iloc[0]
            
            def get_snippet(row):
                if row["parsed_choice"] == "A": return row["snippet_first"]
                if row["parsed_choice"] == "B": return row["snippet_second"]
                return None
            
            s_ab = get_snippet(ab_row)
            s_ba = get_snippet(ba_row)
            
            strict_valid = (s_ab is not None) and (s_ba is not None) and (s_ab == s_ba)
            paired_rows.append({
                "pair_id": pair_id,
                "language": lang,
                "condition": cond,
                "strict_valid": strict_valid
            })
    paired = pd.DataFrame(paired_rows)
    
    for lang in ["cuda", "java", "python"]:
        lang_calls = df[df["language"] == lang]
        flip_rates_all = []
        flip_rates_valid_both = []
        
        for i in range(len(conds)):
            for j in range(i+1, len(conds)):
                c1 = conds[i]
                c2 = conds[j]
                
                for ord_val in ["AB", "BA"]:
                    sub1 = lang_calls[(lang_calls.condition == c1) & (lang_calls.order == ord_val)].set_index("pair_id")
                    sub2 = lang_calls[(lang_calls.condition == c2) & (lang_calls.order == ord_val)].set_index("pair_id")
                    common = sub1.index.intersection(sub2.index)
                    if len(common) > 0:
                        p1 = sub1.loc[common, "parsed_choice"]
                        p2 = sub2.loc[common, "parsed_choice"]
                        valid_mask = p1.notna() & p2.notna()
                        if valid_mask.any():
                            flips = (p1[valid_mask] != p2[valid_mask]).mean()
                            flip_rates_all.append(flips)
                            
                        # valid-both
                        part1 = paired[(paired.language == lang) & (paired.condition == c1)].set_index("pair_id")
                        part2 = paired[(paired.language == lang) & (paired.condition == c2)].set_index("pair_id")
                        val_both_pairs = part1.index[part1.strict_valid & part2.strict_valid]
                        
                        sub_val1 = sub1.loc[sub1.index.intersection(val_both_pairs), "parsed_choice"]
                        sub_val2 = sub2.loc[sub2.index.intersection(val_both_pairs), "parsed_choice"]
                        if len(sub_val1) > 0:
                            val_flips = (sub_val1 != sub_val2).mean()
                            flip_rates_valid_both.append(val_flips)
                            
        median_flip_rate = float(np.median(flip_rates_all)) if flip_rates_all else math.nan
        median_flip_valid_both = float(np.nanmedian(flip_rates_valid_both)) if flip_rates_valid_both else math.nan
        
        # also calculate swap_error to check order_exceeds_rendering
        lang_paired = paired[paired["language"] == lang]
        swap_errors = []
        for c in conds:
            c_paired = lang_paired[lang_paired["condition"] == c]
            if len(c_paired) > 0:
                swap = 1.0 - c_paired["strict_valid"].mean()
                swap_errors.append(swap)
        median_swap_err = float(np.median(swap_errors)) if swap_errors else math.nan
        
        fr_min = min(flip_rates_all)*100
        fr_max = max(flip_rates_all)*100
        
        print(f"[{m_short} {lang}] flip: {median_flip_rate*100:.2f}%, S: {median_swap_err*100:.2f}%")
        results.append({
            "model": m_short,
            "language": lang,
            "median_rendering_flip_rate": median_flip_rate,
            "flip_rate_range": f"{fr_min:.1f}% - {fr_max:.1f}%",
            "median_flip_rate_valid_both": median_flip_valid_both,
            "order_exceeds_rendering": bool(median_swap_err > median_flip_rate)
        })

# load csv and update
csv_path = "/ANON/experiment_root/results/verified/rq2_order_vs_rendering.csv"
df_csv = pd.read_csv(csv_path)

for r in results:
    mask = (df_csv["model"] == r["model"]) & (df_csv["language"] == r["language"])
    if mask.any():
        idx = df_csv[mask].index[0]
        df_csv.loc[idx, "median_rendering_flip_rate"] = r["median_rendering_flip_rate"]
        df_csv.loc[idx, "flip_rate_range"] = r["flip_rate_range"]
        df_csv.loc[idx, "median_flip_rate_valid_both"] = r["median_flip_rate_valid_both"]
        df_csv.loc[idx, "order_exceeds_rendering"] = r["order_exceeds_rendering"]

df_csv.to_csv(csv_path, index=False)
