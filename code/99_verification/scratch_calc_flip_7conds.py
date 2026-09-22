import pandas as pd
import numpy as np
import json
from pathlib import Path

RAW_DIR = Path("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/inference/grid/raw")
RAW_DIR_PERTURB = Path("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/inference/perturbations/raw")

models = [
    ("Qwen2.5-VL-7B", "Qwen__Qwen2.5-VL-7B-Instruct"),
    ("InternVL3-8B", "OpenGVLab__InternVL3-8B")
]

conds = [
    "baseline", # monokai_dark__fs20__wrap80__lnon
    "monokai_dark__fs20__wrap60__lnon",
    "monokai_dark__fs24__wrap80__lnon",
    "mono_light__fs20__wrap80__lnon",
    "no_indent",
    "no_blank_lines",
    "gaussian_sigma_4"
]

def load_jsonl(path):
    calls = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    df = pd.DataFrame(calls)
    if "pair_id" not in df.columns and "protocol_pair_id" in df.columns:
        df["pair_id"] = df["protocol_pair_id"]
    return df

for m_short, m_prefix in models:
    # A_GRID (full)
    path_a = RAW_DIR / f"{m_prefix}__image_only__promptB__seed42__full_20260721.jsonl"
    df_a = load_jsonl(path_a)
    
    # B_PERTURBATION (full)
    path_b = RAW_DIR_PERTURB / f"{m_prefix}__image_only__promptB__seed42__full_20260721.jsonl"
    if path_b.exists():
        df_b = load_jsonl(path_b)
    else:
        # Check for ga0
        p = RAW_DIR_PERTURB / f"{m_prefix}__image_only__promptB__seed42__ga0_20260721.jsonl"
        df_b = load_jsonl(p) if p.exists() else pd.DataFrame()
        
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    df_c = {}
    for c in conds:
        # map baseline
        cond_query = "monokai_dark__fs20__wrap80__lnon" if c == "baseline" else c
        part = df_combined[df_combined["condition"] == cond_query]
        # remove duplicates because baseline exists in both A and B
        part = part.drop_duplicates(subset=["pair_id", "order"])
        df_c[c] = part

    for lang in ["cuda", "java", "python"]:
        flip_rates_all = []
        
        for i in range(len(conds)):
            for j in range(i+1, len(conds)):
                c1 = conds[i]
                c2 = conds[j]
                
                df1 = df_c[c1]
                df2 = df_c[c2]
                
                lang1 = df1[df1["language"] == lang]
                lang2 = df2[df2["language"] == lang]
                
                for ord_val in ["AB", "BA"]:
                    sub1 = lang1[lang1["order"] == ord_val].set_index("pair_id")
                    sub2 = lang2[lang2["order"] == ord_val].set_index("pair_id")
                    common = sub1.index.intersection(sub2.index)
                    if len(common) > 0:
                        p1 = sub1.loc[common, "parsed_choice"]
                        p2 = sub2.loc[common, "parsed_choice"]
                        valid_mask = p1.notna() & p2.notna()
                        if valid_mask.any():
                            flips = (p1[valid_mask] != p2[valid_mask]).mean()
                            flip_rates_all.append(flips)
                            
        if flip_rates_all:
            med = float(np.median(flip_rates_all))
            print(f"{m_short} {lang}: median flips (7 conds) = {med*100:.2f}%")

