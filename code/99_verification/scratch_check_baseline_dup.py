import pandas as pd
import json
from pathlib import Path

path_a = Path("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/inference/grid/raw/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42__full_20260721.jsonl")
path_b = Path("/ANON/experiment_root/results/grounded_protocol_3lang_20260721/inference/perturbation/raw/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42__full_20260721.jsonl")

def load_cond(p, cond):
    res = []
    if p.exists():
        with p.open("r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("condition") == cond:
                        if "pair_id" not in d and "protocol_pair_id" in d:
                            d["pair_id"] = d["protocol_pair_id"]
                        res.append(d)
    return pd.DataFrame(res)

df_a = load_cond(path_a, "monokai_dark__fs20__wrap80__lnon")
df_b = load_cond(path_b, "baseline")

if len(df_b) == 0:
    print("Could not find baseline in path_b")
else:
    df_a = df_a.set_index(["pair_id", "order"])
    df_b = df_b.set_index(["pair_id", "order"])
    
    common = df_a.index.intersection(df_b.index)
    print(f"Overlap: {len(common)} calls")
    
    if "timestamp" in df_a.columns and "timestamp" in df_b.columns:
        ts_match = (df_a.loc[common, "timestamp"] == df_b.loc[common, "timestamp"]).mean()
        print(f"Timestamp match rate: {ts_match*100:.2f}%")
        
    verdict_match = (df_a.loc[common, "parsed_choice"] == df_b.loc[common, "parsed_choice"]).mean()
    print(f"Verdict match rate: {verdict_match*100:.2f}%")

