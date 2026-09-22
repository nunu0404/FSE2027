import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append("/ANON/experiment_root/rq2_eval_reduced/scripts")
import analyze_rq2_reduced as ar2

ROOT = Path("/ANON/experiment_root")
raw_path = ROOT / "results/rq1_model_battery_3lang_20260723/inference/full/qwen/raw.jsonl"
pairs_meta = pd.read_csv(ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv")

calls = []
with raw_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            calls.append(json.loads(line))
calls_df = pd.DataFrame(calls)

# Pair the calls:
# analyze_rq2_reduced expects pair_id, z_i, z_j, human_score_i_z, human_score_j_z, etc.
# Actually, analyze_rq2_reduced.pair_calls assumes the meta has "protocol_pair_id".
# rq1_pairs_9000.csv has "protocol_pair_id" or "pair_id"?
# Let's check columns.
pairs_meta = pairs_meta.rename(columns={"protocol_pair_id": "pair_id"})
if "pair_id" not in pairs_meta.columns:
    pairs_meta["pair_id"] = pairs_meta.index

# Since analyze_rq2_reduced expects certain columns, I'll just do it manually.
paired_rows = []
for pair_id, group in calls_df.groupby("pair_id"):
    if len(group) != 2: continue
    ab = group[group["order"] == "AB"].iloc[0] if sum(group["order"] == "AB") > 0 else None
    ba = group[group["order"] == "BA"].iloc[0] if sum(group["order"] == "BA") > 0 else None
    if ab is None or ba is None: continue
    
    parsed_ab = not pd.isna(ab.get("parsed_choice")) and ab.get("parsed_choice") in ["A", "B"]
    parsed_ba = not pd.isna(ba.get("parsed_choice")) and ba.get("parsed_choice") in ["A", "B"]
    
    parsed_both = parsed_ab and parsed_ba
    
    # valid means parsed both AND choice is swapped (i.e. AB choice == A means snippet_first, BA choice == B means snippet_first)
    # Actually:
    def chosen_snippet(row):
        if row.get("parsed_choice") == "A": return row.get("snippet_first")
        if row.get("parsed_choice") == "B": return row.get("snippet_second")
        return None
        
    s_ab = chosen_snippet(ab)
    s_ba = chosen_snippet(ba)
    
    valid = parsed_both and (s_ab == s_ba) and s_ab is not None
    
    lang = ab.get("language")
    
    paired_rows.append({
        "pair_id": pair_id,
        "language": lang,
        "parsed_both": parsed_both,
        "valid": valid
    })

df = pd.DataFrame(paired_rows)

print("=== RQ1 Qwen2.5-VL-7B Invalidity Decomposition ===")
def print_stats(lang, part):
    n = len(part)
    n_parsed = part["parsed_both"].sum()
    n_valid = part["valid"].sum()
    invalid = n - n_valid
    parsing_failure = n - n_parsed
    parsed_disagreement = n_parsed - n_valid
    
    print(f"Language: {lang}")
    print(f"  Total Pairs: {n}")
    print(f"  Valid Pairs: {n_valid}")
    print(f"  Invalid Pairs: {invalid} ({(invalid/n)*100:.2f}%)")
    print(f"  - Parsing Failure (at least one order): {parsing_failure} ({(parsing_failure/n)*100:.2f}%)")
    print(f"  - Parsed Disagreement (Flip/Tie): {parsed_disagreement} ({(parsed_disagreement/n)*100:.2f}%)")

for lang, part in df.groupby("language"):
    print_stats(lang, part)
print_stats("pooled", df)

