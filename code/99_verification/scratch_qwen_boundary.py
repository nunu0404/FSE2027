import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append("/ANON/experiment_root/rq2_eval_reduced/scripts")
import analyze_rq2_reduced as ar2

ROOT = Path("/ANON/experiment_root")
raw_path = ROOT / "rq2_eval_reduced/anchor_qwen/raw_calls/qwen25_vl_7b_baseline_anchor_raw.jsonl"
pairs_meta = pd.read_csv(ROOT / "results/grounded_protocol_3lang_20260721/data/pairs_seed42_grounded.csv")

calls = []
with raw_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            calls.append(json.loads(line))
calls_df = pd.DataFrame(calls)

paired = ar2.pair_calls(calls_df, pairs_meta)
for lang, part in paired.groupby("language"):
    n = len(part)
    n_parsed = part.parsed_both.sum()
    n_valid = part.strict_valid.sum()
    invalid = n_parsed - n_valid
    
    n_boundary = part.is_boundary.sum()
    # Flips are where sign(b_position) > c_content.abs() ?? No, strict swap invalid is either boundary OR flip.
    # If not boundary, invalid means |b| > |c|
    non_boundary_invalid = ((~part.is_boundary) & (~part.strict_valid) & part.parsed_both).sum()
    
    print(f"Language: {lang}")
    print(f"  Parsed pairs: {n_parsed}")
    print(f"  Valid pairs: {n_valid}")
    print(f"  Invalid pairs (Swap error): {invalid}")
    print(f"  - Boundary pairs: {n_boundary}")
    print(f"  - Non-boundary Flips (|b| > |c|): {non_boundary_invalid}")

