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
    parsed_invalid = part[(part.parsed_both) & (~part.strict_valid)]
    invalid = len(parsed_invalid)
    
    n_boundary = parsed_invalid.is_boundary.sum()
    non_boundary_flip = (parsed_invalid.b_position.abs() > parsed_invalid.c_content.abs()).sum()
    other = invalid - n_boundary - non_boundary_flip
    
    print(f"Language: {lang}")
    print(f"  Parsed Invalid (Swap Error): {invalid}")
    print(f"  - Boundary: {n_boundary} ({(n_boundary/invalid)*100:.1f}%)")
    print(f"  - Non-boundary Flip (|b| > |c|): {non_boundary_flip} ({(non_boundary_flip/invalid)*100:.1f}%)")
    print(f"  - Other: {other}")

