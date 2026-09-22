#!/usr/bin/env python3
"""Automated pre-completion validation script checking all 10 consistency items."""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("/ANON/experiment_root/results/large_model_cross_family_3model_300java_20260910")
PAIR_IDS_CSV = RESULTS_DIR / "pair_ids.csv"
PAIR_LEVEL_CSV = RESULTS_DIR / "pair_level_results.csv"
AGG_CSV = RESULTS_DIR / "aggregate_metrics.csv"
COMP_CSV = RESULTS_DIR / "paired_comparisons.csv"

RAW_FILES = {
    "qwen": RESULTS_DIR / "qwen25_vl_32b_raw.jsonl",
    "gemma": RESULTS_DIR / "gemma3_27b_raw.jsonl",
    "mistral": RESULTS_DIR / "mistral_small_31_24b_raw.jsonl",
}


def verify_all() -> bool:
    print("=" * 60)
    print("RUNNING 10-POINT AUTOMATED PRE-COMPLETION QA CHECKS")
    print("=" * 60)
    all_passed = True
    
    # Check if files exist
    required_files = [PAIR_IDS_CSV, PAIR_LEVEL_CSV, AGG_CSV, COMP_CSV] + list(RAW_FILES.values())
    for f in required_files:
        if not f.exists():
            print(f"❌ [FAIL] Missing file: {f}")
            return False
            
    df_pairs_ref = pd.read_csv(PAIR_IDS_CSV)
    ref_pair_ids = set(df_pairs_ref["pair_id"])
    assert len(ref_pair_ids) == 300, f"Expected 300 pairs, got {len(ref_pair_ids)}"
    
    # 1 & 2 & 3 & 4: Check raw files
    total_calls_all = 0
    raw_dfs = {}
    for m_key, raw_path in RAW_FILES.items():
        rows = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df_m = pd.DataFrame(rows)
        raw_dfs[m_key] = df_m
        n_calls = len(df_m)
        total_calls_all += n_calls
        
        # Check calls count
        if n_calls != 600:
            print(f"❌ [FAIL] {m_key} has {n_calls} calls, expected exactly 600.")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key} has exactly 600 calls.")
            
        # Check pair IDs set
        m_pair_ids = set(df_m["pair_id"])
        if m_pair_ids != ref_pair_ids:
            print(f"❌ [FAIL] {m_key} pair IDs mismatch reference!")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key} pair IDs match reference 300 pairs 100%.")
            
        # Check exactly one AB and one BA per pair
        orders_per_pair = df_m.groupby("pair_id")["order"].apply(lambda s: sorted(list(s))).to_dict()
        bad_orders = [pid for pid, ords in orders_per_pair.items() if ords != ["AB", "BA"]]
        if bad_orders:
            print(f"❌ [FAIL] {m_key} has {len(bad_orders)} pairs without exact ['AB', 'BA'] calls.")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key} has exactly 1 AB and 1 BA per pair.")
            
        # Check duplicate keys
        keys = df_m["model_name"] + "__" + df_m["pair_id"] + "__" + df_m["order"]
        if keys.duplicated().any():
            print(f"❌ [FAIL] {m_key} has duplicate (model, pair_id, order) keys!")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key} has NO duplicate keys.")

    # 5. Check total calls across all models
    if total_calls_all != 1800:
        print(f"❌ [FAIL] Total calls = {total_calls_all}, expected exactly 1,800.")
        all_passed = False
    else:
        print(f"✅ [PASS] Total calls across 3 models = exactly 1,800.")

    # 6 & 7 & 8: Check pair-level results CSV
    df_pair_level = pd.read_csv(PAIR_LEVEL_CSV)
    if len(df_pair_level) != 900:
        print(f"❌ [FAIL] pair_level_results.csv has {len(df_pair_level)} rows, expected 900.")
        all_passed = False
    else:
        print(f"✅ [PASS] pair_level_results.csv has exactly 900 rows (3 models x 300 pairs).")
        
    for m_key in ["qwen", "gemma", "mistral"]:
        sub_pl = df_pair_level[df_pair_level["model_key"] == m_key]
        raw_df = raw_dfs[m_key]
        
        # Verify verdict match with raw JSONL
        for _, r in sub_pl.iterrows():
            pid = r["pair_id"]
            ab_raw = raw_df[(raw_df["pair_id"] == pid) & (raw_df["order"] == "AB")].iloc[0]
            ba_raw = raw_df[(raw_df["pair_id"] == pid) & (raw_df["order"] == "BA")].iloc[0]
            
            # Check parsed
            raw_p_ab = ab_raw["parsed_verdict"] if pd.notna(ab_raw["parsed_verdict"]) else None
            pl_p_ab = r["parsed_ab"] if pd.notna(r["parsed_ab"]) else None
            if raw_p_ab != pl_p_ab:
                print(f"❌ [FAIL] Parsed verdict mismatch for {m_key} {pid} AB: raw={raw_p_ab}, csv={pl_p_ab}")
                all_passed = False
                break
                
            # Check reverse mapping
            sx, sy = r["snippet_x_id"], r["snippet_y_id"]
            if pl_p_ab == "A":
                assert r["choice_ab"] == sx
            elif pl_p_ab == "B":
                assert r["choice_ab"] == sy
            else:
                assert pd.isna(r["choice_ab"])
                
            pl_p_ba = r["parsed_ba"] if pd.notna(r["parsed_ba"]) else None
            if pl_p_ba == "A":
                assert r["choice_ba"] == sy
            elif pl_p_ba == "B":
                assert r["choice_ba"] == sx
            else:
                assert pd.isna(r["choice_ba"])
                
            # Check strict swap valid
            if r["is_valid_strict_swap"]:
                assert pl_p_ab is not None and pl_p_ba is not None
                assert r["choice_ab"] == r["choice_ba"]
                assert not r["pair_parse_failure"]
                
    print(f"✅ [PASS] Raw JSONL verdicts match pair_level_results.csv 100% and reverse mapping is verified.")

    # 9. Effective accuracy == valid accuracy * (valid_pairs / all_pairs)
    df_agg = pd.read_csv(AGG_CSV)
    for m_key in ["qwen", "gemma", "mistral"]:
        sub_agg = df_agg[df_agg["model_key"] == m_key].set_index("metric")
        eff_acc = sub_agg.loc["effective_accuracy", "estimate"]
        val_acc = sub_agg.loc["valid_accuracy", "estimate"]
        val_rate = sub_agg.loc["valid_pairs", "estimate"]
        calculated_eff = val_acc * val_rate
        if not np.isclose(eff_acc, calculated_eff, atol=1e-5):
            print(f"❌ [FAIL] Effective accuracy ({eff_acc}) != Valid acc ({val_acc}) * Valid rate ({val_rate}) for {m_key}")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key}: Effective accuracy exactly equals Valid acc * Valid rate.")
            
    # 10. Parse failures excluded from valid pairs
    for m_key in ["qwen", "gemma", "mistral"]:
        sub_pl = df_pair_level[df_pair_level["model_key"] == m_key]
        failed_as_valid = sub_pl[sub_pl["pair_parse_failure"] & sub_pl["is_valid_strict_swap"]]
        if len(failed_as_valid) > 0:
            print(f"❌ [FAIL] {m_key} has {len(failed_as_valid)} parse failure pairs marked as valid!")
            all_passed = False
        else:
            print(f"✅ [PASS] {m_key}: No parse failure pairs are marked as valid.")
            
    if all_passed:
        print("\n🎉 ALL 10 AUTOMATED QA CHECKS PASSED PERFECTLY! 🎉\n")
    else:
        print("\n❌ SOME QA CHECKS FAILED. PLEASE REVIEW LOGS.\n")
    return all_passed


if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
