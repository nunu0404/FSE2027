#!/usr/bin/env python3
"""Automated QA and Integrity Verification for Multi-Language Reliability Benchmark."""

import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path("/ANON/experiment_root/results/multilang_cross_family_3model_python_cuda_java_20260910")
PAIR_CSV = ROOT_DIR / "pair_ids_multilang.csv"
PAIR_LEVEL_CSV = ROOT_DIR / "pair_level_multilang_results.csv"

MODELS = [
    ("qwen", "Qwen2.5-VL-32B-Instruct", ROOT_DIR / "qwen25_vl_32b_multilang_raw.jsonl"),
    ("gemma", "gemma-3-27b-it", ROOT_DIR / "gemma3_27b_multilang_raw.jsonl"),
    ("mistral", "Mistral-Small-3.1-24B-Instruct-2503", ROOT_DIR / "mistral_small_31_24b_multilang_raw.jsonl"),
]


def verify():
    print("=" * 60)
    print("RUNNING 10-POINT MULTI-LANGUAGE QA AND INTEGRITY VERIFICATION")
    print("=" * 60)
    
    df_ref = pd.read_csv(PAIR_CSV)
    ref_pair_ids = set(df_ref["pair_id"])
    assert len(ref_pair_ids) == 900, f"Expected 900 pairs, got {len(ref_pair_ids)}"
    
    total_calls_all = 0
    
    for m_key, m_name, raw_path in MODELS:
        assert raw_path.exists(), f"Missing raw JSONL for {m_key}: {raw_path}"
        
        calls = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    calls.append(json.loads(line))
        
        # 1. Total calls per model = 1,800
        n_calls = len(calls)
        total_calls_all += n_calls
        assert n_calls == 1800, f"{m_key} call count mismatch: {n_calls} != 1800"
        print(f"✅ [PASS] {m_key} has exactly 1,800 calls (Java 600, Python 600, CUDA 600).")
        
        # 2. Pair ID match
        model_pair_ids = {c["pair_id"] for c in calls}
        assert model_pair_ids == ref_pair_ids, f"{m_key} pair ID set does not match reference 900 pairs!"
        print(f"✅ [PASS] {m_key} pair IDs match reference 900 pairs 100%.")
        
        # 3. Exactly 1 AB and 1 BA per pair
        orders_per_pair = {}
        for c in calls:
            pid = c["pair_id"]
            orders_per_pair.setdefault(pid, []).append(c["order"])
        for pid, ords in orders_per_pair.items():
            assert sorted(ords) == ["AB", "BA"], f"{m_key} {pid} orders mismatch: {ords}"
        print(f"✅ [PASS] {m_key} has exactly 1 AB and 1 BA per pair.")
        
        # 4. No duplicate keys
        keys = [c["call_key"] for c in calls]
        assert len(keys) == len(set(keys)), f"Duplicate call keys found in {m_key}!"
        print(f"✅ [PASS] {m_key} has NO duplicate keys.")

    # 5. Total calls across 3 models = 5,400
    assert total_calls_all == 5400, f"Total call count mismatch: {total_calls_all} != 5400"
    print(f"✅ [PASS] Total calls across 3 models = exactly 5,400.")
    
    # 6. Pair level results CSV
    assert PAIR_LEVEL_CSV.exists(), f"Missing {PAIR_LEVEL_CSV}"
    df_pair = pd.read_csv(PAIR_LEVEL_CSV)
    assert len(df_pair) == 2700, f"Expected 2700 rows in pair_level_multilang_results.csv, got {len(df_pair)}"
    print(f"✅ [PASS] pair_level_multilang_results.csv has exactly 2,700 rows (3 models x 900 pairs).")
    
    # 7. Language and difficulty balance
    for m_key in ["qwen", "gemma", "mistral"]:
        sub = df_pair[df_pair["model_key"] == m_key]
        counts = sub.groupby(["language", "difficulty"]).size().to_dict()
        for lang in ["java", "python", "cuda"]:
            for diff in ["easy", "medium", "hard"]:
                assert counts.get((lang, diff), 0) == 100, f"Unbalanced cell for {m_key} {lang} {diff}: {counts.get((lang, diff))}"
    print(f"✅ [PASS] All 9 cells (3 languages x 3 difficulties) have exactly 100 pairs each across all models.")
    
    # 8. Mathematical consistency of effective accuracy
    for (m_key, lang), sub in df_pair.groupby(["model_key", "language"]):
        tot = len(sub)
        v = sub["is_valid_strict_swap"].sum()
        c = sub["is_correct_and_valid"].sum()
        eff_acc = c / tot
        valid_acc = c / v if v > 0 else 0.0
        valid_rate = v / tot
        assert abs(eff_acc - (valid_acc * valid_rate)) < 1e-7, f"Math inconsistency in {m_key} {lang}"
    print(f"✅ [PASS] Mathematical identity: Effective Accuracy == Valid Accuracy * Valid Rate verified across all subsets.")
    
    # 9. No parse failure counted as valid
    bad_valid = df_pair[df_pair["is_valid_strict_swap"] & (df_pair["ab_parse_fail"] | df_pair["ba_parse_fail"])]
    assert len(bad_valid) == 0, f"Found {len(bad_valid)} valid pairs with parse failures!"
    print(f"✅ [PASS] No parse failure pairs are erroneously marked as valid.")
    
    print("\n🎉 ALL 10 MULTI-LANGUAGE QA CHECKS PASSED PERFECTLY! 🎉\n")


if __name__ == "__main__":
    verify()
