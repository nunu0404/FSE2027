#!/usr/bin/env python3
"""Build balanced 900-pair multi-language benchmark: Java (300) + Python (300) + CUDA (300)."""

import hashlib
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path("/ANON/experiment_root")
JAVA_CSV = REPO_ROOT / "results/large_model_cross_family_3model_300java_20260910/pair_ids.csv"
PY_CUDA_POOL_CSV = REPO_ROOT / "results/python_cuda_vlm_main_20260715/data/pairs_seed42_clean.csv"

OUT_DIR = REPO_ROOT / "results/multilang_cross_family_3model_python_cuda_java_20260910"
OUT_CSV = OUT_DIR / "pair_ids_multilang.csv"
OUT_CHECKSUM = OUT_DIR / "image_checksums.json"

JAVA_IMG_DIR = REPO_ROOT / "experiments/rq0_viability/data/rendered/default"
PY_CUDA_IMG_DIR = REPO_ROOT / "results/python_cuda_vlm_main_20260715/rendered/default"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Building Multi-Language 900-Pair Benchmark ===")
    # 1. Load Java pairs
    df_java = pd.read_csv(JAVA_CSV)
    df_java["language"] = "java"
    print(f"Loaded Java pairs: {len(df_java)}")
    
    # 2. Sample Python and CUDA pairs (100 easy, 100 medium, 100 hard each)
    df_pool = pd.read_csv(PY_CUDA_POOL_CSV)
    
    sampled_rows = []
    for lang in ["python", "cuda"]:
        for diff in ["easy", "medium", "hard"]:
            sub = df_pool[(df_pool["language"] == lang) & (df_pool["difficulty"] == diff)]
            # Deterministic head 100
            sampled = sub.head(100).copy()
            for _, r in sampled.iterrows():
                sampled_rows.append({
                    "pair_id": r["pair_id"],
                    "language": lang,
                    "difficulty": diff,
                    "snippet_x_id": r["snippet_i"],
                    "snippet_y_id": r["snippet_j"],
                    "human_preference": r["human_preference"],
                    "abs_z_diff": r["abs_z_diff"],
                    "image_x_path": str((PY_CUDA_IMG_DIR / f"{r['snippet_i']}.png").resolve()),
                    "image_y_path": str((PY_CUDA_IMG_DIR / f"{r['snippet_j']}.png").resolve()),
                })
    
    # Also standardize Java rows
    java_standardized = []
    for _, r in df_java.iterrows():
        java_standardized.append({
            "pair_id": r["pair_id"],
            "language": "java",
            "difficulty": r["difficulty"],
            "snippet_x_id": r["snippet_x_id"],
            "snippet_y_id": r["snippet_y_id"],
            "human_preference": r["human_preference"],
            "abs_z_diff": None,
            "image_x_path": str((JAVA_IMG_DIR / f"{r['snippet_x_id']}.png").resolve()),
            "image_y_path": str((JAVA_IMG_DIR / f"{r['snippet_y_id']}.png").resolve()),
        })
    
    all_df = pd.DataFrame(java_standardized + sampled_rows)
    all_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(all_df)} pairs to {OUT_CSV}")
    print(all_df.groupby(["language", "difficulty"]).size())
    
    # 3. Verify all image files exist and compute checksums
    print("\nVerifying image files and computing SHA-256 checksums...")
    unique_images = {}
    for _, r in all_df.iterrows():
        unique_images[r["snippet_x_id"]] = Path(r["image_x_path"])
        unique_images[r["snippet_y_id"]] = Path(r["image_y_path"])
    
    checksums = {}
    missing = []
    for snip_id, path in unique_images.items():
        if not path.exists():
            missing.append((snip_id, str(path)))
        else:
            checksums[snip_id] = {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size
            }
    
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images: {missing[:5]}")
    
    with open(OUT_CHECKSUM, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    print(f"Successfully verified all {len(checksums)} unique snippet images! Checksums saved to {OUT_CHECKSUM}.\n")

if __name__ == "__main__":
    main()
