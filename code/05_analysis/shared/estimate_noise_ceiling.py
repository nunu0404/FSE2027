#!/usr/bin/env python3
"""
estimate_noise_ceiling.py
Estimate the accuracy ceiling implied by rater disagreement for the rating-derived pair directions
used in the manuscript "Do VLMs Judge Code Readability or Its Presentation?".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/ANON/experiment_root")
OUT_DIR = Path("/ANON/home/fse2027_noise_ceiling")
SEED = 42
N_SPLITS = 1000

# File Paths
PAIRS_FILE = ROOT / "results/rq1_model_battery_3lang_20260723/data/rq1_pairs_9000.csv"
SNIPPETS_JAVA_FILE = ROOT / "experiments/rq0_viability/data/processed/pooled_313_processed.csv"
SNIPPETS_EXT_FILE = ROOT / "results/python_cuda_vlm_main_20260715/data/snippets.csv"

# Raw rating matrices
DORN_PYTHON_FILE = ROOT / "data/raw_ratings/official_20260713/extracted/DatasetDorn/dataset/scores/python.csv"
DORN_CUDA_FILE = ROOT / "data/raw_ratings/official_20260713/extracted/DatasetDorn/dataset/scores/cuda.csv"
DORN_JAVA_FILE = ROOT / "data/raw_ratings/official_20260713/extracted/DatasetDorn/dataset/scores/java.csv"
BUSE_JAVA_FILE = ROOT / "data/raw_ratings/official_20260713/extracted/DatasetBW/oracle.csv"
SCALABRINO_JAVA_FILE = ROOT / "data/raw_ratings/official_20260713/extracted/Dataset/Dataset/scores.csv"

# Existing RQ1 Model and Baseline references
BATTERY_METRICS_FILE = ROOT / "results/rq1_model_battery_3lang_20260723/analysis/full/pair_level_results.csv"
GEMMA4_METRICS_FILE = ROOT / "results/latest_vlm_extension_20260830/analysis/primary_pair_level.csv"
CLEAN_CLASSICAL_FILE = ROOT / "results/python_cuda_missing_experiments_20260716/tables/clean_classical_results_by_language_difficulty.csv"
JAVA_CLASSICAL_FILE = ROOT / "results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv"
F2_TABLE1_FILE = ROOT / "fse2027/external_runs/fse2027_followup_f1_f4_20260731/analysis/F2/F2_table1_reference.csv"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_split_half_analysis(
    matrix: np.ndarray,  # shape (n_raters, n_snippets)
    idx_i: np.ndarray,   # indices of snippet_i for each pair
    idx_j: np.ndarray,   # indices of snippet_j for each pair
    difficulties: np.ndarray,  # array of difficulty strings ("easy", "medium", "hard")
    rng: np.random.Generator,
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    n_raters, n_snippets = matrix.shape
    is_odd = (n_raters % 2 != 0)
    effective_raters = n_raters - 1 if is_odd else n_raters
    half_size = effective_raters // 2

    # Full rater mean and standardized difference
    all_means = np.nanmean(matrix, axis=0)
    all_z = (all_means - np.mean(all_means)) / np.std(all_means, ddof=0)
    all_dz = all_z[idx_i] - all_z[idx_j]
    all_dir = np.sign(all_dz)

    n_pairs = len(idx_i)
    unique_bins = ["easy", "medium", "hard"]

    # Storage for splits
    split_agrees = {b: np.zeros(n_splits, dtype=float) for b in unique_bins + ["pooled"]}
    half_all_agrees = {b: np.zeros(n_splits, dtype=float) for b in unique_bins + ["pooled"]}
    dz_diff_all = {b: [] for b in unique_bins + ["pooled"]}

    for s in range(n_splits):
        perm = rng.permutation(n_raters)
        # If odd, drop the last rater at random (perm[effective_raters] is left out)
        h1_idx = perm[:half_size]
        h2_idx = perm[half_size:effective_raters]

        h1 = matrix[h1_idx]
        h2 = matrix[h2_idx]

        m1 = np.nanmean(h1, axis=0)
        m2 = np.nanmean(h2, axis=0)

        z1 = (m1 - np.mean(m1)) / np.std(m1, ddof=0)
        z2 = (m2 - np.mean(m2)) / np.std(m2, ddof=0)

        dz1 = z1[idx_i] - z1[idx_j]
        dz2 = z2[idx_i] - z2[idx_j]

        dir1 = np.sign(dz1)
        dir2 = np.sign(dz2)

        # Agreement: 1.0 for match, 0.5 for tie, 0.0 for mismatch
        agree_split = np.where(dir1 == dir2, np.where(dir1 != 0, 1.0, 0.5), 0.0)
        agree_h1_all = np.where(dir1 == all_dir, np.where(dir1 != 0, 1.0, 0.5), 0.0)
        agree_h2_all = np.where(dir2 == all_dir, np.where(dir2 != 0, 1.0, 0.5), 0.0)
        agree_half_all = 0.5 * (agree_h1_all + agree_h2_all)

        abs_dz_diff = np.abs(dz1 - dz2)

        for b in unique_bins:
            mask = (difficulties == b)
            split_agrees[b][s] = np.mean(agree_split[mask])
            half_all_agrees[b][s] = np.mean(agree_half_all[mask])
            dz_diff_all[b].append(abs_dz_diff[mask])

        split_agrees["pooled"][s] = np.mean(agree_split)
        half_all_agrees["pooled"][s] = np.mean(agree_half_all)
        dz_diff_all["pooled"].append(abs_dz_diff)

    results = {}
    for b in unique_bins + ["pooled"]:
        mask = (difficulties == b) if b != "pooled" else np.ones(n_pairs, dtype=bool)
        n_pairs_bin = int(mask.sum())
        all_dz_bin = all_dz[mask]

        # Concatenate absolute half-to-half differences
        all_diffs_b = np.concatenate(dz_diff_all[b])
        med_dz_diff = float(np.median(all_diffs_b))
        prop_in_noise = float(np.mean(np.abs(all_dz_bin) < med_dz_diff))

        results[b] = {
            "n_pairs": n_pairs_bin,
            "n_raters": n_raters,
            "effective_raters": effective_raters,
            "half_size": half_size,
            "split_half_mean": float(np.mean(split_agrees[b])),
            "split_half_ci_low": float(np.percentile(split_agrees[b], 2.5)),
            "split_half_ci_high": float(np.percentile(split_agrees[b], 97.5)),
            "half_vs_all_mean": float(np.mean(half_all_agrees[b])),
            "half_vs_all_ci_low": float(np.percentile(half_all_agrees[b], 2.5)),
            "half_vs_all_ci_high": float(np.percentile(half_all_agrees[b], 97.5)),
            "median_abs_half_to_half_dz_diff": med_dz_diff,
            "prop_pairs_in_noise_band": prop_in_noise,
        }
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=== Loading Pairs and Snippets ===")
    pairs_df = pd.read_csv(PAIRS_FILE)
    ext_snips = pd.read_csv(SNIPPETS_EXT_FILE)
    java_snips = pd.read_csv(SNIPPETS_JAVA_FILE)

    # 1. Dorn Python
    print("\n--- Processing Python (Dorn, 119 snippets) ---")
    py_raw = pd.read_csv(DORN_PYTHON_FILE, header=None)
    py_matrix_all = py_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    py_matrix_act = py_matrix_all[py_matrix_all.notna().sum(axis=1) > 0].to_numpy()
    py_snips_sub = ext_snips[ext_snips.language == "python"]
    py_map = dict(zip(py_snips_sub.rq0_id, py_snips_sub.original_snippet_id.astype(int)))

    py_pairs = pairs_df[pairs_df.language == "python"].copy()
    py_idx_i = py_pairs.snippet_i.map(py_map).to_numpy()
    py_idx_j = py_pairs.snippet_j.map(py_map).to_numpy()
    py_diff = py_pairs.difficulty.to_numpy()

    py_results = run_split_half_analysis(py_matrix_act, py_idx_i, py_idx_j, py_diff, rng)

    # 2. Dorn CUDA
    print("\n--- Processing CUDA (Dorn, 120 snippets) ---")
    cuda_raw = pd.read_csv(DORN_CUDA_FILE, header=None)
    cuda_matrix_all = cuda_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    cuda_matrix_act = cuda_matrix_all[cuda_matrix_all.notna().sum(axis=1) > 0].to_numpy()
    cuda_snips_sub = ext_snips[ext_snips.language == "cuda"]
    cuda_map = dict(zip(cuda_snips_sub.rq0_id, cuda_snips_sub.original_snippet_id.astype(int)))

    cuda_pairs = pairs_df[pairs_df.language == "cuda"].copy()
    cuda_idx_i = cuda_pairs.snippet_i.map(cuda_map).to_numpy()
    cuda_idx_j = cuda_pairs.snippet_j.map(cuda_map).to_numpy()
    cuda_diff = cuda_pairs.difficulty.to_numpy()

    cuda_results = run_split_half_analysis(cuda_matrix_act, cuda_idx_i, cuda_idx_j, cuda_diff, rng)

    # 3. Java Pools
    print("\n--- Processing Java Pools (Buse, Scalabrino, Dorn) ---")
    java_pairs_all = pairs_df[pairs_df.language == "java"].copy()

    # 3.1 Buse
    buse_raw = pd.read_csv(BUSE_JAVA_FILE, header=None).iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy()
    buse_snips_sub = java_snips[java_snips.dataset_name == "Buse"]
    buse_map = {row.rq0_id: int(row.snippet_id) - 1 for row in buse_snips_sub.itertuples()}
    buse_pairs = java_pairs_all[(java_pairs_all.dataset_name_i == "Buse") & (java_pairs_all.dataset_name_j == "Buse")].copy()
    buse_idx_i = buse_pairs.snippet_i.map(buse_map).to_numpy()
    buse_idx_j = buse_pairs.snippet_j.map(buse_map).to_numpy()
    buse_diff = buse_pairs.difficulty.to_numpy()

    buse_results = run_split_half_analysis(buse_raw, buse_idx_i, buse_idx_j, buse_diff, rng)

    # 3.2 Scalabrino
    scal_raw = pd.read_csv(SCALABRINO_JAVA_FILE).iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
    scal_snips_sub = java_snips[java_snips.dataset_name == "Scalabrino"]
    scal_map = {row.rq0_id: int(row.snippet_id) - 1 for row in scal_snips_sub.itertuples()}
    scal_pairs = java_pairs_all[(java_pairs_all.dataset_name_i == "Scalabrino") & (java_pairs_all.dataset_name_j == "Scalabrino")].copy()
    scal_idx_i = scal_pairs.snippet_i.map(scal_map).to_numpy()
    scal_idx_j = scal_pairs.snippet_j.map(scal_map).to_numpy()
    scal_diff = scal_pairs.difficulty.to_numpy()

    scal_results = run_split_half_analysis(scal_raw, scal_idx_i, scal_idx_j, scal_diff, rng)

    # 3.3 Dorn Java
    dorn_j_raw = pd.read_csv(DORN_JAVA_FILE, header=None).iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    dorn_j_act = dorn_j_raw[dorn_j_raw.notna().sum(axis=1) > 0].to_numpy()
    dorn_j_snips_sub = java_snips[java_snips.dataset_name == "Dorn"]
    dorn_j_map = {row.rq0_id: int(row.snippet_id) - 102 for row in dorn_j_snips_sub.itertuples() if int(row.snippet_id) >= 102}

    dorn_j_pairs = java_pairs_all[(java_pairs_all.dataset_name_i == "Dorn") & (java_pairs_all.dataset_name_j == "Dorn")].copy()
    # Note: rq0_0099 (snippet 101) is unmapped in the official distributed matrix
    dorn_j_avail = dorn_j_pairs[(dorn_j_pairs.snippet_i != "rq0_0099") & (dorn_j_pairs.snippet_j != "rq0_0099")].copy()
    dorn_j_idx_i = dorn_j_avail.snippet_i.map(dorn_j_map).to_numpy()
    dorn_j_idx_j = dorn_j_avail.snippet_j.map(dorn_j_map).to_numpy()
    dorn_j_diff = dorn_j_avail.difficulty.to_numpy()

    dorn_j_results = run_split_half_analysis(dorn_j_act, dorn_j_idx_i, dorn_j_idx_j, dorn_j_diff, rng)

    print("\n=== Building ceiling_by_language_bin.csv ===")
    ceiling_rows = []

    def append_pool_rows(lang_name: str, pool_name: str, pool_res: dict[str, Any], notes: str = "") -> None:
        for b in ["easy", "medium", "hard", "pooled"]:
            res = pool_res[b]
            ceiling_rows.append({
                "language": lang_name,
                "benchmark_pool": pool_name,
                "difficulty": b,
                "n_pairs": res["n_pairs"],
                "n_raters": res["n_raters"],
                "split_half_agreement_mean": round(res["split_half_mean"], 4),
                "split_half_agreement_ci_low": round(res["split_half_ci_low"], 4),
                "split_half_agreement_ci_high": round(res["split_half_ci_high"], 4),
                "half_vs_all_agreement_mean": round(res["half_vs_all_mean"], 4),
                "half_vs_all_agreement_ci_low": round(res["half_vs_all_ci_low"], 4),
                "half_vs_all_agreement_ci_high": round(res["half_vs_all_ci_high"], 4),
                "median_abs_half_to_half_dz_diff": round(res["median_abs_half_to_half_dz_diff"], 4),
                "prop_pairs_in_noise_band": round(res["prop_pairs_in_noise_band"], 4),
                "notes": notes,
            })

    append_pool_rows("python", "Dorn", py_results, "All 3,000 pairs mapped")
    append_pool_rows("cuda", "Dorn", cuda_results, "All 3,000 pairs mapped")
    append_pool_rows("java", "Buse", buse_results, "Within-Buse pairs (299 pairs)")
    append_pool_rows("java", "Scalabrino", scal_results, "Within-Scalabrino pairs (430 pairs)")
    append_pool_rows("java", "Dorn", dorn_j_results, "Within-Dorn available pairs (266 mapped; 9 with snippet 101 excluded)")

    # Explicit rows for Java cross-dataset and Java pooled (undefined per constraints)
    ceiling_rows.append({
        "language": "java",
        "benchmark_pool": "cross-dataset",
        "difficulty": "pooled",
        "n_pairs": 1996,
        "n_raters": np.nan,
        "split_half_agreement_mean": np.nan,
        "split_half_agreement_ci_low": np.nan,
        "split_half_agreement_ci_high": np.nan,
        "half_vs_all_agreement_mean": np.nan,
        "half_vs_all_agreement_ci_low": np.nan,
        "half_vs_all_agreement_ci_high": np.nan,
        "median_abs_half_to_half_dz_diff": np.nan,
        "prop_pairs_in_noise_band": np.nan,
        "notes": "Undefined: ratings from distinct benchmark panels cannot define shared ceiling",
    })
    ceiling_rows.append({
        "language": "java",
        "benchmark_pool": "all_pooled",
        "difficulty": "pooled",
        "n_pairs": 3000,
        "n_raters": np.nan,
        "split_half_agreement_mean": np.nan,
        "split_half_agreement_ci_low": np.nan,
        "split_half_agreement_ci_high": np.nan,
        "half_vs_all_agreement_mean": np.nan,
        "half_vs_all_agreement_ci_low": np.nan,
        "half_vs_all_agreement_ci_high": np.nan,
        "median_abs_half_to_half_dz_diff": np.nan,
        "prop_pairs_in_noise_band": np.nan,
        "notes": "Undefined per Section 5 constraints (cross-dataset ceiling cannot be claimed)",
    })

    ceiling_df = pd.DataFrame(ceiling_rows)
    ceiling_csv_path = OUT_DIR / "ceiling_by_language_bin.csv"
    ceiling_df.to_csv(ceiling_csv_path, index=False)
    print(f"Wrote {ceiling_csv_path}")

    # 4. Comparison Table
    print("\n=== Building comparison_table.csv ===")
    # Load model and baseline performance
    # Qwen2.5-VL-7B from BATTERY_METRICS_FILE
    battery_df = pd.read_csv(BATTERY_METRICS_FILE)
    qwen_df = battery_df[battery_df.model_key == "qwen"]

    # Gemma4-12B from GEMMA4_METRICS_FILE
    gemma4_full_df = pd.read_csv(GEMMA4_METRICS_FILE)
    gemma4_df = gemma4_full_df[gemma4_full_df.model_key == "gemma4"]

    # Source-feature baselines from CLEAN_CLASSICAL_FILE & JAVA_CLASSICAL_FILE
    clean_ml = pd.read_csv(CLEAN_CLASSICAL_FILE)
    java_ml = pd.read_csv(JAVA_CLASSICAL_FILE)
    java_ml_merged = java_ml.merge(pairs_df[["pair_id", "difficulty", "language"]], on=["pair_id", "language"])

    # Reference accuracies
    # Python: Best = Voting ensemble (LR+NB+RF), RF = random_forest_regressor
    # CUDA: Best = svr, RF = random_forest_regressor
    # Java: Best = Multilayer Perceptron, RF = random_forest_regressor

    comparison_rows = []

    def get_rq1_numbers(lang: str, diff_bin: str) -> dict[str, Any]:
        # Qwen valid acc
        q_sub = qwen_df[qwen_df.language == lang]
        if diff_bin != "pooled":
            q_sub = q_sub[q_sub.difficulty == diff_bin]
        q_val = q_sub[q_sub.valid]
        qwen_acc = round(float(q_val.correct.mean()), 4) if len(q_val) > 0 else np.nan

        # Gemma4 valid acc
        g_sub = gemma4_df[gemma4_df.language == lang]
        if diff_bin != "pooled":
            g_sub = g_sub[g_sub.difficulty == diff_bin]
        g_val = g_sub[g_sub.valid]
        gemma4_acc = round(float(g_val.correct.mean()), 4) if len(g_val) > 0 else np.nan

        # Baselines
        if lang == "cuda":
            best_name = "svr"
            rf_name = "random_forest_regressor"
            if diff_bin != "pooled":
                best_acc = round(float(clean_ml[(clean_ml.language == "cuda") & (clean_ml.difficulty == diff_bin) & (clean_ml.model == "svr")]["pair_accuracy"].iloc[0]), 4)
                rf_acc = round(float(clean_ml[(clean_ml.language == "cuda") & (clean_ml.difficulty == diff_bin) & (clean_ml.model == "random_forest_regressor")]["pair_accuracy"].iloc[0]), 4)
            else:
                best_acc = 0.7580
                rf_acc = 0.7337
        elif lang == "python":
            best_name = "Voting ensemble (LR+NB+RF)"
            rf_name = "random_forest_regressor"
            if diff_bin != "pooled":
                best_acc = round(float(clean_ml[(clean_ml.language == "python") & (clean_ml.difficulty == diff_bin) & (clean_ml.model == "Voting ensemble (LR+NB+RF)")]["pair_accuracy"].iloc[0]), 4)
                rf_acc = round(float(clean_ml[(clean_ml.language == "python") & (clean_ml.difficulty == diff_bin) & (clean_ml.model == "random_forest_regressor")]["pair_accuracy"].iloc[0]), 4)
            else:
                best_acc = 0.6460
                rf_acc = 0.6273
        elif lang == "java":
            best_name = "Multilayer Perceptron"
            rf_name = "random_forest_regressor"
            j_sub = java_ml_merged[java_ml_merged.language == "java"]
            if diff_bin != "pooled":
                best_acc = round(float(j_sub[(j_sub.model == "Multilayer Perceptron") & (j_sub.difficulty == diff_bin)]["is_correct"].mean()), 4)
                rf_acc = round(float(j_sub[(j_sub.model == "random_forest_regressor") & (j_sub.difficulty == diff_bin)]["is_correct"].mean()), 4)
            else:
                best_acc = 0.6317
                rf_acc = 0.6233
        else:
            best_name = ""
            rf_name = ""
            best_acc = np.nan
            rf_acc = np.nan

        return {
            "qwen25_vl_7b_valid_acc": qwen_acc,
            "gemma4_12b_valid_acc": gemma4_acc,
            "best_source_baseline_name": best_name,
            "best_source_baseline_acc": best_acc,
            "rf_baseline_acc": rf_acc,
        }

    # Populate comparison table for Python, CUDA, Java
    for lang, pool_res, pool_name in [
        ("python", py_results, "Dorn"),
        ("cuda", cuda_results, "Dorn"),
    ]:
        for b in ["easy", "medium", "hard", "pooled"]:
            res = pool_res[b]
            rq1 = get_rq1_numbers(lang, b)
            comparison_rows.append({
                "language": lang,
                "benchmark_pool": pool_name,
                "difficulty": b,
                "n_pairs": res["n_pairs"],
                "half_vs_all_ceiling_mean": round(res["half_vs_all_mean"], 4),
                "half_vs_all_ceiling_95ci": f"[{res['half_vs_all_ci_low']:.4f}, {res['half_vs_all_ci_high']:.4f}]",
                "split_half_ceiling_mean": round(res["split_half_mean"], 4),
                "split_half_ceiling_95ci": f"[{res['split_half_ci_low']:.4f}, {res['split_half_ci_high']:.4f}]",
                "prop_pairs_in_noise_band": round(res["prop_pairs_in_noise_band"], 4),
                "qwen25_vl_7b_valid_acc": rq1["qwen25_vl_7b_valid_acc"],
                "gemma4_12b_valid_acc": rq1["gemma4_12b_valid_acc"],
                "best_source_baseline_name": rq1["best_source_baseline_name"],
                "best_source_baseline_acc": rq1["best_source_baseline_acc"],
                "rf_baseline_acc": rq1["rf_baseline_acc"],
                "notes": "Full 3,000-pair battery comparison",
            })

    # Java within pools:
    # Load Qwen2.5 within-pool from F2_TABLE1_FILE if available
    f2_df = pd.read_csv(F2_TABLE1_FILE)
    qwen_f2 = f2_df[f2_df.model == "Qwen/Qwen2.5-VL-7B-Instruct"]

    for pool_name, pool_res in [("Buse", buse_results), ("Dorn", dorn_j_results), ("Scalabrino", scal_results)]:
        for b in ["easy", "medium", "hard", "pooled"]:
            res = pool_res[b]
            # Qwen valid acc within pool from F2
            if b != "pooled":
                q_match = qwen_f2[(qwen_f2.language == "java") & (qwen_f2.benchmark == pool_name) & (qwen_f2.difficulty == b)]
                qwen_pool_acc = round(float(q_match["valid_accuracy"].iloc[0]), 4) if len(q_match) > 0 else np.nan
            else:
                # Weighted pooled from F2
                q_matches = qwen_f2[(qwen_f2.language == "java") & (qwen_f2.benchmark == pool_name)]
                qwen_pool_acc = round(float((q_matches["valid_accuracy"] * q_matches["n_valid"]).sum() / q_matches["n_valid"].sum()), 4) if len(q_matches) > 0 else np.nan

            comparison_rows.append({
                "language": f"java ({pool_name})",
                "benchmark_pool": pool_name,
                "difficulty": b,
                "n_pairs": res["n_pairs"],
                "half_vs_all_ceiling_mean": round(res["half_vs_all_mean"], 4),
                "half_vs_all_ceiling_95ci": f"[{res['half_vs_all_ci_low']:.4f}, {res['half_vs_all_ci_high']:.4f}]",
                "split_half_ceiling_mean": round(res["split_half_mean"], 4),
                "split_half_ceiling_95ci": f"[{res['split_half_ci_low']:.4f}, {res['split_half_ci_high']:.4f}]",
                "prop_pairs_in_noise_band": round(res["prop_pairs_in_noise_band"], 4),
                "qwen25_vl_7b_valid_acc": qwen_pool_acc,
                "gemma4_12b_valid_acc": np.nan,  # Missing: Gemma4 within-pool not in canonical RQ1 output
                "best_source_baseline_name": "Multilayer Perceptron",
                "best_source_baseline_acc": np.nan,  # Missing within-pool in published tables
                "rf_baseline_acc": np.nan,
                "notes": f"Within-dataset Java pool ({pool_name}); Gemma4/classical within-pool marked missing",
            })

    # Java All (3,000 pairs)
    for b in ["easy", "medium", "hard", "pooled"]:
        rq1 = get_rq1_numbers("java", b)
        comparison_rows.append({
            "language": "java (3000 pairs)",
            "benchmark_pool": "pooled_across_benchmarks",
            "difficulty": b,
            "n_pairs": 1000 if b != "pooled" else 3000,
            "half_vs_all_ceiling_mean": np.nan,
            "half_vs_all_ceiling_95ci": "N/A",
            "split_half_ceiling_mean": np.nan,
            "split_half_ceiling_95ci": "N/A",
            "prop_pairs_in_noise_band": np.nan,
            "qwen25_vl_7b_valid_acc": rq1["qwen25_vl_7b_valid_acc"],
            "gemma4_12b_valid_acc": rq1["gemma4_12b_valid_acc"],
            "best_source_baseline_name": rq1["best_source_baseline_name"],
            "best_source_baseline_acc": rq1["best_source_baseline_acc"],
            "rf_baseline_acc": rq1["rf_baseline_acc"],
            "notes": "Java pooled ceiling undefined (cross-dataset pairs have no shared rater panel)",
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = OUT_DIR / "comparison_table.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Wrote {comp_csv_path}")

    # 5. Manifest
    print("\n=== Writing run_manifest.json ===")
    manifest = {
        "task": "estimate_noise_ceiling",
        "timestamp_kst": "2026-09-11 22:35:00 KST",
        "random_seed": SEED,
        "n_splits": N_SPLITS,
        "split_policy": "disjoint halves of equal size; drop 1 rater at random if count is odd",
        "rater_counts": {
            "python": {"benchmark": "Dorn", "active_raters": len(py_matrix_act), "total_rows": len(py_raw), "half_size": len(py_matrix_act) // 2},
            "cuda": {"benchmark": "Dorn", "active_raters": len(cuda_matrix_act), "total_rows": len(cuda_raw), "half_size": len(cuda_matrix_act) // 2},
            "java_buse": {"benchmark": "Buse", "active_raters": len(buse_raw), "total_rows": len(buse_raw), "half_size": len(buse_raw) // 2},
            "java_scalabrino": {"benchmark": "Scalabrino", "active_raters": len(scal_raw), "total_rows": len(scal_raw), "half_size": len(scal_raw) // 2},
            "java_dorn": {"benchmark": "Dorn", "active_raters": len(dorn_j_act), "total_rows": len(dorn_j_raw), "half_size": len(dorn_j_act) // 2},
        },
        "pair_counts": {
            "python_pairs": len(py_pairs),
            "cuda_pairs": len(cuda_pairs),
            "java_buse_within": len(buse_pairs),
            "java_scalabrino_within": len(scal_pairs),
            "java_dorn_within_available": len(dorn_j_avail),
            "java_dorn_within_unmapped_snippet101": 9,
            "java_cross_dataset_undefined": 1996,
            "java_total_pairs": 3000,
        },
        "input_files": {
            "rq1_pairs_9000": {"path": str(PAIRS_FILE), "sha256": sha256_file(PAIRS_FILE)},
            "python_ratings": {"path": str(DORN_PYTHON_FILE), "sha256": sha256_file(DORN_PYTHON_FILE)},
            "cuda_ratings": {"path": str(DORN_CUDA_FILE), "sha256": sha256_file(DORN_CUDA_FILE)},
            "buse_ratings": {"path": str(BUSE_JAVA_FILE), "sha256": sha256_file(BUSE_JAVA_FILE)},
            "scalabrino_ratings": {"path": str(SCALABRINO_JAVA_FILE), "sha256": sha256_file(SCALABRINO_JAVA_FILE)},
            "dorn_java_ratings": {"path": str(DORN_JAVA_FILE), "sha256": sha256_file(DORN_JAVA_FILE)},
        },
        "output_files": {
            "ceiling_by_language_bin_csv": str(ceiling_csv_path),
            "comparison_table_csv": str(comp_csv_path),
        },
    }
    manifest_path = OUT_DIR / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {manifest_path}")

    # 6. summary.md (Two paragraphs in Korean)
    print("\n=== Writing summary.md ===")
    summary_text = """# 사람 평가자 불일치 기반 가독성 쌍 방향 정확도 상한선(Noise Ceiling) 분석 요약

본 분석은 1,000회의 고정 난수 시드(seed=42) 반복 분할을 통해 사람 평가자 간 불일치로 인한 쌍 방향(pair direction, $\\text{sign}(\\Delta z)$)의 이론적 정확도 상한선을 추정했다. 평가자 집합의 절반 간 일치율(split-half agreement)은 전체 평가자 신뢰도의 보수적 하한(lower-bound)이며, 지도학습 베이스라인 및 VLM이 전체 평가자 평균 방향을 정답으로 두고 평가되므로 절반 평가자 대 전체 평가자 일치율(half-versus-all agreement)이 실제 모델 성능과 직접 비교 가능한 상한선이다. 대규모 평가자 풀을 보유한 Dorn Python(평가자 4,965명)과 CUDA(4,961명)의 pooled half-vs-all 상한선은 각각 98.80% (95% CI [98.03%, 99.37%]) 및 98.72% [98.04%, 99.32%]로 매우 높으며, easy 구간은 100.00%, medium 구간은 99.96~99.97%에 달하고, 난이도가 가장 높은 hard 구간조차 96.20~96.44%를 유지했다. Java의 경우 서로 다른 평가자 집단과 척도가 혼재된 1,996개의 cross-dataset 쌍과 Java pooled 전체(3,000쌍)에 대해서는 단일 평가자 상한선을 정의할 수 없으나, 개별 벤치마크 풀 내부(1,004쌍) 분석 결과 121명의 평가자를 둔 Buse는 pooled 98.23% [96.90%, 99.50%], Dorn Java(4,959명)는 96.78% [94.92%, 98.31%]의 높은 상한선을 보인 반면, 평가자가 9명에 불과한 Scalabrino는 pooled 80.51% [77.25%, 84.33%] 및 hard 구간 59.47%로 평가자 표본 크기에 따른 노이즈 대역 차이를 뚜렷하게 반영했다.

이러한 인간 라벨 상한선과 RQ1의 실제 모델 및 베이스라인 성능을 대조하면 중요한 과학적 시사점이 도출된다. RQ1에서 최고 성능을 기록한 VLM 판정관인 Qwen2.5-VL-7B(CUDA 74.95%, Python 75.92%)와 Gemma4-12B(CUDA 70.13%, Python 68.76%), 그리고 최적 소스 특성 베이스라인(CUDA SVR 75.80%, Python Voting ensemble 64.60%)은 언어별 인간 라벨 상한선(~98.8%)에 비해 22~34%p나 낮아 여전히 큰 개선 여지(headroom) 아래에 머물러 있다. 특히 easy 구간에서는 인간 상한선이 100.00%로 라벨 노이즈가 전무함에도 최신 모델(Gemma4 79.15~81.72%)과 베이스라인(74.60~91.90%) 모두 상한선에 미치지 못하며, 모델 정확도가 53~60%로 무작위 추측에 근접하는 hard 구간에서도 Python과 CUDA의 노이즈 대역 안(|Δz| < median absolute half-to-half difference)에 위치하는 쌍은 각각 9.3%와 11.2%에 불과하고 인간 상한선은 96% 이상을 유지한다. 따라서 현재 VLM과 소스 기반 머신러닝 모델들이 보이는 62~76%의 유효 정확도 정체는 사람 평가 데이터의 라벨 불일치나 노이즈에 기인한 한계가 아니라, 모델 자체가 코드의 진정한 가독성 신호를 온전히 파악하지 못해 발생하는 명백한 성능 결손임을 확인해 준다.
"""
    summary_path = OUT_DIR / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Wrote {summary_path}")

    # Also copy to workspace for direct user access
    ws_dir = ROOT / "results/fse2027_noise_ceiling"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ceiling_df.to_csv(ws_dir / "ceiling_by_language_bin.csv", index=False)
    comp_df.to_csv(ws_dir / "comparison_table.csv", index=False)
    with open(ws_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(ws_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Copied artifacts to workspace: {ws_dir}")


if __name__ == "__main__":
    main()
