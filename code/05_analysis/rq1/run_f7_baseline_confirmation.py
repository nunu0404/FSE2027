#!/usr/bin/env python3
"""F7: verify corrected ML cells and recompute E1 paired comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[3]
BATTERY = ROOT / "results/rq1_model_battery_3lang_20260723"
OUT = BATTERY / "analysis/F7"
PAIRS = BATTERY / "data/rq1_pairs_9000.csv"
OLD_ML = BATTERY / "data/ml_predictions_9models_9000.csv"
CLEAN_EXT_ML = ROOT / "results/python_cuda_missing_experiments_20260716/data/clean_classical_pair_predictions.csv"
VLM_PAIRS = BATTERY / "analysis/full/pair_level_results.csv"
REPLICATION_SUMMARY = BATTERY / "analysis/table1_ml_replication/canonical_pairwise_summary.csv"

SEED = 42
REPLICATES = 10_000
VLM_ORDER = ["qwen", "internvl", "gemma", "ministral", "phi"]
RF_MODEL = "random_forest_regressor"
LANGUAGE_BEST = {
    "java": "Multilayer Perceptron",
    "cuda": "svr",
    "python": "Voting ensemble (LR+NB+RF)",
}
DISPLAY = {
    "Multilayer Perceptron": "Multilayer Perceptron",
    "svr": "Support Vector Regression",
    "Voting ensemble (LR+NB+RF)": "Voting (LR+NB+RF)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def corrected_ml() -> pd.DataFrame:
    old = pd.read_csv(OLD_ML)
    java = old[old["language"].eq("java")].copy()
    clean = pd.read_csv(CLEAN_EXT_ML).rename(columns={
        "model_score_i": "predicted_score_i",
        "model_score_j": "predicted_score_j",
        "model_tie": "prediction_tie",
    })
    clean["source_file"] = rel(CLEAN_EXT_ML)
    keep = [
        "language", "pair_id", "model", "model_preference", "is_correct",
        "predicted_score_i", "predicted_score_j", "prediction_tie", "source_file",
    ]
    result = pd.concat([java[keep], clean[keep]], ignore_index=True)
    require(len(result) == 81_000, f"Expected 81,000 corrected ML rows, found {len(result)}")
    require(not result.duplicated(["language", "pair_id", "model"]).any(), "Duplicate ML key")
    coverage = result.groupby(["language", "model"])["pair_id"].agg(["size", "nunique"])
    require((coverage == 3000).all().all(), "ML coverage is not 3,000 unique pairs per cell")
    return result


def verify_requested_cells(ml: pd.DataFrame) -> pd.DataFrame:
    requested = {
        ("Voting ensemble (LR+NB+RF)", "python"): (1940, 0.6466666666666666, 0.6460),
        ("Multilayer Perceptron", "cuda"): (2069, 0.6896666666666667, 0.6893333333333334),
        ("Multilayer Perceptron", "python"): (1888, 0.6293333333333333, 0.6286666666666667),
        ("svr", "python"): (1786, 0.5953333333333334, 0.5946666666666667),
    }
    old = pd.read_csv(OLD_ML)
    rows = []
    for (model, language), (expected_correct, expected_accuracy, legacy_accuracy) in requested.items():
        corrected = ml[ml["model"].eq(model) & ml["language"].eq(language)]
        legacy = old[old["model"].eq(model) & old["language"].eq(language)]
        correct = int(corrected["is_correct"].sum())
        accuracy = correct / len(corrected)
        require(len(corrected) == 3000 and correct == expected_correct, f"Cell mismatch: {model}/{language}")
        require(np.isclose(accuracy, expected_accuracy), f"Accuracy mismatch: {model}/{language}")
        require(np.isclose(legacy["is_correct"].mean(), legacy_accuracy), f"Legacy mismatch: {model}/{language}")
        rows.append({
            "model": DISPLAY.get(model, model),
            "model_source_name": model,
            "language": language.capitalize() if language != "cuda" else "CUDA",
            "num_pairs": len(corrected),
            "correct_pairs": correct,
            "valid_pairs": len(corrected),
            "valid_accuracy": accuracy,
            "effective_accuracy": accuracy,
            "strict_swap_error": 0.0,
            "legacy_correct_pairs": int(legacy["is_correct"].sum()),
            "legacy_effective_accuracy": float(legacy["is_correct"].mean()),
            "correct_pair_delta": correct - int(legacy["is_correct"].sum()),
            "percentage_point_delta": 100 * (accuracy - float(legacy["is_correct"].mean())),
            "confirmed_value_percent_2dp": round(100 * accuracy, 2),
            "pair_set": "battery frozen clean pair endpoints",
            "cv": "5-fold out-of-fold snippet scores, seed 42",
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows)


def exact_mcnemar(vlm: np.ndarray, baseline: np.ndarray) -> tuple[int, int, float]:
    vlm_only = int(np.sum(vlm & ~baseline))
    baseline_only = int(np.sum(~vlm & baseline))
    discordant = vlm_only + baseline_only
    p = float(binomtest(min(vlm_only, baseline_only), discordant, 0.5).pvalue) if discordant else 1.0
    return vlm_only, baseline_only, p


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def snippet_bootstrap_delta(
    frame: pd.DataFrame, vlm_correct: np.ndarray, baseline_correct: np.ndarray
) -> tuple[float, float, int]:
    snippets = np.array(sorted(set(frame["snippet_i"]) | set(frame["snippet_j"])), dtype=object)
    lookup = {snippet: index for index, snippet in enumerate(snippets)}
    left = frame["snippet_i"].map(lookup).to_numpy(int)
    right = frame["snippet_j"].map(lookup).to_numpy(int)
    delta = vlm_correct.astype(float) - baseline_correct.astype(float)
    rng = np.random.default_rng(SEED)
    estimates = np.empty(REPLICATES, dtype=float)
    used = 0
    for _ in range(REPLICATES):
        sampled = rng.integers(0, len(snippets), size=len(snippets))
        multiplicity = np.bincount(sampled, minlength=len(snippets))
        weights = multiplicity[left] * multiplicity[right]
        denominator = weights.sum()
        if denominator:
            estimates[used] = np.sum(weights * delta) / denominator
            used += 1
    require(used > 0, "All bootstrap replicates had zero denominator")
    low, high = np.quantile(estimates[:used], [0.025, 0.975])
    return float(low), float(high), used


def recompute_e1(ml: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS)
    vlm = pd.read_csv(VLM_PAIRS)
    pair_meta = pairs[["protocol_pair_id", "pair_id", "language", "snippet_i", "snippet_j"]]
    vlm = vlm.merge(
        pair_meta, left_on=["pair_id", "language"], right_on=["protocol_pair_id", "language"],
        how="left", validate="many_to_one", suffixes=("", "_source"),
    )
    require(not vlm["snippet_i"].isna().any(), "VLM-to-pair join failed")
    require(len(vlm) == 45_000, f"Expected 45,000 VLM pair rows, found {len(vlm)}")

    needed = {RF_MODEL, *LANGUAGE_BEST.values()}
    baselines = ml[ml["model"].isin(needed)][["language", "pair_id", "model", "is_correct"]].copy()
    baseline_lookup = baselines.set_index(["language", "pair_id", "model"])["is_correct"]
    rows = []
    for language in ["java", "cuda", "python"]:
        language_best = LANGUAGE_BEST[language]
        for model_key in VLM_ORDER:
            frame = vlm[vlm["language"].eq(language) & vlm["model_key"].eq(model_key)].copy()
            require(len(frame) == 3000, f"VLM coverage mismatch: {model_key}/{language}")
            vlm_correct = frame["debiased_correct"].astype(bool).to_numpy()
            for baseline_type, baseline_model in [("random_forest", RF_MODEL), ("language_best", language_best)]:
                baseline_correct = np.array([
                    bool(baseline_lookup.loc[(language, pair_id, baseline_model)])
                    for pair_id in frame["pair_id_source"]
                ])
                ci_low, ci_high, reps_used = snippet_bootstrap_delta(frame, vlm_correct, baseline_correct)
                vlm_only, baseline_only, p = exact_mcnemar(vlm_correct, baseline_correct)
                rows.append({
                    "model_key": model_key,
                    "model": frame.iloc[0]["model"],
                    "language": language,
                    "n": len(frame),
                    "debiased_correct_pairs": int(vlm_correct.sum()),
                    "debiased_accuracy": float(vlm_correct.mean()),
                    "content_ties": int(frame["content_tie"].sum()),
                    "boundaries_abs_c_eq_abs_b": int(np.isclose(frame["content_margin"].abs(), frame["position_margin"].abs(), rtol=0, atol=0).sum()),
                    "baseline_type": baseline_type,
                    "baseline_model": baseline_model,
                    "baseline_correct_pairs": int(baseline_correct.sum()),
                    "baseline_accuracy": float(baseline_correct.mean()),
                    "diff_debiased_minus_baseline": float(vlm_correct.mean() - baseline_correct.mean()),
                    "snippet_cluster_ci_low": ci_low,
                    "snippet_cluster_ci_high": ci_high,
                    "bootstrap_replicates_requested": REPLICATES,
                    "bootstrap_replicates_used": reps_used,
                    "bootstrap_seed": SEED,
                    "bootstrap_method": "two-endpoint snippet multiplicity-product percentile CI",
                    "vlm_only_correct": vlm_only,
                    "baseline_only_correct": baseline_only,
                    "mcnemar_exact_p": p,
                })
    result = pd.DataFrame(rows)
    require(len(result) == 30, f"Expected 30 E1 comparisons, found {len(result)}")
    result["holm_p_30_comparisons"] = holm_adjust(result["mcnemar_exact_p"].to_numpy(float))
    result["holm_reject_0_05"] = result["holm_p_30_comparisons"] < 0.05
    return result


def legacy_python_best_e1() -> pd.DataFrame:
    old = pd.read_csv(OLD_ML)
    pairs = pd.read_csv(PAIRS)
    vlm = pd.read_csv(VLM_PAIRS)
    meta = pairs[["protocol_pair_id", "pair_id", "language", "snippet_i", "snippet_j"]].rename(
        columns={"pair_id": "pair_id_source"}
    )
    vlm = vlm.merge(meta, left_on=["pair_id", "language"], right_on=["protocol_pair_id", "language"], validate="many_to_one")
    baseline = old[old["language"].eq("python") & old["model"].eq("Voting ensemble (LR+NB+RF)")]
    lookup = baseline.set_index("pair_id")["is_correct"]
    rows = []
    for model_key in VLM_ORDER:
        frame = vlm[vlm["language"].eq("python") & vlm["model_key"].eq(model_key)].copy()
        v = frame["debiased_correct"].astype(bool).to_numpy()
        b = frame["pair_id_source"].map(lookup).astype(bool).to_numpy()
        low, high, used = snippet_bootstrap_delta(frame, v, b)
        rows.append({
            "model_key": model_key,
            "legacy_baseline_correct_pairs": int(b.sum()),
            "legacy_baseline_accuracy": float(b.mean()),
            "legacy_diff": float(v.mean() - b.mean()),
            "legacy_ci_low": low,
            "legacy_ci_high": high,
            "bootstrap_replicates_used": used,
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ml = corrected_ml()
    confirmation = verify_requested_cells(ml)
    e1 = recompute_e1(ml)
    best = e1[e1["baseline_type"].eq("language_best")].copy()
    legacy_python = legacy_python_best_e1()
    python_update = best[best["language"].eq("python")].merge(legacy_python, on="model_key", validate="one_to_one")
    python_update["baseline_correct_pair_change"] = python_update["baseline_correct_pairs"] - python_update["legacy_baseline_correct_pairs"]
    python_update["diff_change"] = python_update["diff_debiased_minus_baseline"] - python_update["legacy_diff"]
    python_update["ci_low_change"] = python_update["snippet_cluster_ci_low"] - python_update["legacy_ci_low"]
    python_update["ci_high_change"] = python_update["snippet_cluster_ci_high"] - python_update["legacy_ci_high"]

    confirmation.to_csv(OUT / "F7_baseline_cell_confirmation.csv", index=False)
    e1.to_csv(OUT / "F7_E1_all_30_comparisons.csv", index=False)
    best.to_csv(OUT / "F7_E1_debiased_vs_language_best.csv", index=False)
    python_update.to_csv(OUT / "F7_E1_python_legacy_vs_corrected.csv", index=False)

    report = [
        "# F7 baseline confirmation and E1 correction",
        "",
        "No new model inference was performed. The frozen clean pair endpoints were rescored from stored five-fold OOF snippet predictions.",
        "",
        "## Decision",
        "",
        "Voting (LR+NB+RF) on Python is 1,940/3,000 = 64.6667% (64.67%), not 1,938/3,000 = 64.60%.",
        "The old battery ML aggregation joined pre-clean predictions to reused pair IDs after three endpoint pairs changed (one CUDA and two Python).",
        "",
        "## Requested cells",
        "",
        confirmation[["model", "language", "correct_pairs", "num_pairs", "effective_accuracy", "confirmed_value_percent_2dp"]].to_markdown(index=False),
        "",
        "## Corrected E1: debiased minus language-best",
        "",
        best[["model_key", "language", "debiased_accuracy", "baseline_model", "baseline_accuracy", "diff_debiased_minus_baseline", "snippet_cluster_ci_low", "snippet_cluster_ci_high", "mcnemar_exact_p", "holm_p_30_comparisons"]].to_markdown(index=False),
        "",
        "## Method",
        "",
        "The CI uses 10,000 percentile bootstrap replicates with seed 42. Unique snippets are sampled with replacement within each language; a pair receives the product of its two endpoint multiplicities. Exact McNemar uses matched pair outcomes. Holm adjustment is recomputed over the prespecified 30 E1 comparisons (5 VLMs x 3 languages x RF/language-best).",
    ]
    (OUT / "F7_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    sources = [PAIRS, OLD_ML, CLEAN_EXT_ML, VLM_PAIRS, REPLICATION_SUMMARY, Path(__file__).resolve()]
    manifest = {
        "run_id": "F7_battery_ml_confirmation_20260819",
        "new_model_calls": 0,
        "seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_method": "two-endpoint snippet cluster bootstrap using endpoint multiplicity products",
        "primary_correction": "Voting Python effective accuracy = 1940/3000 = 0.6466666667",
        "holm_family": "30 prespecified E1 comparisons",
        "sources": {rel(path): sha256(path) for path in sources},
    }
    (OUT / "F7_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")
    print(confirmation.to_string(index=False))
    print("\nCorrected Python E1 language-best comparisons:")
    print(python_update[["model_key", "debiased_accuracy", "baseline_accuracy", "diff_debiased_minus_baseline", "snippet_cluster_ci_low", "snippet_cluster_ci_high", "legacy_diff", "legacy_ci_low", "legacy_ci_high"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
