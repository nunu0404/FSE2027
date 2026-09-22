#!/usr/bin/env python3
"""Statistical analysis and artifact generation for larger-model cross-family reliability experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

EXP_ROOT = Path("/ANON/experiment_root")
RESULTS_DIR = EXP_ROOT / "results/large_model_cross_family_3model_300java_20260910"
PAIR_IDS_CSV = RESULTS_DIR / "pair_ids.csv"

MODELS = [
    ("qwen", "Qwen2.5-VL-32B-Instruct", RESULTS_DIR / "qwen25_vl_32b_raw.jsonl"),
    ("gemma", "gemma-3-27b-it", RESULTS_DIR / "gemma3_27b_raw.jsonl"),
    ("mistral", "Mistral-Small-3.1-24B-Instruct-2503", RESULTS_DIR / "mistral_small_31_24b_raw.jsonl"),
]


def load_raw_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build_pair_level_data(models: list[tuple[str, str, Path]]) -> pd.DataFrame:
    all_pair_rows = []
    
    for m_key, m_name, raw_path in models:
        df_raw = load_raw_jsonl(raw_path)
        
        # Group by pair_id
        for pair_id, group in df_raw.groupby("pair_id"):
            ab_row = group[group["order"] == "AB"].iloc[0]
            ba_row = group[group["order"] == "BA"].iloc[0]
            
            p_ab = ab_row["parsed_verdict"]
            p_ba = ba_row["parsed_verdict"]
            c_ab = ab_row["actual_choice_snippet"]
            c_ba = ba_row["actual_choice_snippet"]
            pref = ab_row["human_preference"]
            diff = ab_row["difficulty"]
            sx = ab_row["snippet_x_id"]
            sy = ab_row["snippet_y_id"]
            
            parsed_both = (p_ab is not None) and (p_ba is not None)
            is_valid = parsed_both and (c_ab is not None) and (c_ba is not None) and (c_ab == c_ba)
            is_correct = bool(is_valid and (c_ab == pref))
            
            fail_calls = int(p_ab is None) + int(p_ba is None)
            pair_fail = fail_calls > 0
            
            all_pair_rows.append({
                "model_key": m_key,
                "model_name": m_name,
                "pair_id": pair_id,
                "snippet_x_id": sx,
                "snippet_y_id": sy,
                "difficulty": diff,
                "human_preference": pref,
                "parsed_ab": p_ab,
                "parsed_ba": p_ba,
                "choice_ab": c_ab,
                "choice_ba": c_ba,
                "parsed_both": parsed_both,
                "is_valid_strict_swap": is_valid,
                "is_correct_and_valid": is_correct,
                "parse_failures_count": fail_calls,
                "pair_parse_failure": pair_fail,
            })
            
    return pd.DataFrame(all_pair_rows)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = weights.sum()
    return float(np.sum(values * weights) / w_sum) if w_sum > 0 else 0.0


def run_cluster_bootstrap(
    df_pairs: pd.DataFrame, n_draws: int = 10000, seed: int = 42
) -> dict[str, dict[str, tuple[float, float]]]:
    """10,000-replicate snippet-aware crossed cluster bootstrap for all metrics."""
    print(f"[BOOTSTRAP] Running {n_draws} snippet-aware crossed cluster bootstrap replicates (seed={seed})...")
    rng = np.random.default_rng(seed)
    
    snippets = sorted(set(df_pairs["snippet_x_id"]).union(set(df_pairs["snippet_y_id"])))
    snippet_idx = {s: i for i, s in enumerate(snippets)}
    n_snippets = len(snippets)
    
    # Pre-index snippets
    s_x = df_pairs["snippet_x_id"].map(snippet_idx).to_numpy()
    s_y = df_pairs["snippet_y_id"].map(snippet_idx).to_numpy()
    
    models = df_pairs["model_key"].unique()
    
    # Prepare bootstrap storage
    boot_metrics: dict[str, dict[str, list[float]]] = {
        m: {
            "effective_accuracy": [],
            "valid_accuracy": [],
            "strict_swap_error": [],
            "pair_parse_failure_rate": [],
        }
        for m in models
    }
    
    # Store paired differences for model comparisons
    boot_diffs: dict[tuple[str, str], dict[str, list[float]]] = {
        (m1, m2): {"effective_acc_diff": [], "swap_error_diff": [], "valid_acc_diff": []}
        for m1, m2 in [("qwen", "gemma"), ("qwen", "mistral"), ("gemma", "mistral")]
    }

    # Group data by model
    model_groups = {m: df_pairs[df_pairs["model_key"] == m].reset_index(drop=True) for m in models}
    
    # Verify all models share identical pair indexing
    base_pair_order = model_groups[models[0]]["pair_id"].to_numpy()
    for m in models[1:]:
        assert np.array_equal(model_groups[m]["pair_id"].to_numpy(), base_pair_order)
    
    # Fast numpy arrays per model
    m_valid = {m: model_groups[m]["is_valid_strict_swap"].to_numpy(dtype=float) for m in models}
    m_correct = {m: model_groups[m]["is_correct_and_valid"].to_numpy(dtype=float) for m in models}
    m_fail = {m: model_groups[m]["pair_parse_failure"].to_numpy(dtype=float) for m in models}
    
    # Pair indices in base
    base_s_x = model_groups[models[0]]["snippet_x_id"].map(snippet_idx).to_numpy()
    base_s_y = model_groups[models[0]]["snippet_y_id"].map(snippet_idx).to_numpy()

    for draw in range(n_draws):
        sampled = rng.integers(0, n_snippets, size=n_snippets)
        counts = np.bincount(sampled, minlength=n_snippets)
        
        # Crossed endpoint product weight
        weights = counts[base_s_x] * counts[base_s_y]
        w_sum = weights.sum()
        if w_sum == 0:
            continue
        
        cur_stats: dict[str, dict[str, float]] = {}
        for m in models:
            eff_acc = float(np.sum(m_correct[m] * weights) / w_sum)
            val_rate = float(np.sum(m_valid[m] * weights) / w_sum)
            valid_w = m_valid[m] * weights
            valid_w_sum = valid_w.sum()
            val_acc = float(np.sum(m_correct[m] * valid_w) / valid_w_sum) if valid_w_sum > 0 else 0.0
            swap_err = 1.0 - val_rate
            fail_rate = float(np.sum(m_fail[m] * weights) / w_sum)
            
            boot_metrics[m]["effective_accuracy"].append(eff_acc)
            boot_metrics[m]["valid_accuracy"].append(val_acc)
            boot_metrics[m]["strict_swap_error"].append(swap_err)
            boot_metrics[m]["pair_parse_failure_rate"].append(fail_rate)
            
            cur_stats[m] = {
                "eff_acc": eff_acc,
                "swap_err": swap_err,
                "val_acc": val_acc,
            }
            
        for m1, m2 in boot_diffs.keys():
            boot_diffs[(m1, m2)]["effective_acc_diff"].append(cur_stats[m1]["eff_acc"] - cur_stats[m2]["eff_acc"])
            boot_diffs[(m1, m2)]["swap_error_diff"].append(cur_stats[m1]["swap_err"] - cur_stats[m2]["swap_err"])
            boot_diffs[(m1, m2)]["valid_acc_diff"].append(cur_stats[m1]["val_acc"] - cur_stats[m2]["val_acc"])

    # Compute 95% CIs
    ci_results: dict[str, Any] = {"metrics": {}, "comparisons": {}}
    for m in models:
        ci_results["metrics"][m] = {}
        for metric, vals in boot_metrics[m].items():
            low, high = np.percentile(vals, [2.5, 97.5])
            ci_results["metrics"][m][metric] = (float(low), float(high))
            
    for pair_key, diff_dict in boot_diffs.items():
        pair_str = f"{pair_key[0]}_vs_{pair_key[1]}"
        ci_results["comparisons"][pair_str] = {}
        for metric, vals in diff_dict.items():
            low, high = np.percentile(vals, [2.5, 97.5])
            ci_results["comparisons"][pair_str][metric] = (float(low), float(high))

    print("[BOOTSTRAP] Completed successfully.")
    return ci_results


def compute_mcnemar_exact(s1: np.ndarray, s2: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test for paired binary outcomes."""
    # s1, s2 are boolean arrays (True = valid, False = swap error)
    b = int(np.sum(s1 & (~s2)))  # s1 valid, s2 error
    c = int(np.sum((~s1) & s2))  # s1 error, s2 valid
    res = binomtest(b, b + c, p=0.5, alternative="two-sided")
    return b, c, float(res.pvalue)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction over a family of hypotheses."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = cum_max
    return adjusted


def main() -> None:
    print("=" * 60)
    print("STATISTICAL ANALYSIS: CONTROLLED LARGER-MODEL RELIABILITY")
    print("=" * 60)
    
    # 1. Build pair-level dataframe
    df_pairs = build_pair_level_data(MODELS)
    pair_level_path = RESULTS_DIR / "pair_level_results.csv"
    df_pairs.to_csv(pair_level_path, index=False)
    print(f"[OK] Saved pair-level results ({len(df_pairs)} rows) to {pair_level_path}")
    
    # 2. Bootstrap CIs
    ci_data = run_cluster_bootstrap(df_pairs, n_draws=10000, seed=42)
    
    # 3. Aggregate metrics
    agg_rows = []
    models_order = ["qwen", "gemma", "mistral"]
    model_labels = {
        "qwen": "Qwen2.5-VL-32B-Instruct",
        "gemma": "gemma-3-27b-it",
        "mistral": "Mistral-Small-3.1-24B-Instruct-2503",
    }
    
    for m in models_order:
        sub = df_pairs[df_pairs["model_key"] == m]
        n_pairs = len(sub)
        n_calls = n_pairs * 2
        
        valid_pairs = sub["is_valid_strict_swap"].sum()
        correct_valid = sub["is_correct_and_valid"].sum()
        swap_errors = n_pairs - valid_pairs
        failed_calls = sub["parse_failures_count"].sum()
        failed_pairs = sub["pair_parse_failure"].sum()
        
        eff_acc = correct_valid / n_pairs
        val_acc = (correct_valid / valid_pairs) if valid_pairs > 0 else np.nan
        swap_err = swap_errors / n_pairs
        call_fail_rate = failed_calls / n_calls
        pair_fail_rate = failed_pairs / n_pairs
        
        # Retrieve CIs
        cis = ci_data["metrics"][m]
        
        metrics_dict = [
            ("num_pairs", n_pairs, n_pairs, float(n_pairs), None, None),
            ("num_calls", n_calls, n_calls, float(n_calls), None, None),
            ("valid_pairs", valid_pairs, n_pairs, valid_pairs / n_pairs, None, None),
            ("strict_swap_error", swap_errors, n_pairs, swap_err, cis["strict_swap_error"][0], cis["strict_swap_error"][1]),
            ("correct_and_valid_pairs", correct_valid, n_pairs, correct_valid / n_pairs, None, None),
            ("valid_accuracy", correct_valid, valid_pairs, val_acc, cis["valid_accuracy"][0], cis["valid_accuracy"][1]),
            ("effective_accuracy", correct_valid, n_pairs, eff_acc, cis["effective_accuracy"][0], cis["effective_accuracy"][1]),
            ("call_parse_failure_rate", failed_calls, n_calls, call_fail_rate, None, None),
            ("pair_parse_failure_rate", failed_pairs, n_pairs, pair_fail_rate, cis["pair_parse_failure_rate"][0], cis["pair_parse_failure_rate"][1]),
        ]
        
        for metric_name, num, den, val, ci_low, ci_high in metrics_dict:
            agg_rows.append({
                "model_key": m,
                "model_name": model_labels[m],
                "metric": metric_name,
                "numerator": num,
                "denominator": den,
                "estimate": round(val, 6) if val is not None else None,
                "ci_95_low": round(ci_low, 6) if ci_low is not None else None,
                "ci_95_high": round(ci_high, 6) if ci_high is not None else None,
            })
            
    df_agg = pd.DataFrame(agg_rows)
    agg_path = RESULTS_DIR / "aggregate_metrics.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[OK] Saved aggregate metrics to {agg_path}")

    # 4. Paired Comparisons
    comparisons = [
        ("qwen", "gemma", "Qwen2.5-VL-32B vs Gemma 3-27B"),
        ("qwen", "mistral", "Qwen2.5-VL-32B vs Mistral Small 3.1-24B"),
        ("gemma", "mistral", "Gemma 3-27B vs Mistral Small 3.1-24B"),
    ]
    
    comp_rows = []
    raw_p_mcnemar = []
    
    for m1, m2, label in comparisons:
        sub1 = df_pairs[df_pairs["model_key"] == m1].set_index("pair_id")
        sub2 = df_pairs[df_pairs["model_key"] == m2].set_index("pair_id")
        
        # Paired differences
        # Effective accuracy:
        c1 = sub1["is_correct_and_valid"].to_numpy(dtype=float)
        c2 = sub2["is_correct_and_valid"].to_numpy(dtype=float)
        eff_diff = float(np.mean(c1 - c2))
        
        # Strict swap error:
        e1 = (~sub1["is_valid_strict_swap"]).to_numpy(dtype=float)
        e2 = (~sub2["is_valid_strict_swap"]).to_numpy(dtype=float)
        swap_diff = float(np.mean(e1 - e2))
        
        # Valid accuracy diff
        v1_acc = sub1["is_correct_and_valid"].sum() / sub1["is_valid_strict_swap"].sum()
        v2_acc = sub2["is_correct_and_valid"].sum() / sub2["is_valid_strict_swap"].sum()
        val_acc_diff = float(v1_acc - v2_acc)
        
        # McNemar test on swap validity
        v1 = sub1["is_valid_strict_swap"].to_numpy(dtype=bool)
        v2 = sub2["is_valid_strict_swap"].to_numpy(dtype=bool)
        b, c, p_val = compute_mcnemar_exact(v1, v2)
        raw_p_mcnemar.append(p_val)
        
        pair_key = f"{m1}_vs_{m2}"
        ci_comp = ci_data["comparisons"][pair_key]
        
        comp_rows.append({
            "comparison": label,
            "model_1": model_labels[m1],
            "model_2": model_labels[m2],
            "effective_acc_diff_pp": round(eff_diff * 100, 3),
            "effective_acc_ci_low": round(ci_comp["effective_acc_diff"][0] * 100, 3),
            "effective_acc_ci_high": round(ci_comp["effective_acc_diff"][1] * 100, 3),
            "valid_acc_diff_pp": round(val_acc_diff * 100, 3),
            "valid_acc_ci_low": round(ci_comp["valid_acc_diff"][0] * 100, 3),
            "valid_acc_ci_high": round(ci_comp["valid_acc_diff"][1] * 100, 3),
            "swap_error_diff_pp": round(swap_diff * 100, 3),
            "swap_error_ci_low": round(ci_comp["swap_error_diff"][0] * 100, 3),
            "swap_error_ci_high": round(ci_comp["swap_error_diff"][1] * 100, 3),
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_raw_p": p_val,
        })
        
    holm_p = holm_bonferroni(raw_p_mcnemar)
    for i, r in enumerate(comp_rows):
        r["mcnemar_holm_p"] = holm_p[i]
        
    df_comp = pd.DataFrame(comp_rows)
    comp_path = RESULTS_DIR / "paired_comparisons.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"[OK] Saved paired comparisons to {comp_path}")
    
    # 5. Write analysis.md
    write_analysis_markdown(df_agg, df_comp, RESULTS_DIR / "analysis.md")
    print(f"[OK] Analysis complete! Artifacts generated in {RESULTS_DIR}")


def write_analysis_markdown(df_agg: pd.DataFrame, df_comp: pd.DataFrame, out_path: Path) -> None:
    # Build summary markdown table
    piv = df_agg.pivot(index="model_name", columns="metric", values="estimate")
    
    md_content = f"""# Controlled Larger-Model Cross-Family Reliability Experiment Report

## 1. 개요 및 연구 목적
본 보고서는 FSE 2027 투고 논문("Do VLMs Judge Code Readability or Its Presentation? A Reliability Study of Pixels-Only Code Assessment")의 Figure 5(c)를 위한 **대형 오픈 VLM 3종 교차 계열 신뢰성 평가 실험**의 통계 분석 결과입니다.

- **데이터**: 기존 Figure 5 Qwen2.5-VL-32B 실험에서 사용된 정확히 동일한 300개 Java pair (`rq0_300_pair_pilot.csv`, easy/medium/hard 각 100쌍 균형)
- **모달리티**: `image-only` (두 snippet의 원본 코드 렌더링 이미지와 frozen A/B prompt만 제공, separate 2-image packaging)
- **평가 모델 3종**:
  1. `Qwen/Qwen2.5-VL-32B-Instruct` (BF16, greedy)
  2. `google/gemma-3-27b-it` (BF16, greedy)
  3. `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (BF16, greedy)

---

## 2. 모델별 핵심 평가 결과 (Aggregate Metrics)

| Model | Valid Acc. [95% CI] | Effective Acc. [95% CI] | Strict-Swap Error [95% CI] | Call Parse Failure | Pair Parse Failure |
|---|---:|---:|---:|---:|---:|
"""
    for m_name in df_agg["model_name"].unique():
        sub = df_agg[df_agg["model_name"] == m_name].set_index("metric")
        va = f"{sub.loc['valid_accuracy', 'estimate']:.3f} [{sub.loc['valid_accuracy', 'ci_95_low']:.3f}, {sub.loc['valid_accuracy', 'ci_95_high']:.3f}]"
        ea = f"{sub.loc['effective_accuracy', 'estimate']:.3f} [{sub.loc['effective_accuracy', 'ci_95_low']:.3f}, {sub.loc['effective_accuracy', 'ci_95_high']:.3f}]"
        se = f"{sub.loc['strict_swap_error', 'estimate']:.3f} [{sub.loc['strict_swap_error', 'ci_95_low']:.3f}, {sub.loc['strict_swap_error', 'ci_95_high']:.3f}]"
        cpf = f"{sub.loc['call_parse_failure_rate', 'estimate']:.1%}"
        ppf = f"{sub.loc['pair_parse_failure_rate', 'estimate']:.1%}"
        md_content += f"| **{m_name}** | {va} | {ea} | {se} | {cpf} | {ppf} |\n"

    md_content += """
> [!NOTE]
> - **Effective Accuracy**: Correct and valid pairs / All pairs (300 pairs)
> - **Valid Accuracy**: Correct and valid pairs / Strict-swap valid pairs
> - **Strict-swap Error**: 1 - (Strict-swap valid pairs / All pairs)
> - 모든 95% 신뢰구간은 273개 고유 snippet의 종속성을 보존하는 **10,000회 snippet-aware crossed cluster bootstrap**으로 계산되었습니다.

---

## 3. 모델 간 Paired 비교 및 가설 검정 (Paired Comparisons)

| Comparison | Effective Acc. Diff (pp) [95% CI] | Strict-Swap Error Diff (pp) [95% CI] | Exact McNemar (b/c) | Raw p-value | Holm-adjusted p-value |
|---|---:|---:|---:|---:|---:|
"""
    for _, row in df_comp.iterrows():
        c_label = row["comparison"]
        eff_diff = f"{row['effective_acc_diff_pp']:+.2f} pp [{row['effective_acc_ci_low']:+.2f}, {row['effective_acc_ci_high']:+.2f}]"
        swap_diff = f"{row['swap_error_diff_pp']:+.2f} pp [{row['swap_error_ci_low']:+.2f}, {row['swap_error_ci_high']:+.2f}]"
        mc = f"{row['mcnemar_b']} / {row['mcnemar_c']}"
        raw_p = f"{row['mcnemar_raw_p']:.4e}" if row['mcnemar_raw_p'] < 0.001 else f"{row['mcnemar_raw_p']:.4f}"
        holm_p = f"{row['mcnemar_holm_p']:.4e}" if row['mcnemar_holm_p'] < 0.001 else f"{row['mcnemar_holm_p']:.4f}"
        md_content += f"| {c_label} | {eff_diff} | {swap_diff} | {mc} | {raw_p} | {holm_p} |\n"

    md_content += """
---

## 4. 논문 작성 지침 및 허용/금지 주장

### 허용 가능한 주장 (Allowed Claims):
- "Under an exact matched protocol on 300 Java snippet pairs (image-only, BF16 greedy, separate 2-image packaging), substantial order instability (strict-swap error) persists across three 24B–32B open VLMs from distinct families."
- 세 모델 모두에서 상당한 비율의 strict-swap error가 관찰되므로, 순서 불안정성이 7B급에 국한되지 않고 24B–32B급 오픈 VLM에서도 견고하게 유지된다는 강건성 증거 제시.
- 모델 간 신뢰성 차이가 관찰될 경우 파라미터 수보다 모델 아키텍처 및 얼라인먼트 기법이 신뢰성에 큰 영향을 미친다는 탐색적 증거로 기술.

### 금지되는 주장 (Disallowed Claims):
- ❌ "Larger models are less reliable." (인과적 스케일링 효과 주장 금지)
- ❌ "Increasing model size does not improve reliability." (크기 증가의 인과성 부정 금지)
- ❌ "24B–32B models represent all frontier VLMs." (최첨단 프론티어 전체로의 무리한 일반화 금지)
- ❌ "The experiment establishes a scaling law." (스케일링 법칙 확립 주장 금지)

---

## 5. 산출물 파일 목록 및 위치
- Experiment Manifest: `experiment_manifest.json`
- Pair IDs & Ground Truth: `pair_ids.csv`
- Raw Outputs: `qwen25_vl_32b_raw.jsonl`, `gemma3_27b_raw.jsonl`, `mistral_small_31_24b_raw.jsonl`
- Pair-level Results: `pair_level_results.csv`
- Aggregate Metrics: `aggregate_metrics.csv`
- Paired Comparisons: `paired_comparisons.csv`
"""
    out_path.write_text(md_content, encoding="utf-8")


if __name__ == "__main__":
    main()
