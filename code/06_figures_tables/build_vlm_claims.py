import csv
import pandas as pd

claims = []

def add_claim(claim_id, task, scope, metric, value, ci_low="", ci_high="", p_value="", source="", description=""):
    claims.append({
        "claim_id": claim_id,
        "task": task,
        "scope": scope,
        "metric": metric,
        "value": f"{value:.4f}" if isinstance(value, (int, float)) else str(value),
        "ci_low": f"{ci_low:.4f}" if isinstance(ci_low, (int, float)) else str(ci_low),
        "ci_high": f"{ci_high:.4f}" if isinstance(ci_high, (int, float)) else str(ci_high),
        "p_value": f"{p_value:.4e}" if isinstance(p_value, (int, float)) else str(p_value),
        "source": source,
        "description": description
    })

# --- Task 0: 45,000 Identity ---
add_claim("CLM-001", "Task 0", "margin-diagnostic set", "boundary_count", 5577, "", "", "", "pair_level_results.csv", "Count of |c|==|b| in 5-model battery (45k rows)")
add_claim("CLM-002", "Task 0", "margin-diagnostic set", "boundary_valid_count", 2576, "", "", "", "pair_level_results.csv", "Count of valid pairs on |c|==|b| boundary in 5-model battery")
add_claim("CLM-003", "Task 0", "primary RQ1 set", "boundary_count", 1150, "", "", "", "A+B primary set", "Count of |c|==|b| in primary 5 models (45k rows)")
add_claim("CLM-004", "Task 0", "primary RQ1 set", "boundary_valid_count", 924, "", "", "", "A+B primary set", "Count of valid pairs on |c|==|b| boundary in primary 5 models")

# --- Task 0b: Closed Model Pilot Pairs ---
add_claim("CLM-005", "Task 0b", "gpt-5.4-mini pilot", "java_pairs", 300, "", "", "", "openai_vlm_pilot_20260707", "Unique Java pairs in pilot (100 easy, 100 med, 100 hard)")
add_claim("CLM-006", "Task 0b", "gpt-5.4 pilot", "java_pairs", 300, "", "", "", "openai_vlm_pilot_20260708", "Unique Java pairs in pilot (100 easy, 100 med, 100 hard)")
add_claim("CLM-007", "Task 0b", "gpt-5.5 pilot", "java_pairs", 300, "", "", "", "openai_vlm_pilot_20260708_gpt55_fixed", "Unique Java pairs in pilot (100 easy, 100 med, 100 hard)")

# --- Task 1: Boundary Case Sensitivity ---
# Equivalence
add_claim("CLM-010", "Task 1", "margin-diagnostic set", "boundary_equiv_m0", 1.0, "", "", 0.0, "pair_level_results.csv", "Equivalence of |c|==|b| with (m_AB==0 or m_BA==0) is 100%")
# Qwen2.5-VL-7B abstract numbers
add_claim("CLM-011", "Task 1", "Qwen2.5-VL-7B pooled", "valid_acc_permissive", 71.1680, "", "", "", "pair_level_results.csv", "Permissive valid accuracy (matches abstract 71.17%)")
add_claim("CLM-012", "Task 1", "Qwen2.5-VL-7B pooled", "valid_acc_conservative", 72.5527, "", "", "", "pair_level_results.csv", "Conservative valid accuracy (boundary invalid)")
add_claim("CLM-013", "Task 1", "Qwen2.5-VL-7B pooled", "swap_err_permissive", 45.7778, "", "", "", "pair_level_results.csv", "Permissive strict-swap error (matches abstract 45.78%)")
add_claim("CLM-014", "Task 1", "Qwen2.5-VL-7B pooled", "swap_err_conservative", 52.5556, "", "", "", "pair_level_results.csv", "Conservative strict-swap error (+6.78%p increase)")
# Pooled 5 models
add_claim("CLM-015", "Task 1", "Pooled 5 models", "valid_acc_permissive", 56.9771, "", "", "", "pair_level_results.csv", "Pooled permissive valid accuracy")
add_claim("CLM-016", "Task 1", "Pooled 5 models", "valid_acc_conservative", 57.5503, "", "", "", "pair_level_results.csv", "Pooled conservative valid accuracy")
add_claim("CLM-017", "Task 1", "Pooled 5 models", "eff_acc_permissive", 36.3311, "", "", "", "pair_level_results.csv", "Pooled permissive effective accuracy")
add_claim("CLM-018", "Task 1", "Pooled 5 models", "eff_acc_conservative", 33.4022, "", "", "", "pair_level_results.csv", "Pooled conservative effective accuracy (-2.93%p)")
add_claim("CLM-019", "Task 1", "Pooled 5 models", "swap_err_permissive", 36.2356, "", "", "", "pair_level_results.csv", "Pooled permissive strict-swap error")
add_claim("CLM-020", "Task 1", "Pooled 5 models", "swap_err_conservative", 41.9600, "", "", "", "pair_level_results.csv", "Pooled conservative strict-swap error (+5.72%p)")

# --- Task 2: Rendering Change vs Swap Stability ---
add_claim("CLM-021", "Task 2", "4 models (w/o InternVL3)", "spearman_rho", 0.4000, -1.0000, 1.0000, 0.6000, "rq2_eval_reduced + rq1", "Spearman rho between RQ1 S and max |ΔS| in reduced grid")
add_claim("CLM-022", "Task 2", "4 models (w/o InternVL3)", "monte_carlo_p", 0.6141, "", "", "", "10k MC reps", "Percentile 38.59%tile in null distribution (p=0.6141)")
add_claim("CLM-023", "Task 2", "5 models (with InternVL3)", "spearman_rho", 0.5000, -1.0000, 1.0000, 0.3910, "rq2_eval_reduced + rq1", "Spearman rho 5 models")
add_claim("CLM-024", "Task 2", "5 models (with InternVL3)", "monte_carlo_p", 0.4617, "", "", "", "10k MC reps", "Percentile 53.83%tile in null distribution (p=0.4617)")
# Max |ΔS| by model
add_claim("CLM-025", "Task 2", "InternVL3-8B", "max_abs_delta_s", 15.47, "", "", "", "reduced grid", "Max |ΔS| pooled at gaussian_sigma_4")
add_claim("CLM-026", "Task 2", "Qwen2.5-VL-7B", "max_abs_delta_s", 10.77, "", "", "", "reduced grid", "Max |ΔS| pooled at monokai_dark__fs24__wrap80__lnon")
add_claim("CLM-027", "Task 2", "InternVL3.5-8B", "max_abs_delta_s", 28.13, "", "", "", "reduced grid", "Max |ΔS| pooled at gaussian_sigma_4")
add_claim("CLM-028", "Task 2", "Gemma-3-12B", "max_abs_delta_s", 11.67, "", "", "", "reduced grid", "Max |ΔS| pooled at gaussian_sigma_4")
add_claim("CLM-029", "Task 2", "Gemma4-12B", "max_abs_delta_s", 3.70, "", "", "", "reduced grid", "Max |ΔS| pooled at gaussian_sigma_4")

# --- Task 3: InternVL3 CUDA Blur Swap Drop & Visual Regression ---
add_claim("CLM-031", "Task 3", "InternVL3-8B CUDA Baseline", "swap_error", 69.40, "", "", "", "A_B_PAIR_LEVEL.csv", "Baseline strict-swap error")
add_claim("CLM-032", "Task 3", "InternVL3-8B CUDA Blur", "swap_error", 36.20, "", "", "", "A_B_PAIR_LEVEL.csv", "Gaussian sigma 4 strict-swap error (-33.20%p drop)")
add_claim("CLM-033", "Task 3", "InternVL3-8B CUDA Baseline", "always_first_rate", 69.10, "", "", "", "A_B_PAIR_LEVEL.csv", "Always 1st choice rate explaining 99.6% of swap error")
add_claim("CLM-034", "Task 3", "InternVL3-8B CUDA Blur", "always_first_rate", 34.70, "", "", "", "A_B_PAIR_LEVEL.csv", "Always 1st choice rate in blur condition (halved)")
add_claim("CLM-035", "Task 3", "InternVL3-8B CUDA Baseline", "visual_reg_pseudo_r2", 0.2101, "", "", 0.0, "CUDA rendered images", "McFadden pseudo R2 from visual features (area, width)")
add_claim("CLM-036", "Task 3", "InternVL3-8B CUDA Blur", "visual_reg_pseudo_r2", 0.1127, "", "", 0.0, "CUDA rendered images", "McFadden pseudo R2 in blur condition (halved)")
add_claim("CLM-037", "Task 3", "InternVL3-8B CUDA Baseline", "visual_reg_auc", 0.8291, "", "", "", "CUDA rendered images", "ROC AUC for predicting choice from visual features")
add_claim("CLM-038", "Task 3", "InternVL3-8B CUDA Blur", "visual_reg_auc", 0.7249, "", "", "", "CUDA rendered images", "ROC AUC in blur condition")

# --- Task 4: H-B1 (No-Indent) Unconditional Effect & TOST ---
# 89-pair cell identification
add_claim("CLM-040", "Task 4", "InternVL3-8B / Qwen2.5 Java Dorn", "cell_n_pairs", 89, "", "", "", "B_DORN_PRIMARY_RESULTS.csv", "Exact 89-pair cell in Java Dorn perturbation experiments")
# Unconditional effective accuracy difference (pooled)
add_claim("CLM-041", "Task 4", "Qwen2.5-VL-7B pooled", "hb1_delta_e", -1.8333, -3.1238, -0.5429, 0.0382, "A_B_PAIR_LEVEL.csv", "ΔE no_indent vs baseline (TOST p=0.0382, equivalent <3%p)")
add_claim("CLM-042", "Task 4", "InternVL3.5-8B pooled", "hb1_delta_e", -2.2333, -3.5557, -0.9110, 0.1279, "InternVL3.5-8B_pair_level.csv", "ΔE no_indent vs baseline")
add_claim("CLM-043", "Task 4", "Gemma-3-12B pooled", "hb1_delta_e", -2.4667, -3.9826, -0.9508, 0.2453, "Gemma-3-12B_pair_level.csv", "ΔE no_indent vs baseline")
add_claim("CLM-044", "Task 4", "Gemma4-12B pooled", "hb1_delta_e", -2.8000, -4.1763, -1.4237, 0.3879, "Gemma4-12B_pair_level.csv", "ΔE no_indent vs baseline")
add_claim("CLM-045", "Task 4", "InternVL3-8B pooled", "hb1_delta_e", 3.0667, 1.7256, 4.4077, 0.5388, "A_B_PAIR_LEVEL.csv", "ΔE no_indent vs baseline (+3.07%p due to bias reduction)")
# MDE
add_claim("CLM-046", "Task 4", "Cell level (N=1000)", "mde_pp", 3.40, "", "", "", "power calculation", "Minimum Detectable Effect at alpha=0.05, power=0.80 (3.0%~3.8%p)")
add_claim("CLM-047", "Task 4", "Pooled level (N=3000)", "mde_pp", 1.95, "", "", "", "power calculation", "Minimum Detectable Effect at alpha=0.05, power=0.80 (1.8%~2.2%p)")

# --- Task 5: H-B2 Full Table & Directional Concordance ---
add_claim("CLM-051", "Task 5", "Gemma-3-12B blur=4", "concordance_sign", "-", "", "", 0.25, "Gemma-3-12B_summary_metrics.csv", "All 3 languages show negative ΔE (CUDA -5.0, Java -17.0, Py -0.9)")
add_claim("CLM-052", "Task 5", "Gemma4-12B blur=4", "concordance_sign", "-", "", "", 0.25, "Gemma4-12B_summary_metrics.csv", "All 3 languages show negative ΔE (CUDA -5.5, Java -4.1, Py -0.4)")
add_claim("CLM-053", "Task 5", "InternVL3.5-8B blur=4", "concordance_sign", "-", "", "", 0.25, "InternVL3.5-8B_summary_metrics.csv", "All 3 languages show negative ΔE (CUDA -20.5, Java -17.0, Py -27.7)")
add_claim("CLM-054", "Task 5", "InternVL3-8B blur=4", "concordance_sign", "+", "", "", 0.25, "B_PERTURBATION_RESULTS.csv", "All 3 languages show positive ΔE (CUDA +23.4, Java +2.1, Py +15.3)")
add_claim("CLM-055", "Task 5", "Qwen2.5-VL-7B blur=4", "concordance_sign", "discordant", "", "", 1.0, "B_PERTURBATION_RESULTS.csv", "Discordant ΔE (CUDA -12.6, Java -13.6, Python +0.8)")

# --- Task 6: Noise Ceiling Normalization ---
add_claim("CLM-061", "Task 6", "Cross-language rank", "rank_change_count", 0, "", "", "", "11 models comparison", "Rankings of all 11 models (3 ML + 8 VLM) are 100% unchanged after ceiling norm")
add_claim("CLM-062", "Task 6", "CUDA Best VLM vs ML", "norm_gap_pp", -17.97, "", "", "", "norm by 98.7%", "Normalized gap between Gemma4-12B (58.83%) and SVR (76.80%)")
add_claim("CLM-063", "Task 6", "Java Best VLM vs ML", "norm_gap_pp", -11.01, "", "", "", "norm by 80.5%", "Normalized gap between Gemma4-12B (67.45%) and MLP (78.47%)")
add_claim("CLM-064", "Task 6", "Python Best VLM vs ML", "norm_gap_pp", -6.48, "", "", "", "norm by 98.8%", "Normalized gap between Gemma4-12B (58.97%) and Voting (65.45%)")

# --- Task 7: Within-Dataset Java Recalculation ---
add_claim("CLM-071", "Task 7", "Java Within n=1004", "buse_pairs", 299, "", "", "", "rq1_pairs_9000.csv", "Within-Buse pair count")
add_claim("CLM-072", "Task 7", "Java Within n=1004", "dorn_pairs", 275, "", "", "", "rq1_pairs_9000.csv", "Within-Dorn pair count")
add_claim("CLM-073", "Task 7", "Java Within n=1004", "scalabrino_pairs", 430, "", "", "", "rq1_pairs_9000.csv", "Within-Scalabrino pair count")
add_claim("CLM-074", "Task 7", "Gemma4-12B Java", "within_valid_acc", 64.83, "", "", "", "primary_pair_level.csv", "Valid Acc on within-dataset Java (+2.94%p vs mixed 61.89%)")
add_claim("CLM-075", "Task 7", "Gemma4-12B Java", "within_swap_err", 15.04, "", "", "", "primary_pair_level.csv", "Strict-swap error on within Java (+2.77%p vs mixed 12.27%)")
add_claim("CLM-076", "Task 7", "Gemma-3-12B Java", "within_valid_acc", 66.21, "", "", "", "pair_level_results.csv", "Valid Acc on within-dataset Java (+3.67%p vs mixed 62.54%)")
add_claim("CLM-077", "Task 7", "Gemma-3-12B Java", "within_swap_err", 20.72, "", "", "", "pair_level_results.csv", "Strict-swap error on within Java (+1.95%p vs mixed 18.77%)")
add_claim("CLM-078", "Task 7", "Qwen2.5-VL-7B Java", "within_swap_err", 51.10, "", "", "", "pair_level_results.csv", "Strict-swap error on within Java (+3.60%p vs mixed 47.50%)")
add_claim("CLM-079", "Task 7", "Gemma4-12B Pooled Recalc", "recalc_e_7004", 57.72, "", "", "", "n=7004 pooled", "Pooled Effective Acc with Java within (+0.85%p vs 56.88%)")

# --- Task 8: Baseline Selection Bias (9 Classical Baselines) ---
add_claim("CLM-081", "Task 8", "9 Baselines CUDA", "expected_mean_acc", 70.71, "", "", "", "ml_predictions_9models_9000.csv", "Mean accuracy across 9 baselines on CUDA")
add_claim("CLM-082", "Task 8", "9 Baselines CUDA", "median_acc", 71.06, "", "", "", "ml_predictions_9models_9000.csv", "Median accuracy across 9 baselines on CUDA")
add_claim("CLM-083", "Task 8", "9 Baselines Java", "expected_mean_acc", 60.31, "", "", "", "ml_predictions_9models_9000.csv", "Mean accuracy across 9 baselines on Java")
add_claim("CLM-084", "Task 8", "9 Baselines Java", "median_acc", 60.66, "", "", "", "ml_predictions_9models_9000.csv", "Median accuracy across 9 baselines on Java")
add_claim("CLM-085", "Task 8", "9 Baselines Python", "expected_mean_acc", 62.84, "", "", "", "ml_predictions_9models_9000.csv", "Mean accuracy across 9 baselines on Python")
add_claim("CLM-086", "Task 8", "9 Baselines Python", "median_acc", 62.85, "", "", "", "ml_predictions_9models_9000.csv", "Median accuracy across 9 baselines on Python")
add_claim("CLM-087", "Task 8", "9 Baselines 3-Lang Pooled", "expected_mean_acc", 64.62, "", "", "", "ml_predictions_9models_9000.csv", "Mean pooled accuracy across 9 baselines")
add_claim("CLM-088", "Task 8", "9 Baselines 3-Lang Pooled", "median_acc", 64.92, "", "", "", "ml_predictions_9models_9000.csv", "Median pooled accuracy across 9 baselines")

# --- Task 9: Closed-Model Pilot (Figure 5) Bootstrap Intervals ---
# Contrasts with 0 included
add_claim("CLM-091", "Task 9", "gpt-5.4 Text+Image vs Image", "separate_delta_e", -1.0000, -14.6743, 13.5377, "", "openai_vlm_pilot_20260708", "ΔE separate Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-092", "Task 9", "gpt-5.4 Text+Image vs Image", "separate_delta_s", 9.6667, -3.5142, 22.6709, "", "openai_vlm_pilot_20260708", "ΔS separate Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-093", "Task 9", "gpt-5.4 Text+Image vs Image", "combined_delta_e", -1.3333, -16.0764, 14.3698, "", "openai_vlm_pilot_20260708", "ΔE combined Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-094", "Task 9", "gpt-5.4 Text+Image vs Image", "combined_delta_s", 8.3333, -4.5461, 21.1624, "", "openai_vlm_pilot_20260708", "ΔS combined Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-095", "Task 9", "gpt-5.5 Text+Image vs Image", "separate_delta_e", -5.6667, -21.2942, 9.4652, "", "openai_vlm_pilot_20260708_gpt55_fixed", "ΔE separate Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-096", "Task 9", "gpt-5.5 Text+Image vs Image", "separate_delta_s", 6.6667, -7.2802, 21.5477, "", "openai_vlm_pilot_20260708_gpt55_fixed", "ΔS separate Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-097", "Task 9", "gpt-5.5 Text+Image vs Image", "combined_delta_e", -0.3333, -16.6245, 14.8597, "", "openai_vlm_pilot_20260708_gpt55_fixed", "ΔE combined Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-098", "Task 9", "gpt-5.5 Text+Image vs Image", "combined_delta_s", -2.3333, -16.2903, 12.7908, "", "openai_vlm_pilot_20260708_gpt55_fixed", "ΔS combined Text+Image vs Image-Only (95% CI includes 0)")
add_claim("CLM-099", "Task 9", "gpt-5.4 Reasoning Budget", "image_only_delta_e", 4.3333, -1.7778, 11.6885, "", "gpt54_reasoning_budget_pilot_90_20260709", "ΔE High vs Low mot (image_only) (95% CI includes 0)")
add_claim("CLM-100", "Task 9", "gpt-5.4 Reasoning Budget", "text_img_delta_e", 1.0000, -6.0071, 8.5821, "", "gpt54_reasoning_budget_pilot_90_20260709", "ΔE High vs Low mot (text+img) (95% CI includes 0)")

df_claims = pd.DataFrame(claims)
df_claims.to_csv("notes/vlm_claims.csv", index=False)
print(f"Successfully generated notes/vlm_claims.csv with {len(df_claims)} claims.")
