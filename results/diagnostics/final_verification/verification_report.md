# FSE 2027 Final Experiment and Statistical Verification

Generated from frozen local artifacts on 2026-08-20 UTC. No original result, log, manifest, prediction file, or LaTeX file was modified. The requested manuscript path `/ASE'26 VLM-as-a-Judge (Copy)/main.tex` was not present on this host, so section/table references below follow the supplied verification prompt; exact LaTeX line numbers could not be checked.

## F1. Java OCR loss

- **판정:** 9.37 percentage points.
- **최종 값:** Source-access RF = 1,870/3,000 = 0.6233333333333333; RapidOCR+SVR = 1,589/3,000 = 0.5296666666666666; difference = 281/3,000 = 0.09366666666666668 = **9.366666666666667 pp**, rounded to **9.37 pp**.
- **근거:** `/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/data/ml_predictions_9models_9000.csv`, fields `language`, `model`, `is_correct`; `/ANON/scratch_rq1/deploy_v2_logit_20260731/data/F5_OCRML_GATE.csv`, fields `language`, `ocr_engine`, `model`, `correct`, `n`, `effective_accuracy`.
- **관련 코드:** `/ANON/experiment_root/results/rq1_model_battery_3lang_20260723/code/build_table1_ml_replication.py`; `/ANON/scratch_rq1/deploy_v2_logit_20260731/code/validate_ocrml_gate.py`.
- **논문 수정 필요 여부:** The loss value is correct. Keep 9.37 and, if space permits, state it is computed from unrounded counts. Subtracting the independently rounded displays (62.33 - 52.97) produces 9.36, explaining the apparent mismatch.
- **기존 결론 유지 여부:** Yes.

Full machine-readable audit: `/ANON/experiment_root/results/fse2027_final_verification/ocr_loss_audit.csv`.

## F2. Qwen3.5

- **판정:** Exact model ID and client protocol are recoverable; backend runtime details are insufficient because this was a hosted NVIDIA API run. The supplied description "300 Java pairs" is false.
- **정확한 모델:** `qwen/qwen3.5-122b-a10b`, a multimodal MoE model documented by Qwen as 122B total and 10B activated parameters.
- **근거:** `/ANON/experiment_root/results/nvidia_qwen35_extension_pilot_20260713/manifest.json`, `calls.jsonl`, `pilot_pairs.csv`, `summary.csv`, and `run_stats.json`; runner `/ANON/experiment_root/experiments/rq0_viability/scripts/run_nvidia_qwen_extension_pilot.py`.
- **표에 사용할 이름:** `Qwen3.5-122B-A10B (MoE, 10B active; non-thinking, NVIDIA API)`.
- **프로토콜:** NVIDIA endpoint `https://integrate.api.nvidia.com/v1/chat/completions`; user message only; `enable_thinking=false`; temperature 0; max tokens 64; seed 42; non-streaming. No `top_p` was sent. Existing PNG bytes were sent without client resizing/cropping, as base64 or NVIDIA assets depending on a 180,000-byte threshold.
- **표본:** 300 total pairs = Java 150 (Dorn 75 + Sergeyuk2024 75), Python 75, CUDA 75; each dataset-language contributes 25 easy, 25 medium, and 25 hard pairs. Two modalities and AB/BA yield 1,200 final successful calls.
- **기존 수치 재현 여부:** Image-only effective = 69/300 = **23.00%** and swap error = 195/300 = **65.00%**, both reproduced. Image-only parse failures = 65/600 calls = **10.8333%**, so **11.00% is not an exact two-decimal result**. Across both modalities, 131/1,200 = 10.9167% (10.92% to two decimals).
- **기록 불충분 항목:** checkpoint revision, server engine/version, server `transformers` version, dtype, quantization, exact server chat template, and provider-side image processor are not exposed in the saved API response. They must remain `not recorded`, not inferred from a current model card.

Official model references are recorded in `/ANON/experiment_root/results/fse2027_final_verification/qwen35_run_metadata.json`.

## F3. Mann-Whitney independence

- **판정:** **방향성은 유지되지만 p-value와 분석 계보 설명을 교체해야 함.** The manuscript prompt conflates two distinct analyses.
- **기존 분석:** The actual four `p<1e-62` tests are model (InternVL/Qwen) x component (`|b|`/`|c|`) at one 3,000-pair baseline condition, with no boundary exclusion. Raw p-values are 1.803e-71, 6.836e-249, 2.702e-144, and 2.630e-274; all remain below 1e-62 under Holm over four. Source: `/ANON/experiment_root/results/protocol_unified_3lang_20260720/a1/analysis/logit_component_mannwhitney.csv`.
- **99,832 observation lineage:** `/ANON/experiment_root/results/grounded_protocol_3lang_20260721/analysis/logit_decomposition/LOGIT_PAIR_LEVEL.csv` has 108,000 rendering/perturbation observations, of which 8,168 satisfy `|b|=|c|` and 99,832 are non-boundary. Its existing script writes 216 condition/language/model/component Mann-Whitney rows, not the four baseline tests. There is no original artifact that simultaneously supports "99,832 non-boundary", "four tests", and the historical p-values.
- **재분석 groups:** grid-InternVL (33,274 non-boundary), grid-Qwen (33,523), perturbation-InternVL (16,347), perturbation-Qwen (16,688). The eight tests are four groups x two components.
- **boundary/ties:** New analysis excludes exactly `logit_boundary=True`, defined as `|b|=|c|`; exact AB/BA logit ties are retained unless they induce that boundary. Strict valid/swap comes from the frozen `strict_valid` field.
- **effect:** Mean difference and Cliff's delta are `swap - valid`; therefore the expected signs are positive for `|b|` and negative for `|c|`.

| Group | metric | swap n / valid n | swap mean (median) | valid mean (median) | mean diff | Cliff's delta | pair-cluster 95% CI | snippet-aware 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| grid InternVL | `|b|` | 17,939 / 15,335 | 0.9992 (1.0000) | 0.6413 (0.5625) | 0.3579 | 0.4468 | [0.3362, 0.3797] | [0.3001, 0.4139] |
| grid InternVL | `|c|` | 17,939 / 15,335 | 0.4218 (0.3125) | 1.2637 (1.1875) | -0.8419 | -0.7769 | [-0.8691, -0.8152] | [-0.9192, -0.7704] |
| grid Qwen | `|b|` | 16,916 / 16,607 | 0.9948 (0.9375) | 0.5250 (0.5000) | 0.4698 | 0.6167 | [0.4512, 0.4891] | [0.4228, 0.5176] |
| grid Qwen | `|c|` | 16,916 / 16,607 | 0.4215 (0.3750) | 1.2591 (1.1875) | -0.8376 | -0.8101 | [-0.8656, -0.8103] | [-0.9042, -0.7764] |
| perturbation InternVL | `|b|` | 6,954 / 9,393 | 0.8696 (0.8125) | 0.5983 (0.5000) | 0.2713 | 0.3605 | [0.2513, 0.2912] | [0.2216, 0.3229] |
| perturbation InternVL | `|c|` | 6,954 / 9,393 | 0.3887 (0.3125) | 1.2849 (1.1875) | -0.8962 | -0.7776 | [-0.9218, -0.8698] | [-0.9661, -0.8324] |
| perturbation Qwen | `|b|` | 9,112 / 7,576 | 0.9041 (0.8750) | 0.5270 (0.5000) | 0.3771 | 0.5595 | [0.3592, 0.3951] | [0.3329, 0.4217] |
| perturbation Qwen | `|c|` | 9,112 / 7,576 | 0.3742 (0.3125) | 1.1906 (1.1250) | -0.8164 | -0.8412 | [-0.8414, -0.7913] | [-0.8767, -0.7595] |

- **cluster-aware 결과:** Pair bootstrap clusters all conditions by `pair_id` (3,000 clusters/group). The snippet-aware crossed Bayesian/pigeonhole bootstrap gives each endpoint an independent exponential weight and weights a pair by the product (552 endpoint clusters/group). Both use 10,000 draws, base seed 20260820, and preserve all eight directions.
- **p-values:** The centered-bootstrap raw p-value reaches its 10,000-draw Monte Carlo floor, 1/10,001 = 0.00009999, for all tests; Holm over eight gives 0.00079992. Report `Holm p <= .0008 at 10,000-resample resolution`, not `p<1e-62` for this dependence-aware analysis.
- **결론 유지 여부:** The substantive mechanism conclusion is robust. The astronomical p-value and the statement that it came from 99,832 non-boundary observations in four tests are not.

Reproducible data: `/ANON/experiment_root/results/fse2027_final_verification/logit_cluster_statistics.csv` and `original_logit_mannwhitney_audit.csv`; script `code/recompute_logit_cluster_statistics.py`.

## F4. Rendering contrasts

- **판정:** §5 and §6 use different 66-count definitions.
- **§5의 66:** 11 non-baseline grid conditions x six model-language combinations = 66 rows. Baseline is `monokai_dark__fs20__wrap80__lnon`. Exactly 22 rows have `holm_reject_0_05=True`.
- **§6의 66:** All `C(12,2)=66` unordered condition pairs **within each** model-language combination; six combinations produce 396 condition pairs. Three preference metrics yield 1,188 rows.
- **Holm family:** §5 correction is separate over 11 contrasts within each model-language block, not over all 66. §6 correction is separate over 66 pairs within each model-language-metric block.
- **96 후속 수치:** 66 grid baseline rows + 30 perturbation baseline rows (five perturbations x six model-language combinations).
- **논문에 사용할 설명:** "The 66 RQ2 rows comprise 11 baseline comparisons in each of six model-language combinations, with Holm adjustment applied separately to each 11-test model-language family. The localization analysis is distinct: it evaluates all C(12,2)=66 unordered rendering-condition pairs within each model-language combination (396 condition pairs overall)."

All 12 conditions and the exact 22 significant rows are in `/ANON/experiment_root/results/fse2027_final_verification/rendering_contrast_definition.json`.

## F5. 632-feature diagnostics

- **판정:** Correct the feature count and qualify leakage. The headline AUCs reproduce, but they are pair-level OOF, not held-out and not snippet-disjoint.
- **split:** Five-fold stratified pair-row CV, shuffled, seed 42. No nested tuning. Same snippet can occur in train and validation; exact/reversed pairs do not occur as separate rows.
- **features:** 632 is total table columns. Predictors are 164 source + 104 pixel visual/layout = 268 engineered features, plus three numeric controls and categorical difficulty. OCR features = 0; VLM-derived features = 0.
- **reproduced AUC:** Java 0.8892/0.7992/0.8800 for RF failure/rescue/valid correctness; Python+CUDA pooled 0.9499/0.8580/0.9187.
- **visual increment:** Five of eight effective-rescue conditions have ordinary row-bootstrap, **unadjusted** p<.01 with 2,000 draws. Only one remains p<.05 after Holm over eight.
- **snippet-disjoint 결과:** Strict endpoint-disjoint sensitivity lowers Java AUCs to 0.8560/0.6559/0.6913 on 571/200/155 evaluable rows and pooled Python/CUDA to 0.9180/0.7282/0.8109 on 1,155/387/283 rows. Crossing edges are excluded, so denominator/coverage must accompany these values.
- **Methods용 설명:** "We trained gradient-boosting classifiers on 268 engineered pair features (164 source and 104 pixel-derived visual/layout features), three pair-level controls, and difficulty. Reported AUCs are five-fold stratified pair-level OOF estimates (Java: n=3,000; Python/CUDA pooled: n=6,000; seed 42); because snippets recur across pair folds, we treat them as explanatory and report a separate endpoint-disjoint sensitivity analysis."

Full audit: `/ANON/experiment_root/results/fse2027_final_verification/mechanism_model_audit.md`.

## F6. Difficulty bins

- **판정:** Describe as **study-specific descriptive/legacy bins**, not direct Cohen thresholds.
- **`|Delta z|` 정의:** `|[(rating_i - dataset_mean)/dataset_population_SD] - [(rating_j - dataset_mean)/dataset_population_SD]|`; implementation uses `ddof=0`.
- **threshold 근거:** 0.2/0.5/1.0 are present in pair-construction code before the 2026-06-19 full-pair artifact, but no contemporaneous rationale or external standard for 1.0 was found. Local mtime is not external preregistration.
- **Cohen 인용 유지 여부:** No. This statistic is not Cohen's d, and 1.0 is not the conventional 0.8 large-effect boundary.
- **논문에 사용할 설명:** "We analyze difficulty primarily as continuous |Delta z|. For balanced sampling and legacy reproducibility only, we report study-specific bins: hard [0.2,0.5), medium [0.5,1.0), and easy >=1.0; pairs below 0.2 are excluded."

Full provenance: `/ANON/experiment_root/results/fse2027_final_verification/difficulty_bin_provenance.md`.

## F7. Preregistration

- **판정:** Use `prespecified before rerun`.
- **사전 문서:** `/ANON/scratch_rq1/deploy_v2_logit_20260731/config/F5_PREREGISTRATION.md`; execution lock `config/F5_EXECUTION_LOCK.json`.
- **시간/hash 근거:** protocol birth/mtime 2026-07-31 01:58:54 UTC; lock 02:01:13.544493 UTC; first call 02:01:53.953237 UTC. Current protocol SHA-256 `c008a88f...34a7` exactly matches the lock.
- **내용:** Models/pipelines, 54,000 calls, D and tie/boundary definitions, comparator, 10,000 endpoint-cluster bootstrap, exact McNemar, Holm over nine, and categories (a)/(b)/(c) are all fixed.
- **실행 후 수정:** No hash mismatch detected. However, there is no Git repository or external/public immutable registration timestamp, so local timestamps cannot fully support conventional preregistration language.
- **익명 package:** Artifacts are includable after copying from `/data/tmp` and preserving hashes; machine-specific paths may need anonymization.
- **preregistered 표현 유지 여부:** Weaken to "prespecified in a SHA-256-locked protocol before the rerun."

Full audit: `/ANON/experiment_root/results/fse2027_final_verification/deploy_v2_preregistration_audit.md`.

## Summary

| 항목 | 확인 | 정정 필요 | 재분석 필요 | 논문 결론 영향 |
|---|---|---|---|---|
| F1 OCR loss | Yes, 9.37 pp | No; explain unrounded subtraction | No | None |
| F2 Qwen3.5 | Partial runtime reconstruction | Yes: exact name, sample composition, parse rate/backend unknowns | No new calls | Protocol table/limitations |
| F3 margin statistics | Direction confirmed | Yes: separate baseline-four-test and 99,832 lineages; replace clustered p | Completed | Mechanism stays; inferential wording changes |
| F4 contrasts | Yes | Yes: Holm family wording | No | Clarifies multiplicity only |
| F5 mechanism | AUCs reproduced | Yes: 632 columns, 268 features; leakage and unadjusted p | Snippet-disjoint completed | Magnitudes weaken; qualitative predictability remains |
| F6 difficulty | Formula/bins confirmed | Yes: remove direct Cohen claim | No | Method rationale only |
| F7 deploy-v2 | Prespecification confirmed | Yes: weaken `preregistered` | No | Transparency wording only |

## Execution

Commands, seeds, denominators, and hash generation are in `/ANON/experiment_root/results/fse2027_final_verification/commands.sh`. Generated-file hashes are in `SHA256SUMS`.

