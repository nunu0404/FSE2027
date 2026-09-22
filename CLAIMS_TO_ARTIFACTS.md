# Claims → Artifacts

Every quantitative claim in the manuscript, mapped to the file that contains it
and the script that produced that file. Paths are relative to this package.

Values marked **[verified]** were recomputed from the shipped raw data during
assembly of this package and matched the manuscript. Values marked
**[verified, discrepancy]** did not; see [Known discrepancies](#known-discrepancies).

---

## Abstract

| Claim | Value | Artifact | Producer |
| --- | --- | --- | --- |
| Pairs of rendered snippets | 9,000 | `data/pairs/rq1_pairs_9000_seed42.csv` **[verified]** | `code/00_data_preparation/build_pair_sets.py` |
| Qwen2.5-VL matches human ratings on consistent pairs | 71% | `results/rq1_viability/pair_level/pair_level_5model_battery_45000.csv` → 71.17% **[verified]** | `code/05_analysis/rq1/analyze_full_results.py` |
| Its two answers disagree | 46% of pairs | same file → S = 45.78% **[verified]** | same |
| Order disagreement exceeds single-order flips | all 15 model×language groups | `results/rq2_reliability/verified_recomputation/rq2_order_vs_rendering.csv` **[verified]** | `code/99_verification/scratch_calc_flip_6groups.py`, `scratch_calc_flip_7conds.py` |
| Two-order averaged accuracy over all pairs | 60–65% | `results/rq1_viability/bootstrap_intervals/fig_data/fig4_data.csv` → 59.9–64.6% **[verified]** | `code/05_analysis/rq1/generate_rq1_bundle.py` |
| Pooled source-feature reference | 68% | `data/features_and_baseline_predictions/java/table1_ml_replication/canonical_pairwise_summary.csv` → 67.88% **[verified]** | `code/03_source_feature_baselines/` |
| Both tested models prefer original code in most pairs | — | `results/rq3_content_vs_appearance/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv` **[verified]** | `code/05_analysis/rq3/analyze_rq3_image_only.py` |
| All three paired CIs include zero | — | `results/rq4_image_only/image_only_pipelines/F5_deploy_debiased.csv` + `recomputed_intervals_20260920.csv` **[verified]** | `code/05_analysis/rq4/analyze_f5.py`, `code/99_verification/scratch_compute_ci.py` |

---

## Section 3 — Study design

| Claim | Artifact |
| --- | --- |
| Metric definitions `A_valid`, `E`, `S` (Eq. 1–3) | `code/05_analysis/rq1/analyze_full_results.py`; column semantics in `results/rq1_viability/analysis_5model_battery/ANALYSIS_METADATA.json` |
| Two-order averaged accuracy `D`, ties counted as errors | `debiased_correct` column of both pair-level files; loop in `code/05_analysis/rq1/generate_rq1_bundle.py` L140–235 |
| Logit-margin decomposition `b`, `c` (Eq. 4) | `results/rq2_reliability/logit_margin_diagnostics/logit_decomposition.csv` | 
| Validity ⟺ `\|c\| > \|b\|` off-boundary, zero identity violations | `results/rq2_reliability/logit_margin_diagnostics/INTEGRITY.json` |
| Judging rubric (Figure 3) | `config/prompts/prompt_B_readability_judge.txt`, SHA-256 `1eb486ba…978ae1`. All three frozen copies in the original runs are byte-identical — see `config/prompts/README.md` |

---

## Section 4 — Experimental setup

### 4.1 Benchmarks and pair construction

| Claim | Value | Artifact |
| --- | --- | --- |
| Java snippets: Buse / Dorn / Scalabrino | 99 / 90 / 124 = 313 | `data/snippets/snippets_552_with_human_ratings.csv` **[verified]** |
| Dorn Python / CUDA | 119 / 120 | same **[verified]** |
| Total snippets | 552 | same **[verified]** |
| Java pairs within-dataset / cross-dataset | 1,004 / 1,996 | `data/pairs/rq1_pairs_9000_seed42.csv` **[verified]** |
| Pairs per language, per stratum | 3,000 / 1,000 | same **[verified]** |
| Preferred snippet appears first | 48.2–51.5% of strata | same **[verified]** |
| Strata: hard / medium / easy | 0.2≤\|Δz\|<0.5 / 0.5≤\|Δz\|<1.0 / \|Δz\|≥1.0 | same **[verified]** |
| Seed | 42 | `config/protocols/protocol_rq1_model_battery.json` |
| Half-panel vs full-panel direction agreement, Python | 98.8% | `results/diagnostics/noise_ceiling/ceiling_by_language_bin.csv` → 98.82% **[verified]** |
| Same, CUDA | 98.7% | same → 98.69% **[verified]** |
| Same, hard pairs | 96.1–96.5% | same → 96.12 / 96.48% **[verified]** |
| Same, nine-rater Scalabrino pool | 80.5% | same → 80.49% **[verified]** |
| 1,000 random rater splits | — | `code/05_analysis/shared/estimate_noise_ceiling.py`, `results/diagnostics/noise_ceiling/run_manifest.json` |

### 4.2 Models and baselines (Table 1)

Table 1 is descriptive. Its scope is fixed by the pre-registered protocols:

| Row | Protocol / manifest |
| --- | --- |
| RQ1 primary (5 judges) | `config/protocols/protocol_rq1_model_battery.json`, `protocol_rq1_latest_vlm_extension.json` |
| Logit-margin diagnostic (5 judges) | `results/rq1_viability/raw_inference/{qwen,gemma,internvl,ministral,phi}/manifest.json` |
| RQ2 full grid (2 judges, 17 conditions) | `config/protocols/protocol_rq2_rq3_grounded_3lang.json` |
| RQ2 reduced grid (3 judges, 7 conditions) | `config/protocols/rq2_reduced_manifests/*.json` |
| RQ3 image-only (2 judges, 6 contrasts) | `data/rq3_variants/RQ3_PREPARATION_MANIFEST.json` |
| RQ4 direct route and text ablation | `config/protocols/protocol_rq4_deploy_execution_lock.json` |
| Protocol checks | `results/protocol_checks/*/[...]_manifest.json`, `experiment_manifest.json` |

Exact checkpoint revisions for all 12 open-weight models:
`environment/MODEL_REVISIONS.json`.

Nine supervised source-feature predictors (RF, MLP, SVR, voting, linear and tree
ensembles): trained by `code/03_source_feature_baselines/`, out-of-fold pair
predictions in `data/features_and_baseline_predictions/ml_predictions_9models_9000.csv`
(Java from `java/`, Python and CUDA from `python_cuda/`).

### 4.3 Prompting and execution

| Claim | Artifact |
| --- | --- |
| BF16, greedy, temperature 0, 24-token limit | `config/protocols/*.json`; `environment/MODEL_REVISIONS.json` |
| Environment capture | `environment/env_rq1_model_battery.json` (Python 3.13.11, torch 2.11.0, transformers 5.9.0; separate Phi-4 env recorded), `environment/latest_vlm_extension/`, `environment/env_review_defense.json` |
| Logits collected at the verdict step and checked against generated signs | `code/02_inference/rq1/run_rq1_vlm.py`; validation in `results/rq1_viability/raw_inference/*/validation.json` |

### 4.4 Experiment-specific configuration

| Claim | Value | Artifact |
| --- | --- | --- |
| RQ2 full grid observations | 102,000 (17 unique conditions) | `results/rq2_reliability/full_grid_2models/A_B_PAIR_LEVEL.csv` holds 108,000 rows over 18 condition labels; the `baseline` label duplicates `monokai_dark__fs20__wrap80__lnon` exactly **[verified]** — evidence in `code/99_verification/scratch_check_baseline_dup.py`, `docs/statistical_verification_20260919_ko.md` §2.3 item 3 |
| 12 rendering conditions = 3 themes × 2 font sizes × 2 wrap widths | — | `data/render_metadata/render_conditions.csv`; `images/rq2_rendering_grid/` (12 dirs × 552) |
| 5 perturbations + baseline | — | `images/rq2_perturbation/` (6 dirs × 552) |
| RQ2 reduced grid, 7 conditions | 21,000 rows/judge | `results/rq2_reliability/reduced_grid_3models/*/pair_level/*.csv` **[verified]** |
| RQ3: 100 originals/language, 6 contrasts, 2 models, both orders | 7,200 calls | `results/rq3_content_vs_appearance/raw_inference/raw/` **[verified: 3,600 pair rows × 2 orders]** |
| RQ3 transformations, 5–333 mutations per snippet | — | `data/rq3_variants/rq3_mutation_log.csv`, `rq3_variant_sources.csv` |
| Diagnostic set: 45,000 pairs (5 judges × 9,000) | — | `results/rq1_viability/pair_level/pair_level_5model_battery_45000.csv` **[verified]** |
| Closed-model checks: 300 pairs per condition | — | `results/protocol_checks/closed_models_gpt/` |
| Bootstrap: 10,000 replicates, seed 42, product snippet weights | — | `code/05_analysis/rq1/generate_rq1_bundle.py` L140–235; `code/05_analysis/review_followups/analyze_e1.py` |
| McNemar + Holm over 66 rendering and 30 perturbation contrasts | — | `results/diagnostics/review_defense_e1_e8/analysis/E8/E8_multiplicity_families.csv` |

---

## Section 5.1 — RQ1 (Figures 6, 7; Table 2)

### Best source-feature baselines

| Language | Model | Manuscript | Artifact value |
| --- | --- | --- | --- |
| Java | Multilayer Perceptron | 63.17% | 1895/3000 = **63.17%** **[verified]** |
| Python | Voting ensemble (LR+NB+RF) | 64.67% | 1940/3000 = **64.67%** **[verified]** |
| CUDA | SVR | 75.80% | 2274/3000 = **75.80%** **[verified]** |
| Equally weighted reference | | 67.9% | 67.88% **[verified]** |

Source of record:
`data/features_and_baseline_predictions/java/table1_ml_replication/canonical_pairwise_summary.csv`.
**Not** `ml_predictions_9models_9000.csv`, which is stale for three pairs — see
[A trap for replicators](#a-trap-for-replicators).

### Figure 7 — pooled `E`, `D`, `S` over 9,000 pairs

Recomputed from `results/rq1_viability/pair_level/*.csv`; all eight judges match
the manuscript exactly **[verified]**.

| Judge | E | D | S |
| --- | --- | --- | --- |
| Gemma4-12B | 56.88 | 64.62 | 14.91 |
| Gemma-3-12B | 50.36 | 62.01 | 21.79 |
| Qwen3-VL-8B | 44.50 | 59.91 | 28.40 |
| InternVL3.5-8B | 43.94 | 62.30 | 33.10 |
| Qwen2.5-VL-7B | 38.59 | 64.16 | 45.78 |
| Ministral-3-8B | 36.74 | 54.30 | 33.29 |
| InternVL3-8B | 24.73 | 51.00 | 49.93 |
| Phi-4-multimodal | 31.23 | 37.83 | 30.39 |

95% intervals: `figures_and_tables/fig07_order_robustness/fig07_data.csv` and
`results/rq1_viability/bootstrap_intervals/rq1_intervals.csv` (128 rows: 8 judges
× 4 scopes × 4 metrics). LaTeX source in
`figures_and_tables/fig07_order_robustness/vlm_tab_rq1_intervals.tex`.

### Figure 6 — per-language valid and effective accuracy

`figures_and_tables/fig06_rq1_by_language/fig06_data.csv` (24 rows).
Three cells differ from the manuscript by exactly 0.1 pp through boundary
half-rounding; the exact unrounded ratios are tabulated in
`figures_and_tables/FIGURE_NUMBER_HISTORY.md`.

### Other Section 5.1 claims

| Claim | Value | Artifact |
| --- | --- | --- |
| Policy gap `D − E`, primary set | 7.7–25.6 pp | `figures_and_tables/fig07_order_robustness/fig07_data.csv` **[verified]** |
| Including three earlier checkpoints | 6.6–26.3 pp | same **[verified]** |
| Qwen Python `D` vs RF | 66.77%, +4.0 pp, CI [−4.31, +12.04] | `results/diagnostics/review_defense_e1_e8/analysis/E1/E1_debiased_vs_baseline.csv` **[verified]** |
| Qwen Python vs best source baseline | +2.1 pp, CI [−6.18, +10.23] | same **[verified]**. Note: `analyze_e1.py` recorded `language_best_accuracy = 0.646`, the stale value, so the interval is computed against 64.60% rather than the corrected 64.67%. The point estimate the manuscript prints, +2.1 pp, is 66.77 − 64.67 and is correct |
| All 18,000 Qwen RQ1 calls parsed; 45.78% invalidity is parsed disagreement | — | `code/99_verification/scratch_rq1_qwen_parse.py`; `docs/statistical_verification_20260919_ko.md` §2.2 |
| Phi-4 exact BF16 candidate-oriented ties (`c = 0`) | 1,418 / 9,000 | `results/diagnostics/vlm_claims.csv`; `figures_and_tables/fig09_logit_margin/fig09_data_5_diagnostic_judges.csv` (phi decile 1: n = 2,177, c_ties = 1,418) **[verified]** |
| Call-level first-position selection rate | 24.6–78.1% | `results/rq1_viability/analysis_5model_battery/metrics_by_scope.csv` |
| Within-dataset Java check; baseline−Gemma4 gap 8.86 vs 8.87 pp | — | `results/rq1_viability/bootstrap_intervals/java_within.csv`; `code/06_figures_tables/task7_java_within_recalc.py` |

### Table 2 — protocol checks

**(a) Closed models, Java, 300 pairs/condition.** Source:
`results/protocol_checks/closed_models_gpt/*/[model]__pilot_summary.csv`.
Spot-checked GPT-5.4-mini: image 63.8 / 48.7 / 23.7, text+image 67.5 / 37.3 / 44.7,
combined 63.1 / 47.3 / 25.0, combined+text 62.6 / 30.7 / 51.0 **[verified]**.
Consolidated in `figures_and_tables/tab02_protocol_checks/E7_closed_model_pilot.csv`.
Bootstrap intervals: `fig05_closed_model_bootstrap.csv`, produced by
`code/06_figures_tables/task9_figure5_closed_bootstrap.py`.

**(b) Open 24B–32B models, 300 pairs/language.** Source:
`results/protocol_checks/large_open_models/multilang_300_20260910/aggregate_metrics_multilang.csv`
and `pair_level_multilang_results.csv` (2,700 rows, verdicts only — logit hooks
were disabled; see `EXCLUSIONS.md`).

GPT-5.4 reasoning-budget check (low vs high, 4,096 tokens):
`results/protocol_checks/reasoning_budget/`.

---

## Section 5.2 — RQ2 (Figures 8, 9)

| Claim | Value | Artifact |
| --- | --- | --- |
| Gemma-3-12B parse-failed pairs | 0 / 21,000 | **[verified]** `results/rq2_reliability/reduced_grid_3models/gemma/` |
| InternVL3.5-8B parse-failed pairs | 204 / 21,000 (from 247 / 42,000 calls) | **[verified]** `results/rq2_reliability/reduced_grid_3models/internvl3_5/`. 182 of the 204 are in `gaussian_sigma_4` |
| "the **two** reduced-grid runs with retained parsing counts" | — | **[verified, discrepancy]** all **three** retained them; Gemma4-12B has 33 / 21,000 parse-failed pairs from 34 / 42,000 calls, unreported. See [discrepancy 5](#5-section-52-says-two-reduced-grid-runs-retained-parsing-counts-it-was-three) and `docs/PARSE_FAILURE_AUDIT.md` |
| Wrap 80 improves `E` by 1.7–5.4 pp, lowers `S` by 2.8–14.2 pp | ΔE +1.65…+5.35, ΔS −14.25…−2.80 | **[verified]** `results/rq2_reliability/full_grid_2models/A_GRID_RESULTS.csv`, averaged over 3 themes × 2 font sizes for each of the 6 model–language groups |
| 22 of 66 baseline-relative contrasts significant after Holm | — | `results/rq2_reliability/full_grid_2models/A_GRID_MCNEMAR.csv`; families in `results/diagnostics/review_defense_e1_e8/analysis/E8/E8_multiplicity_families.csv` |
| Figure 8(a) single-condition Monokai contrasts | — | `figures_and_tables/fig08_rendering_perturbation/*_fig6_input.csv` (5 judges) |
| Removing indentation lowers `E` by 1.8–2.8 pp in four above-chance judges | Gemma4 −2.80, Gemma-3 −2.47, InternVL3.5 −2.23, Qwen2.5-VL −1.83 | **[verified]** Pooled across the three languages per judge: reduced-grid judges from `figures_and_tables/fig08_rendering_perturbation/*_matched_contrasts.csv` (`condition = no_indent`), Qwen2.5-VL-7B from `results/rq2_reliability/full_grid_2models/A_B_PAIR_LEVEL.csv`. InternVL3-8B is excluded as below-chance in RQ1; it *gains* 3.07 pp |
| Figure 8(b), valid-only changes, incl. the 89-pair Qwen Java cell | Qwen +1.6/−0.03/−1.1, InternVL3 −2.6/−2.9/+3.1 | **[verified: all 6 cells]** `results/rq2_reliability/full_grid_2models/B_DORN_PRIMARY_RESULTS.csv`, `valid_accuracy` column, `baseline` vs `no_indent`. **Java uses the Dorn-only 89-pair subset**, per the protocol's "language interactions use the Dorn-only three-language subset"; Python and CUDA are 1,000 pairs each. Pooled Java gives visibly different values — see the note in `figures_and_tables/README.md` |
| Figure 8(c): Qwen Java `E` 33.1 → 19.5; InternVL3 CUDA `S` 69.4 → 36.2 | — | **[verified]** `results/rq2_reliability/full_grid_2models/A_B_PAIR_LEVEL.csv`, `baseline` vs `gaussian_sigma_4` |
| Combining Java snippets raises both open models' swap error | Qwen 37.77→51.40%, InternVL3 54.80→60.07% | `results/rq2_reliability/packaging_and_modality/packaging_modality_summary_8conditions.csv` **[verified]**; InternVL re-run check in `results/diagnostics/review_defense_e1_e8/analysis/E5/E5B_INTERNVL_PACKAGING_CELL.csv` |
| GPT-5.4 high reasoning, adding source text: `S` 11.33% → 29.00% | — | `figures_and_tables/tab02_protocol_checks/E7_gpt54_text_swap_check.csv` **[verified]** |
| **Order vs rendering, all 15 groups** | `order_exceeds_rendering = True` in every group | `results/rq2_reliability/verified_recomputation/rq2_order_vs_rendering.csv` **[verified]** |
| Full-grid median rendering flips | 9.8–16.9% | same **[verified]** |
| Full-grid median within-condition swap error | 38.6–69.3% | same **[verified]** |
| Full-grid flips restricted to valid-both | 0.6–7.1% | same **[verified]** |
| Reduced-grid median swap error | 12.5–37.7% | same **[verified]** |
| Reduced-grid median input-condition flips | 10.0–19.3% | same **[verified]** |
| Reduced-grid valid-both flips | 3.0–8.7% | same **[verified]** |
| Smallest gap: Gemma-3-12B CUDA | 0.7 pp | same (18.95 − 18.25) **[verified]** |
| Figure 9, ten ordered \|c\| groups, 5 diagnostic judges | — | `figures_and_tables/fig09_logit_margin/fig09_data_5_diagnostic_judges.csv` **[verified]** |
| Held-out AUC for averaged decisions | 0.52–0.63 | `figures_and_tables/fig09_logit_margin/isotonic_auc_and_trend_summary.csv`, `isotonic_auroc` column: 0.5206–0.6292 **[verified]** |
| Ties produce 758–2,177 observations per group | — | `fig09_data_5_diagnostic_judges.csv` **[verified: max 2,177 at phi decile 1]** |
| Isotonic calibration, language-stratified snippet-disjoint 50/50 split | — | `figures_and_tables/fig09_logit_margin/B_snippet_disjoint_split.csv` (`calibration_snippet_overlap = 0`) |
| Boundary rule: `S` 45.78% → 52.56% (6.78 pp) | — | `code/06_figures_tables/task1_boundary_sensitivity.py`, `code/99_verification/scratch_qwen_boundary.py` |
| Boundary cases = 12.4% of diagnostic observations | 5,577 / 45,000 | `results/diagnostics/vlm_claims.csv` (CLM-001) **[verified]** |
| Encoder top-one retrieval: InternVL3 72.2–89.5%, Qwen 92.2–98.8%, chance below 1% | — | `results/rq2_reliability/encoder_retrieval/d_embedding_stability.csv`, `same_snippet_top1` and `retrieval_chance` columns **[verified: 72.2–89.5 / 92.2–98.8; chance 0.319–0.840%]**. Per-condition breakdown in `d_retrieval_by_condition.csv`; perturbation-condition distances in `d_perturbation_distance.csv` **[see discrepancy 4]** |

---

## Section 5.3 — RQ3 (Figure 10)

All 7,200 calls parsed. Source for every cell of Figure 10:
`results/rq3_content_vs_appearance/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv`
**[verified: all 12 cells match]**.

Variant-name mapping between the manuscript and the data files:

| Manuscript | Data file `variant_*` value |
| --- | --- |
| `Beautiful_gold` | `golden` |
| `Beautiful_trash` | `beautiful_trash` |
| `Ugly_gold` | `ugly_gold` |
| `Ugly_trash` | `ugly_trash` |

| Contrast | Qwen2.5-VL-7B (pref / S) | InternVL3-8B (pref / S) |
| --- | --- | --- |
| Beautiful_gold vs Beautiful_trash | 98.3 / 1.7 | 91.7 / 8.3 |
| Beautiful_gold vs Ugly_trash | 97.7 / 2.3 | 99.0 / 1.0 |
| Ugly_gold vs Ugly_trash | 94.0 / 6.0 | 97.7 / 2.0 |
| **Ugly_gold vs Beautiful_trash** (key conflict) | **91.7 / 8.3** | **82.0 / 17.3** |
| Beautiful_gold vs Ugly_gold | 56.0 / 43.3 | 57.3 / 39.7 |
| Beautiful_trash vs Ugly_trash | 8.3 / 91.7 | 60.3 / 36.7 |

Per-language conflict preference (66–91% InternVL3, 89–94% Qwen):
`RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv`. Conditional preference 99.2–100%:
`target_preference_valid` column. The 25-of-300 Qwen appearance-only cell:
same file, `visual_only_trash` row.

---

## Section 5.4 — RQ4 (Table 3)

### Table 3(a) — frozen Java RF, OCR feature substitution **[verified]**

`figures_and_tables/tab03_rq4/table3a_frozen_rf_ocr_substitution.csv`
(rows `Oracle source random_forest_regressor`, `OCR(rapidocr)+random_forest_regressor`,
`OCR(easyocr)+…`, `OCR(easyocr_preprocessed)+…`):
62.33 / 51.87 / 48.23 / 51.73 → changes −10.46 / −14.10 / −10.60 pp.

### Table 3(b) — image-only pipelines **[verified]**

| Pipeline | Java | Python | CUDA | Artifact |
| --- | --- | --- | --- | --- |
| Direct Qwen2.5-VL-7B (`D`) | 57.17 | 63.07 | 65.57 | `tab03_rq4/Table5_deploy_v2.csv` |
| Best OCR + ML | 52.97 (RapidOCR+SVR) | 59.53 (RapidOCR+GBM) | 61.37 (EasyOCR+LR) | Java: `table3a_…csv`; Python/CUDA: `results/rq4_image_only/ocr_substitution_frozen_rf/python_cuda/ocr_classical_pair_predictions.csv` |
| RapidOCR + Qwen2.5-Coder (`D`) | 50.10 | 55.60 | 54.33 | `tab03_rq4/Table5_deploy_v2.csv` |
| Direct − best OCR+ML | +4.20 | +3.53 | +4.20 | `tab03_rq4/F5_deploy_debiased.csv`, `recomputed_intervals.csv` |

Paired 95% CIs [−2.3, +10.6], [−5.9, +12.9], [−2.6, +11.0]:
the Java and Python intervals are the `ci_lo`/`ci_hi` columns of
`F5_deploy_debiased.csv`. **The CUDA interval is not in that file** — it compares
against RapidOCR+LR (60.73), not Table 3(b)'s EasyOCR+LR (61.37). The CUDA
interval was computed separately by joining pair-level `is_correct` from the
Python/CUDA OCR predictions (3,000 matching `pair_id`s, no new inference); see
`tab03_rq4/recomputed_intervals.csv` and
`docs/interval_provenance_20260920_ko.md`.

### Table 3(c) — Qwen2.5-VL-7B input ablation **[verified]**

`figures_and_tables/tab03_rq4/table3c_input_ablation_summary.csv`
(row `scope = overall`).

| Input | E | A_valid | D | S |
| --- | --- | --- | --- | --- |
| Source text only | 44.78 | 58.04 | 56.30 | 22.86 |
| OCR text only | 27.96 | 54.76 | 52.58 | 48.94 |
| Images only | 41.30 | 68.35 | 61.93 | 39.58 |

**Note on `D`.** The file's `debiased_accuracy` column divides by
`n_D_defined` (ties excluded): 0.5734 and 0.5427. The manuscript counts ties as
errors, i.e. divides by 9,000: 0.5734 × 8,836 / 9,000 = 56.30% and
0.5427 × 8,720 / 9,000 = 52.58%. The image-only `D` of 61.93% is the mean of the
three per-language `D_main` values in `Table5_deploy_v2.csv`. Both conventions
are reproduced by `code/99_verification/scratch_check_table3.py`.

### Other Section 5.4 claims

| Claim | Artifact |
| --- | --- |
| Strict-rejection `E` for RapidOCR + Qwen2.5-Coder: 28.47 / 9.87 / 21.57 | `tab03_rq4/Table5_deploy_v2.csv` |
| Python `A_valid` 57.42→62.71, `S` 54.60→84.27, `E` 26.07→9.87 | same |
| Preprocessed EasyOCR lowers `E` in every language | `results/rq4_image_only/ocr_text_llm_judge/` |
| RapidOCR leads token F1; EasyOCR leads CUDA pipeline accuracy | `results/rq4_image_only/ocr_transcription_quality/ocr_quality_summary.csv` |
| CUDA reverses source–image ordering (E 48.27 vs 49.97) | `table3c_input_ablation_summary.csv`, `scope = language` rows |
| Neither text condition has parsing failures | same, `parse_failure_rate = 0.0` |

---

## Section 8 — Threats to validity

| Claim | Artifact |
| --- | --- |
| 90,000 primary RQ1 calls | 5 primary judges × 18,000; `results/rq1_viability/raw_inference/*/raw.jsonl` **[verified: 18,000 lines each for all 8 judges]** |
| Boundary rule shifts Qwen `S` by 6.78 pp | `code/06_figures_tables/task1_boundary_sensitivity.py` |
| RQ1 vs deployment Qwen Python `D`: 66.77% vs 63.07%, 3.70 pp cross-configuration | `E1_debiased_vs_baseline.csv` vs `Table5_deploy_v2.csv`; run settings differ per `results/rq4_image_only/image_only_pipelines/F5_ANALYSIS_AUDIT.json` |
| Same-environment and changed-environment repeat runs | `results/diagnostics/review_defense_e1_e8/analysis/E4/`, `runs/repeat_same_env/`, `runs/repeat_changed_env/` |
| Cross-run determinism check | `results/diagnostics/followup_f1_f4/analysis/F3/F3_nondeterminism_report.md` |

---

## Known discrepancies

Recomputation during package assembly found one substantive mismatch and two
cosmetic ones. The artifact values are correct; the manuscript should be
corrected before camera-ready.

### 1. One confidence interval uses the superseded baseline value

Not a manuscript error. The manuscript's Python baseline of 64.67% is correct
(see [A trap for replicators](#a-trap-for-replicators)), but the interval it
quotes for "Qwen reaches 66.77% on Python … 2.1 above the best source baseline",
[−6.18, +10.23], was computed by `analyze_e1.py` against the superseded 64.60%
(`E1_debiased_vs_baseline.csv`, `language_best_accuracy = 0.646`).

The effect is 0.07 pp on one endpoint of an interval that spans 16 pp and
contains zero either way, so no reported conclusion changes. Re-running
`analyze_e1.py` against `canonical_pairwise_summary.csv` would make the interval
exactly consistent with the printed point estimate.

### 2. Figure 6, three valid-accuracy cells off by 0.1 pp

Qwen3-VL-8B CUDA (62.75% → manuscript 62.8), Qwen2.5-VL-7B Java (62.35% →
62.4), Qwen2.5-VL-7B CUDA (74.95% → 75.0). These are boundary half-rounding
differences, documented with exact unrounded ratios in
`figures_and_tables/FIGURE_NUMBER_HISTORY.md`.

### 3. Section 5.4, "+3.53" vs "+3.54"

Table 3(b)'s Python row shows +3.53 pp. Subtracting the rounded table entries
(63.07 − 59.53) gives 3.54. The unrounded difference in `F5_deploy_debiased.csv`
is 0.03533, so +3.53 is correct.

### 4. Encoder retrieval: "across 17 conditions" is imprecise

Sections 4.4.2 and 5.2 both say top-one retrieval was measured "across 17
rendering and perturbation conditions". The embeddings do cover all 17
(`D_INTEGRITY.json`: 9,384 rows = 552 snippets × 17 conditions per model), but
the retrieval statistic that yields the quoted 72.2–89.5% and 92.2–98.8% ranges
is computed over the **12-condition rendering grid** only —
`d_embedding_stability.csv` and `d_retrieval_by_condition.csv` both record
`n_conditions = 12`.

The five perturbation conditions are analysed by cosine distance to the
baseline embedding (`d_perturbation_distance.csv`, 75 rows), not by retrieval.
The numbers are correct; the scope sentence should say that retrieval uses the
12-condition grid while the embedding extraction spans all 17.

### 5. Section 5.2 says "two" reduced-grid runs retained parsing counts; it was three

Section 5.2 opens: "Parse failures are uncommon in the two reduced-grid runs
with retained parsing counts." All three retained them. Gemma4-12B's pair-level
file in fact carries the most detailed per-call parse columns of the three
(`parsed_choice_ab`, `parsed_choice_ba`, `argmax_matches_parsed_*`), and it has
**33 / 21,000 parse-failed pairs (0.16%) from 34 / 42,000 calls (0.08%)**,
spread over all seven conditions.

The reported Gemma-3-12B and InternVL3.5-8B counts are exact. Adding Gemma4-12B
does not change the paper's point — parse failures stay uncommon — but the
sentence should say three, and the count should appear.

### 6. Table 2(a): GPT-5.5's strict-swap error is largely parse failure

GPT-5.5 is the only model in Table 2 with nonzero parse failures, and the table
gives no hint of it:

| Condition | `S` | parse failure | disagreement | PF share of `S` |
| --- | ---: | ---: | ---: | ---: |
| combined image | 23.0% | 16.0 pp | 7.0 pp | **70%** |
| image | 18.7% | 8.7 pp | 10.0 pp | **46%** |
| combined+text | 20.7% | 6.0 pp | 14.7 pp | 29% |
| text+image | 25.3% | 6.7 pp | 18.7 pp | 26% |

GPT-5.4-mini and GPT-5.4 have zero parse failures in all four conditions, so a
reader comparing the three models' `S` columns is not comparing like with like.
Section 5.1 does say that `S` includes parsing failures, but a footnote on the
GPT-5.5 rows would prevent the misreading. Full audit in
`docs/PARSE_FAILURE_AUDIT.md`.

---

## A trap for replicators

`data/features_and_baseline_predictions/ml_predictions_9models_9000.csv` is the
convenient file — nine predictors, three languages, 27,000 rows, one place. It
is also **wrong for three of the 9,000 pairs**, and the first author of this
review pass was caught by it.

A clean-pair rebuild reused three `pair_id`s (one CUDA, two Python) after their
endpoint snippets had changed:

| pair_id | language | frozen endpoints | endpoints the stale rows were scored against |
| --- | --- | --- | --- |
| `ext_dorn_cuda_hard_02718` | cuda | `dorn_cuda_094`, `dorn_cuda_105` | `dorn_cuda_099`, `dorn_cuda_059` |
| `ext_dorn_python_medium_07974` | python | `dorn_python_045`, `dorn_python_115` | `dorn_python_029`, `dorn_python_054` |
| `ext_dorn_python_hard_08056` | python | `dorn_python_057`, `dorn_python_065` | `dorn_python_108`, `dorn_python_063` |

`ml_predictions_9models_9000.csv` inherits those three rows from its upstream
source (`results/dataset_extension_20260713/data/pair_predictions.csv`) and
therefore under-counts by up to 2 correct pairs per language.

**What is affected.** Only the source-feature predictor accuracies, and only in
the third decimal place. The one value that reaches the manuscript is the Python
best baseline: 1938/3000 = 64.60% stale, **1940/3000 = 64.67% corrected**. Best-
predictor selection is unchanged in all three languages, since the nearest
runner-up is 0.6 pp or more behind.

**What is not affected.** Every VLM result in this package.
`data/pairs/rq1_pairs_9000_seed42.csv` carries the **frozen** endpoints, so all
judges saw the correct images, and every pair-level outcome file, raw JSONL,
bootstrap interval and figure derived from them is correct.

**Use instead:**
`data/features_and_baseline_predictions/java/table1_ml_replication/canonical_pairwise_summary.csv`,
which re-scores the frozen pairs from the stored out-of-fold snippet
predictions. `legacy_reported_values_audit.csv` in the same directory gives the
stale value, the corrected value and the delta for all nine cells, and
`pair_id_reuse_audit.csv` lists the three pairs. `code/99_verification/recompute_headline_numbers.py`
prints both columns side by side.

We kept the 9-model file because it is the only one covering all nine
predictors, which the paper's "nine supervised models" sentence refers to. It
should not be used to recompute a reported baseline accuracy.

---

## Items the manuscript describes that are outside this package

| Item | Status |
| --- | --- |
| Figure 1 panel illustrations | Schematic; drawn from cited prior work, no underlying data |
| Figures 2, 4, 5 | Design diagrams; the conditions they depict are in `data/render_metadata/render_conditions.csv` and `data/rq3_variants/` |
| Figure rendering source (TikZ/PGF) | Held with the manuscript source, not with the experiments. Every figure's numeric input is in `figures_and_tables/` |
