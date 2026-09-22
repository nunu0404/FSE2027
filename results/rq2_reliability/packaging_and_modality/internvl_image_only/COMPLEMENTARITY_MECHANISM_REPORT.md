# Complementarity Mechanism Report

## 1. Purpose
Explain when RF fails and when the best VLM condition rescues RF errors in the RQ0 pairwise readability task.

## 2. Relationship to existing RQ0 result
The analysis rebuilds the 3,000-pair RF/VLM mechanism dataset from clean inputs. The VLM condition is `OpenGVLab/InternVL3-8B / image_only / promptB`. This is explanatory, not a new performance claim.

## 3. Data and outcome groups
| outcome_group           |   count |
|:------------------------|--------:|
| RF_correct__VLM_correct |     407 |
| RF_correct__VLM_wrong   |     441 |
| RF_correct__VLM_invalid |    1022 |
| RF_wrong__VLM_correct   |     345 |
| RF_wrong__VLM_wrong     |     163 |
| RF_wrong__VLM_invalid   |     622 |

## 4. Feature extraction
Source features reuse the RQ0 handcrafted feature extractor and add simple counts for braces, calls, control flow, and exception handling. Visual features are extracted from rendered PNGs by thresholding non-background pixels and measuring density, bounding boxes, row/column structure, margins, contrast, and color diversity. No OCR is used for visual features.

## 5. Descriptive group analysis
See `group_descriptive_summary.md` and `group_descriptive_stats.csv`.

## 6. RF failure characteristics
RF failure is assessed by comparing RF-correct and RF-wrong pairs using nonparametric tests and explanatory models. See `feature_stat_tests_summary.md` and `rf_failure_model_results.md`.

## 7. VLM rescue characteristics
VLM rescue group size is 345 out of 3000 pairs, and 345 out of 1130 RF-wrong pairs. Bottom-30% RF-margin share among rescue cases: 32.75%.

Top rescue-vs-nonrescue feature differences among RF-wrong cases:
| feature                                          |   mean_diff_a_minus_b |   cliffs_delta |     fdr_p |
|:-------------------------------------------------|----------------------:|---------------:|----------:|
| visual_aspect_ratio_gold_minus_loser             |               -1.758  |        -0.5744 | 4.823e-51 |
| visual_aspect_ratio_max                          |                1.064  |         0.4527 | 9.598e-32 |
| source_loc_nonempty_gold_minus_loser             |               12.74   |         0.4382 | 6.588e-30 |
| visual_aspect_ratio_mean                         |                0.6422 |         0.4315 | 4.18e-29  |
| source_loc_total_gold_minus_loser                |               14.16   |         0.4282 | 9.115e-29 |
| source_method_like_count_gold_minus_loser        |                2.157  |         0.4221 | 2.337e-28 |
| source_brace_count_gold_minus_loser              |                5.757  |         0.4157 | 2.706e-27 |
| visual_code_bounding_box_height_gold_minus_loser |              403.5    |         0.41   | 1.155e-26 |
| visual_estimated_line_height_gold_minus_loser    |              403.5    |         0.41   | 1.155e-26 |
| visual_image_height_gold_minus_loser             |              403.5    |         0.41   | 1.155e-26 |
| visual_aspect_ratio_diff_abs                     |                0.8435 |         0.3942 | 9.794e-25 |
| source_nesting_max_gold_minus_loser              |                1.228  |         0.3931 | 2.695e-25 |

## 8. Explanatory models
Model reports are saved in `rf_failure_model_results.md`, `vlm_rescue_model_results.md`, and `vlm_rescue_valid_only_model_results.md`. These are sanity-check explanatory models and should not be read as deployment classifiers.

## 9. Rule/taxonomy analysis
See `rescue_failure_taxonomy.md`.

## 10. Case studies
See `case_studies.md` and `case_studies_viewer.html`.

## 11. Threats to validity
- `human_preference` is a score-derived proxy, not direct pairwise annotation.
- Many pairs are cross-dataset after within-dataset z-normalization.
- VLM correctness is evaluated under strict-swap protocol.
- Invalid VLM outputs are separated from wrong valid outputs.
- Feature analysis is correlational and does not prove causal mechanisms.
- Pair-level observations are not independent because snippets repeat across pairs.
- Visual features are proxies and may miss human-perceived layout qualities.
- Results are based on evaluated open-weight VLMs and should not generalize to all VLMs.

## 12. Implications for better judge design
The most defensible design implication is selective fallback: VLMs may be more useful in RF-uncertain regions, but strict-swap reliability must be audited and invalid outputs cannot be ignored.

## 13. Safe paper-ready wording
Most VLM rescue cases are concentrated in lower RF-margin regions, suggesting that VLM is most useful as an uncertainty-triggered fallback rather than a general replacement for source-based readability predictors. Some visual/layout proxy features differ across rescue and non-rescue cases, but the pattern is correlational and not strong enough to claim a causal mechanism. This supports a selective delegation strategy while preserving the main conclusion that current VLM judges remain fragile under strict-swap evaluation.

## 14. Reproduction commands
See `README.md` in this output directory for the exact command and clean input paths.

