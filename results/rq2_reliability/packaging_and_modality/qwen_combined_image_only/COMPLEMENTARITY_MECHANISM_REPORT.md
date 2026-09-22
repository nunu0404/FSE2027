# Complementarity Mechanism Report

## 1. Purpose
Explain when RF fails and when the best VLM condition rescues RF errors in the RQ0 pairwise readability task.

## 2. Relationship to existing RQ0 result
The analysis rebuilds the 3,000-pair RF/VLM mechanism dataset from clean inputs. The VLM condition is `Qwen/Qwen2.5-VL-7B-Instruct / combined_labeled_image_only / promptB`. This is explanatory, not a new performance claim.

## 3. Data and outcome groups
| outcome_group           |   count |
|:------------------------|--------:|
| RF_correct__VLM_correct |     608 |
| RF_correct__VLM_wrong   |     295 |
| RF_correct__VLM_invalid |     967 |
| RF_wrong__VLM_correct   |     293 |
| RF_wrong__VLM_wrong     |     262 |
| RF_wrong__VLM_invalid   |     575 |

## 4. Feature extraction
Source features reuse the RQ0 handcrafted feature extractor and add simple counts for braces, calls, control flow, and exception handling. Visual features are extracted from rendered PNGs by thresholding non-background pixels and measuring density, bounding boxes, row/column structure, margins, contrast, and color diversity. No OCR is used for visual features.

## 5. Descriptive group analysis
See `group_descriptive_summary.md` and `group_descriptive_stats.csv`.

## 6. RF failure characteristics
RF failure is assessed by comparing RF-correct and RF-wrong pairs using nonparametric tests and explanatory models. See `feature_stat_tests_summary.md` and `rf_failure_model_results.md`.

## 7. VLM rescue characteristics
VLM rescue group size is 293 out of 3000 pairs, and 293 out of 1130 RF-wrong pairs. Bottom-30% RF-margin share among rescue cases: 44.37%.

Top rescue-vs-nonrescue feature differences among RF-wrong cases:
| feature                                              |   mean_diff_a_minus_b |   cliffs_delta |     fdr_p |
|:-----------------------------------------------------|----------------------:|---------------:|----------:|
| visual_horizontal_whitespace_ratio_mean              |            -0.001629  |        -0.2847 | 5.183e-11 |
| visual_non_background_pixel_density_mean             |             0.001629  |         0.2847 | 5.183e-11 |
| visual_horizontal_whitespace_ratio_max               |            -0.002409  |        -0.2622 | 2.05e-09  |
| source_comment_line_ratio_gold_minus_loser           |             0.1058    |         0.2576 | 2.05e-09  |
| source_comment_line_ratio_mean                       |            -0.04901   |        -0.2464 | 1.113e-08 |
| source_indent_mean_gold_minus_loser                  |            -2.171     |        -0.2437 | 2.322e-08 |
| visual_non_background_pixel_density_max              |             0.0008485 |         0.2419 | 2.547e-08 |
| source_comment_line_ratio_max                        |            -0.07906   |        -0.2386 | 2.547e-08 |
| source_comment_line_count_gold_minus_loser           |             2.167     |         0.2246 | 1.805e-07 |
| source_comment_line_count_mean                       |            -1.284     |        -0.2169 | 5.763e-07 |
| source_comment_line_ratio_diff_abs                   |            -0.06011   |        -0.2073 | 2.18e-06  |
| visual_non_background_pixel_density_gold_minus_loser |            -0.001966  |        -0.2068 | 2.716e-06 |

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

