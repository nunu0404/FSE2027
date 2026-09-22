# Complementarity Mechanism Report

## 1. Purpose
Explain when RF fails and when the best VLM condition rescues RF errors in the RQ0 pairwise readability task.

## 2. Relationship to existing RQ0 result
The analysis rebuilds the 3,000-pair RF/VLM mechanism dataset from clean inputs. The VLM condition is `Qwen/Qwen2.5-VL-7B-Instruct / image_only / promptB`. This is explanatory, not a new performance claim.

## 3. Data and outcome groups
| outcome_group           |   count |
|:------------------------|--------:|
| RF_correct__VLM_correct |     833 |
| RF_correct__VLM_wrong   |     337 |
| RF_correct__VLM_invalid |     700 |
| RF_wrong__VLM_correct   |     327 |
| RF_wrong__VLM_wrong     |     370 |
| RF_wrong__VLM_invalid   |     433 |

## 4. Feature extraction
Source features reuse the RQ0 handcrafted feature extractor and add simple counts for braces, calls, control flow, and exception handling. Visual features are extracted from rendered PNGs by thresholding non-background pixels and measuring density, bounding boxes, row/column structure, margins, contrast, and color diversity. No OCR is used for visual features.

## 5. Descriptive group analysis
See `group_descriptive_summary.md` and `group_descriptive_stats.csv`.

## 6. RF failure characteristics
RF failure is assessed by comparing RF-correct and RF-wrong pairs using nonparametric tests and explanatory models. See `feature_stat_tests_summary.md` and `rf_failure_model_results.md`.

## 7. VLM rescue characteristics
VLM rescue group size is 327 out of 3000 pairs, and 327 out of 1130 RF-wrong pairs. Bottom-30% RF-margin share among rescue cases: 45.57%.

Top rescue-vs-nonrescue feature differences among RF-wrong cases:
| feature                                     |   mean_diff_a_minus_b |   cliffs_delta |     fdr_p |
|:--------------------------------------------|----------------------:|---------------:|----------:|
| source_indent_mean_gold_minus_loser         |            -3.016     |        -0.3266 | 1.818e-15 |
| source_indent_max_gold_minus_loser          |            -5.286     |        -0.2715 | 8.136e-11 |
| source_cyclomatic_estimate_gold_minus_loser |            -1.216     |        -0.2427 | 9.316e-09 |
| source_comment_line_ratio_gold_minus_loser  |             0.108     |         0.2376 | 1.535e-08 |
| visual_non_background_pixel_density_max     |             0.0007431 |         0.2346 | 3.228e-08 |
| source_halstead_distinct_operands_mean      |            -5.017     |        -0.2286 | 7.3e-08   |
| source_halstead_length_mean                 |           -22.27      |        -0.2258 | 8.54e-08  |
| source_halstead_volume_mean                 |          -139.9       |        -0.2256 | 8.54e-08  |
| source_token_count_mean                     |           -23.86      |        -0.2251 | 8.54e-08  |
| source_blank_line_ratio_gold_minus_loser    |             0.07117   |         0.2215 | 1.239e-07 |
| source_operator_count_mean                  |           -12.5       |        -0.2208 | 1.271e-07 |
| source_identifier_count_mean                |            -9.028     |        -0.2201 | 1.304e-07 |

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

