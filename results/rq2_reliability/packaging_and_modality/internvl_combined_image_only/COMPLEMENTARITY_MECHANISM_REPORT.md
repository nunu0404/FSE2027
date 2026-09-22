# Complementarity Mechanism Report

## 1. Purpose
Explain when RF fails and when the best VLM condition rescues RF errors in the RQ0 pairwise readability task.

## 2. Relationship to existing RQ0 result
The analysis rebuilds the 3,000-pair RF/VLM mechanism dataset from clean inputs. The VLM condition is `OpenGVLab/InternVL3-8B / combined_labeled_image_only / promptB`. This is explanatory, not a new performance claim.

## 3. Data and outcome groups
| outcome_group           |   count |
|:------------------------|--------:|
| RF_correct__VLM_correct |     414 |
| RF_correct__VLM_wrong   |     321 |
| RF_correct__VLM_invalid |    1135 |
| RF_wrong__VLM_correct   |     248 |
| RF_wrong__VLM_wrong     |     215 |
| RF_wrong__VLM_invalid   |     667 |

## 4. Feature extraction
Source features reuse the RQ0 handcrafted feature extractor and add simple counts for braces, calls, control flow, and exception handling. Visual features are extracted from rendered PNGs by thresholding non-background pixels and measuring density, bounding boxes, row/column structure, margins, contrast, and color diversity. No OCR is used for visual features.

## 5. Descriptive group analysis
See `group_descriptive_summary.md` and `group_descriptive_stats.csv`.

## 6. RF failure characteristics
RF failure is assessed by comparing RF-correct and RF-wrong pairs using nonparametric tests and explanatory models. See `feature_stat_tests_summary.md` and `rf_failure_model_results.md`.

## 7. VLM rescue characteristics
VLM rescue group size is 248 out of 3000 pairs, and 248 out of 1130 RF-wrong pairs. Bottom-30% RF-margin share among rescue cases: 37.10%.

Top rescue-vs-nonrescue feature differences among RF-wrong cases:
| feature                                 |   mean_diff_a_minus_b |   cliffs_delta |   fdr_p |
|:----------------------------------------|----------------------:|---------------:|--------:|
| visual_non_background_pixel_density_max |             0.0003985 |         0.1518 | 0.06972 |
| visual_image_width_diff_abs             |            18.2       |         0.1318 | 0.08043 |
| visual_code_bounding_box_width_diff_abs |            18.2       |         0.1318 | 0.08043 |
| visual_line_length_mean_diff_abs        |            18.2       |         0.1318 | 0.08043 |
| visual_line_length_max_diff_abs         |            18.2       |         0.1318 | 0.08043 |
| source_token_entropy_mean               |            -0.08911   |        -0.1269 | 0.08918 |
| source_identifier_unique_count_mean     |            -2.496     |        -0.1266 | 0.08918 |
| source_nesting_mean_max                 |            -0.1491    |        -0.1221 | 0.1028  |
| source_nesting_mean_mean                |            -0.09861   |        -0.1203 | 0.1028  |
| source_halstead_vocabulary_mean         |            -2.796     |        -0.1197 | 0.1028  |
| source_identifier_count_mean            |            -4.939     |        -0.1177 | 0.1028  |
| source_token_count_mean                 |           -11.19      |        -0.1156 | 0.1028  |

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

