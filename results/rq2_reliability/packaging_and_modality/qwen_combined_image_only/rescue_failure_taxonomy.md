# Rescue and Failure Taxonomy

This taxonomy is descriptive and correlational.

## Supported Categories
1. RF low-margin ambiguous cases: 44.37% of rescue cases fall in the bottom 30% of RF absolute-margin percentiles.
2. VLM strict-swap instability cases: invalid VLM groups remain large and are kept separate from valid wrong cases.
3. Visual/layout-difference cases: visual feature proxies show measurable but not necessarily causal differences in some rescue comparisons.
4. Both-system hard cases: RF_wrong__VLM_wrong and RF_wrong__VLM_invalid groups indicate cases not recovered by the evaluated VLM.

## Top visual features in rescue vs other RF-wrong cases
| feature                                      |   mean_diff_a_minus_b |   cliffs_delta |     fdr_p |
|:---------------------------------------------|----------------------:|---------------:|----------:|
| visual_non_background_pixel_density_diff_abs |             -0.001561 |        -0.1874 | 2.651e-05 |
| visual_horizontal_whitespace_ratio_diff_abs  |             -0.001561 |        -0.1874 | 2.651e-05 |
| visual_aspect_ratio_diff_abs                 |              0.3102   |         0.1456 | 0.001542  |
| visual_code_bounding_box_width_diff_abs      |             24.05     |         0.1228 | 0.008615  |
| visual_line_length_mean_diff_abs             |             24.05     |         0.1228 | 0.008615  |
| visual_image_width_diff_abs                  |             24.05     |         0.1228 | 0.008615  |
| visual_line_length_max_diff_abs              |             24.05     |         0.1228 | 0.008615  |
| visual_contrast_score_diff_abs               |              0.2541   |         0.0275 | 0.8111    |

## Interpretation
The evidence supports a cautious taxonomy: VLM rescue is partly associated with RF uncertainty and some visual/layout proxies, but not enough to claim a stable causal mechanism.
