# Rescue and Failure Taxonomy

This taxonomy is descriptive and correlational.

## Supported Categories
1. RF low-margin ambiguous cases: 32.75% of rescue cases fall in the bottom 30% of RF absolute-margin percentiles.
2. VLM strict-swap instability cases: invalid VLM groups remain large and are kept separate from valid wrong cases.
3. Visual/layout-difference cases: visual feature proxies show measurable but not necessarily causal differences in some rescue comparisons.
4. Both-system hard cases: RF_wrong__VLM_wrong and RF_wrong__VLM_invalid groups indicate cases not recovered by the evaluated VLM.

## Top visual features in rescue vs other RF-wrong cases
| feature                                      |   mean_diff_a_minus_b |   cliffs_delta |     fdr_p |
|:---------------------------------------------|----------------------:|---------------:|----------:|
| visual_aspect_ratio_diff_abs                 |              0.8435   |        0.3942  | 9.794e-25 |
| visual_image_width_diff_abs                  |             26.46     |        0.1344  | 0.0008083 |
| visual_code_bounding_box_width_diff_abs      |             26.46     |        0.1344  | 0.0008083 |
| visual_line_length_mean_diff_abs             |             26.46     |        0.1344  | 0.0008083 |
| visual_line_length_max_diff_abs              |             26.46     |        0.1344  | 0.0008083 |
| visual_non_background_pixel_density_diff_abs |             -0.001117 |       -0.0858  | 0.04028   |
| visual_horizontal_whitespace_ratio_diff_abs  |             -0.001117 |       -0.0858  | 0.04028   |
| visual_estimated_line_height_diff_abs        |             49.61     |        0.06236 | 0.1539    |

## Interpretation
The evidence supports a cautious taxonomy: VLM rescue is partly associated with RF uncertainty and some visual/layout proxies, but not enough to claim a stable causal mechanism.
