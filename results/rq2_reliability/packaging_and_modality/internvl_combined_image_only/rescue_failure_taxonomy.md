# Rescue and Failure Taxonomy

This taxonomy is descriptive and correlational.

## Supported Categories
1. RF low-margin ambiguous cases: 37.10% of rescue cases fall in the bottom 30% of RF absolute-margin percentiles.
2. VLM strict-swap instability cases: invalid VLM groups remain large and are kept separate from valid wrong cases.
3. Visual/layout-difference cases: visual feature proxies show measurable but not necessarily causal differences in some rescue comparisons.
4. Both-system hard cases: RF_wrong__VLM_wrong and RF_wrong__VLM_invalid groups indicate cases not recovered by the evaluated VLM.

## Top visual features in rescue vs other RF-wrong cases
| feature                                 |   mean_diff_a_minus_b |   cliffs_delta |   fdr_p |
|:----------------------------------------|----------------------:|---------------:|--------:|
| visual_image_width_diff_abs             |               18.2    |        0.1318  | 0.08043 |
| visual_code_bounding_box_width_diff_abs |               18.2    |        0.1318  | 0.08043 |
| visual_line_length_max_diff_abs         |               18.2    |        0.1318  | 0.08043 |
| visual_line_length_mean_diff_abs        |               18.2    |        0.1318  | 0.08043 |
| visual_contrast_score_diff_abs          |                0.5811 |        0.07382 | 0.3156  |
| visual_color_diversity_diff_abs         |               -0.5109 |       -0.04566 | 0.6407  |
| visual_clutter_score_diff_abs           |               -0.503  |       -0.04419 | 0.6564  |
| visual_image_height_diff_abs            |               -4.883  |       -0.02274 | 1       |

## Interpretation
The evidence supports a cautious taxonomy: VLM rescue is partly associated with RF uncertainty and some visual/layout proxies, but not enough to claim a stable causal mechanism.
