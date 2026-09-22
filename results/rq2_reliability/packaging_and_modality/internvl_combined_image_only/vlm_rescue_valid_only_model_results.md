# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| logistic_regression  |   0.6722 |                 0.6348 | 463 |          0.5356 |
| random_forest        |   0.6602 |                 0.6048 | 463 |          0.5356 |
| gradient_boosting    |   0.6544 |                 0.5993 | 463 |          0.5356 |
| decision_tree_depth3 |   0.6261 |                 0.5804 | 463 |          0.5356 |

## Top Feature Importances

| model                | feature                                                  |   importance |   abs_importance |
|:---------------------|:---------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_identifier_unique_count_gold_minus_loser     |      0.3582  |          0.3582  |
| decision_tree_depth3 | num__source_token_entropy_gold_minus_loser               |      0.3271  |          0.3271  |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                 |      0.2612  |          0.2612  |
| decision_tree_depth3 | num__source_avg_identifier_length_diff_abs               |      0.05348 |          0.05348 |
| decision_tree_depth3 | num__abs_human_score_gap                                 |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                       |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                           |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                               |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                   |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                        |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                             |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_mean                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_gold_minus_loser                |      0       |          0       |
| gradient_boosting    | num__source_identifier_unique_count_gold_minus_loser     |      0.07148 |          0.07148 |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                 |      0.04754 |          0.04754 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser |      0.02592 |          0.02592 |
| gradient_boosting    | num__source_avg_token_length_mean                        |      0.02111 |          0.02111 |
| gradient_boosting    | num__source_token_entropy_gold_minus_loser               |      0.01952 |          0.01952 |
| gradient_boosting    | num__visual_image_width_gold_minus_loser                 |      0.01844 |          0.01844 |
| gradient_boosting    | num__source_avg_line_length_diff_abs                     |      0.01812 |          0.01812 |
| gradient_boosting    | num__source_halstead_length_max                          |      0.01801 |          0.01801 |
| gradient_boosting    | num__source_nesting_mean_mean                            |      0.01782 |          0.01782 |
| gradient_boosting    | num__source_blank_line_count_max                         |      0.01722 |          0.01722 |
| gradient_boosting    | num__source_token_count_gold_minus_loser                 |      0.01657 |          0.01657 |
| gradient_boosting    | num__source_avg_identifier_length_mean                   |      0.01503 |          0.01503 |
| gradient_boosting    | num__rf_margin_abs                                       |      0.01465 |          0.01465 |
| gradient_boosting    | num__source_line_length_std_diff_abs                     |      0.01434 |          0.01434 |
| gradient_boosting    | num__visual_non_background_pixel_density_max             |      0.01345 |          0.01345 |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                |      1.514   |          1.514   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                   |      1.332   |          1.332   |
| logistic_regression  | num__source_max_line_length_gold_minus_loser             |      1.145   |          1.145   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser           |      0.8665  |          0.8665  |
| logistic_regression  | num__source_if_count_gold_minus_loser                    |     -0.8378  |          0.8378  |
| logistic_regression  | num__source_avg_nonempty_line_length_mean                |     -0.8338  |          0.8338  |
| logistic_regression  | num__source_max_line_length_diff_abs                     |     -0.7803  |          0.7803  |
| logistic_regression  | num__source_blank_line_count_mean                        |      0.76    |          0.76    |
| logistic_regression  | num__source_indent_std_gold_minus_loser                  |      0.7596  |          0.7596  |
| logistic_regression  | num__source_brace_count_gold_minus_loser                 |     -0.7448  |          0.7448  |
| logistic_regression  | num__visual_image_height_gold_minus_loser                |     -0.7433  |          0.7433  |
| logistic_regression  | num__visual_code_bounding_box_height_gold_minus_loser    |     -0.7433  |          0.7433  |
| logistic_regression  | num__visual_estimated_line_height_gold_minus_loser       |     -0.7433  |          0.7433  |
| logistic_regression  | num__source_line_length_std_gold_minus_loser             |     -0.7404  |          0.7404  |
| logistic_regression  | num__source_token_entropy_diff_abs                       |     -0.7261  |          0.7261  |
| random_forest        | num__source_indent_mean_gold_minus_loser                 |      0.02006 |          0.02006 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser     |      0.01749 |          0.01749 |
| random_forest        | num__source_halstead_vocabulary_gold_minus_loser         |      0.01663 |          0.01663 |
| random_forest        | num__visual_image_width_gold_minus_loser                 |      0.01659 |          0.01659 |
| random_forest        | num__source_halstead_distinct_operands_gold_minus_loser  |      0.01635 |          0.01635 |
| random_forest        | num__source_halstead_volume_gold_minus_loser             |      0.01536 |          0.01536 |
| random_forest        | num__source_max_line_length_gold_minus_loser             |      0.01408 |          0.01408 |
| random_forest        | num__visual_line_length_max_gold_minus_loser             |      0.01394 |          0.01394 |
| random_forest        | num__source_halstead_length_max                          |      0.01391 |          0.01391 |
| random_forest        | num__source_identifier_count_gold_minus_loser            |      0.0139  |          0.0139  |
| random_forest        | num__source_halstead_volume_max                          |      0.01337 |          0.01337 |
| random_forest        | num__source_halstead_volume_mean                         |      0.01306 |          0.01306 |
| random_forest        | num__source_identifier_count_mean                        |      0.01236 |          0.01236 |
| random_forest        | num__source_token_entropy_gold_minus_loser               |      0.01226 |          0.01226 |
| random_forest        | num__visual_code_bounding_box_height_max                 |      0.01221 |          0.01221 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_identifier_unique_count_gold_minus_loser <= 51.50
|   |--- num__source_token_entropy_gold_minus_loser <= 1.00
|   |   |--- num__source_indent_mean_gold_minus_loser <= -4.38
|   |   |   |--- class: 1
|   |   |--- num__source_indent_mean_gold_minus_loser >  -4.38
|   |   |   |--- class: 0
|   |--- num__source_token_entropy_gold_minus_loser >  1.00
|   |   |--- num__source_avg_identifier_length_diff_abs <= 3.38
|   |   |   |--- class: 1
|   |   |--- num__source_avg_identifier_length_diff_abs >  3.38
|   |   |   |--- class: 1
|--- num__source_identifier_unique_count_gold_minus_loser >  51.50
|   |--- class: 0

```
