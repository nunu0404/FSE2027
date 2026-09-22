# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.8040 |                 0.6396 | 1130 |          0.2894 |
| logistic_regression  |   0.8025 |                 0.7251 | 1130 |          0.2894 |
| random_forest        |   0.7663 |                 0.6874 | 1130 |          0.2894 |
| decision_tree_depth3 |   0.6479 |                 0.6114 | 1130 |          0.2894 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.3624  |          0.3624  |
| decision_tree_depth3 | num__visual_non_background_pixel_density_max              |      0.2378  |          0.2378  |
| decision_tree_depth3 | num__source_brace_count_max                               |      0.1608  |          0.1608  |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0.09468 |          0.09468 |
| decision_tree_depth3 | num__source_halstead_distinct_operands_mean               |      0.06221 |          0.06221 |
| decision_tree_depth3 | num__source_operator_count_gold_minus_loser               |      0.05381 |          0.05381 |
| decision_tree_depth3 | num__source_avg_identifier_length_mean                    |      0.02832 |          0.02832 |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |      0       |          0       |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.1072  |          0.1072  |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.06195 |          0.06195 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.05054 |          0.05054 |
| gradient_boosting    | num__source_blank_line_ratio_gold_minus_loser             |      0.03282 |          0.03282 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_max               |      0.02373 |          0.02373 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.02316 |          0.02316 |
| gradient_boosting    | num__source_avg_token_length_mean                         |      0.023   |          0.023   |
| gradient_boosting    | num__abs_human_score_gap                                  |      0.02152 |          0.02152 |
| gradient_boosting    | num__source_max_line_length_gold_minus_loser              |      0.0204  |          0.0204  |
| gradient_boosting    | num__source_avg_identifier_length_mean                    |      0.01675 |          0.01675 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.0163  |          0.0163  |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.01613 |          0.01613 |
| gradient_boosting    | num__source_token_entropy_diff_abs                        |      0.01581 |          0.01581 |
| gradient_boosting    | num__source_indent_max_mean                               |      0.01488 |          0.01488 |
| gradient_boosting    | num__source_avg_line_length_max                           |      0.01472 |          0.01472 |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.863   |          1.863   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.752   |          1.752   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.209   |          1.209   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.164   |          1.164   |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -1.151   |          1.151   |
| logistic_regression  | num__visual_image_height_gold_minus_loser                 |     -1.039   |          1.039   |
| logistic_regression  | num__visual_code_bounding_box_height_gold_minus_loser     |     -1.039   |          1.039   |
| logistic_regression  | num__visual_estimated_line_height_gold_minus_loser        |     -1.039   |          1.039   |
| logistic_regression  | num__rf_margin_abs                                        |      0.9376  |          0.9376  |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      0.9156  |          0.9156  |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -0.9135  |          0.9135  |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.909   |          0.909   |
| logistic_regression  | num__rf_margin_percentile                                 |     -0.8373  |          0.8373  |
| logistic_regression  | num__source_for_count_mean                                |     -0.8364  |          0.8364  |
| logistic_regression  | num__source_api_call_count_mean                           |     -0.7196  |          0.7196  |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.05496 |          0.05496 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.03024 |          0.03024 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.02444 |          0.02444 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.02356 |          0.02356 |
| random_forest        | num__source_indent_max_gold_minus_loser                   |      0.02212 |          0.02212 |
| random_forest        | num__source_blank_line_ratio_gold_minus_loser             |      0.02121 |          0.02121 |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |      0.01583 |          0.01583 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.01471 |          0.01471 |
| random_forest        | num__source_avg_identifier_length_mean                    |      0.01426 |          0.01426 |
| random_forest        | num__visual_non_background_pixel_density_mean             |      0.01371 |          0.01371 |
| random_forest        | num__source_avg_token_length_mean                         |      0.01335 |          0.01335 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.01308 |          0.01308 |
| random_forest        | num__visual_image_width_gold_minus_loser                  |      0.01251 |          0.01251 |
| random_forest        | num__source_halstead_vocabulary_mean                      |      0.01236 |          0.01236 |
| random_forest        | num__source_max_line_length_max                           |      0.01217 |          0.01217 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_indent_mean_gold_minus_loser <= -3.16
|   |--- num__source_halstead_distinct_operands_mean <= 31.75
|   |   |--- num__source_avg_identifier_length_mean <= 11.46
|   |   |   |--- class: 1
|   |   |--- num__source_avg_identifier_length_mean >  11.46
|   |   |   |--- class: 1
|   |--- num__source_halstead_distinct_operands_mean >  31.75
|   |   |--- num__source_operator_count_gold_minus_loser <= 57.00
|   |   |   |--- class: 1
|   |   |--- num__source_operator_count_gold_minus_loser >  57.00
|   |   |   |--- class: 0
|--- num__source_indent_mean_gold_minus_loser >  -3.16
|   |--- num__visual_non_background_pixel_density_max <= 1.00
|   |   |--- num__source_brace_count_max <= 7.50
|   |   |   |--- class: 0
|   |   |--- num__source_brace_count_max >  7.50
|   |   |   |--- class: 0
|   |--- num__visual_non_background_pixel_density_max >  1.00
|   |   |--- num__abs_human_score_gap <= 0.51
|   |   |   |--- class: 0
|   |   |--- num__abs_human_score_gap >  0.51
|   |   |   |--- class: 1

```
