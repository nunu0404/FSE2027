# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| logistic_regression  |   0.8062 |                 0.7336 | 1130 |          0.2593 |
| gradient_boosting    |   0.7823 |                 0.6227 | 1130 |          0.2593 |
| random_forest        |   0.7625 |                 0.6854 | 1130 |          0.2593 |
| decision_tree_depth3 |   0.6752 |                 0.6531 | 1130 |          0.2593 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_ratio_gold_minus_loser           |      0.4201  |          0.4201  |
| decision_tree_depth3 | num__source_identifier_count_gold_minus_loser             |      0.2162  |          0.2162  |
| decision_tree_depth3 | num__source_token_entropy_max                             |      0.1461  |          0.1461  |
| decision_tree_depth3 | num__source_avg_token_length_mean                         |      0.1031  |          0.1031  |
| decision_tree_depth3 | num__source_indent_mean_max                               |      0.06745 |          0.06745 |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.04712 |          0.04712 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |      0       |          0       |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.07902 |          0.07902 |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.05269 |          0.05269 |
| gradient_boosting    | num__source_indent_std_mean                               |      0.03239 |          0.03239 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.03156 |          0.03156 |
| gradient_boosting    | num__source_nesting_mean_gold_minus_loser                 |      0.02968 |          0.02968 |
| gradient_boosting    | num__source_avg_token_length_mean                         |      0.02867 |          0.02867 |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.02479 |          0.02479 |
| gradient_boosting    | num__source_catch_count_gold_minus_loser                  |      0.0213  |          0.0213  |
| gradient_boosting    | num__source_loc_total_max                                 |      0.02035 |          0.02035 |
| gradient_boosting    | num__source_identifier_count_gold_minus_loser             |      0.01927 |          0.01927 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.01917 |          0.01917 |
| gradient_boosting    | num__source_halstead_distinct_operands_gold_minus_loser   |      0.01899 |          0.01899 |
| gradient_boosting    | num__source_indent_mean_max                               |      0.01753 |          0.01753 |
| gradient_boosting    | num__visual_non_background_pixel_density_gold_minus_loser |      0.01719 |          0.01719 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_mean              |      0.01647 |          0.01647 |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.473   |          1.473   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.392   |          1.392   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.338   |          1.338   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.286   |          1.286   |
| logistic_regression  | num__source_avg_token_length_gold_minus_loser             |      1.087   |          1.087   |
| logistic_regression  | num__visual_image_height_gold_minus_loser                 |     -1.046   |          1.046   |
| logistic_regression  | num__visual_code_bounding_box_height_gold_minus_loser     |     -1.046   |          1.046   |
| logistic_regression  | num__visual_estimated_line_height_gold_minus_loser        |     -1.046   |          1.046   |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -1.016   |          1.016   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.015   |          1.015   |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -0.9931  |          0.9931  |
| logistic_regression  | num__source_nesting_max_mean                              |      0.931   |          0.931   |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -0.8975  |          0.8975  |
| logistic_regression  | num__source_avg_token_length_mean                         |     -0.8365  |          0.8365  |
| logistic_regression  | num__source_nesting_max_diff_abs                          |     -0.8176  |          0.8176  |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.04376 |          0.04376 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.03997 |          0.03997 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.03446 |          0.03446 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.03207 |          0.03207 |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.02976 |          0.02976 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.02194 |          0.02194 |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |      0.01903 |          0.01903 |
| random_forest        | num__visual_horizontal_whitespace_ratio_max               |      0.01772 |          0.01772 |
| random_forest        | num__source_comment_line_ratio_mean                       |      0.01729 |          0.01729 |
| random_forest        | num__visual_non_background_pixel_density_mean             |      0.01536 |          0.01536 |
| random_forest        | num__source_halstead_volume_gold_minus_loser              |      0.01238 |          0.01238 |
| random_forest        | num__source_avg_token_length_mean                         |      0.01232 |          0.01232 |
| random_forest        | num__source_halstead_distinct_operands_gold_minus_loser   |      0.01146 |          0.01146 |
| random_forest        | num__visual_aspect_ratio_max                              |      0.01122 |          0.01122 |
| random_forest        | num__source_identifier_count_gold_minus_loser             |      0.01114 |          0.01114 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_ratio_gold_minus_loser <= -0.07
|   |--- num__source_identifier_count_gold_minus_loser <= -74.50
|   |   |--- class: 1
|   |--- num__source_identifier_count_gold_minus_loser >  -74.50
|   |   |--- num__source_indent_mean_gold_minus_loser <= -1.06
|   |   |   |--- class: 0
|   |   |--- num__source_indent_mean_gold_minus_loser >  -1.06
|   |   |   |--- class: 0
|--- num__source_comment_line_ratio_gold_minus_loser >  -0.07
|   |--- num__source_token_entropy_max <= 5.49
|   |   |--- num__source_avg_token_length_mean <= 3.64
|   |   |   |--- class: 1
|   |   |--- num__source_avg_token_length_mean >  3.64
|   |   |   |--- class: 1
|   |--- num__source_token_entropy_max >  5.49
|   |   |--- num__source_indent_mean_max <= 10.86
|   |   |   |--- class: 0
|   |   |--- num__source_indent_mean_max >  10.86
|   |   |   |--- class: 1

```
