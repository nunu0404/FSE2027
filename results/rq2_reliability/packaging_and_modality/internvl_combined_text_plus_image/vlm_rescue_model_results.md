# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.8097 |                 0.6965 | 1130 |          0.3460 |
| logistic_regression  |   0.7915 |                 0.7147 | 1130 |          0.3460 |
| random_forest        |   0.7703 |                 0.7067 | 1130 |          0.3460 |
| decision_tree_depth3 |   0.6472 |                 0.6054 | 1130 |          0.3460 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.31    |          0.31    |
| decision_tree_depth3 | num__source_avg_token_length_gold_minus_loser             |      0.254   |          0.254   |
| decision_tree_depth3 | num__source_max_line_length_mean                          |      0.1407  |          0.1407  |
| decision_tree_depth3 | num__source_halstead_volume_max                           |      0.09518 |          0.09518 |
| decision_tree_depth3 | num__source_keyword_count_gold_minus_loser                |      0.08592 |          0.08592 |
| decision_tree_depth3 | num__source_avg_identifier_length_gold_minus_loser        |      0.07054 |          0.07054 |
| decision_tree_depth3 | num__source_literal_count_gold_minus_loser                |      0.04361 |          0.04361 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.06492 |          0.06492 |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.05035 |          0.05035 |
| gradient_boosting    | num__source_keyword_count_gold_minus_loser                |      0.03491 |          0.03491 |
| gradient_boosting    | num__source_avg_token_length_max                          |      0.03076 |          0.03076 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.02883 |          0.02883 |
| gradient_boosting    | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.02631 |          0.02631 |
| gradient_boosting    | num__abs_human_score_gap                                  |      0.02224 |          0.02224 |
| gradient_boosting    | num__source_halstead_vocabulary_max                       |      0.02221 |          0.02221 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.02205 |          0.02205 |
| gradient_boosting    | num__source_max_line_length_mean                          |      0.02128 |          0.02128 |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.02047 |          0.02047 |
| gradient_boosting    | num__source_avg_identifier_length_mean                    |      0.0187  |          0.0187  |
| gradient_boosting    | num__source_keyword_count_max                             |      0.01743 |          0.01743 |
| gradient_boosting    | num__visual_aspect_ratio_mean                             |      0.01699 |          0.01699 |
| gradient_boosting    | num__visual_contrast_score_max                            |      0.01699 |          0.01699 |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -2.241   |          2.241   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.914   |          1.914   |
| logistic_regression  | num__source_identifier_count_gold_minus_loser             |     -1.166   |          1.166   |
| logistic_regression  | num__source_halstead_length_diff_abs                      |      1.094   |          1.094   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.016   |          1.016   |
| logistic_regression  | num__source_operator_count_diff_abs                       |     -0.9038  |          0.9038  |
| logistic_regression  | num__source_indent_max_mean                               |     -0.8947  |          0.8947  |
| logistic_regression  | num__source_identifier_count_mean                         |     -0.8879  |          0.8879  |
| logistic_regression  | num__source_halstead_volume_diff_abs                      |     -0.8582  |          0.8582  |
| logistic_regression  | num__source_keyword_count_gold_minus_loser                |      0.7948  |          0.7948  |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -0.7526  |          0.7526  |
| logistic_regression  | num__rf_margin_abs                                        |      0.7448  |          0.7448  |
| logistic_regression  | num__source_operator_count_gold_minus_loser               |      0.7402  |          0.7402  |
| logistic_regression  | num__source_indent_std_gold_minus_loser                   |      0.7263  |          0.7263  |
| logistic_regression  | num__source_indent_std_mean                               |      0.7208  |          0.7208  |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.02956 |          0.02956 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.02569 |          0.02569 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02511 |          0.02511 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.02146 |          0.02146 |
| random_forest        | num__visual_aspect_ratio_mean                             |      0.02021 |          0.02021 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.01618 |          0.01618 |
| random_forest        | num__source_avg_token_length_max                          |      0.01569 |          0.01569 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.01548 |          0.01548 |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.01488 |          0.01488 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01487 |          0.01487 |
| random_forest        | num__source_avg_identifier_length_max                     |      0.01437 |          0.01437 |
| random_forest        | num__source_avg_identifier_length_mean                    |      0.01405 |          0.01405 |
| random_forest        | num__source_line_length_std_gold_minus_loser              |      0.01335 |          0.01335 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |      0.01299 |          0.01299 |
| random_forest        | num__source_halstead_distinct_operands_gold_minus_loser   |      0.01232 |          0.01232 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__source_avg_identifier_length_gold_minus_loser <= 0.51
|   |   |--- num__source_literal_count_gold_minus_loser <= -0.50
|   |   |   |--- class: 0
|   |   |--- num__source_literal_count_gold_minus_loser >  -0.50
|   |   |   |--- class: 0
|   |--- num__source_avg_identifier_length_gold_minus_loser >  0.51
|   |   |--- num__source_keyword_count_gold_minus_loser <= -3.50
|   |   |   |--- class: 0
|   |   |--- num__source_keyword_count_gold_minus_loser >  -3.50
|   |   |   |--- class: 1
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_avg_token_length_gold_minus_loser <= 0.35
|   |   |--- num__source_halstead_volume_max <= 1265.42
|   |   |   |--- class: 1
|   |   |--- num__source_halstead_volume_max >  1265.42
|   |   |   |--- class: 0
|   |--- num__source_avg_token_length_gold_minus_loser >  0.35
|   |   |--- num__source_max_line_length_mean <= 86.75
|   |   |   |--- class: 1
|   |   |--- num__source_max_line_length_mean >  86.75
|   |   |   |--- class: 1

```
