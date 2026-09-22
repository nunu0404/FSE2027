# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.8536 |                 0.7637 | 769 |          0.5696 |
| random_forest        |   0.8332 |                 0.7477 | 769 |          0.5696 |
| logistic_regression  |   0.8307 |                 0.7570 | 769 |          0.5696 |
| decision_tree_depth3 |   0.7598 |                 0.6804 | 769 |          0.5696 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.5296  |          0.5296  |
| decision_tree_depth3 | num__source_avg_token_length_gold_minus_loser             |      0.1956  |          0.1956  |
| decision_tree_depth3 | num__source_if_count_max                                  |      0.07852 |          0.07852 |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.06542 |          0.06542 |
| decision_tree_depth3 | num__source_halstead_distinct_operands_max                |      0.06253 |          0.06253 |
| decision_tree_depth3 | num__source_api_call_count_mean                           |      0.05538 |          0.05538 |
| decision_tree_depth3 | num__source_literal_count_max                             |      0.01301 |          0.01301 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.1632  |          0.1632  |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.06724 |          0.06724 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.06206 |          0.06206 |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.05232 |          0.05232 |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser                 |      0.02492 |          0.02492 |
| gradient_boosting    | num__source_api_call_count_gold_minus_loser               |      0.02412 |          0.02412 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.02368 |          0.02368 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02312 |          0.02312 |
| gradient_boosting    | num__source_blank_line_ratio_mean                         |      0.01911 |          0.01911 |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.01893 |          0.01893 |
| gradient_boosting    | num__source_halstead_distinct_operands_max                |      0.01557 |          0.01557 |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.01442 |          0.01442 |
| gradient_boosting    | num__visual_contrast_score_max                            |      0.01401 |          0.01401 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.01353 |          0.01353 |
| gradient_boosting    | num__visual_non_background_pixel_density_gold_minus_loser |      0.01277 |          0.01277 |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.334   |          1.334   |
| logistic_regression  | num__source_nesting_max_gold_minus_loser                  |     -1.295   |          1.295   |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -1.264   |          1.264   |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.227   |          1.227   |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      1.203   |          1.203   |
| logistic_regression  | num__source_identifier_count_diff_abs                     |      1.086   |          1.086   |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser              |     -1.076   |          1.076   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.073   |          1.073   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      1.024   |          1.024   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.005   |          1.005   |
| logistic_regression  | num__source_loc_total_diff_abs                            |      0.9463  |          0.9463  |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -0.9207  |          0.9207  |
| logistic_regression  | num__source_halstead_distinct_operators_gold_minus_loser  |      0.9132  |          0.9132  |
| logistic_regression  | num__source_keyword_count_gold_minus_loser                |      0.8867  |          0.8867  |
| logistic_regression  | num__source_nesting_max_mean                              |      0.8084  |          0.8084  |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.05712 |          0.05712 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.05205 |          0.05205 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.05057 |          0.05057 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.04498 |          0.04498 |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.04011 |          0.04011 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.02655 |          0.02655 |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.02503 |          0.02503 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.0226  |          0.0226  |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |      0.01375 |          0.01375 |
| random_forest        | num__source_identifier_count_gold_minus_loser             |      0.01262 |          0.01262 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01197 |          0.01197 |
| random_forest        | num__visual_aspect_ratio_max                              |      0.0113  |          0.0113  |
| random_forest        | num__source_avg_line_length_gold_minus_loser              |      0.01068 |          0.01068 |
| random_forest        | num__source_loc_nonempty_gold_minus_loser                 |      0.01065 |          0.01065 |
| random_forest        | num__source_comment_line_count_mean                       |      0.01046 |          0.01046 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__source_indent_mean_gold_minus_loser <= 2.84
|   |   |--- num__source_api_call_count_mean <= 7.25
|   |   |   |--- class: 0
|   |   |--- num__source_api_call_count_mean >  7.25
|   |   |   |--- class: 1
|   |--- num__source_indent_mean_gold_minus_loser >  2.84
|   |   |--- num__source_literal_count_max <= 3.50
|   |   |   |--- class: 0
|   |   |--- num__source_literal_count_max >  3.50
|   |   |   |--- class: 0
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_avg_token_length_gold_minus_loser <= 0.35
|   |   |--- num__source_if_count_max <= 4.50
|   |   |   |--- class: 0
|   |   |--- num__source_if_count_max >  4.50
|   |   |   |--- class: 1
|   |--- num__source_avg_token_length_gold_minus_loser >  0.35
|   |   |--- num__source_halstead_distinct_operands_max <= 54.50
|   |   |   |--- class: 1
|   |   |--- num__source_halstead_distinct_operands_max >  54.50
|   |   |   |--- class: 1

```
