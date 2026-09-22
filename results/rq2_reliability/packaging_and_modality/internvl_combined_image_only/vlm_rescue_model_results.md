# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| random_forest        |   0.6287 |                 0.5924 | 1130 |          0.2195 |
| logistic_regression  |   0.6280 |                 0.6056 | 1130 |          0.2195 |
| gradient_boosting    |   0.6137 |                 0.5314 | 1130 |          0.2195 |
| decision_tree_depth3 |   0.5628 |                 0.5464 | 1130 |          0.2195 |

## Top Feature Importances

| model                | feature                                                 |   importance |   abs_importance |
|:---------------------|:--------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_halstead_distinct_operands_gold_minus_loser |      0.321   |          0.321   |
| decision_tree_depth3 | num__visual_non_background_pixel_density_max            |      0.2557  |          0.2557  |
| decision_tree_depth3 | num__source_token_entropy_gold_minus_loser              |      0.2258  |          0.2258  |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                |      0.1526  |          0.1526  |
| decision_tree_depth3 | num__source_exception_handling_count_gold_minus_loser   |      0.0449  |          0.0449  |
| decision_tree_depth3 | num__abs_human_score_gap                                |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                      |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                               |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                          |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                               |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                              |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                  |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                       |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_mean                           |      0       |          0       |
| gradient_boosting    | num__source_indent_std_mean                             |      0.03382 |          0.03382 |
| gradient_boosting    | num__visual_non_background_pixel_density_max            |      0.02589 |          0.02589 |
| gradient_boosting    | num__source_line_length_std_diff_abs                    |      0.02582 |          0.02582 |
| gradient_boosting    | num__source_avg_token_length_mean                       |      0.02558 |          0.02558 |
| gradient_boosting    | num__source_token_entropy_max                           |      0.02207 |          0.02207 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser           |      0.01963 |          0.01963 |
| gradient_boosting    | num__source_avg_identifier_length_diff_abs              |      0.01923 |          0.01923 |
| gradient_boosting    | num__source_loc_total_max                               |      0.01897 |          0.01897 |
| gradient_boosting    | num__source_avg_line_length_max                         |      0.01874 |          0.01874 |
| gradient_boosting    | num__visual_contrast_score_gold_minus_loser             |      0.01751 |          0.01751 |
| gradient_boosting    | num__source_avg_nonempty_line_length_diff_abs           |      0.01731 |          0.01731 |
| gradient_boosting    | num__visual_clutter_score_diff_abs                      |      0.01717 |          0.01717 |
| gradient_boosting    | num__source_blank_line_count_diff_abs                   |      0.01673 |          0.01673 |
| gradient_boosting    | num__source_token_entropy_gold_minus_loser              |      0.01653 |          0.01653 |
| gradient_boosting    | num__source_max_line_length_gold_minus_loser            |      0.01653 |          0.01653 |
| logistic_regression  | num__source_avg_nonempty_line_length_mean               |     -0.9035  |          0.9035  |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser   |      0.8422  |          0.8422  |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser            |     -0.8411  |          0.8411  |
| logistic_regression  | num__source_try_count_mean                              |     -0.8045  |          0.8045  |
| logistic_regression  | num__source_try_count_diff_abs                          |      0.7951  |          0.7951  |
| logistic_regression  | num__source_blank_line_count_mean                       |      0.7699  |          0.7699  |
| logistic_regression  | num__source_avg_line_length_mean                        |      0.7566  |          0.7566  |
| logistic_regression  | num__source_avg_identifier_length_mean                  |     -0.7384  |          0.7384  |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser               |      0.7359  |          0.7359  |
| logistic_regression  | num__source_loc_total_gold_minus_loser                  |      0.6711  |          0.6711  |
| logistic_regression  | num__source_parenthesis_count_mean                      |      0.6684  |          0.6684  |
| logistic_regression  | num__source_identifier_unique_count_gold_minus_loser    |     -0.6366  |          0.6366  |
| logistic_regression  | num__source_halstead_distinct_operands_diff_abs         |      0.622   |          0.622   |
| logistic_regression  | num__source_operator_count_gold_minus_loser             |     -0.6186  |          0.6186  |
| logistic_regression  | num__source_max_line_length_gold_minus_loser            |      0.6114  |          0.6114  |
| random_forest        | num__visual_non_background_pixel_density_max            |      0.01902 |          0.01902 |
| random_forest        | num__source_halstead_distinct_operands_gold_minus_loser |      0.01774 |          0.01774 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser    |      0.01671 |          0.01671 |
| random_forest        | num__source_halstead_volume_gold_minus_loser            |      0.01595 |          0.01595 |
| random_forest        | num__source_halstead_vocabulary_gold_minus_loser        |      0.01367 |          0.01367 |
| random_forest        | num__source_halstead_volume_max                         |      0.01289 |          0.01289 |
| random_forest        | num__visual_image_height_max                            |      0.01253 |          0.01253 |
| random_forest        | num__visual_line_length_max_gold_minus_loser            |      0.01244 |          0.01244 |
| random_forest        | num__source_identifier_count_gold_minus_loser           |      0.01238 |          0.01238 |
| random_forest        | num__source_avg_token_length_mean                       |      0.01202 |          0.01202 |
| random_forest        | num__visual_line_length_max_diff_abs                    |      0.01187 |          0.01187 |
| random_forest        | num__source_token_count_gold_minus_loser                |      0.01129 |          0.01129 |
| random_forest        | num__visual_image_width_diff_abs                        |      0.01107 |          0.01107 |
| random_forest        | num__source_nesting_mean_max                            |      0.01083 |          0.01083 |
| random_forest        | num__source_halstead_length_max                         |      0.01052 |          0.01052 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_halstead_distinct_operands_gold_minus_loser <= 52.50
|   |--- num__visual_non_background_pixel_density_max <= 1.00
|   |   |--- num__source_token_entropy_gold_minus_loser <= 1.15
|   |   |   |--- class: 0
|   |   |--- num__source_token_entropy_gold_minus_loser >  1.15
|   |   |   |--- class: 1
|   |--- num__visual_non_background_pixel_density_max >  1.00
|   |   |--- num__source_indent_mean_gold_minus_loser <= -6.81
|   |   |   |--- class: 1
|   |   |--- num__source_indent_mean_gold_minus_loser >  -6.81
|   |   |   |--- class: 1
|--- num__source_halstead_distinct_operands_gold_minus_loser >  52.50
|   |--- num__source_exception_handling_count_gold_minus_loser <= 0.50
|   |   |--- class: 0
|   |--- num__source_exception_handling_count_gold_minus_loser >  0.50
|   |   |--- class: 0

```
