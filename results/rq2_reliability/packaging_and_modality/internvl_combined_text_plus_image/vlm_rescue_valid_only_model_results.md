# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.8513 |                 0.7645 | 714 |          0.5476 |
| logistic_regression  |   0.8370 |                 0.7484 | 714 |          0.5476 |
| random_forest        |   0.8141 |                 0.7331 | 714 |          0.5476 |
| decision_tree_depth3 |   0.6998 |                 0.6533 | 714 |          0.5476 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |     0.3849   |         0.3849   |
| decision_tree_depth3 | num__source_avg_token_length_gold_minus_loser             |     0.259    |         0.259    |
| decision_tree_depth3 | num__abs_human_score_gap                                  |     0.1136   |         0.1136   |
| decision_tree_depth3 | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |     0.1046   |         0.1046   |
| decision_tree_depth3 | num__source_avg_line_length_diff_abs                      |     0.08334  |         0.08334  |
| decision_tree_depth3 | num__source_keyword_count_mean                            |     0.05445  |         0.05445  |
| decision_tree_depth3 | num__rf_margin_abs                                        |     0        |         0        |
| decision_tree_depth3 | num__rf_margin_percentile                                 |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_max                                 |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_mean                                |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_mean                             |     0        |         0        |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |     0.105    |         0.105    |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |     0.06921  |         0.06921  |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |     0.04544  |         0.04544  |
| gradient_boosting    | num__source_avg_token_length_max                          |     0.03784  |         0.03784  |
| gradient_boosting    | num__source_avg_nonempty_line_length_gold_minus_loser     |     0.03737  |         0.03737  |
| gradient_boosting    | num__source_avg_identifier_length_gold_minus_loser        |     0.03337  |         0.03337  |
| gradient_boosting    | num__source_keyword_count_max                             |     0.02943  |         0.02943  |
| gradient_boosting    | num__visual_contrast_score_max                            |     0.02314  |         0.02314  |
| gradient_boosting    | num__source_identifier_unique_count_gold_minus_loser      |     0.02257  |         0.02257  |
| gradient_boosting    | num__source_keyword_count_gold_minus_loser                |     0.02151  |         0.02151  |
| gradient_boosting    | num__abs_human_score_gap                                  |     0.01801  |         0.01801  |
| gradient_boosting    | num__source_avg_line_length_max                           |     0.01602  |         0.01602  |
| gradient_boosting    | num__source_line_length_std_gold_minus_loser              |     0.01507  |         0.01507  |
| gradient_boosting    | num__visual_contrast_score_gold_minus_loser               |     0.01413  |         0.01413  |
| gradient_boosting    | num__source_avg_line_length_diff_abs                      |     0.01386  |         0.01386  |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |    -1.614    |         1.614    |
| logistic_regression  | num__source_operator_count_diff_abs                       |    -1.419    |         1.419    |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |     1.349    |         1.349    |
| logistic_regression  | num__source_literal_count_gold_minus_loser                |    -1.33     |         1.33     |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |     1.295    |         1.295    |
| logistic_regression  | num__source_exception_handling_count_mean                 |    -1.184    |         1.184    |
| logistic_regression  | num__source_indent_max_mean                               |    -1.054    |         1.054    |
| logistic_regression  | num__source_identifier_count_gold_minus_loser             |    -1.052    |         1.052    |
| logistic_regression  | num__source_indent_std_gold_minus_loser                   |     0.9945   |         0.9945   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |     0.9915   |         0.9915   |
| logistic_regression  | num__source_nesting_max_gold_minus_loser                  |    -0.9331   |         0.9331   |
| logistic_regression  | num__source_exception_handling_count_diff_abs             |     0.9204   |         0.9204   |
| logistic_regression  | num__source_catch_count_mean                              |     0.8938   |         0.8938   |
| logistic_regression  | num__source_keyword_count_gold_minus_loser                |     0.8275   |         0.8275   |
| logistic_regression  | num__source_indent_std_mean                               |     0.8216   |         0.8216   |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |     0.03636  |         0.03636  |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |     0.03538  |         0.03538  |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |     0.03429  |         0.03429  |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |     0.03154  |         0.03154  |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |     0.02824  |         0.02824  |
| random_forest        | num__source_literal_count_gold_minus_loser                |     0.02381  |         0.02381  |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |     0.02183  |         0.02183  |
| random_forest        | num__source_avg_nonempty_line_length_gold_minus_loser     |     0.02145  |         0.02145  |
| random_forest        | num__source_line_length_std_gold_minus_loser              |     0.01888  |         0.01888  |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |     0.01862  |         0.01862  |
| random_forest        | num__visual_non_background_pixel_density_max              |     0.01502  |         0.01502  |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |     0.01247  |         0.01247  |
| random_forest        | num__source_keyword_count_gold_minus_loser                |     0.01079  |         0.01079  |
| random_forest        | num__source_halstead_distinct_operands_gold_minus_loser   |     0.0105   |         0.0105   |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |     0.009931 |         0.009931 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__source_avg_line_length_diff_abs <= 3.17
|   |   |--- class: 1
|   |--- num__source_avg_line_length_diff_abs >  3.17
|   |   |--- num__source_keyword_count_mean <= 10.25
|   |   |   |--- class: 0
|   |   |--- num__source_keyword_count_mean >  10.25
|   |   |   |--- class: 0
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_avg_token_length_gold_minus_loser <= 0.29
|   |   |--- num__abs_human_score_gap <= 1.13
|   |   |   |--- class: 0
|   |   |--- num__abs_human_score_gap >  1.13
|   |   |   |--- class: 1
|   |--- num__source_avg_token_length_gold_minus_loser >  0.29
|   |   |--- num__visual_horizontal_whitespace_ratio_gold_minus_loser <= -0.00
|   |   |   |--- class: 0
|   |   |--- num__visual_horizontal_whitespace_ratio_gold_minus_loser >  -0.00
|   |   |   |--- class: 1

```
