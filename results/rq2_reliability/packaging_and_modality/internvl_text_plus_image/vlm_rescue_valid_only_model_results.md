# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.8581 |                 0.7816 | 814 |          0.5319 |
| logistic_regression  |   0.8289 |                 0.7595 | 814 |          0.5319 |
| random_forest        |   0.8000 |                 0.7247 | 814 |          0.5319 |
| decision_tree_depth3 |   0.7006 |                 0.6486 | 814 |          0.5319 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.2676  |          0.2676  |
| decision_tree_depth3 | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.2016  |          0.2016  |
| decision_tree_depth3 | num__source_blank_line_ratio_max                          |      0.149   |          0.149   |
| decision_tree_depth3 | num__source_avg_identifier_length_gold_minus_loser        |      0.1425  |          0.1425  |
| decision_tree_depth3 | num__source_blank_line_count_diff_abs                     |      0.1134  |          0.1134  |
| decision_tree_depth3 | num__visual_horizontal_whitespace_ratio_max               |      0.09433 |          0.09433 |
| decision_tree_depth3 | num__source_exception_handling_count_gold_minus_loser     |      0.0315  |          0.0315  |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__source_avg_token_length_max                          |      0.06575 |          0.06575 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.05803 |          0.05803 |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.04807 |          0.04807 |
| gradient_boosting    | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.0457  |          0.0457  |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.03768 |          0.03768 |
| gradient_boosting    | num__source_avg_identifier_length_gold_minus_loser        |      0.03619 |          0.03619 |
| gradient_boosting    | num__source_keyword_count_max                             |      0.02879 |          0.02879 |
| gradient_boosting    | num__source_blank_line_ratio_max                          |      0.02865 |          0.02865 |
| gradient_boosting    | num__source_blank_line_count_max                          |      0.02852 |          0.02852 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02648 |          0.02648 |
| gradient_boosting    | num__visual_contrast_score_gold_minus_loser               |      0.02617 |          0.02617 |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.01994 |          0.01994 |
| gradient_boosting    | num__source_cyclomatic_estimate_mean                      |      0.01514 |          0.01514 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.01512 |          0.01512 |
| gradient_boosting    | num__visual_aspect_ratio_max                              |      0.01487 |          0.01487 |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      2.097   |          2.097   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      1.818   |          1.818   |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -1.639   |          1.639   |
| logistic_regression  | num__source_identifier_count_gold_minus_loser             |     -1.214   |          1.214   |
| logistic_regression  | num__source_exception_handling_count_mean                 |     -1.196   |          1.196   |
| logistic_regression  | num__source_exception_handling_count_diff_abs             |      1.174   |          1.174   |
| logistic_regression  | num__source_literal_count_gold_minus_loser                |     -1.053   |          1.053   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.042   |          1.042   |
| logistic_regression  | num__source_indent_max_mean                               |     -0.972   |          0.972   |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -0.9311  |          0.9311  |
| logistic_regression  | num__source_catch_count_mean                              |      0.924   |          0.924   |
| logistic_regression  | num__rf_margin_abs                                        |      0.8734  |          0.8734  |
| logistic_regression  | num__source_halstead_distinct_operators_gold_minus_loser  |      0.7384  |          0.7384  |
| logistic_regression  | num__rf_margin_percentile                                 |     -0.7357  |          0.7357  |
| logistic_regression  | num__source_try_count_diff_abs                            |     -0.7331  |          0.7331  |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.02892 |          0.02892 |
| random_forest        | num__source_literal_count_gold_minus_loser                |      0.02628 |          0.02628 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02586 |          0.02586 |
| random_forest        | num__source_parenthesis_count_gold_minus_loser            |      0.02457 |          0.02457 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.01976 |          0.01976 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01955 |          0.01955 |
| random_forest        | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.01952 |          0.01952 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.01941 |          0.01941 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |      0.01748 |          0.01748 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.01619 |          0.01619 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.01552 |          0.01552 |
| random_forest        | num__source_keyword_count_gold_minus_loser                |      0.01472 |          0.01472 |
| random_forest        | num__source_avg_line_length_gold_minus_loser              |      0.01345 |          0.01345 |
| random_forest        | num__source_line_length_std_gold_minus_loser              |      0.0122  |          0.0122  |
| random_forest        | num__source_blank_line_count_max                          |      0.01216 |          0.01216 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__visual_horizontal_whitespace_ratio_max <= 0.01
|   |   |--- num__source_exception_handling_count_gold_minus_loser <= 0.50
|   |   |   |--- class: 0
|   |   |--- num__source_exception_handling_count_gold_minus_loser >  0.50
|   |   |   |--- class: 0
|   |--- num__visual_horizontal_whitespace_ratio_max >  0.01
|   |   |--- num__source_blank_line_count_diff_abs <= 2.50
|   |   |   |--- class: 1
|   |   |--- num__source_blank_line_count_diff_abs >  2.50
|   |   |   |--- class: 0
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_avg_nonempty_line_length_gold_minus_loser <= 3.03
|   |   |--- num__source_blank_line_ratio_max <= 0.28
|   |   |   |--- class: 1
|   |   |--- num__source_blank_line_ratio_max >  0.28
|   |   |   |--- class: 0
|   |--- num__source_avg_nonempty_line_length_gold_minus_loser >  3.03
|   |   |--- num__source_avg_identifier_length_gold_minus_loser <= -2.06
|   |   |   |--- class: 0
|   |   |--- num__source_avg_identifier_length_gold_minus_loser >  -2.06
|   |   |   |--- class: 1

```
