# Rf Failure Model Results

Subset: All 3,000 pairs; target=RF wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.8892 |                 0.7729 | 3000 |          0.3767 |
| logistic_regression  |   0.8767 |                 0.7957 | 3000 |          0.3767 |
| random_forest        |   0.8695 |                 0.7799 | 3000 |          0.3767 |
| decision_tree_depth3 |   0.8004 |                 0.7273 | 3000 |          0.3767 |

## Top Feature Importances

| model                | feature                                                  |   importance |   abs_importance |
|:---------------------|:---------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_avg_line_length_gold_minus_loser             |      0.5625  |          0.5625  |
| decision_tree_depth3 | num__rf_margin_percentile                                |      0.2021  |          0.2021  |
| decision_tree_depth3 | num__source_max_line_length_gold_minus_loser             |      0.1154  |          0.1154  |
| decision_tree_depth3 | num__source_indent_max_gold_minus_loser                  |      0.05029 |          0.05029 |
| decision_tree_depth3 | num__rf_margin_abs                                       |      0.04132 |          0.04132 |
| decision_tree_depth3 | num__source_avg_nonempty_line_length_max                 |      0.02839 |          0.02839 |
| decision_tree_depth3 | num__abs_human_score_gap                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                           |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                               |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                   |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                        |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                             |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_mean                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_gold_minus_loser                |      0       |          0       |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser             |      0.2067  |          0.2067  |
| gradient_boosting    | num__source_max_line_length_gold_minus_loser             |      0.07134 |          0.07134 |
| gradient_boosting    | num__rf_margin_percentile                                |      0.0675  |          0.0675  |
| gradient_boosting    | num__source_line_length_std_gold_minus_loser             |      0.06526 |          0.06526 |
| gradient_boosting    | num__source_avg_nonempty_line_length_gold_minus_loser    |      0.05858 |          0.05858 |
| gradient_boosting    | num__rf_margin_abs                                       |      0.04142 |          0.04142 |
| gradient_boosting    | num__source_indent_max_gold_minus_loser                  |      0.03938 |          0.03938 |
| gradient_boosting    | num__source_operator_count_gold_minus_loser              |      0.03545 |          0.03545 |
| gradient_boosting    | num__source_nesting_mean_gold_minus_loser                |      0.03343 |          0.03343 |
| gradient_boosting    | num__source_indent_std_gold_minus_loser                  |      0.02231 |          0.02231 |
| gradient_boosting    | num__source_avg_line_length_mean                         |      0.02019 |          0.02019 |
| gradient_boosting    | num__source_api_call_count_gold_minus_loser              |      0.0192  |          0.0192  |
| gradient_boosting    | num__source_line_length_std_mean                         |      0.01854 |          0.01854 |
| gradient_boosting    | num__source_avg_line_length_max                          |      0.01715 |          0.01715 |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser          |      0.01463 |          0.01463 |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser             |      2.286   |          2.286   |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser             |      1.585   |          1.585   |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                |      1.364   |          1.364   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                   |      1.258   |          1.258   |
| logistic_regression  | num__source_identifier_unique_count_gold_minus_loser     |     -0.9853  |          0.9853  |
| logistic_regression  | num__source_operator_count_gold_minus_loser              |     -0.9193  |          0.9193  |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                |      0.8857  |          0.8857  |
| logistic_regression  | num__source_if_count_gold_minus_loser                    |      0.7309  |          0.7309  |
| logistic_regression  | num__source_max_line_length_gold_minus_loser             |      0.7239  |          0.7239  |
| logistic_regression  | num__visual_aspect_ratio_gold_minus_loser                |      0.711   |          0.711   |
| logistic_regression  | num__source_halstead_distinct_operands_gold_minus_loser  |      0.7054  |          0.7054  |
| logistic_regression  | num__source_brace_count_gold_minus_loser                 |     -0.7014  |          0.7014  |
| logistic_regression  | num__source_line_length_std_gold_minus_loser             |      0.6733  |          0.6733  |
| logistic_regression  | num__source_halstead_vocabulary_diff_abs                 |      0.6542  |          0.6542  |
| logistic_regression  | num__source_halstead_distinct_operators_gold_minus_loser |     -0.6177  |          0.6177  |
| random_forest        | num__source_avg_line_length_gold_minus_loser             |      0.09508 |          0.09508 |
| random_forest        | num__source_avg_nonempty_line_length_gold_minus_loser    |      0.07002 |          0.07002 |
| random_forest        | num__source_max_line_length_gold_minus_loser             |      0.06676 |          0.06676 |
| random_forest        | num__source_line_length_std_gold_minus_loser             |      0.06075 |          0.06075 |
| random_forest        | num__source_parenthesis_count_gold_minus_loser           |      0.0395  |          0.0395  |
| random_forest        | num__source_operator_count_gold_minus_loser              |      0.03439 |          0.03439 |
| random_forest        | num__rf_margin_abs                                       |      0.03157 |          0.03157 |
| random_forest        | num__source_halstead_length_gold_minus_loser             |      0.02693 |          0.02693 |
| random_forest        | num__rf_margin_percentile                                |      0.02539 |          0.02539 |
| random_forest        | num__source_api_call_count_gold_minus_loser              |      0.02365 |          0.02365 |
| random_forest        | num__source_indent_std_gold_minus_loser                  |      0.02046 |          0.02046 |
| random_forest        | num__source_halstead_volume_gold_minus_loser             |      0.01944 |          0.01944 |
| random_forest        | num__source_nesting_mean_gold_minus_loser                |      0.01941 |          0.01941 |
| random_forest        | num__visual_code_bounding_box_height_gold_minus_loser    |      0.01912 |          0.01912 |
| random_forest        | num__source_token_count_gold_minus_loser                 |      0.01641 |          0.01641 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_avg_line_length_gold_minus_loser <= -1.12
|   |--- num__rf_margin_percentile <= 0.55
|   |   |--- num__rf_margin_abs <= 0.26
|   |   |   |--- class: 1
|   |   |--- num__rf_margin_abs >  0.26
|   |   |   |--- class: 0
|   |--- num__rf_margin_percentile >  0.55
|   |   |--- num__source_max_line_length_gold_minus_loser <= 6.50
|   |   |   |--- class: 0
|   |   |--- num__source_max_line_length_gold_minus_loser >  6.50
|   |   |   |--- class: 0
|--- num__source_avg_line_length_gold_minus_loser >  -1.12
|   |--- num__source_max_line_length_gold_minus_loser <= 20.50
|   |   |--- num__source_indent_max_gold_minus_loser <= -3.50
|   |   |   |--- class: 0
|   |   |--- num__source_indent_max_gold_minus_loser >  -3.50
|   |   |   |--- class: 1
|   |--- num__source_max_line_length_gold_minus_loser >  20.50
|   |   |--- num__source_avg_nonempty_line_length_max <= 32.65
|   |   |   |--- class: 1
|   |   |--- num__source_avg_nonempty_line_length_max >  32.65
|   |   |   |--- class: 1

```
