# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.8804 |                 0.7907 | 697 |          0.4692 |
| logistic_regression  |   0.8737 |                 0.7894 | 697 |          0.4692 |
| random_forest        |   0.8318 |                 0.7426 | 697 |          0.4692 |
| decision_tree_depth3 |   0.7241 |                 0.6686 | 697 |          0.4692 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.4523  |          0.4523  |
| decision_tree_depth3 | num__source_comment_line_ratio_gold_minus_loser           |      0.1611  |          0.1611  |
| decision_tree_depth3 | num__source_indent_mean_max                               |      0.1297  |          0.1297  |
| decision_tree_depth3 | num__source_avg_identifier_length_mean                    |      0.1246  |          0.1246  |
| decision_tree_depth3 | num__source_halstead_distinct_operands_gold_minus_loser   |      0.06872 |          0.06872 |
| decision_tree_depth3 | num__visual_clutter_score_mean                            |      0.06371 |          0.06371 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |      0       |          0       |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.1693  |          0.1693  |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.08129 |          0.08129 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.04468 |          0.04468 |
| gradient_boosting    | num__source_indent_mean_max                               |      0.03889 |          0.03889 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.03842 |          0.03842 |
| gradient_boosting    | num__source_indent_mean_mean                              |      0.03306 |          0.03306 |
| gradient_boosting    | num__source_max_line_length_gold_minus_loser              |      0.03039 |          0.03039 |
| gradient_boosting    | num__source_blank_line_ratio_gold_minus_loser             |      0.02828 |          0.02828 |
| gradient_boosting    | num__source_avg_identifier_length_mean                    |      0.02617 |          0.02617 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.02353 |          0.02353 |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.02291 |          0.02291 |
| gradient_boosting    | num__source_avg_identifier_length_gold_minus_loser        |      0.02205 |          0.02205 |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.01495 |          0.01495 |
| gradient_boosting    | num__abs_human_score_gap                                  |      0.01397 |          0.01397 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_max               |      0.01369 |          0.01369 |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.669   |          1.669   |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.466   |          1.466   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.429   |          1.429   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.413   |          1.413   |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -1.389   |          1.389   |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -1.358   |          1.358   |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      1.319   |          1.319   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.172   |          1.172   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      1.104   |          1.104   |
| logistic_regression  | num__source_line_length_std_gold_minus_loser              |     -1.047   |          1.047   |
| logistic_regression  | num__source_identifier_count_diff_abs                     |      1.034   |          1.034   |
| logistic_regression  | num__source_for_count_mean                                |     -0.9762  |          0.9762  |
| logistic_regression  | num__source_keyword_count_gold_minus_loser                |      0.9222  |          0.9222  |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -0.9039  |          0.9039  |
| logistic_regression  | num__source_cyclomatic_estimate_diff_abs                  |     -0.8069  |          0.8069  |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.07316 |          0.07316 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.04219 |          0.04219 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.03271 |          0.03271 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.03091 |          0.03091 |
| random_forest        | num__source_indent_max_gold_minus_loser                   |      0.02958 |          0.02958 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02776 |          0.02776 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.02074 |          0.02074 |
| random_forest        | num__visual_horizontal_whitespace_ratio_max               |      0.01667 |          0.01667 |
| random_forest        | num__visual_image_width_gold_minus_loser                  |      0.01662 |          0.01662 |
| random_forest        | num__visual_line_length_max_gold_minus_loser              |      0.01645 |          0.01645 |
| random_forest        | num__source_literal_count_gold_minus_loser                |      0.01618 |          0.01618 |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |      0.01516 |          0.01516 |
| random_forest        | num__source_indent_std_gold_minus_loser                   |      0.0135  |          0.0135  |
| random_forest        | num__source_blank_line_ratio_gold_minus_loser             |      0.01333 |          0.01333 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.01199 |          0.01199 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_indent_mean_gold_minus_loser <= 0.04
|   |--- num__source_indent_mean_max <= 10.82
|   |   |--- num__visual_clutter_score_mean <= 45.03
|   |   |   |--- class: 0
|   |   |--- num__visual_clutter_score_mean >  45.03
|   |   |   |--- class: 1
|   |--- num__source_indent_mean_max >  10.82
|   |   |--- num__source_indent_mean_gold_minus_loser <= -3.22
|   |   |   |--- class: 1
|   |   |--- num__source_indent_mean_gold_minus_loser >  -3.22
|   |   |   |--- class: 1
|--- num__source_indent_mean_gold_minus_loser >  0.04
|   |--- num__source_comment_line_ratio_gold_minus_loser <= -0.00
|   |   |--- num__source_halstead_distinct_operands_gold_minus_loser <= -3.50
|   |   |   |--- class: 0
|   |   |--- num__source_halstead_distinct_operands_gold_minus_loser >  -3.50
|   |   |   |--- class: 0
|   |--- num__source_comment_line_ratio_gold_minus_loser >  -0.00
|   |   |--- num__source_avg_identifier_length_mean <= 8.56
|   |   |   |--- class: 1
|   |   |--- num__source_avg_identifier_length_mean >  8.56
|   |   |   |--- class: 0

```
