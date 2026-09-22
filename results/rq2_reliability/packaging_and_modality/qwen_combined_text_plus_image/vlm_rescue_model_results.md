# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.8114 |                 0.7253 | 1130 |          0.3876 |
| logistic_regression  |   0.8035 |                 0.7434 | 1130 |          0.3876 |
| random_forest        |   0.7899 |                 0.7258 | 1130 |          0.3876 |
| decision_tree_depth3 |   0.7169 |                 0.6633 | 1130 |          0.3876 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.4781  |          0.4781  |
| decision_tree_depth3 | num__source_avg_token_length_gold_minus_loser             |      0.2113  |          0.2113  |
| decision_tree_depth3 | num__source_indent_mean_mean                              |      0.0986  |          0.0986  |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.07123 |          0.07123 |
| decision_tree_depth3 | num__source_nesting_mean_mean                             |      0.06197 |          0.06197 |
| decision_tree_depth3 | num__visual_non_background_pixel_density_max              |      0.05759 |          0.05759 |
| decision_tree_depth3 | num__source_blank_line_ratio_diff_abs                     |      0.02124 |          0.02124 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.09922 |          0.09922 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.0693  |          0.0693  |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.04905 |          0.04905 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.03506 |          0.03506 |
| gradient_boosting    | num__source_indent_mean_mean                              |      0.03268 |          0.03268 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.03139 |          0.03139 |
| gradient_boosting    | num__source_api_call_count_gold_minus_loser               |      0.03115 |          0.03115 |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.0278  |          0.0278  |
| gradient_boosting    | num__source_avg_token_length_mean                         |      0.02724 |          0.02724 |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser                 |      0.02653 |          0.02653 |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.02056 |          0.02056 |
| gradient_boosting    | num__source_avg_identifier_length_gold_minus_loser        |      0.01947 |          0.01947 |
| gradient_boosting    | num__visual_non_background_pixel_density_gold_minus_loser |      0.0185  |          0.0185  |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.01655 |          0.01655 |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.0157  |          0.0157  |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.204   |          1.204   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.055   |          1.055   |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser              |     -1.047   |          1.047   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      0.9944  |          0.9944  |
| logistic_regression  | num__source_identifier_unique_count_gold_minus_loser      |      0.8431  |          0.8431  |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -0.7682  |          0.7682  |
| logistic_regression  | num__source_loc_nonempty_mean                             |     -0.7547  |          0.7547  |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -0.6976  |          0.6976  |
| logistic_regression  | num__source_token_entropy_gold_minus_loser                |     -0.6897  |          0.6897  |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      0.6744  |          0.6744  |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      0.6693  |          0.6693  |
| logistic_regression  | num__source_loc_total_diff_abs                            |      0.6591  |          0.6591  |
| logistic_regression  | num__source_identifier_unique_count_mean                  |      0.628   |          0.628   |
| logistic_regression  | num__source_while_count_mean                              |      0.6107  |          0.6107  |
| logistic_regression  | num__source_halstead_distinct_operators_gold_minus_loser  |      0.6036  |          0.6036  |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.05493 |          0.05493 |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.0497  |          0.0497  |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.04673 |          0.04673 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.04379 |          0.04379 |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.02542 |          0.02542 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.02498 |          0.02498 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.0247  |          0.0247  |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.01709 |          0.01709 |
| random_forest        | num__source_indent_mean_mean                              |      0.01557 |          0.01557 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01488 |          0.01488 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |      0.01268 |          0.01268 |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |      0.01229 |          0.01229 |
| random_forest        | num__visual_non_background_pixel_density_mean             |      0.01148 |          0.01148 |
| random_forest        | num__visual_clutter_score_mean                            |      0.01114 |          0.01114 |
| random_forest        | num__source_avg_line_length_gold_minus_loser              |      0.01092 |          0.01092 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__source_indent_mean_gold_minus_loser <= 2.84
|   |   |--- num__source_nesting_mean_mean <= 0.84
|   |   |   |--- class: 0
|   |   |--- num__source_nesting_mean_mean >  0.84
|   |   |   |--- class: 1
|   |--- num__source_indent_mean_gold_minus_loser >  2.84
|   |   |--- num__source_blank_line_ratio_diff_abs <= 0.07
|   |   |   |--- class: 0
|   |   |--- num__source_blank_line_ratio_diff_abs >  0.07
|   |   |   |--- class: 0
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_avg_token_length_gold_minus_loser <= 0.63
|   |   |--- num__source_indent_mean_mean <= 11.18
|   |   |   |--- class: 0
|   |   |--- num__source_indent_mean_mean >  11.18
|   |   |   |--- class: 1
|   |--- num__source_avg_token_length_gold_minus_loser >  0.63
|   |   |--- num__visual_non_background_pixel_density_max <= 1.00
|   |   |   |--- class: 1
|   |   |--- num__visual_non_background_pixel_density_max >  1.00
|   |   |   |--- class: 1

```
