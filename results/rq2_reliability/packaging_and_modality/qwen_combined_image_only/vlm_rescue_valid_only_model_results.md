# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| logistic_regression  |   0.8991 |                 0.8117 | 555 |          0.5279 |
| gradient_boosting    |   0.8867 |                 0.8074 | 555 |          0.5279 |
| random_forest        |   0.8460 |                 0.7577 | 555 |          0.5279 |
| decision_tree_depth3 |   0.7703 |                 0.7083 | 555 |          0.5279 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_ratio_gold_minus_loser           |     0.4725   |         0.4725   |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |     0.1783   |         0.1783   |
| decision_tree_depth3 | num__source_halstead_distinct_operands_max                |     0.1503   |         0.1503   |
| decision_tree_depth3 | num__source_identifier_count_gold_minus_loser             |     0.1359   |         0.1359   |
| decision_tree_depth3 | num__source_indent_mean_max                               |     0.06301  |         0.06301  |
| decision_tree_depth3 | num__abs_human_score_gap                                  |     0        |         0        |
| decision_tree_depth3 | num__rf_margin_abs                                        |     0        |         0        |
| decision_tree_depth3 | num__rf_margin_percentile                                 |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_max                                 |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_mean                                |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_mean                             |     0        |         0        |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |     0.184    |         0.184    |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |     0.1175   |         0.1175   |
| gradient_boosting    | num__source_identifier_count_gold_minus_loser             |     0.04746  |         0.04746  |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |     0.04641  |         0.04641  |
| gradient_boosting    | num__source_indent_mean_max                               |     0.04041  |         0.04041  |
| gradient_boosting    | num__source_nesting_mean_gold_minus_loser                 |     0.03294  |         0.03294  |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |     0.03292  |         0.03292  |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |     0.02349  |         0.02349  |
| gradient_boosting    | num__source_avg_token_length_mean                         |     0.02315  |         0.02315  |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |     0.02128  |         0.02128  |
| gradient_boosting    | num__source_indent_mean_mean                              |     0.01911  |         0.01911  |
| gradient_boosting    | num__source_nesting_mean_max                              |     0.01756  |         0.01756  |
| gradient_boosting    | num__source_halstead_distinct_operands_max                |     0.01743  |         0.01743  |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |     0.01572  |         0.01572  |
| gradient_boosting    | num__source_loc_total_diff_abs                            |     0.01387  |         0.01387  |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |     2.37     |         2.37     |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |    -1.863    |         1.863    |
| logistic_regression  | num__source_catch_count_gold_minus_loser                  |    -1.723    |         1.723    |
| logistic_regression  | num__source_avg_token_length_gold_minus_loser             |     1.586    |         1.586    |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |     1.311    |         1.311    |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |     1.305    |         1.305    |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |     1.274    |         1.274    |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |     1.186    |         1.186    |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |    -1.122    |         1.122    |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |     1.102    |         1.102    |
| logistic_regression  | num__source_halstead_distinct_operators_gold_minus_loser  |     0.965    |         0.965    |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |    -0.9263   |         0.9263   |
| logistic_regression  | num__source_nesting_max_mean                              |     0.9015   |         0.9015   |
| logistic_regression  | num__source_avg_token_length_mean                         |    -0.8673   |         0.8673   |
| logistic_regression  | num__visual_contrast_score_gold_minus_loser               |     0.8591   |         0.8591   |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |     0.06048  |         0.06048  |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |     0.05497  |         0.05497  |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |     0.04071  |         0.04071  |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |     0.03855  |         0.03855  |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |     0.03631  |         0.03631  |
| random_forest        | num__visual_non_background_pixel_density_max              |     0.02917  |         0.02917  |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |     0.02105  |         0.02105  |
| random_forest        | num__visual_non_background_pixel_density_mean             |     0.01827  |         0.01827  |
| random_forest        | num__visual_horizontal_whitespace_ratio_max               |     0.01672  |         0.01672  |
| random_forest        | num__source_comment_line_ratio_diff_abs                   |     0.01376  |         0.01376  |
| random_forest        | num__source_comment_line_ratio_mean                       |     0.01291  |         0.01291  |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |     0.01209  |         0.01209  |
| random_forest        | num__visual_image_width_gold_minus_loser                  |     0.01151  |         0.01151  |
| random_forest        | num__source_comment_line_ratio_max                        |     0.01102  |         0.01102  |
| random_forest        | num__source_indent_max_gold_minus_loser                   |     0.009061 |         0.009061 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_ratio_gold_minus_loser <= -0.07
|   |--- num__source_identifier_count_gold_minus_loser <= -74.50
|   |   |--- class: 1
|   |--- num__source_identifier_count_gold_minus_loser >  -74.50
|   |   |--- num__source_indent_mean_gold_minus_loser <= 0.97
|   |   |   |--- class: 0
|   |   |--- num__source_indent_mean_gold_minus_loser >  0.97
|   |   |   |--- class: 0
|--- num__source_comment_line_ratio_gold_minus_loser >  -0.07
|   |--- num__source_halstead_distinct_operands_max <= 54.50
|   |   |--- num__source_indent_mean_gold_minus_loser <= 0.48
|   |   |   |--- class: 1
|   |   |--- num__source_indent_mean_gold_minus_loser >  0.48
|   |   |   |--- class: 1
|   |--- num__source_halstead_distinct_operands_max >  54.50
|   |   |--- num__source_indent_mean_max <= 9.68
|   |   |   |--- class: 0
|   |   |--- num__source_indent_mean_max >  9.68
|   |   |   |--- class: 1

```
