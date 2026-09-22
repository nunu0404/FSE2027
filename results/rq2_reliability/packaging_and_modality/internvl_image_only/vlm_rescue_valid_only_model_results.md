# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.9591 |                 0.8748 | 508 |          0.6791 |
| logistic_regression  |   0.9557 |                 0.8874 | 508 |          0.6791 |
| random_forest        |   0.9556 |                 0.8814 | 508 |          0.6791 |
| decision_tree_depth3 |   0.9185 |                 0.8589 | 508 |          0.6791 |

## Top Feature Importances

| model                | feature                                               |   importance |   abs_importance |
|:---------------------|:------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__visual_aspect_ratio_gold_minus_loser             |     0.8171   |         0.8171   |
| decision_tree_depth3 | num__source_brace_count_gold_minus_loser              |     0.08659  |         0.08659  |
| decision_tree_depth3 | num__source_blank_line_ratio_max                      |     0.04769  |         0.04769  |
| decision_tree_depth3 | num__source_api_call_count_gold_minus_loser           |     0.03411  |         0.03411  |
| decision_tree_depth3 | num__source_identifier_count_gold_minus_loser         |     0.01315  |         0.01315  |
| decision_tree_depth3 | num__source_halstead_length_mean                      |     0.001336 |         0.001336 |
| decision_tree_depth3 | num__abs_human_score_gap                              |     0        |         0        |
| decision_tree_depth3 | num__rf_margin_abs                                    |     0        |         0        |
| decision_tree_depth3 | num__rf_margin_percentile                             |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_diff_abs                        |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_max                             |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_mean                            |     0        |         0        |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                     |     0        |         0        |
| decision_tree_depth3 | num__source_loc_nonempty_max                          |     0        |         0        |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser             |     0.3583   |         0.3583   |
| gradient_boosting    | num__source_method_like_count_gold_minus_loser        |     0.07557  |         0.07557  |
| gradient_boosting    | num__source_loc_nonempty_gold_minus_loser             |     0.07467  |         0.07467  |
| gradient_boosting    | num__source_brace_count_gold_minus_loser              |     0.04864  |         0.04864  |
| gradient_boosting    | num__visual_aspect_ratio_mean                         |     0.02873  |         0.02873  |
| gradient_boosting    | num__source_indent_std_gold_minus_loser               |     0.02537  |         0.02537  |
| gradient_boosting    | num__source_nesting_mean_gold_minus_loser             |     0.02006  |         0.02006  |
| gradient_boosting    | num__source_token_entropy_gold_minus_loser            |     0.01954  |         0.01954  |
| gradient_boosting    | num__source_loc_total_gold_minus_loser                |     0.0177   |         0.0177   |
| gradient_boosting    | num__source_blank_line_ratio_max                      |     0.01753  |         0.01753  |
| gradient_boosting    | num__visual_non_background_pixel_density_max          |     0.0175   |         0.0175   |
| gradient_boosting    | num__source_indent_max_gold_minus_loser               |     0.01608  |         0.01608  |
| gradient_boosting    | num__source_avg_identifier_length_mean                |     0.01533  |         0.01533  |
| gradient_boosting    | num__source_halstead_volume_max                       |     0.01387  |         0.01387  |
| gradient_boosting    | num__abs_human_score_gap                              |     0.01259  |         0.01259  |
| logistic_regression  | num__visual_aspect_ratio_gold_minus_loser             |    -1.745    |         1.745    |
| logistic_regression  | num__source_blank_line_count_gold_minus_loser         |     1.445    |         1.445    |
| logistic_regression  | num__source_token_entropy_gold_minus_loser            |     1.353    |         1.353    |
| logistic_regression  | num__source_blank_line_ratio_gold_minus_loser         |    -1.231    |         1.231    |
| logistic_regression  | cat__difficulty_medium                                |    -1.123    |         1.123    |
| logistic_regression  | num__abs_human_score_gap                              |     1.02     |         1.02     |
| logistic_regression  | num__visual_aspect_ratio_diff_abs                     |     1.007    |         1.007    |
| logistic_regression  | num__source_exception_handling_count_diff_abs         |    -0.9622   |         0.9622   |
| logistic_regression  | num__source_while_count_gold_minus_loser              |    -0.882    |         0.882    |
| logistic_regression  | num__source_keyword_count_gold_minus_loser            |     0.8779   |         0.8779   |
| logistic_regression  | num__visual_contrast_score_gold_minus_loser           |     0.8384   |         0.8384   |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser          |    -0.8282   |         0.8282   |
| logistic_regression  | num__source_catch_count_mean                          |     0.8227   |         0.8227   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser       |    -0.7829   |         0.7829   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                |     0.7635   |         0.7635   |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser             |     0.08434  |         0.08434  |
| random_forest        | num__source_loc_total_gold_minus_loser                |     0.05852  |         0.05852  |
| random_forest        | num__visual_code_bounding_box_height_gold_minus_loser |     0.04996  |         0.04996  |
| random_forest        | num__source_loc_nonempty_gold_minus_loser             |     0.04959  |         0.04959  |
| random_forest        | num__source_brace_count_gold_minus_loser              |     0.04578  |         0.04578  |
| random_forest        | num__source_method_like_count_gold_minus_loser        |     0.04329  |         0.04329  |
| random_forest        | num__visual_estimated_line_height_gold_minus_loser    |     0.03932  |         0.03932  |
| random_forest        | num__source_parenthesis_count_gold_minus_loser        |     0.03041  |         0.03041  |
| random_forest        | num__source_nesting_mean_gold_minus_loser             |     0.02952  |         0.02952  |
| random_forest        | num__visual_image_height_gold_minus_loser             |     0.02605  |         0.02605  |
| random_forest        | num__source_identifier_count_gold_minus_loser         |     0.02299  |         0.02299  |
| random_forest        | num__source_token_count_gold_minus_loser              |     0.02186  |         0.02186  |
| random_forest        | num__source_nesting_max_gold_minus_loser              |     0.02134  |         0.02134  |
| random_forest        | num__source_halstead_length_gold_minus_loser          |     0.01918  |         0.01918  |
| random_forest        | num__source_token_entropy_gold_minus_loser            |     0.01836  |         0.01836  |

## Depth-3 Decision Tree Rules

```text
|--- num__visual_aspect_ratio_gold_minus_loser <= -0.70
|   |--- num__source_api_call_count_gold_minus_loser <= -0.50
|   |   |--- class: 1
|   |--- num__source_api_call_count_gold_minus_loser >  -0.50
|   |   |--- num__source_halstead_length_mean <= 149.75
|   |   |   |--- class: 1
|   |   |--- num__source_halstead_length_mean >  149.75
|   |   |   |--- class: 1
|--- num__visual_aspect_ratio_gold_minus_loser >  -0.70
|   |--- num__source_brace_count_gold_minus_loser <= -1.50
|   |   |--- num__source_identifier_count_gold_minus_loser <= 3.50
|   |   |   |--- class: 0
|   |   |--- num__source_identifier_count_gold_minus_loser >  3.50
|   |   |   |--- class: 0
|   |--- num__source_brace_count_gold_minus_loser >  -1.50
|   |   |--- num__source_blank_line_ratio_max <= 0.19
|   |   |   |--- class: 1
|   |   |--- num__source_blank_line_ratio_max >  0.19
|   |   |   |--- class: 0

```
