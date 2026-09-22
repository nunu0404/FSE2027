# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.7945 |                 0.6929 | 1130 |          0.3832 |
| logistic_regression  |   0.7842 |                 0.7264 | 1130 |          0.3832 |
| random_forest        |   0.7473 |                 0.6831 | 1130 |          0.3832 |
| decision_tree_depth3 |   0.6643 |                 0.6192 | 1130 |          0.3832 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.3156  |          0.3156  |
| decision_tree_depth3 | num__source_halstead_volume_mean                          |      0.1837  |          0.1837  |
| decision_tree_depth3 | num__source_line_length_std_gold_minus_loser              |      0.1445  |          0.1445  |
| decision_tree_depth3 | num__source_keyword_count_gold_minus_loser                |      0.1266  |          0.1266  |
| decision_tree_depth3 | num__visual_contrast_score_gold_minus_loser               |      0.1185  |          0.1185  |
| decision_tree_depth3 | num__source_literal_count_gold_minus_loser                |      0.08669 |          0.08669 |
| decision_tree_depth3 | num__source_keyword_count_max                             |      0.02448 |          0.02448 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.04951 |          0.04951 |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.04678 |          0.04678 |
| gradient_boosting    | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.04296 |          0.04296 |
| gradient_boosting    | num__source_avg_token_length_max                          |      0.04123 |          0.04123 |
| gradient_boosting    | num__visual_contrast_score_gold_minus_loser               |      0.03027 |          0.03027 |
| gradient_boosting    | num__visual_contrast_score_max                            |      0.02807 |          0.02807 |
| gradient_boosting    | num__source_keyword_count_gold_minus_loser                |      0.02788 |          0.02788 |
| gradient_boosting    | num__source_halstead_volume_mean                          |      0.0273  |          0.0273  |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.02552 |          0.02552 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.02175 |          0.02175 |
| gradient_boosting    | num__source_blank_line_ratio_mean                         |      0.01977 |          0.01977 |
| gradient_boosting    | num__visual_clutter_score_max                             |      0.01776 |          0.01776 |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser                 |      0.01613 |          0.01613 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.01602 |          0.01602 |
| gradient_boosting    | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.0157  |          0.0157  |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.8     |          1.8     |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -1.745   |          1.745   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      1.254   |          1.254   |
| logistic_regression  | num__source_identifier_count_gold_minus_loser             |     -1.046   |          1.046   |
| logistic_regression  | num__source_method_like_count_gold_minus_loser            |      1.021   |          1.021   |
| logistic_regression  | num__source_operator_count_gold_minus_loser               |      0.9514  |          0.9514  |
| logistic_regression  | num__source_literal_count_gold_minus_loser                |     -0.7938  |          0.7938  |
| logistic_regression  | num__source_indent_max_mean                               |     -0.7838  |          0.7838  |
| logistic_regression  | num__source_avg_token_length_diff_abs                     |      0.7611  |          0.7611  |
| logistic_regression  | num__source_avg_token_length_mean                         |     -0.7071  |          0.7071  |
| logistic_regression  | num__source_brace_count_gold_minus_loser                  |     -0.6711  |          0.6711  |
| logistic_regression  | num__source_halstead_vocabulary_mean                      |     -0.6413  |          0.6413  |
| logistic_regression  | num__source_identifier_unique_count_mean                  |      0.6392  |          0.6392  |
| logistic_regression  | num__source_operator_count_diff_abs                       |     -0.6264  |          0.6264  |
| logistic_regression  | num__source_halstead_distinct_operands_mean               |     -0.6259  |          0.6259  |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.02539 |          0.02539 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.02378 |          0.02378 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.02224 |          0.02224 |
| random_forest        | num__source_parenthesis_count_gold_minus_loser            |      0.02042 |          0.02042 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.01985 |          0.01985 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.01961 |          0.01961 |
| random_forest        | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.01579 |          0.01579 |
| random_forest        | num__source_identifier_unique_count_gold_minus_loser      |      0.01552 |          0.01552 |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.01522 |          0.01522 |
| random_forest        | num__visual_aspect_ratio_mean                             |      0.01418 |          0.01418 |
| random_forest        | num__source_halstead_length_mean                          |      0.01394 |          0.01394 |
| random_forest        | num__visual_aspect_ratio_max                              |      0.01392 |          0.01392 |
| random_forest        | num__source_halstead_volume_mean                          |      0.01378 |          0.01378 |
| random_forest        | num__source_literal_count_gold_minus_loser                |      0.0133  |          0.0133  |
| random_forest        | num__source_line_length_std_gold_minus_loser              |      0.01243 |          0.01243 |

## Depth-3 Decision Tree Rules

```text
|--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |--- num__source_keyword_count_gold_minus_loser <= -4.50
|   |   |--- num__source_keyword_count_max <= 19.50
|   |   |   |--- class: 0
|   |   |--- num__source_keyword_count_max >  19.50
|   |   |   |--- class: 0
|   |--- num__source_keyword_count_gold_minus_loser >  -4.50
|   |   |--- num__source_literal_count_gold_minus_loser <= 0.50
|   |   |   |--- class: 1
|   |   |--- num__source_literal_count_gold_minus_loser >  0.50
|   |   |   |--- class: 0
|--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |--- num__source_halstead_volume_mean <= 861.90
|   |   |--- num__source_line_length_std_gold_minus_loser <= -1.39
|   |   |   |--- class: 0
|   |   |--- num__source_line_length_std_gold_minus_loser >  -1.39
|   |   |   |--- class: 1
|   |--- num__source_halstead_volume_mean >  861.90
|   |   |--- num__visual_contrast_score_gold_minus_loser <= 1.39
|   |   |   |--- class: 0
|   |   |--- num__visual_contrast_score_gold_minus_loser >  1.39
|   |   |   |--- class: 1

```
