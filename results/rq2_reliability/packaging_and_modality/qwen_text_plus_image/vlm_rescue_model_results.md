# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| gradient_boosting    |   0.7992 |                 0.7259 | 1130 |          0.4372 |
| logistic_regression  |   0.7936 |                 0.7236 | 1130 |          0.4372 |
| random_forest        |   0.7667 |                 0.6808 | 1130 |          0.4372 |
| decision_tree_depth3 |   0.7022 |                 0.6523 | 1130 |          0.4372 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__visual_aspect_ratio_gold_minus_loser                 |      0.4078  |          0.4078  |
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.1916  |          0.1916  |
| decision_tree_depth3 | num__source_avg_identifier_length_gold_minus_loser        |      0.1151  |          0.1151  |
| decision_tree_depth3 | num__source_token_count_max                               |      0.1082  |          0.1082  |
| decision_tree_depth3 | num__source_indent_mean_gold_minus_loser                  |      0.0817  |          0.0817  |
| decision_tree_depth3 | num__source_literal_count_gold_minus_loser                |      0.06069 |          0.06069 |
| decision_tree_depth3 | num__source_parenthesis_count_mean                        |      0.03482 |          0.03482 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser                 |      0.1031  |          0.1031  |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.06577 |          0.06577 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.05994 |          0.05994 |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.04255 |          0.04255 |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.02282 |          0.02282 |
| gradient_boosting    | num__source_avg_identifier_length_mean                    |      0.01999 |          0.01999 |
| gradient_boosting    | num__visual_non_background_pixel_density_max              |      0.01995 |          0.01995 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.01884 |          0.01884 |
| gradient_boosting    | num__source_api_call_count_gold_minus_loser               |      0.01867 |          0.01867 |
| gradient_boosting    | num__visual_clutter_score_mean                            |      0.01713 |          0.01713 |
| gradient_boosting    | num__visual_non_background_pixel_density_gold_minus_loser |      0.01686 |          0.01686 |
| gradient_boosting    | num__source_token_count_max                               |      0.01651 |          0.01651 |
| gradient_boosting    | num__source_halstead_distinct_operands_max                |      0.01622 |          0.01622 |
| gradient_boosting    | num__source_nesting_mean_max                              |      0.01583 |          0.01583 |
| gradient_boosting    | num__source_avg_line_length_gold_minus_loser              |      0.01582 |          0.01582 |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.262   |          1.262   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.13    |          1.13    |
| logistic_regression  | num__source_nesting_max_gold_minus_loser                  |     -1.126   |          1.126   |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      1.102   |          1.102   |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -1.031   |          1.031   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.015   |          1.015   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      0.9529  |          0.9529  |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -0.9395  |          0.9395  |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser              |     -0.8304  |          0.8304  |
| logistic_regression  | num__source_brace_count_mean                              |     -0.7925  |          0.7925  |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      0.7869  |          0.7869  |
| logistic_regression  | num__source_literal_count_diff_abs                        |      0.6907  |          0.6907  |
| logistic_regression  | num__source_indent_max_gold_minus_loser                   |      0.6731  |          0.6731  |
| logistic_regression  | num__source_loc_total_diff_abs                            |      0.6585  |          0.6585  |
| logistic_regression  | num__source_literal_count_mean                            |     -0.6318  |          0.6318  |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.0394  |          0.0394  |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.03258 |          0.03258 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.02751 |          0.02751 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.0243  |          0.0243  |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.02404 |          0.02404 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.02384 |          0.02384 |
| random_forest        | num__source_parenthesis_count_gold_minus_loser            |      0.01932 |          0.01932 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01438 |          0.01438 |
| random_forest        | num__source_loc_total_gold_minus_loser                    |      0.01317 |          0.01317 |
| random_forest        | num__source_loc_nonempty_gold_minus_loser                 |      0.01222 |          0.01222 |
| random_forest        | num__visual_horizontal_whitespace_ratio_mean              |      0.0118  |          0.0118  |
| random_forest        | num__source_identifier_count_gold_minus_loser             |      0.01178 |          0.01178 |
| random_forest        | num__visual_aspect_ratio_mean                             |      0.01176 |          0.01176 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.01096 |          0.01096 |
| random_forest        | num__visual_code_bounding_box_height_gold_minus_loser     |      0.0107  |          0.0107  |

## Depth-3 Decision Tree Rules

```text
|--- num__visual_aspect_ratio_gold_minus_loser <= -0.63
|   |--- num__source_token_count_max <= 253.50
|   |   |--- num__source_indent_mean_gold_minus_loser <= -1.83
|   |   |   |--- class: 1
|   |   |--- num__source_indent_mean_gold_minus_loser >  -1.83
|   |   |   |--- class: 1
|   |--- num__source_token_count_max >  253.50
|   |   |--- num__source_parenthesis_count_mean <= 25.75
|   |   |   |--- class: 0
|   |   |--- num__source_parenthesis_count_mean >  25.75
|   |   |   |--- class: 1
|--- num__visual_aspect_ratio_gold_minus_loser >  -0.63
|   |--- num__source_comment_line_count_gold_minus_loser <= -0.50
|   |   |--- num__source_literal_count_gold_minus_loser <= -4.50
|   |   |   |--- class: 0
|   |   |--- num__source_literal_count_gold_minus_loser >  -4.50
|   |   |   |--- class: 0
|   |--- num__source_comment_line_count_gold_minus_loser >  -0.50
|   |   |--- num__source_avg_identifier_length_gold_minus_loser <= -1.08
|   |   |   |--- class: 0
|   |   |--- num__source_avg_identifier_length_gold_minus_loser >  -1.08
|   |   |   |--- class: 1

```
