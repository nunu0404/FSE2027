# Vlm Rescue Valid Only Model Results

Subset: RF-wrong and VLM-valid subset; target=VLM correct vs VLM wrong

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |   n |   positive_rate |
|:---------------------|---------:|-----------------------:|----:|----------------:|
| gradient_boosting    |   0.8800 |                 0.8028 | 856 |          0.5771 |
| logistic_regression  |   0.8480 |                 0.7669 | 856 |          0.5771 |
| random_forest        |   0.8266 |                 0.7471 | 856 |          0.5771 |
| decision_tree_depth3 |   0.7504 |                 0.7137 | 856 |          0.5771 |

## Top Feature Importances

| model                | feature                                                   |   importance |   abs_importance |
|:---------------------|:----------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__visual_aspect_ratio_gold_minus_loser                 |      0.45    |          0.45    |
| decision_tree_depth3 | num__source_comment_line_count_gold_minus_loser           |      0.1805  |          0.1805  |
| decision_tree_depth3 | num__source_avg_identifier_length_gold_minus_loser        |      0.1212  |          0.1212  |
| decision_tree_depth3 | num__visual_clutter_score_mean                            |      0.1177  |          0.1177  |
| decision_tree_depth3 | num__source_loc_total_max                                 |      0.0827  |          0.0827  |
| decision_tree_depth3 | num__source_literal_count_gold_minus_loser                |      0.04792 |          0.04792 |
| decision_tree_depth3 | num__abs_human_score_gap                                  |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                        |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                                 |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                    |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                         |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                              |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_mean                             |      0       |          0       |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser                 |      0.1333  |          0.1333  |
| gradient_boosting    | num__source_comment_line_count_gold_minus_loser           |      0.06738 |          0.06738 |
| gradient_boosting    | num__source_parenthesis_count_gold_minus_loser            |      0.04101 |          0.04101 |
| gradient_boosting    | num__source_indent_mean_gold_minus_loser                  |      0.03768 |          0.03768 |
| gradient_boosting    | num__source_avg_identifier_length_gold_minus_loser        |      0.03096 |          0.03096 |
| gradient_boosting    | num__visual_clutter_score_mean                            |      0.02905 |          0.02905 |
| gradient_boosting    | num__source_avg_token_length_gold_minus_loser             |      0.02836 |          0.02836 |
| gradient_boosting    | num__source_literal_count_gold_minus_loser                |      0.02713 |          0.02713 |
| gradient_boosting    | num__visual_contrast_score_gold_minus_loser               |      0.01998 |          0.01998 |
| gradient_boosting    | num__source_blank_line_count_mean                         |      0.01978 |          0.01978 |
| gradient_boosting    | num__source_cyclomatic_estimate_mean                      |      0.01896 |          0.01896 |
| gradient_boosting    | num__source_cyclomatic_estimate_max                       |      0.01873 |          0.01873 |
| gradient_boosting    | num__visual_non_background_pixel_density_gold_minus_loser |      0.01809 |          0.01809 |
| gradient_boosting    | num__source_avg_identifier_length_mean                    |      0.01747 |          0.01747 |
| gradient_boosting    | num__source_comment_line_ratio_gold_minus_loser           |      0.0166  |          0.0166  |
| logistic_regression  | num__source_nesting_max_gold_minus_loser                  |     -1.518   |          1.518   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser            |      1.453   |          1.453   |
| logistic_regression  | num__source_comment_line_ratio_gold_minus_loser           |      1.435   |          1.435   |
| logistic_regression  | num__source_loc_nonempty_gold_minus_loser                 |      1.361   |          1.361   |
| logistic_regression  | num__source_indent_mean_gold_minus_loser                  |     -1.144   |          1.144   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser     |      1.121   |          1.121   |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser                 |      1.115   |          1.115   |
| logistic_regression  | num__source_avg_line_length_gold_minus_loser              |     -1.103   |          1.103   |
| logistic_regression  | num__source_loc_total_diff_abs                            |      1.073   |          1.073   |
| logistic_regression  | num__source_loc_total_gold_minus_loser                    |      1.066   |          1.066   |
| logistic_regression  | num__source_halstead_volume_gold_minus_loser              |     -0.9299  |          0.9299  |
| logistic_regression  | num__source_identifier_count_diff_abs                     |      0.9098  |          0.9098  |
| logistic_regression  | num__source_keyword_count_gold_minus_loser                |      0.8751  |          0.8751  |
| logistic_regression  | num__source_catch_count_mean                              |      0.8603  |          0.8603  |
| logistic_regression  | num__source_literal_count_gold_minus_loser                |     -0.7956  |          0.7956  |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser                 |      0.0462  |          0.0462  |
| random_forest        | num__source_comment_line_count_gold_minus_loser           |      0.03244 |          0.03244 |
| random_forest        | num__source_comment_line_ratio_gold_minus_loser           |      0.02867 |          0.02867 |
| random_forest        | num__source_parenthesis_count_gold_minus_loser            |      0.02609 |          0.02609 |
| random_forest        | num__source_avg_token_length_gold_minus_loser             |      0.02498 |          0.02498 |
| random_forest        | num__visual_non_background_pixel_density_gold_minus_loser |      0.02245 |          0.02245 |
| random_forest        | num__visual_horizontal_whitespace_ratio_gold_minus_loser  |      0.01994 |          0.01994 |
| random_forest        | num__visual_non_background_pixel_density_max              |      0.01673 |          0.01673 |
| random_forest        | num__source_avg_identifier_length_gold_minus_loser        |      0.01662 |          0.01662 |
| random_forest        | num__source_loc_nonempty_gold_minus_loser                 |      0.01621 |          0.01621 |
| random_forest        | num__source_identifier_count_gold_minus_loser             |      0.01464 |          0.01464 |
| random_forest        | num__source_indent_mean_gold_minus_loser                  |      0.01363 |          0.01363 |
| random_forest        | num__visual_estimated_line_height_gold_minus_loser        |      0.01328 |          0.01328 |
| random_forest        | num__source_loc_total_gold_minus_loser                    |      0.01264 |          0.01264 |
| random_forest        | num__visual_aspect_ratio_mean                             |      0.0121  |          0.0121  |

## Depth-3 Decision Tree Rules

```text
|--- num__visual_aspect_ratio_gold_minus_loser <= -0.54
|   |--- num__visual_clutter_score_mean <= 41.35
|   |   |--- class: 0
|   |--- num__visual_clutter_score_mean >  41.35
|   |   |--- num__source_loc_total_max <= 50.50
|   |   |   |--- class: 1
|   |   |--- num__source_loc_total_max >  50.50
|   |   |   |--- class: 0
|--- num__visual_aspect_ratio_gold_minus_loser >  -0.54
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
