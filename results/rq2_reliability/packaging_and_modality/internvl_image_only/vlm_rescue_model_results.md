# Vlm Rescue Model Results

Subset: RF-wrong subset; target=VLM strict-swap valid and correct vs wrong or invalid

## Cross-validated Metrics

| model                |   cv_auc |   cv_balanced_accuracy |    n |   positive_rate |
|:---------------------|---------:|-----------------------:|-----:|----------------:|
| logistic_regression  |   0.8973 |                 0.8123 | 1130 |          0.3053 |
| gradient_boosting    |   0.8898 |                 0.7819 | 1130 |          0.3053 |
| random_forest        |   0.8754 |                 0.7948 | 1130 |          0.3053 |
| decision_tree_depth3 |   0.8325 |                 0.7583 | 1130 |          0.3053 |

## Top Feature Importances

| model                | feature                                               |   importance |   abs_importance |
|:---------------------|:------------------------------------------------------|-------------:|-----------------:|
| decision_tree_depth3 | num__visual_aspect_ratio_gold_minus_loser             |      0.6019  |          0.6019  |
| decision_tree_depth3 | num__source_loc_total_max                             |      0.1334  |          0.1334  |
| decision_tree_depth3 | num__visual_aspect_ratio_mean                         |      0.1296  |          0.1296  |
| decision_tree_depth3 | num__source_blank_line_ratio_gold_minus_loser         |      0.07299 |          0.07299 |
| decision_tree_depth3 | num__source_line_length_std_mean                      |      0.0417  |          0.0417  |
| decision_tree_depth3 | num__source_avg_line_length_gold_minus_loser          |      0.02036 |          0.02036 |
| decision_tree_depth3 | num__abs_human_score_gap                              |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_abs                                    |      0       |          0       |
| decision_tree_depth3 | num__rf_margin_percentile                             |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_diff_abs                        |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_mean                            |      0       |          0       |
| decision_tree_depth3 | num__source_loc_total_gold_minus_loser                |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_diff_abs                     |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_max                          |      0       |          0       |
| decision_tree_depth3 | num__source_loc_nonempty_mean                         |      0       |          0       |
| gradient_boosting    | num__visual_aspect_ratio_gold_minus_loser             |      0.2869  |          0.2869  |
| gradient_boosting    | num__visual_aspect_ratio_mean                         |      0.07813 |          0.07813 |
| gradient_boosting    | num__source_method_like_count_gold_minus_loser        |      0.05011 |          0.05011 |
| gradient_boosting    | num__abs_human_score_gap                              |      0.03596 |          0.03596 |
| gradient_boosting    | num__source_indent_std_gold_minus_loser               |      0.03502 |          0.03502 |
| gradient_boosting    | num__visual_non_background_pixel_density_max          |      0.03377 |          0.03377 |
| gradient_boosting    | num__source_avg_token_length_mean                     |      0.0277  |          0.0277  |
| gradient_boosting    | num__source_loc_total_max                             |      0.02177 |          0.02177 |
| gradient_boosting    | num__source_blank_line_ratio_gold_minus_loser         |      0.01608 |          0.01608 |
| gradient_boosting    | num__visual_code_bounding_box_height_mean             |      0.01417 |          0.01417 |
| gradient_boosting    | num__source_avg_nonempty_line_length_mean             |      0.01282 |          0.01282 |
| gradient_boosting    | num__visual_line_length_mean_diff_abs                 |      0.01129 |          0.01129 |
| gradient_boosting    | num__source_identifier_count_gold_minus_loser         |      0.01105 |          0.01105 |
| gradient_boosting    | num__visual_image_width_mean                          |      0.0101  |          0.0101  |
| gradient_boosting    | num__source_nesting_mean_gold_minus_loser             |      0.00977 |          0.00977 |
| logistic_regression  | num__source_blank_line_ratio_gold_minus_loser         |     -1.72    |          1.72    |
| logistic_regression  | num__source_blank_line_count_gold_minus_loser         |      1.695   |          1.695   |
| logistic_regression  | num__source_brace_count_gold_minus_loser              |      1.142   |          1.142   |
| logistic_regression  | num__source_token_entropy_gold_minus_loser            |      1.036   |          1.036   |
| logistic_regression  | num__source_identifier_count_mean                     |     -1.033   |          1.033   |
| logistic_regression  | num__source_avg_nonempty_line_length_gold_minus_loser |      1.021   |          1.021   |
| logistic_regression  | num__visual_aspect_ratio_gold_minus_loser             |     -1.021   |          1.021   |
| logistic_regression  | num__abs_human_score_gap                              |      0.9416  |          0.9416  |
| logistic_regression  | num__source_identifier_unique_count_mean              |      0.8971  |          0.8971  |
| logistic_regression  | num__source_cyclomatic_estimate_gold_minus_loser      |      0.8844  |          0.8844  |
| logistic_regression  | num__source_nesting_mean_gold_minus_loser             |      0.8787  |          0.8787  |
| logistic_regression  | num__source_method_like_count_mean                    |     -0.865   |          0.865   |
| logistic_regression  | num__source_parenthesis_count_gold_minus_loser        |      0.8453  |          0.8453  |
| logistic_regression  | num__source_catch_count_mean                          |      0.8141  |          0.8141  |
| logistic_regression  | num__source_halstead_volume_diff_abs                  |      0.7533  |          0.7533  |
| random_forest        | num__visual_aspect_ratio_gold_minus_loser             |      0.06746 |          0.06746 |
| random_forest        | num__visual_aspect_ratio_max                          |      0.04615 |          0.04615 |
| random_forest        | num__visual_aspect_ratio_mean                         |      0.04504 |          0.04504 |
| random_forest        | num__source_loc_total_gold_minus_loser                |      0.03654 |          0.03654 |
| random_forest        | num__source_method_like_count_gold_minus_loser        |      0.03244 |          0.03244 |
| random_forest        | num__visual_aspect_ratio_diff_abs                     |      0.03216 |          0.03216 |
| random_forest        | num__source_brace_count_gold_minus_loser              |      0.03125 |          0.03125 |
| random_forest        | num__visual_code_bounding_box_height_gold_minus_loser |      0.0307  |          0.0307  |
| random_forest        | num__source_loc_nonempty_gold_minus_loser             |      0.03032 |          0.03032 |
| random_forest        | num__visual_estimated_line_height_gold_minus_loser    |      0.03017 |          0.03017 |
| random_forest        | num__visual_image_height_mean                         |      0.02264 |          0.02264 |
| random_forest        | num__source_identifier_count_gold_minus_loser         |      0.02185 |          0.02185 |
| random_forest        | num__source_loc_nonempty_mean                         |      0.02107 |          0.02107 |
| random_forest        | num__source_halstead_volume_gold_minus_loser          |      0.01902 |          0.01902 |
| random_forest        | num__visual_code_bounding_box_height_mean             |      0.01891 |          0.01891 |

## Depth-3 Decision Tree Rules

```text
|--- num__visual_aspect_ratio_gold_minus_loser <= -0.72
|   |--- num__visual_aspect_ratio_mean <= 1.52
|   |   |--- num__source_line_length_std_mean <= 23.07
|   |   |   |--- class: 1
|   |   |--- num__source_line_length_std_mean >  23.07
|   |   |   |--- class: 0
|   |--- num__visual_aspect_ratio_mean >  1.52
|   |   |--- num__visual_aspect_ratio_mean <= 2.05
|   |   |   |--- class: 1
|   |   |--- num__visual_aspect_ratio_mean >  2.05
|   |   |   |--- class: 1
|--- num__visual_aspect_ratio_gold_minus_loser >  -0.72
|   |--- num__source_loc_total_max <= 30.50
|   |   |--- num__source_blank_line_ratio_gold_minus_loser <= 0.11
|   |   |   |--- class: 1
|   |   |--- num__source_blank_line_ratio_gold_minus_loser >  0.11
|   |   |   |--- class: 0
|   |--- num__source_loc_total_max >  30.50
|   |   |--- num__source_avg_line_length_gold_minus_loser <= -9.88
|   |   |   |--- class: 0
|   |   |--- num__source_avg_line_length_gold_minus_loser >  -9.88
|   |   |   |--- class: 0

```
