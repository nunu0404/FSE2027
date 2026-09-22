# Strict-swap error by human-score gap

## Inventory

|   cases |   pair_run_rows | families                                                                                                                                                         | difficulty_definition                                                                                                | excluded                                                                                                                                                                                                                                                                                                    |
|--------:|----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      50 |           46560 | {'gpt54_reasoning_budget': 8, 'max_token_300': 16, 'ocr_text_llm_full3000': 4, 'openai_standard_300': 12, 'primary_clean_full3000': 8, 'protocol_sanity_300': 2} | {'easy': 'abs human z gap >= 1.0', 'medium': '0.5 <= abs human z gap < 1.0', 'hard': '0.2 <= abs human z gap < 0.5'} | ['pre-clean/stale/corrupted-render runs', 'smoke and duplicate result-tree copies', 'failed partial GPT-5.5 run (422 rows); fixed complete run used', 'presentation perturbation cases without score-gap difficulty strata', 'classical ML swap error, structurally zero by score-difference construction'] |

## Primary clean 3,000-pair cases

| family                 | model                       | case                             |   pairs | source_file                                                                                                                                                      |   hard_swap_error |   medium_swap_error |   easy_swap_error |   easy_minus_hard | expected_monotonic_hard_gt_medium_gt_easy   |
|:-----------------------|:----------------------------|:---------------------------------|--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|--------------------:|------------------:|------------------:|:--------------------------------------------|
| primary_clean_full3000 | OpenGVLab/InternVL3-8B      | combined_labeled_image_only      |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__combined_labeled_image_only__promptB__seed42.jsonl           |          0.590000 |            0.617000 |          0.595000 |          0.005000 | False                                       |
| primary_clean_full3000 | OpenGVLab/InternVL3-8B      | combined_labeled_text_plus_image |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__combined_labeled_text_plus_image__promptB__seed42.jsonl      |          0.374000 |            0.367000 |          0.377000 |          0.003000 | False                                       |
| primary_clean_full3000 | OpenGVLab/InternVL3-8B      | image_only                       |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__image_only__promptB__seed42.jsonl                            |          0.523000 |            0.576000 |          0.545000 |          0.022000 | False                                       |
| primary_clean_full3000 | OpenGVLab/InternVL3-8B      | text_plus_image                  |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/OpenGVLab__InternVL3-8B__text_plus_image__promptB__seed42.jsonl                       |          0.290000 |            0.294000 |          0.275000 |         -0.015000 | False                                       |
| primary_clean_full3000 | Qwen/Qwen2.5-VL-7B-Instruct | combined_labeled_image_only      |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__combined_labeled_image_only__promptB__seed42.jsonl      |          0.510000 |            0.545000 |          0.487000 |         -0.023000 | False                                       |
| primary_clean_full3000 | Qwen/Qwen2.5-VL-7B-Instruct | combined_labeled_text_plus_image |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__combined_labeled_text_plus_image__promptB__seed42.jsonl |          0.352000 |            0.296000 |          0.277000 |         -0.075000 | True                                        |
| primary_clean_full3000 | Qwen/Qwen2.5-VL-7B-Instruct | image_only                       |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__image_only__promptB__seed42.jsonl                       |          0.390000 |            0.385000 |          0.358000 |         -0.032000 | True                                        |
| primary_clean_full3000 | Qwen/Qwen2.5-VL-7B-Instruct | text_plus_image                  |    3000 | experiments/rq0_viability/outputs/vlm_judges/full_factorial_clean_20260703/Qwen__Qwen2.5-VL-7B-Instruct__text_plus_image__promptB__seed42.jsonl                  |          0.244000 |            0.256000 |          0.246000 |          0.002000 | False                                       |

## Family summary

| family                 |   cases |   mean_hard_swap |   mean_medium_swap |   mean_easy_swap |   hard_greater_easy_cases |   expected_monotonic_cases |
|:-----------------------|--------:|-----------------:|-------------------:|-----------------:|--------------------------:|---------------------------:|
| gpt54_reasoning_budget |       8 |         0.309167 |           0.296667 |         0.326667 |                         3 |                          0 |
| max_token_300          |      16 |         0.436250 |           0.405000 |         0.331250 |                        14 |                         10 |
| ocr_text_llm_full3000  |       4 |         0.511750 |           0.545250 |         0.550000 |                         0 |                          0 |
| openai_standard_300    |      12 |         0.260000 |           0.260000 |         0.258333 |                         6 |                          1 |
| primary_clean_full3000 |       8 |         0.409125 |           0.417000 |         0.395000 |                         4 |                          2 |
| protocol_sanity_300    |       2 |         0.430000 |           0.385000 |         0.385000 |                         2 |                          1 |

## Primary pooled models

| analysis                                                 | contrast                         |   coefficient |   cluster_robust_se |   odds_ratio |   or_ci_low |   or_ci_high |   p_value |   observations |   pair_clusters |
|:---------------------------------------------------------|:---------------------------------|--------------:|--------------------:|-------------:|------------:|-------------:|----------:|---------------:|----------------:|
| continuous_gap_condition_FE_pair_clustered               | +1 abs human z gap               |     -0.073429 |            0.024627 |     0.929202 |    0.885417 |     0.975153 |  0.002867 |          24000 |            3000 |
| difficulty_condition_FE_pair_clustered                   | medium_vs_hard                   |      0.034649 |            0.042219 |     1.035256 |    0.953038 |     1.124566 |  0.411829 |          24000 |            3000 |
| difficulty_condition_FE_pair_clustered                   | easy_vs_hard                     |     -0.062652 |            0.043558 |     0.939270 |    0.862410 |     1.022981 |  0.150335 |          24000 |            3000 |
| within_difficulty_continuous_condition_FE_pair_clustered | +1 abs human z gap within hard   |     -0.023974 |            0.335572 |     0.976312 |    0.505765 |     1.884638 |  0.943047 |           8000 |            1000 |
| within_difficulty_continuous_condition_FE_pair_clustered | +1 abs human z gap within medium |     -0.596791 |            0.216543 |     0.550576 |    0.360160 |     0.841664 |  0.005851 |           8000 |            1000 |
| within_difficulty_continuous_condition_FE_pair_clustered | +1 abs human z gap within easy   |     -0.109474 |            0.048921 |     0.896306 |    0.814357 |     0.986501 |  0.025235 |           8000 |            1000 |

## Primary gap deciles

|   gap_decile |   unique_pairs |   pair_run_rows |   score_gap_min |   score_gap_max |   score_gap_mean |   strict_swap_error |
|-------------:|---------------:|----------------:|----------------:|----------------:|-----------------:|--------------------:|
|     1.000000 |     300.000000 |     2400.000000 |        0.200215 |        0.305487 |         0.250824 |            0.387500 |
|     2.000000 |     300.000000 |     2400.000000 |        0.305506 |        0.375703 |         0.344543 |            0.449583 |
|     3.000000 |     301.000000 |     2408.000000 |        0.375711 |        0.474057 |         0.423656 |            0.396595 |
|     4.000000 |     299.000000 |     2392.000000 |        0.474057 |        0.574790 |         0.520984 |            0.441472 |
|     5.000000 |     307.000000 |     2456.000000 |        0.576905 |        0.726028 |         0.664252 |            0.425489 |
|     6.000000 |     293.000000 |     2344.000000 |        0.726028 |        0.901498 |         0.811415 |            0.385239 |
|     7.000000 |     300.000000 |     2400.000000 |        0.902346 |        1.092964 |         0.976880 |            0.414583 |
|     8.000000 |     300.000000 |     2400.000000 |        1.093527 |        1.535173 |         1.313825 |            0.394167 |
|     9.000000 |     300.000000 |     2400.000000 |        1.535710 |        2.136082 |         1.823937 |            0.409583 |
|    10.000000 |     300.000000 |     2400.000000 |        2.137528 |        4.279799 |         2.670859 |            0.365417 |

## Easy-stratum gap quartiles

|   easy_gap_quartile |   unique_pairs |   pair_run_rows |   score_gap_min |   score_gap_max |   score_gap_mean |   strict_swap_error |
|--------------------:|---------------:|----------------:|----------------:|----------------:|-----------------:|--------------------:|
|            1.000000 |     250.000000 |     2000.000000 |        1.000064 |        1.305431 |         1.151623 |            0.411000 |
|            2.000000 |     250.000000 |     2000.000000 |        1.305673 |        1.714442 |         1.499854 |            0.410500 |
|            3.000000 |     250.000000 |     2000.000000 |        1.717579 |        2.257351 |         1.974748 |            0.397500 |
|            4.000000 |     250.000000 |     2000.000000 |        2.257464 |        4.279799 |         2.767506 |            0.361000 |

## Snippet-cluster bootstrap

| estimand                           |   point_estimate |   bootstrap_mean |    ci_low |   ci_high |   replicates |
|:-----------------------------------|-----------------:|-----------------:|----------:|----------:|-------------:|
| easy_minus_hard_swap               |        -0.014125 |        -0.014245 | -0.052184 |  0.023824 |        10000 |
| linear_probability_slope_per_abs_z |        -0.016498 |        -0.016601 | -0.040201 |  0.006565 |        10000 |
