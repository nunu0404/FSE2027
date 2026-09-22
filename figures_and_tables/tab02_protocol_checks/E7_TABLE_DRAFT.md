# Closed-model pilot table

| model_generation   | reasoning_effort               |   max_output_tokens | condition       |   n |   valid_n |   valid_accuracy |   effective_accuracy |   strict_swap_error |   parse_failure_rate |   rho_valid_only |
|:-------------------|:-------------------------------|--------------------:|:----------------|----:|----------:|-----------------:|---------------------:|--------------------:|---------------------:|-----------------:|
| gpt-5.4-mini       | API default/not explicitly set |                  24 | image_only      | 300 |       229 |           0.6376 |               0.4867 |              0.2367 |               0.0000 |           0.3409 |
| gpt-5.4-mini       | API default/not explicitly set |                  24 | text_plus_image | 300 |       166 |           0.6747 |               0.3733 |              0.4467 |               0.0000 |           0.2969 |
| gpt-5.4            | API default/not explicitly set |                  24 | image_only      | 300 |       249 |           0.6064 |               0.5033 |              0.1700 |               0.0000 |           0.2825 |
| gpt-5.4            | API default/not explicitly set |                  24 | text_plus_image | 300 |       220 |           0.6727 |               0.4933 |              0.2667 |               0.0000 |           0.3876 |
| gpt-5.5            | API default/not explicitly set |                  24 | image_only      | 300 |       244 |           0.6844 |               0.5567 |              0.1867 |               0.0483 |           0.4202 |
| gpt-5.5            | API default/not explicitly set |                  24 | text_plus_image | 300 |       224 |           0.6696 |               0.5000 |              0.2533 |               0.0367 |           0.3664 |
| gpt-5.4-2026-03-05 | low                            |                4096 | image_only      | 300 |       256 |           0.6094 |               0.5200 |              0.1467 |               0.0000 |           0.2967 |
| gpt-5.4-2026-03-05 | low                            |                4096 | text_plus_image | 300 |       219 |           0.6621 |               0.4833 |              0.2700 |               0.0000 |           0.3416 |
| gpt-5.4-2026-03-05 | high                           |                4096 | image_only      | 300 |       266 |           0.6353 |               0.5633 |              0.1133 |               0.0000 |           0.3286 |
| gpt-5.4-2026-03-05 | high                           |                4096 | text_plus_image | 300 |       213 |           0.6948 |               0.4933 |              0.2900 |               0.0000 |           0.4101 |

**Caption.** Protocol-check results on 300 Java pairs (100 per legacy difficulty stratum). The rendering grid was not applied, so rows are not a controlled model-scale comparison. Qwen2.5-VL-32B additionally required forced single-character output.

GPT-5.4 high/4096 cross-modality check: swap error 11.33% -> 29.00%; exact McNemar p=1.18e-07 (discordant n=101).
