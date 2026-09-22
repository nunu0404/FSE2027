# RQ3 Image-Only Results

## Method

The completed design contains 300 bases (100 each in Java, Python, and CUDA), four semantic-integrity x visual-quality cells per base, and all six unordered contrasts. Both pinned VLMs judged 1,800 contrasts in AB and BA order, producing 3,600 calls per model and 7,200 calls total. Inputs were two ordered rendered PNGs plus frozen Prompt B; source text was not supplied. Inference used BF16, greedy decoding, temperature 0, inactive top-p 1.0, max 24 generated tokens, and seed 42.

There is no independent human pairwise gold for generated variants. The outcome is therefore target preference, not accuracy. A pair is strict-valid only when AB and BA select the same underlying variant. `target_preference_valid` conditions on valid pairs; `effective_target_preference` divides strict-valid target selections by all pairs; `strict_swap_error` is the invalid fraction. The target is the clean variant when semantics are equal and the semantics-preserving variant when semantic integrity differs.

## Six Contrasts

| model                       | variant_i       | variant_j       |   n_pairs |   valid_pairs |   target_selected_valid_pairs | target_preference_valid   | effective_target_preference   | strict_swap_error   |
|:----------------------------|:----------------|:----------------|----------:|--------------:|------------------------------:|:--------------------------|:------------------------------|:--------------------|
| OpenGVLab/InternVL3-8B      | golden          | beautiful_trash |       300 |           275 |                           275 | 100.00%                   | 91.67%                        | 8.33%               |
| OpenGVLab/InternVL3-8B      | golden          | ugly_trash      |       300 |           297 |                           297 | 100.00%                   | 99.00%                        | 1.00%               |
| OpenGVLab/InternVL3-8B      | ugly_gold       | ugly_trash      |       300 |           294 |                           293 | 99.66%                    | 97.67%                        | 2.00%               |
| OpenGVLab/InternVL3-8B      | ugly_gold       | beautiful_trash |       300 |           248 |                           246 | 99.19%                    | 82.00%                        | 17.33%              |
| OpenGVLab/InternVL3-8B      | golden          | ugly_gold       |       300 |           181 |                           172 | 95.03%                    | 57.33%                        | 39.67%              |
| OpenGVLab/InternVL3-8B      | beautiful_trash | ugly_trash      |       300 |           190 |                           181 | 95.26%                    | 60.33%                        | 36.67%              |
| Qwen/Qwen2.5-VL-7B-Instruct | golden          | beautiful_trash |       300 |           295 |                           295 | 100.00%                   | 98.33%                        | 1.67%               |
| Qwen/Qwen2.5-VL-7B-Instruct | golden          | ugly_trash      |       300 |           293 |                           293 | 100.00%                   | 97.67%                        | 2.33%               |
| Qwen/Qwen2.5-VL-7B-Instruct | ugly_gold       | ugly_trash      |       300 |           282 |                           282 | 100.00%                   | 94.00%                        | 6.00%               |
| Qwen/Qwen2.5-VL-7B-Instruct | ugly_gold       | beautiful_trash |       300 |           275 |                           275 | 100.00%                   | 91.67%                        | 8.33%               |
| Qwen/Qwen2.5-VL-7B-Instruct | golden          | ugly_gold       |       300 |           170 |                           168 | 98.82%                    | 56.00%                        | 43.33%              |
| Qwen/Qwen2.5-VL-7B-Instruct | beautiful_trash | ugly_trash      |       300 |            25 |                            25 | 100.00%                   | 8.33%                         | 91.67%              |

## Primary Conflict by Language

The primary conflict is `ugly_gold` versus `beautiful_trash`: preserved semantics with degraded presentation versus destroyed semantics with clean presentation.

| model                       | language   |   n_pairs |   valid_pairs |   target_selected_valid_pairs | target_preference_valid   | effective_target_preference   | strict_swap_error   |
|:----------------------------|:-----------|----------:|--------------:|------------------------------:|:--------------------------|:------------------------------|:--------------------|
| OpenGVLab/InternVL3-8B      | cuda       |       100 |            66 |                            66 | 100.00%                   | 66.00%                        | 34.00%              |
| OpenGVLab/InternVL3-8B      | java       |       100 |            91 |                            89 | 97.80%                    | 89.00%                        | 9.00%               |
| OpenGVLab/InternVL3-8B      | python     |       100 |            91 |                            91 | 100.00%                   | 91.00%                        | 9.00%               |
| Qwen/Qwen2.5-VL-7B-Instruct | cuda       |       100 |            89 |                            89 | 100.00%                   | 89.00%                        | 11.00%              |
| Qwen/Qwen2.5-VL-7B-Instruct | java       |       100 |            94 |                            94 | 100.00%                   | 94.00%                        | 6.00%               |
| Qwen/Qwen2.5-VL-7B-Instruct | python     |       100 |            92 |                            92 | 100.00%                   | 92.00%                        | 8.00%               |

## Integrity and Interpretation

Both full manifests are complete. All 7,200 calls parsed, all parsed verdicts matched the captured A/B logit argmax, and every generation used 7 tokens. High valid-only target preference must not be read without its valid denominator. In particular, Qwen's `beautiful_trash` versus `ugly_trash` result has 25/300 valid pairs and 91.67% strict-swap error, so its 100% valid-only clean preference corresponds to only 8.33% effective preference over all pairs.

Verdict-logit decomposition is secondary. It uses `c=(m_AB-m_BA)/2` and `b=(m_AB+m_BA)/2`; it does not replace strict-swap results.

