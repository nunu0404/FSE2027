# E5 Completion report

## Historical failure

The old InternVL RQ3 text+image GA0 failed reproducibly on one output,
`FINAL_VERDDICT: B`. The old combined-packaging full run completed 3,000 pairs
but failed completeness on 6 calls across
4 pairs because InternVL emitted `VERDDICT`, `VERDDIC`, or `VERDD`.
The E5 adapter changes only output-format parsing for these unambiguous prefixes;
it does not modify prompts, generated strings, A/B tokens, or logits.

## Completeness

    E5-A attempted 3600 pair rows (1800 per model). Qwen
passes the full gate. InternVL retry 2 still has one unrecoverable call,
`FINAL_VERD`, with no explicit A/B; its cell therefore remains missing and its
diagnostic estimates are not promoted to the main table. E5-B contains
3000 Java pairs and parse-failure rate 0.0000%.
No old and new runs were merged.

## RQ3 text+image

| model                       |    n |   valid_n |   target_correct_n |   effective_target_preference |   valid_conditional_target_preference |   strict_swap_error |   parse_failure_pairs |   debiased_ties |   debiased_target_main_tie_incorrect |   debiased_target_excluding_ties |   debiased_target_half_credit |   boundary_abs_c_eq_abs_b | completeness_status   | reportable   | failure_reason                                                                               |
|:----------------------------|-----:|----------:|-------------------:|------------------------------:|--------------------------------------:|--------------------:|----------------------:|----------------:|-------------------------------------:|---------------------------------:|------------------------------:|--------------------------:|:----------------------|:-------------|:---------------------------------------------------------------------------------------------|
| OpenGVLab/InternVL3-8B      | 1800 |  nan      |           nan      |                      nan      |                              nan      |            nan      |              nan      |        nan      |                             nan      |                         nan      |                      nan      |                  nan      | fail                  | False        | one deterministic call emitted FINAL_VERD without an explicit A/B; full GA0 parse_failures=1 |
| Qwen/Qwen2.5-VL-7B-Instruct | 1800 | 1392.0000 |          1392.0000 |                        0.7733 |                                1.0000 |              0.2267 |                0.0000 |          5.0000 |                               0.9928 |                           0.9955 |                        0.9942 |                   29.0000 | pass                  | True         |                                                                                              |

## Packaging cell

| run_id                                         | model                       | language   |    n |   valid_n |   correct_n |   valid_accuracy |   effective_accuracy |   strict_swap_error |   parse_failure_rate | packaging        | modality        |
|:-----------------------------------------------|:----------------------------|:-----------|-----:|----------:|------------:|-----------------:|---------------------:|--------------------:|---------------------:|:-----------------|:----------------|
| clean_full_factorial_20260703                  | Qwen/Qwen2.5-VL-7B-Instruct | java       | 3000 |      2075 |        1242 |           0.5986 |               0.4140 |              0.3083 |               0.0000 | combined_labeled | text_plus_image |
| E5B_internvl_combined_text_plus_image_20260730 | OpenGVLab/InternVL3-8B      | java       | 3000 |      1884 |        1029 |           0.5462 |               0.3430 |              0.3720 |               0.0000 | combined_labeled | text_plus_image |
