# Clean Multi-OCR Screenshot Experiment

Generated at: 2026-07-07T04:32:14.215438+00:00

## Scope

- Corrected render metadata: `/ANON/experiment_root/experiments/rq0_viability/outputs/render_metadata/default_render_metadata.csv`
- Pair set: `/ANON/experiment_root/experiments/rq0_viability/data/pairs/full_pair_set_rq0.csv`
- OCR engines attempted: easyocr, rapidocr, easyocr_preprocessed
- Direct VLM baseline: clean full-factorial image-only outputs from `full_factorial_clean_20260703`
- Text-only OCR LLM status: not_run: no local Qwen/Qwen2.5-Coder-7B-Instruct checkpoint is currently present in Hugging Face cache

## OCR Quality

| ocr_engine           |   ok_rate |   token_f1 |   char_error_rate |   runtime_sec |
|:---------------------|----------:|-----------:|------------------:|--------------:|
| easyocr              |    1.0000 |     0.7265 |            0.3158 |        1.9698 |
| easyocr_preprocessed |    1.0000 |     0.7765 |            0.3572 |        1.7179 |
| rapidocr             |    1.0000 |     0.8119 |            0.2378 |        0.9540 |

## Final Comparison

| system                                                      | kind                    | model                                            | ocr_engine           |   effective_accuracy |   valid_accuracy |   strict_swap_error |   delta_vs_oracle_rf |   delta_vs_best_ocr_rf |
|:------------------------------------------------------------|:------------------------|:-------------------------------------------------|:---------------------|---------------------:|-----------------:|--------------------:|---------------------:|-----------------------:|
| Oracle source gradient_boosting_regressor                   | oracle_source_classical | gradient_boosting_regressor                      |                      |               0.6200 |           0.6200 |            nan      |              -0.0033 |                 0.1013 |
| Oracle source linear_regression                             | oracle_source_classical | linear_regression                                |                      |               0.5967 |           0.5967 |            nan      |              -0.0267 |                 0.0780 |
| Oracle source random_forest_regressor                       | oracle_source_classical | random_forest_regressor                          |                      |               0.6233 |           0.6233 |            nan      |               0.0000 |                 0.1047 |
| Oracle source svr                                           | oracle_source_classical | svr                                              |                      |               0.6127 |           0.6127 |            nan      |              -0.0107 |                 0.0940 |
| OCR(easyocr)+gradient_boosting_regressor                    | ocr_classical           | gradient_boosting_regressor                      | easyocr              |               0.4977 |           0.4977 |            nan      |              -0.1257 |                -0.0210 |
| OCR(easyocr)+linear_regression                              | ocr_classical           | linear_regression                                | easyocr              |               0.4830 |           0.4830 |            nan      |              -0.1403 |                -0.0357 |
| OCR(easyocr)+random_forest_regressor                        | ocr_classical           | random_forest_regressor                          | easyocr              |               0.4823 |           0.4823 |            nan      |              -0.1410 |                -0.0363 |
| OCR(easyocr)+svr                                            | ocr_classical           | svr                                              | easyocr              |               0.5013 |           0.5013 |            nan      |              -0.1220 |                -0.0173 |
| OCR(easyocr_preprocessed)+gradient_boosting_regressor       | ocr_classical           | gradient_boosting_regressor                      | easyocr_preprocessed |               0.5087 |           0.5087 |            nan      |              -0.1147 |                -0.0100 |
| OCR(easyocr_preprocessed)+linear_regression                 | ocr_classical           | linear_regression                                | easyocr_preprocessed |               0.4910 |           0.4910 |            nan      |              -0.1323 |                -0.0277 |
| OCR(easyocr_preprocessed)+random_forest_regressor           | ocr_classical           | random_forest_regressor                          | easyocr_preprocessed |               0.5173 |           0.5173 |            nan      |              -0.1060 |                -0.0013 |
| OCR(easyocr_preprocessed)+svr                               | ocr_classical           | svr                                              | easyocr_preprocessed |               0.5103 |           0.5103 |            nan      |              -0.1130 |                -0.0083 |
| OCR(rapidocr)+gradient_boosting_regressor                   | ocr_classical           | gradient_boosting_regressor                      | rapidocr             |               0.5223 |           0.5223 |            nan      |              -0.1010 |                 0.0037 |
| OCR(rapidocr)+linear_regression                             | ocr_classical           | linear_regression                                | rapidocr             |               0.5103 |           0.5103 |            nan      |              -0.1130 |                -0.0083 |
| OCR(rapidocr)+random_forest_regressor                       | ocr_classical           | random_forest_regressor                          | rapidocr             |               0.5187 |           0.5187 |            nan      |              -0.1047 |                 0.0000 |
| OCR(rapidocr)+svr                                           | ocr_classical           | svr                                              | rapidocr             |               0.5297 |           0.5297 |            nan      |              -0.0937 |                 0.0110 |
| Direct VLM OpenGVLab/InternVL3-8B / image_only / clean      | direct_vlm              | OpenGVLab/InternVL3-8B / image_only / clean      |                      |               0.2507 |           0.5546 |              0.5480 |              -0.3727 |                -0.2680 |
| Direct VLM Qwen/Qwen2.5-VL-7B-Instruct / image_only / clean | direct_vlm              | Qwen/Qwen2.5-VL-7B-Instruct / image_only / clean |                      |               0.3867 |           0.6213 |              0.3777 |              -0.2367 |                -0.1320 |

## Interpretation Guardrails

- OCR rows are screenshot-deployable only through OCR text and do not use raw source at prediction time.
- Oracle source rows are upper bounds for the same pair set, not screenshot-only systems.
- Direct VLM rows reuse clean rerendered image-only strict-swap outputs; no stale June VLM rows are used.
- OCR text LLM is not included unless a local text-only checkpoint is available and rerun on the newly extracted clean OCR text.
