# Clean Multi-OCR Statistical Tests

| comparison                                                                      |   n_pairs |   accuracy_a |   accuracy_b |   delta_b_minus_a |   ci_low |   ci_high |   mcnemar_a_only |   mcnemar_b_only |   mcnemar_p |
|:--------------------------------------------------------------------------------|----------:|-------------:|-------------:|------------------:|---------:|----------:|-----------------:|-----------------:|------------:|
| Oracle RF -> OCR(easyocr) RF                                                    |      3000 |       0.6233 |       0.4823 |           -0.1410 |  -0.1647 |   -0.1160 |              905 |              482 |      0.0000 |
| Oracle RF -> OCR(easyocr_preprocessed) RF                                       |      3000 |       0.6233 |       0.5173 |           -0.1060 |  -0.1277 |   -0.0840 |              729 |              411 |      0.0000 |
| Oracle RF -> OCR(rapidocr) RF                                                   |      3000 |       0.6233 |       0.5187 |           -0.1047 |  -0.1270 |   -0.0817 |              760 |              446 |      0.0000 |
| OCR(rapidocr) RF -> Direct VLM OpenGVLab/InternVL3-8B / image_only / clean      |      3000 |       0.5187 |       0.2507 |           -0.2680 |  -0.2917 |   -0.2447 |             1145 |              341 |      0.0000 |
| OCR(rapidocr) RF -> Direct VLM Qwen/Qwen2.5-VL-7B-Instruct / image_only / clean |      3000 |       0.5187 |       0.3867 |           -0.1320 |  -0.1557 |   -0.1083 |              909 |              513 |      0.0000 |
