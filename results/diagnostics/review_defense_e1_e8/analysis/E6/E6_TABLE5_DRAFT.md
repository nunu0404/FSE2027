# Table 5 replacement draft

| language   | system                       |   n_total_pairs |   n_valid_pairs |   n_correct_pairs |   effective_accuracy |   valid_accuracy |   strict_swap_error |   parse_failure_rate |
|:-----------|:-----------------------------|----------------:|----------------:|------------------:|---------------------:|-----------------:|--------------------:|---------------------:|
| java       | Source-access RF             |            3000 |            3000 |              1870 |               0.6233 |           0.6233 |              0.0000 |               0.0000 |
| java       | RapidOCR + best ML           |            3000 |            3000 |              1589 |               0.5297 |           0.5297 |              0.0000 |               0.0000 |
| java       | Direct Qwen image-only       |            3000 |            1867 |              1160 |               0.3867 |           0.6213 |              0.3777 |               0.0000 |
| java       | Source-text Qwen2.5-Coder    |            3000 |            1796 |               999 |               0.3330 |           0.5562 |              0.4013 |               0.0000 |
| java       | Best OCR-text LLM (RapidOCR) |            3000 |            1530 |               854 |               0.2847 |           0.5582 |              0.4900 |               0.0000 |
| python     | Source-access RF             |            3000 |            3000 |              1884 |               0.6280 |           0.6280 |              0.0000 |               0.0000 |
| python     | RapidOCR + best ML           |            3000 |            3000 |              1786 |               0.5953 |           0.5953 |              0.0000 |               0.0000 |
| python     | Direct Qwen image-only       |            3000 |            1473 |              1058 |               0.3527 |           0.7183 |              0.5090 |               0.0000 |
| python     | Source-text Qwen2.5-Coder    |            3000 |            1362 |               782 |               0.2607 |           0.5742 |              0.5460 |               0.0000 |
| python     | Best OCR-text LLM (RapidOCR) |            3000 |             472 |               296 |               0.0987 |           0.6271 |              0.8427 |               0.0000 |
| cuda       | Source-access RF             |            3000 |            3000 |              2202 |               0.7340 |           0.7340 |              0.0000 |               0.0000 |
| cuda       | RapidOCR + best ML           |            3000 |            3000 |              1822 |               0.6073 |           0.6073 |              0.0000 |               0.0000 |
| cuda       | Direct Qwen image-only       |            3000 |            2098 |              1499 |               0.4997 |           0.7145 |              0.3007 |               0.0000 |
| cuda       | Source-text Qwen2.5-Coder    |            3000 |            1724 |               996 |               0.3320 |           0.5777 |              0.4253 |               0.0000 |
| cuda       | Best OCR-text LLM (RapidOCR) |            3000 |            1075 |               647 |               0.2157 |           0.6019 |              0.6417 |               0.0000 |

All deterministic source/OCR-ML rows have valid coverage 100%, strict-swap error 0%, and parse-failure rate 0% by construction. VLM/LLM effective accuracy counts invalid pairs as incorrect.
