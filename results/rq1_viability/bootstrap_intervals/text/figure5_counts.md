# Closed Model Pilot Pair Counts and Metric Verification

This document reports unique pairs, call counts per presentation order (AB and BA), and recomputed metrics across all closed-model pilot result folders under `~/experiment_26_v1/results/`.

### Summary of Conditions and Verification Table

| Folder | File | Model (Condition) | Condition Name | Unique Pairs | Calls (AB / BA) | Total Calls | Valid Pairs | Correct Pairs | Valid Acc (%) | Effective Acc E (%) | Strict Swap Err S (%) | Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | combined_labeled_image_only | 300 | 300 / 300 | 600 | 225 | 142 | 63.11 | 47.33 | 25.00 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 147 | 92 | 62.59 | 30.67 | 51.00 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | image_only | 300 | 300 / 300 | 600 | 229 | 146 | 63.76 | 48.67 | 23.67 | OK_300 |
| `openai_vlm_pilot_20260707` | `gpt-5.4-mini__pilot_raw.jsonl` | gpt-5.4-mini | text_plus_image | 300 | 300 / 300 | 600 | 166 | 112 | 67.47 | 37.33 | 44.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | combined_labeled_image_only | 300 | 300 / 300 | 600 | 259 | 159 | 61.39 | 53.00 | 13.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 234 | 155 | 66.24 | 51.67 | 22.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | image_only | 300 | 300 / 300 | 600 | 249 | 151 | 60.64 | 50.33 | 17.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.4__pilot_raw.jsonl` | gpt-5.4 | text_plus_image | 300 | 300 / 300 | 600 | 220 | 148 | 67.27 | 49.33 | 26.67 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (buggy prompt) | image_only | 300 | 300 / 300 | 600 | 0 | 0 | NaN | 0.00 | 100.00 | OK_300 |
| `openai_vlm_pilot_20260708` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (buggy prompt) | text_plus_image | 122 | 122 / 122 | 244 | 0 | 0 | NaN | 0.00 | 100.00 | FLAG_NOT_300 (122 pairs) |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | combined_labeled_image_only | 300 | 300 / 300 | 600 | 231 | 155 | 67.10 | 51.67 | 23.00 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | combined_labeled_text_plus_image | 300 | 300 / 300 | 600 | 238 | 154 | 64.71 | 51.33 | 20.67 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | image_only | 300 | 300 / 300 | 600 | 244 | 167 | 68.44 | 55.67 | 18.67 | OK_300 |
| `openai_vlm_pilot_20260708_gpt55_fixed` | `gpt-5.5__pilot_raw.jsonl` | gpt-5.5 (fixed) | text_plus_image | 300 | 300 / 300 | 600 | 224 | 150 | 66.96 | 50.00 | 25.33 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_256_raw.jsonl` | gpt-5.4 (high, mot 256) | image_only | 90 | 90 / 90 | 180 | 19 | 14 | 73.68 | 15.56 | 78.89 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_256_raw.jsonl` | gpt-5.4 (high, mot 256) | text_plus_image | 90 | 90 / 90 | 180 | 48 | 35 | 72.92 | 38.89 | 46.67 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl` | gpt-5.4 (high, mot 4096) | image_only | 300 | 300 / 300 | 600 | 266 | 169 | 63.53 | 56.33 | 11.33 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_high__mot_4096_raw.jsonl` | gpt-5.4 (high, mot 4096) | text_plus_image | 300 | 300 / 300 | 600 | 213 | 148 | 69.48 | 49.33 | 29.00 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_256_raw.jsonl` | gpt-5.4 (low, mot 256) | image_only | 90 | 90 / 90 | 180 | 76 | 45 | 59.21 | 50.00 | 15.56 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_256_raw.jsonl` | gpt-5.4 (low, mot 256) | text_plus_image | 90 | 90 / 90 | 180 | 67 | 44 | 65.67 | 48.89 | 25.56 | FLAG_NOT_300 (90 pairs) |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_4096_raw.jsonl` | gpt-5.4 (low, mot 4096) | image_only | 300 | 300 / 300 | 600 | 256 | 156 | 60.94 | 52.00 | 14.67 | OK_300 |
| `gpt54_reasoning_budget_pilot_90_20260709` | `gpt-5.4-2026-03-05__reasoning_low__mot_4096_raw.jsonl` | gpt-5.4 (low, mot 4096) | text_plus_image | 300 | 300 / 300 | 600 | 219 | 145 | 66.21 | 48.33 | 27.00 | OK_300 |

### Meaning of "90" in Folder Name
The "90" in `gpt54_reasoning_budget_pilot_90_20260709` denotes that for the reduced reasoning budget condition (`mot_256`), 30 pairs per difficulty bin (easy, medium, hard) were evaluated, resulting in exactly 90 unique pairs per condition (180 calls each). By contrast, full-budget conditions (`mot_4096`) evaluated all 300 Java pairs (600 calls each).

### Flagged Conditions (FLAG_NOT_300)
1. `openai_vlm_pilot_20260708` -> `gpt-5.5` `text_plus_image`: Aborted at 122 pairs (244 calls).
2. `gpt54_reasoning_budget_pilot_90_20260709` -> All four `mot_256` conditions: Evaluated on 90 pairs each (180 calls each).
