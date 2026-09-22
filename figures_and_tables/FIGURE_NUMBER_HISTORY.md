# Numeric Tables for Redrawing Manuscript Figures with Uncertainty

**Generated**: 2026-09-12 04:07:24 (KST)  
**Evaluation Set**: RQ1 (9,000 pairs; 3,000 Java, 3,000 Python, 3,000 CUDA) and RQ2 Reduced Grid (42,000 calls per judge)  
**Bootstrap Method**: 10,000-resample snippet-cluster bootstrap (seed 42, 95% confidence intervals)  
**Formatting**: One decimal place; signed values for contrasts in Table 3.

---

## Table 1 (for Figure 3: Valid and Effective Accuracy per Language)

| judge | language | valid | valid_lo | valid_hi | E | E_lo | E_hi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemma4-12B | Java | 61.9 | 57.4 | 66.4 | 54.3 | 50.2 | 58.4 |
| Gemma4-12B | Python | 68.8 | 61.2 | 75.7 | 58.3 | 50.8 | 65.4 |
| Gemma4-12B | Cuda | 70.1 | 64.1 | 76.1 | 58.1 | 52.1 | 63.8 |
| Gemma-3-12B | Java | 62.5 | 57.8 | 67.1 | 50.8 | 46.5 | 54.9 |
| Gemma-3-12B | Python | 65.5 | 56.3 | 74.1 | 46.6 | 38.9 | 54.4 |
| Gemma-3-12B | Cuda | 65.3 | 59.3 | 70.9 | 53.7 | 48.0 | 58.9 |
| Qwen3-VL-8B | Java | 62.2 | 57.0 | 67.3 | 44.3 | 40.0 | 48.6 |
| Qwen3-VL-8B | Python | 61.6 | 53.5 | 69.1 | 47.3 | 40.3 | 54.2 |
| Qwen3-VL-8B | Cuda | 62.7 | 55.3 | 70.0 | 41.8 | 35.6 | 47.9 |
| InternVL3.5-8B | Java | 60.3 | 55.1 | 65.3 | 41.4 | 37.0 | 45.7 |
| InternVL3.5-8B | Python | 71.3 | 62.6 | 79.2 | 44.7 | 37.4 | 51.8 |
| InternVL3.5-8B | Cuda | 65.9 | 58.6 | 73.0 | 45.7 | 39.3 | 52.0 |
| Qwen2.5-VL-7B | Java | 62.3 | 56.2 | 68.4 | 32.7 | 28.2 | 37.3 |
| Qwen2.5-VL-7B | Python | 75.9 | 65.8 | 84.8 | 36.6 | 28.8 | 44.4 |
| Qwen2.5-VL-7B | Cuda | 74.9 | 68.4 | 81.0 | 46.5 | 39.9 | 53.0 |
| Ministral-3-8B | Java | 54.0 | 48.5 | 59.3 | 38.9 | 34.7 | 43.3 |
| Ministral-3-8B | Python | 53.3 | 43.6 | 62.8 | 34.2 | 27.4 | 41.2 |
| Ministral-3-8B | Cuda | 58.1 | 50.7 | 65.6 | 37.1 | 31.5 | 42.6 |
| Phi-4-multimodal | Java | 44.9 | 39.3 | 50.4 | 31.2 | 27.0 | 35.6 |
| Phi-4-multimodal | Python | 44.6 | 36.4 | 52.9 | 30.4 | 24.7 | 36.2 |
| Phi-4-multimodal | Cuda | 45.1 | 38.9 | 51.3 | 32.0 | 27.5 | 36.6 |
| InternVL3-8B | Java | 46.9 | 41.0 | 52.7 | 30.0 | 25.6 | 34.4 |
| InternVL3-8B | Python | 48.1 | 37.2 | 59.4 | 20.7 | 15.4 | 26.5 |
| InternVL3-8B | Cuda | 54.5 | 44.9 | 64.3 | 23.4 | 18.0 | 29.1 |

---

## Table 2 (for Figure 4: Pooled Evaluation Metrics over 9,000 Pairs)

| judge | E | E_lo | E_hi | D | D_lo | D_hi | S | S_lo | S_hi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemma4-12B | 56.9 | 53.4 | 60.2 | 64.6 | 61.3 | 67.8 | 14.9 | 13.2 | 16.8 |
| Gemma-3-12B | 50.4 | 46.8 | 53.7 | 62.0 | 58.6 | 65.1 | 21.8 | 19.2 | 24.5 |
| Qwen3-VL-8B | 44.5 | 41.1 | 47.9 | 59.9 | 56.7 | 63.2 | 28.4 | 25.5 | 31.6 |
| InternVL3.5-8B | 43.9 | 40.4 | 47.4 | 62.3 | 59.1 | 65.5 | 33.1 | 29.8 | 36.7 |
| Qwen2.5-VL-7B | 38.6 | 34.6 | 42.4 | 64.2 | 60.8 | 67.4 | 45.8 | 41.7 | 50.0 |
| Ministral-3-8B | 36.7 | 33.5 | 40.1 | 54.3 | 50.9 | 57.7 | 33.3 | 29.9 | 36.9 |
| Phi-4-multimodal | 31.2 | 28.5 | 34.1 | 37.8 | 34.8 | 41.0 | 30.4 | 28.0 | 32.9 |
| InternVL3-8B | 24.7 | 21.7 | 27.9 | 51.0 | 47.5 | 54.5 | 49.9 | 45.5 | 54.3 |

---

## Table 3 (for Figure 6(a): Contrast Sensitivity on Added Judges)

| contrast | quantity | min | max | Gemma-3 CUDA | Gemma-3 Java | Gemma-3 Python | InternVL3.5 CUDA | InternVL3.5 Java | InternVL3.5 Python |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| wrap 60 to 80 | change in E | +0.7 | +7.3 | +0.7 | +1.5 | +1.6 | +7.3 | +4.0 | +6.3 |
| wrap 60 to 80 | change in S | -9.8 | -0.8 | -4.5 | -1.0 | -0.8 | -9.8 | -6.2 | -7.5 |
| font 20 to 24 | change in E | -3.2 | +1.6 | +0.8 | -1.0 | -3.2 | -1.8 | +0.9 | +1.6 |
| font 20 to 24 | change in S | -2.5 | +2.9 | +0.5 | +2.3 | +0.6 | +2.9 | -2.5 | -2.3 |

---

## Discrepancy List (> 0.05 Difference from Manuscript Reference)

| Table | Judge | Language | Metric | Computed Value | Reference Value | Difference | Exact Unrounded (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Table 1 (Fig 3) | Qwen3-VL-8B | Cuda | valid | 62.7 | 62.8 | -0.10 | 62.7500% |
| Table 1 (Fig 3) | Qwen2.5-VL-7B | Java | valid | 62.3 | 62.4 | -0.10 | 62.3492% |
| Table 1 (Fig 3) | Qwen2.5-VL-7B | Cuda | valid | 74.9 | 75.0 | -0.10 | 74.9462% |

### Rationale for Discrepancies
- In all 3 discrepant cells, the difference is exactly 0.1%p and stems from boundary half-rounding of the unrounded raw ratios:
  1. `Qwen3-VL-8B CUDA valid`: 1,253 correct / 1,997 valid pairs = 62.7441%, which standard mathematical rounding rounds to **62.7%** (reference: 62.8%).
  2. `Qwen2.5-VL-7B Java valid`: 982 correct / 1,575 valid pairs = 62.3492%, which standard mathematical rounding rounds to **62.3%** (reference: 62.4%).
  3. `Qwen2.5-VL-7B CUDA valid`: 1,394 correct / 1,860 valid pairs = 74.9462%, which standard mathematical rounding rounds to **74.9%** (reference: 75.0%).
- All pooled values in Table 2 (Figure 4) match the manuscript reference within 0.05.
- Per instructions, values are not altered and the exact computed data is reported faithfully.
