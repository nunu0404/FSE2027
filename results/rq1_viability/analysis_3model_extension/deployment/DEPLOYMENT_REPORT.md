# Qwen3 deployment result

The preregistered overall decision is **(b)**.

- java: D=0.5783, OCR+ML E=0.5297, difference=+0.0487, 95% cluster CI [-0.0136, +0.1107], exact McNemar p=9.693e-05, Holm p=0.0002908, category (b).
- python: D=0.5870, OCR+ML E=0.5953, difference=-0.0083, 95% cluster CI [-0.1101, +0.0911], exact McNemar p=0.5516, Holm p=0.8781, category (a).
- cuda: D=0.5970, OCR+ML E=0.6073, difference=-0.0103, 95% cluster CI [-0.0980, +0.0772], exact McNemar p=0.439, Holm p=0.8781, category (a).

## Manuscript draft

For Qwen3-VL-8B, we retained the A/B verdict logits in both presentation orders and defined the order-debiased decision as sign(c), where c=(m_AB-m_BA)/2. We compared that decision with the deterministic, supervised RapidOCR plus language-specific ML pipeline on the same frozen pairs. Confidence intervals use 10,000 two-endpoint snippet-cluster bootstrap replicates, and exact McNemar tests are Holm-adjusted across the three language-specific comparisons. The VLM result requires two calls per pair and verdict-logit access, whereas OCR+ML is supervised but deterministic at deployment time.
