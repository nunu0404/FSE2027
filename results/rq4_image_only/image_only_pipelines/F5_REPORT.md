# F5 deployment-v2 result

The preregistered overall decision is **(b)**.

## Direct Qwen versus the same-run RapidOCR+ML comparator

- java: D=0.5717, OCR+ML E=0.5297, difference=+0.0420, 95% cluster CI [-0.0219, +0.1065], exact McNemar p=0.0008657, Holm p=0.004328, category (b).
- python: D=0.6307, OCR+ML E=0.5953, difference=+0.0353, 95% cluster CI [-0.0591, +0.1276], exact McNemar p=0.005088, Holm p=0.01527, category (b).
- cuda: D=0.6557, OCR+ML E=0.6073, difference=+0.0483, 95% cluster CI [-0.0264, +0.1222], exact McNemar p=7.793e-05, Holm p=0.0004676, category (b).

## English manuscript draft

In the preregistered deployment-v2 rerun, we retained the A/B verdict logits for both presentation orders and computed the order-debiased decision as sign(c), where c=(m_AB-m_BA)/2. Comparisons against the deterministic RapidOCR plus supervised-ML pipeline use the same frozen pairs and are reported separately by language. The debiased VLM requires two model calls per pair and access to verdict logits, which in practice favors open or logprob-accessible models; the OCR+ML comparator is supervised whereas the VLM judge is zero-shot. All confidence intervals use 10,000 two-endpoint snippet-cluster bootstrap replicates, and exact McNemar tests are Holm-adjusted across the nine preregistered comparisons.
