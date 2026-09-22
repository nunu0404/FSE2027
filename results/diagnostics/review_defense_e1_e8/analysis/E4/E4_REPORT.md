# E4 Reproducibility report

Same-environment pair-state disagreement ranges from
5.80% to
8.70% across six cells. The controlled
GPU-UUID change ranges from 5.80% to
8.70%. Logit bit identity is reported
separately because identical verdicts need not have identical BF16 logits.

For the battery repeat, 15/15
model-language cells meet the preregistered >=99.5% pair-state agreement
threshold. Original Table 1/2 values are not overwritten under either outcome.

## Draft language

We measured the noise floor by repeating all six baseline grid cells under the
same frozen environment and by changing only the physical GPU. We report
pair-level state disagreement, call-level verdict agreement, and bit-level logit
identity separately. The full five-model battery was then repeated under its
original concurrency schedule. Under the preregistered rule, cells below 99.5%
agreement are reported as reproducibility findings and both runs are retained;
the original headline table is not silently replaced.
