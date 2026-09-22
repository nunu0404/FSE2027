# Parse-failure audit

Every experiment reported in the paper, audited for calls whose output did not
yield `FINAL_VERDICT: A` or `FINAL_VERDICT: B`. Counts are recomputed from the
raw JSONL and pair-level files in this package, not copied from the manuscript.

A parse failure always makes its pair invalid, so it is inside the strict-swap
error `S` and can never be counted correct. `S` therefore mixes two different
things — formatting failures and genuine order disagreements — which the paper
states in Section 3.3 but decomposes for only one run.

Reproduce with `code/99_verification/audit_parse_failures.py`.

---

## Summary

| Experiment | Scope | Calls | Failed calls | Failed pairs | In the paper? |
| --- | --- | ---: | ---: | ---: | --- |
| **RQ1** primary + diagnostic | 8 judges × 9,000 pairs | 144,000 | **12** | **12** | no |
| **RQ2** full grid | 2 judges × 18 labels × 3,000 | 216,000 | 0 | 0 | — |
| **RQ2** reduced grid | 3 judges × 7 cond. × 3,000 | 126,000 | **281** | **237** | partly |
| **RQ2** packaging / modality | 8 cond. × 3,000 Java pairs | 48,000 | **39** | **37** | no |
| **RQ3** image-only primary | 2 judges × 6 contrasts × 300 | 7,200 | 0 | 0 | yes |
| **RQ3** Qwen text+image (suppl.) | 6 contrasts × 300 | 3,600 | 0 | 0 | — |
| **RQ3** InternVL text+image GA0 | 2 gate runs × 100 | 200 | **2** | **2** | indirectly |
| **RQ4** text ablation | 2 inputs × 9,000 pairs | 36,000 | 0 | 0 | yes |
| **RQ4** OCR-text LLM judge | 4 text inputs | — | 0 | 0 | — |
| **Table 2(a)** GPT-5.4-mini, GPT-5.4 | 2 models × 4 cond. × 300 | 4,800 | 0 | 0 | — |
| **Table 2(a)** GPT-5.5 | 4 cond. × 300 pairs | 2,400 | **122** | **112** | no |
| **Table 2(b)** 24B–32B open models | 3 models × 3 lang. × 300 | 5,400 | 0 | 0 | — |
| Excluded GPT-5.5 pilot (aborted) | 2 cond., 422 pairs | 844 | **844** | **422** | as "aborted" |
| Excluded GPT-5.4 256-token pilot | 4 cond. × 90 pairs | 720 | **143** | — | as "excluded" |

---

## RQ1 — viability (Figures 6, 7)

Recomputed from `results/rq1_viability/raw_inference/*/raw.jsonl`, 18,000 calls
per judge.

| Judge | Failed calls | Failed pairs | By language | `S` | Share of `S` |
| --- | ---: | ---: | --- | ---: | ---: |
| Gemma4-12B | 11 | 11 | CUDA 7, Python 4 | 14.91% | 0.82% |
| InternVL3.5-8B | 1 | 1 | Java 1 | 33.10% | 0.03% |
| Qwen2.5-VL-7B | 0 | 0 | — | 45.78% | 0 |
| Qwen3-VL-8B | 0 | 0 | — | 28.40% | 0 |
| Gemma-3-12B | 0 | 0 | — | 21.79% | 0 |
| InternVL3-8B | 0 | 0 | — | 49.93% | 0 |
| Ministral-3-8B | 0 | 0 | — | 33.29% | 0 |
| Phi-4-multimodal | 0 | 0 | — | 30.39% | 0 |

Section 5.1 states only that all 18,000 Qwen2.5-VL-7B calls parsed, so its
45.78% invalidity is entirely parsed disagreement. That is correct. The paper
does not mention that Gemma4-12B and InternVL3.5-8B each have a small number of
failures; they are inside `S` and inside the `E` denominator. At 0.82% and 0.03%
of `S` they change no reported conclusion.

Note that `pair_level_5model_battery_45000.csv` has **no parse-failure column**
— validity there is inferred from the parsed choices alone. Its zero count is
confirmed directly from the raw JSONL.

## RQ2 — full rendering grid

`results/rq2_reliability/full_grid_2models/A_B_PAIR_LEVEL.csv`, `parsed_both`:
**0 failures in 108,000 pairs / 216,000 calls** for Qwen2.5-VL-7B and
InternVL3-8B across all 18 condition labels. Every point of their swap error in
Figure 8 is parsed disagreement.

## RQ2 — reduced grid

21,000 pairs / 42,000 calls per judge, seven conditions.

| Judge | Failed calls | Failed pairs | Rate (pairs) | Concentration |
| --- | ---: | ---: | ---: | --- |
| Gemma-3-12B | 0 | 0 | 0.00% | — |
| **Gemma4-12B** | **34** | **33** | **0.16%** | spread over all 7 conditions; CUDA 19, Python 10, Java 4 |
| InternVL3.5-8B | 247 | 204 | 0.97% | **182 of 204 (89%) in `gaussian_sigma_4`**; Java 127, Python 66, CUDA 11 |

Section 5.2 reports Gemma-3-12B's 0/21,000 and InternVL3.5-8B's 204/21,000 and
247/42,000 exactly, and its "four rendering conditions alone, 12/12,000 (0.10%)"
also reproduces.

> **Correction needed.** Section 5.2 opens with "Parse failures are uncommon in
> the *two* reduced-grid runs with retained parsing counts." All **three**
> retained them — Gemma4-12B's pair-level file carries the most detailed
> per-call parse columns of the three (`parsed_choice_ab`, `parsed_choice_ba`,
> `argmax_matches_parsed_*`). Its 33 parse-failed pairs are simply unreported.

Failure modes differ by model. InternVL3.5-8B emits `<think>` reasoning that
runs past the 24-token limit before reaching a verdict, which is why blur level
4 dominates. Gemma4-12B emits prose description instead of the verdict line.

## RQ2 — packaging and modality

8 conditions × 3,000 Java pairs
(`results/rq2_reliability/packaging_and_modality/*/pair_level_analysis_table.csv`).

| Condition | Failed pairs |
| --- | ---: |
| InternVL3-8B, text+image, separate | 35 |
| InternVL3-8B, text+image, combined | 2 |
| InternVL3-8B, image-only (both packagings) | 0 |
| Qwen2.5-VL-7B, all four conditions | 0 |

Adding source text is what triggers failure here, and only for InternVL3-8B.

## RQ3 — content versus appearance

| Arm | Calls | Failed |
| --- | ---: | ---: |
| Image-only primary, 2 judges × 6 contrasts × 300 × 2 orders | 7,200 | **0** |
| Qwen2.5-VL-7B text+image, supplementary | 3,600 | 0 |
| InternVL3-8B text+image, GA0 gate | 100 | **1** |
| InternVL3-8B text+image, GA0 gate retry | 100 | **1** |

Section 5.3's "All 7,200 RQ3 calls parsed" is exact.

The two gate failures are the same call in both runs — pair
`rq3g_cuda_008__c2`, order BA, CUDA — and the same malformed output:

```
FINAL_VERDDICT: B
```

A doubled `D`. One mistyped token, deterministic under greedy decoding, so the
retry reproduced it exactly. That single call failed the 100-call gate twice and
the InternVL3-8B text+image arm was never promoted to full inference — the
blocked arm recorded in `results/rq3_content_vs_appearance/analysis/historical_alignment/RQ3_ARM_STATUS.csv`
and listed in `EXCLUSIONS.md`.

## RQ4 — image-only pipelines

| Route | Failed |
| --- | ---: |
| Qwen2.5-VL-7B on source text, 9,000 pairs | 0 |
| Qwen2.5-VL-7B on OCR text, 9,000 pairs | 0 |
| Qwen2.5-VL-7B direct image route | 0 |
| Qwen2.5-Coder-7B on source / RapidOCR / EasyOCR / preprocessed EasyOCR | 0 |
| OCR + ML predictors | n/a — deterministic scorers, no parsing stage |

Section 5.4's "Neither text condition has parsing failures" is exact.

## Protocol checks

### Table 2(a), closed models, 300 Java pairs per condition

| Model | Condition | Failed calls / 600 | Rate | Failed pairs / 300 |
| --- | --- | ---: | ---: | ---: |
| GPT-5.4-mini | all four | 0 | 0.00% | 0 |
| GPT-5.4 | all four | 0 | 0.00% | 0 |
| **GPT-5.5** | image | 29 | 4.83% | 26 |
| **GPT-5.5** | text+image | 22 | 3.67% | 20 |
| **GPT-5.5** | combined image | 53 | 8.83% | 48 |
| **GPT-5.5** | combined+text | 18 | 3.00% | 18 |

> **Caveat the paper does not state.** GPT-5.5 is the only model in Table 2 with
> nonzero parse failures, and in two cells they dominate its strict-swap error:

| Condition | `S` | of which parse failure | of which disagreement | PF share of `S` |
| --- | ---: | ---: | ---: | ---: |
| combined image | 23.0% | 16.0 pp | 7.0 pp | **70%** |
| image | 18.7% | 8.7 pp | 10.0 pp | **46%** |
| combined+text | 20.7% | 6.0 pp | 14.7 pp | 29% |
| text+image | 25.3% | 6.7 pp | 18.7 pp | 26% |

Read literally, GPT-5.5's combined-image row is mostly an output-format problem,
not an order-consistency problem. Section 5.1's sentence "Strict-swap error `S`
includes parsing failures and parsed disagreements, rather than pure reversals"
covers this in principle, but a reader comparing GPT-5.5 against GPT-5.4 on `S`
is not comparing like with like. Consider adding the decomposition to Table 2(a)
or a footnote.

### Table 2(b), 24B–32B open models

Qwen2.5-VL-32B, Mistral-Small-3.1-24B and Gemma-3-27B: **0 parse failures** in
all nine model–language cells (900 pairs each), and 0 in the earlier Java-only
300-pair run of the same three models.

### Excluded pilots (Section 4.4.1)

| Pilot | Result |
| --- | --- |
| GPT-5.5 aborted run (`openai_vlm_pilot_20260708`) | image-only: **all 600 calls failed**; text+image: **aborted after 122 pairs**, all 244 calls failed. This is the paper's "one aborted 122-pair GPT-5.5 run". A response-format problem, fixed in the `gpt55_fixed` rerun |
| GPT-5.4 reasoning-budget, 256 max output tokens | high reasoning: **58.89%** image-only, **20.00%** text+image; low reasoning: 0.56% image-only |
| GPT-5.4 reasoning-budget, 4,096 max output tokens | 0.00% in every condition |

The 256-token pilot is why the reported reasoning-budget check uses 4,096
tokens: at 256, a high-reasoning model spends its entire budget on hidden
reasoning and never emits the verdict line.

---

## What this implies for the manuscript

1. **Section 5.2, first sentence of "Parsing outcomes":** "two reduced-grid runs
   with retained parsing counts" should be "three". Gemma4-12B has 33/21,000
   parse-failed pairs (0.16%) from 34/42,000 calls (0.08%).
2. **Table 2(a):** GPT-5.5's `S` is up to 70% parse failure. Worth a footnote,
   since the table invites a direct `S` comparison across the three GPT models.
3. **Optional, Section 5.1:** the paper decomposes invalidity into parse failure
   and disagreement for Qwen2.5-VL-7B only. The same decomposition is available
   for all eight RQ1 judges and is 0 for six of them, which strengthens rather
   than weakens the order-consistency argument.
