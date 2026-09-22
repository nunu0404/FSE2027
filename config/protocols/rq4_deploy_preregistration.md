# F5 deployment-v2 logit preregistration

- Run ID: `deploy_v2_logit_20260731`
- Registered before inference: 2026-07-31 KST
- New inference budget: exactly 54,000 calls
- Systems: Direct Qwen2.5-VL-7B image-only; source-text
  Qwen2.5-Coder-7B; RapidOCR-text Qwen2.5-Coder-7B
- Population: the unmodified deployment pairs, 3,000 each for Java, Python,
  and CUDA; AB and BA order for every pair.
- Run isolation: this run is not merged with any previous deployment run.

## H-F5

For each language, the debiased accuracy of Direct Qwen image-only exceeds the
effective accuracy of the best RapidOCR+ML system computed on the same frozen
pairs. The directional expectation is strongest for Python, then CUDA, with
Java expected to be closest to the boundary.

## Comparisons

The primary comparisons are, separately for Java, Python, and CUDA:

`D(Direct Qwen image-only) - E(best RapidOCR+ML)`.

Secondary comparisons replace Direct Qwen with source-text Qwen2.5-Coder and
RapidOCR-text Qwen2.5-Coder. Exact McNemar p-values are Holm-adjusted across
all nine row-language comparisons. Conclusions use only deployment-v2 values;
old deployment outputs are used solely for reproducibility comparisons.

## Fixed decision rules

- `m_AB` and `m_BA` are first-position-minus-second-position verdict margins.
- `c = (m_AB - m_BA) / 2`; `b = (m_AB + m_BA) / 2`.
- Main D accuracy uses `sign(c)`; `c=0` receives zero credit.
- Sensitivities: exclude `c=0`, and give `c=0` half credit.
- `|c|=|b|` with `c!=0` is a boundary and is counted separately.
- E is strict-valid correct pairs divided by all pairs.
- V is correct divided by strict-valid pairs.
- S is order-inconsistent or unparsable pairs divided by all pairs.
- AB-only uses the parsed AB decision without swap filtering.
- Confidence intervals use 10,000 two-endpoint snippet-cluster bootstrap
  replicates. All pair-level comparisons use the same frozen pairs.

The preregistered conclusion is:

- (a) Direct Qwen D exceeds OCR+ML E in no language.
- (b) A point estimate exceeds OCR+ML E, but its 95% cluster CI includes zero.
- (c) A point estimate exceeds OCR+ML E and its 95% cluster CI excludes zero.

If languages differ, the overall label is the strongest category reached and
the language-specific category is always reported. Category (c) requires
revising Section 8.1.

## Frozen inference environment

- One Python environment and one physical GPU are used for all 54,000 calls.
- Execution is sequential: Direct Qwen first; one Coder load then source text
  followed by RapidOCR text.
- Direct Transformers generation, batch size one, no continuous batching,
  TP, or PP.
- BF16, greedy decoding (`do_sample=false`), inactive `top_p=1.0`,
  `max_new_tokens=24`, seed 42.
- Model revisions are pinned in the environment manifest.
- A read-only LM-head hook records A/B verdict logits without changing the
  generation arguments.
- No package, model, input, prompt, driver, or environment update is permitted
  after GA0 starts.

## Frozen inputs

The exact pair CSVs, PNGs, source-text tables, RapidOCR-text tables, Prompt B,
old outputs, and deterministic OCR+ML prediction assets are enumerated by
`data/F5_FROZEN_INPUT_INVENTORY.csv`. Images and OCR text are never
regenerated in F5.

## Stop gates

Before the full run:

1. All frozen inputs must exist and match their recorded SHA-256.
2. Each language must contain exactly 3,000 unique pairs and 6,000 calls per
   system.
3. GA0 must emit parseable A/B verdicts, non-null A/B logits, and a verdict
   argmax matching the parsed verdict on every pilot call.
4. RapidOCR+ML recomputation must reproduce Java 52.97%, Python 59.53%, and
   CUDA 60.73%; otherwise inference is not launched.

No threshold, tie rule, comparison family, or conclusion category may be
changed after observing F5 outcomes.
