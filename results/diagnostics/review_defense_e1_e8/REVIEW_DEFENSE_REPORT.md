# FSE 2027 Review-defense analyses E1-E8

## Scope and integrity

All analyses use immutable stored judgments or isolated new run IDs. GA0 passed
for 234,000 existing grid/battery calls before analysis: pair IDs, AB/BA
allocation, verdict/logit argmax, missingness, duplication, and finite-logit
checks all passed. Existing and repeated runs were never merged. Language-level
results are primary; pooled values are supplementary. Debiased ties (`c=0`) are
incorrect in main estimates, excluded in sensitivity A, and worth 0.5 in
sensitivity B. Strict-validity boundaries (`|c|=|b|`) are counted separately.

## E1: fair VLM-baseline comparison

Conclusion **(b)**: at least one exceeds a baseline, but no positive difference excludes zero. Qwen/Python
is the only point-estimate exception, at 66.77% debiased accuracy versus 62.73%
RF and 64.60% language-best Voting. Its two-way snippet-cluster confidence
intervals include zero for both differences. The manuscript must weaken “no VLM
exceeds a baseline in any language”; it may state that no excess is
distinguishable under cluster uncertainty.

## E2: run separation

The battery, clean-default, grid, RQ3, and deployment numbers are separate
runs. Battery and clean-default contain the same 3,000 Java pair IDs. Their
large InternVL swap difference (35.93% versus 54.80%) is therefore not sampling:
the selected-snippet/invalid state differs on
40.17%
of pairs. Run labels must appear in every affected table and figure.

## E3: numerical corrections

- Java OCR loss: 9.37 pp, not 10.5 pp.
- Five-model debiasing recovery: 6.60-26.27 pp, not 11.7-25.6 pp.
- 108,000 decomposition observations = 72,000 grid + 36,000 perturbation.
  There are 2,946 ties and 8,168 boundaries; 99,832 is the non-boundary count.
- OR 0.55/0.90 models strict-swap invalidity, not correct verdict. On a +0.1 z
  scale the ORs are 0.942 and 0.989.

## E4: reproducibility

Same-environment pair-state disagreement spans
5.80-
8.70% across six grid cells. Changing
only the physical GPU spans 5.80-
8.70%. The full battery has
15/15 cells at or above
the preregistered 99.5% pair-state agreement threshold. Per the fixed rule,
original Table 1/2 values are retained and repeats are reported separately.

## E5: missing cells

RQ3 text+image attempted the full 3600 contrasts and
7200 calls. Qwen passed completeness; InternVL retry 2 retained
one `FINAL_VERD` call without A/B and therefore remains missing rather than
being imputed. InternVL combined-labeled text+image separately completed on
3000 Java pairs with zero parse failures. The documented
adapter changes parsing only for malformed but unambiguous verdict prefixes.

## E6: deployment

# E6 conclusion

The deployment conclusion remains true only when stated as end-to-end effective accuracy, not conditional valid accuracy.

- cuda: RapidOCR+ML effective=60.73%; best neural judge effective=49.97%. Conclusion retained=True.
- java: RapidOCR+ML effective=52.97%; best neural judge effective=38.67%. Conclusion retained=True.
- python: RapidOCR+ML effective=59.53%; best neural judge effective=35.27%. Conclusion retained=True.


## E7: closed-model pilots

The package exposes all existing 300-pair Java closed-model generations and
separates standard 24-token runs from GPT-5.4 low/high 4,096-token runs. These
remain protocol checks, not controlled scale comparisons. The GPT-5.4
high/4,096 cross-modality result reproduces swap error 11.33% to 29.00%,
exact McNemar p=1.18e-7.

## E8: restructuring inputs

All 66 rendering and 30 perturbation contrasts retain effect, cluster CI,
exact McNemar p, Holm p, and direction. Coverage/consistency is the larger
effective-accuracy component in
70/
96 cells. On easy-only and Java-within-dataset slices, no VLM
exceeds the best ML baseline. The appearance-only RQ3 contrast remains fragile:
Qwen beautiful-trash versus ugly-trash has 25/300 valid pairs and 8.33%
effective target preference.

## Required manuscript edits

1. Weaken the universal E1 baseline claim and insert the language-specific
   debiased comparison with cluster intervals.
2. Add `battery`, `clean_default`, `grid`, `rq3`, and `deploy` labels; never
   compare their cells as controlled treatments.
3. Correct the four E3 statements exactly as listed above.
4. Add E4 noise-floor and battery reproducibility results without replacing
   original Table 1/2 values.
5. Fill E5 cells and document the output-format adapter and old failures.
6. Replace Table 5 with the denominator-complete E6 table.
7. Add the E7 numeric pilot table with its protocol-check caption.
8. Put the multiplicity family table in Section 3.7 and use the four-row RQ3
   compact view.

## Reproduction

Scripts are under `code/`; immutable results are under `analysis/`; isolated new
runs are under `runs/`; logs and environment snapshots are under `logs/` and
`env/`. `SHA256SUMS` covers every regular package artifact except itself.
