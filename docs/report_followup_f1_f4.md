# FSE 2027 follow-up F1-F4 final report

Run package: `fse2027_followup_f1_f4_20260731`

New model-inference calls: **0**.

## Executive corrections

1. F1 cannot be computed from the deployment run because its AB/BA assets do
   not store verdict-token logits. D is reported as not measured in all nine
   neural row-language cells; no run was merged.
2. E4-A was mislabelled as a same-environment repeat. The original grid used
   Python 3.13.11 and E4-A used Python 3.10.12. Its disagreement is therefore
   cross-environment, not a same-environment floor.
3. Direct reaggregation of all 66 grid rendering cells gives a both-valid flip
   range of 0.00%--8.31%, not the manuscript's 0.6%--7.1%. The detailed
   condition-level table is authoritative until the manuscript scope is
   reconciled.
4. The public Dorn Java rater matrix reproduces 89/90 selected proxy means.
   Snippet `rq0_0099` has no reproducible rating column, so its nine
   within-Dorn pairs are explicitly unavailable.

## F1 deployment debiased column

| Language | Row | E | V | S | D=sign(c) | OCR+ML E |
|---|---|---|---|---|---|---|
| java | Direct Qwen image-only | 38.67% | 62.13% | 37.77% | not measured | 52.97% |
| java | Source-text Qwen2.5-Coder | 33.30% | 55.62% | 40.13% | not measured | 52.97% |
| java | Best OCR-text LLM (RapidOCR) | 28.47% | 55.82% | 49.00% | not measured | 52.97% |
| python | Direct Qwen image-only | 35.27% | 71.83% | 50.90% | not measured | 59.53% |
| python | Source-text Qwen2.5-Coder | 26.07% | 57.42% | 54.60% | not measured | 59.53% |
| python | Best OCR-text LLM (RapidOCR) | 9.87% | 62.71% | 84.27% | not measured | 59.53% |
| cuda | Direct Qwen image-only | 49.97% | 71.45% | 30.07% | not measured | 60.73% |
| cuda | Source-text Qwen2.5-Coder | 33.20% | 57.77% | 42.53% | not measured | 60.73% |
| cuda | Best OCR-text LLM (RapidOCR) | 21.57% | 60.19% | 64.17% | not measured | 60.73% |

**Answer:** the stored assets cannot determine whether the strongest
screenshot-only system changes after adding D. The currently measurable
effective winners remain RapidOCR+SVR 52.97% (Java), RapidOCR+GB 59.53%
(Python), and RapidOCR+Linear 60.73% (CUDA). A valid D rerun would require
54,000 calls and must preserve the two-call cost, logit-access/open-model
requirement, and supervised-versus-zero-shot asymmetry.

## F4 both-valid floor

`Original vs E4-A` is a cross-environment comparison. `E4-A vs E4-B` isolates
the GPU change inside the repeat runtime.

| Model | Lang | Orig/E4A mismatches | Cross-env floor | E4A/E4B mismatches | GPU-only floor | Rendering flip range |
|---|---|---|---|---|---|---|
| Qwen | cuda | 0/599 | 0.00% | 0/629 | 0.00% | 0.00%--1.20% |
| Qwen | java | 0/484 | 0.00% | 0/512 | 0.00% | 0.24%--3.81% |
| Qwen | python | 1/393 | 0.25% | 0/429 | 0.00% | 0.30%--1.56% |
| InternVL | cuda | 0/270 | 0.00% | 0/305 | 0.00% | 0.00%--3.45% |
| InternVL | java | 0/593 | 0.00% | 0/640 | 0.00% | 0.36%--8.31% |
| InternVL | python | 2/429 | 0.47% | 0/467 | 0.00% | 0.90%--5.13% |

There is no defensible same-environment grid floor in the stored runs.
Nevertheless, strict-valid decisions are highly stable across the runtime
change: four cells have 0 disagreements, Qwen-Python has 1/393 (0.25%), and
InternVL-Python has 2/429 (0.47%). E4-A and E4-B have zero disagreements in all
six both-valid comparisons, so the physical GPU change adds no observed
effect. This does not rescue the original “same environment” label.

## F3 nondeterminism cause

| Model | Lang | Changed calls | Rate | |margin| changed | |margin| same | |c| changed | |b| changed | logit diff med. | logit diff max | tie/boundary calls |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | cuda | 62/2000 | 3.10% | 0.125 | 0.875 | 0.406 | 0.438 | 0.250 | 0.875 | 3/26 |
| Qwen | java | 60/2000 | 3.00% | 0.125 | 0.875 | 0.469 | 0.531 | 0.250 | 0.875 | 2/19 |
| Qwen | python | 75/2000 | 3.75% | 0.125 | 1.000 | 0.688 | 0.750 | 0.250 | 1.000 | 2/27 |
| InternVL | cuda | 73/2000 | 3.65% | 0.125 | 0.875 | 0.688 | 0.688 | 1.250 | 3.125 | 2/24 |
| InternVL | java | 87/2000 | 4.35% | 0.125 | 0.875 | 0.500 | 0.438 | 1.250 | 3.125 | 9/25 |
| InternVL | python | 70/2000 | 3.50% | 0.125 | 0.875 | 0.688 | 0.625 | 1.375 | 3.250 | 1/22 |

Changed calls are concentrated at absolute verdict margins of 0.125, versus
0.875--1.000 for unchanged calls. Qwen's changed-call maximum logit difference
is at most 1.000 across cells; InternVL's is 3.125--3.250. The immediate cause
of the earlier contradiction is confirmed as an invalid environment label:
the grid repeat changed Python/runtime and workload, while the exact battery
repeat retained Python 3.13.11. The precise lower-level kernel/library cause
remains unconfirmed; continuous batching is excluded because this is a
single-call direct-Transformers runner. A true original-environment repeat
would require 12,000 new calls; a controlled 3.10-vs-3.13 attribution would
require 24,000 total. Neither was executed.

## F2 Py/CUDA proxy-label agreement ceiling

| Lang | Difficulty | Estimator | Agreement | 95% cluster CI | Pairs | Draw-tie probability |
|---|---|---|---|---|---|---|
| cuda | easy | k=1 | 75.90% | [74.37%, 77.39%] | 1000 | 18.72% |
| cuda | easy | k=panel | 100.00% | [100.00%, 100.00%] | 1000 | 0.00% |
| cuda | hard | k=1 | 55.39% | [55.20%, 55.60%] | 1000 | 25.70% |
| cuda | hard | k=panel | 96.19% | [95.71%, 96.67%] | 1000 | 0.01% |
| cuda | medium | k=1 | 61.31% | [60.97%, 61.68%] | 1000 | 24.67% |
| cuda | medium | k=panel | 99.97% | [99.95%, 99.98%] | 1000 | 0.00% |
| python | easy | k=1 | 72.67% | [71.48%, 73.82%] | 1000 | 21.39% |
| python | easy | k=panel | 100.00% | [100.00%, 100.00%] | 1000 | 0.00% |
| python | hard | k=1 | 55.64% | [55.41%, 55.89%] | 1000 | 27.32% |
| python | hard | k=panel | 96.53% | [95.97%, 97.02%] | 1000 | 0.01% |
| python | medium | k=1 | 61.69% | [61.31%, 62.07%] | 1000 | 25.83% |
| python | medium | k=panel | 99.97% | [99.96%, 99.98%] | 1000 | 0.00% |

## F2 Java within-benchmark ceiling

| Benchmark | Difficulty | Estimator | Agreement | 95% cluster CI | Used/requested | Unavailable |
|---|---|---|---|---|---|---|
| Buse | easy | k=1 | 77.87% | [75.26%, 80.47%] | 106/106 | 0 |
| Buse | easy | k=panel | 100.00% | [100.00%, 100.00%] | 106/106 | 0 |
| Buse | hard | k=1 | 55.85% | [55.27%, 56.45%] | 99/99 | 0 |
| Buse | hard | k=panel | 93.67% | [92.05%, 95.12%] | 99/99 | 0 |
| Buse | medium | k=1 | 61.97% | [61.04%, 62.90%] | 94/94 | 0 |
| Buse | medium | k=panel | 99.84% | [99.72%, 99.92%] | 94/94 | 0 |
| Dorn | easy | k=1 | 69.22% | [67.04%, 71.76%] | 77/79 | 2 |
| Dorn | easy | k=panel | 100.00% | [100.00%, 100.00%] | 77/79 | 2 |
| Dorn | hard | k=1 | 53.89% | [53.48%, 54.28%] | 91/96 | 5 |
| Dorn | hard | k=panel | 91.02% | [89.06%, 92.91%] | 91/96 | 5 |
| Dorn | medium | k=1 | 58.75% | [58.09%, 59.39%] | 98/100 | 2 |
| Dorn | medium | k=panel | 99.73% | [99.55%, 99.87%] | 98/100 | 2 |
| Scalabrino | easy | k=1 | 80.39% | [77.63%, 82.94%] | 159/159 | 0 |
| Scalabrino | easy | k=panel | 98.38% | [97.53%, 99.04%] | 159/159 | 0 |
| Scalabrino | hard | k=1 | 56.89% | [55.87%, 57.88%] | 81/81 | 0 |
| Scalabrino | hard | k=panel | 70.54% | [69.00%, 72.42%] | 81/81 | 0 |
| Scalabrino | medium | k=1 | 63.03% | [61.98%, 64.06%] | 190/190 | 0 |
| Scalabrino | medium | k=panel | 84.76% | [82.90%, 86.58%] | 190/190 | 0 |
| Buse | all | k=1 | 65.58% | [63.39%, 68.09%] | 299/299 | 0 |
| Buse | all | k=panel | 97.85% | [97.07%, 98.52%] | 299/299 | 0 |
| Dorn | all | k=1 | 60.12% | [58.51%, 62.00%] | 266/275 | 9 |
| Dorn | all | k=panel | 96.83% | [95.65%, 97.86%] | 266/275 | 9 |
| Scalabrino | all | k=1 | 68.29% | [66.10%, 70.69%] | 430/430 | 0 |
| Scalabrino | all | k=panel | 87.12% | [84.91%, 89.25%] | 430/430 | 0 |

Java cross-dataset pairs: **1,996, undefined**. They join distinct benchmark
panels and scales, so no same-rater or same-panel ceiling was interpolated.

## Table 1 reference check

This table shows the highest battery valid accuracy in each available stratum,
not “human accuracy.”

| Lang | Benchmark | Difficulty | Best model valid | Panel proxy ceiling | Ratio |
|---|---|---|---|---|---|
| cuda | Dorn | easy | 87.99% | 100.00% | 87.99% |
| cuda | Dorn | hard | 60.27% | 96.19% | 62.66% |
| cuda | Dorn | medium | 73.57% | 99.97% | 73.60% |
| java | Buse | easy | 86.81% | 100.00% | 86.81% |
| java | Buse | hard | 62.96% | 93.67% | 67.22% |
| java | Buse | medium | 67.27% | 99.84% | 67.38% |
| java | Dorn | easy | 50.91% | 100.00% | 50.91% |
| java | Dorn | hard | 46.38% | 91.02% | 50.95% |
| java | Dorn | medium | 57.78% | 99.73% | 57.93% |
| java | Scalabrino | easy | 96.55% | 98.38% | 98.15% |
| java | Scalabrino | hard | 68.18% | 70.54% | 96.65% |
| java | Scalabrino | medium | 81.13% | 84.76% | 95.72% |
| python | Dorn | easy | 84.44% | 100.00% | 84.44% |
| python | Dorn | hard | 65.82% | 96.53% | 68.18% |
| python | Dorn | medium | 73.72% | 99.97% | 73.74% |

All easy-stratum model values are below the panel proxy-agreement reference.
Where a model approaches or exceeds a finite Monte Carlo reference in another
stratum, it should be described as alignment with a fixed mean-score proxy,
not as superhuman performance.

## Statistical definitions

- `k=1`: exact agreement probability after independently drawing one released
  rating for each snippet; rating ties receive 0.5 credit.
- `k=panel`: 10,000 bootstrap panel means per snippet at that snippet's
  original panel size; ties receive 0.5 credit.
- CI: 10,000 two-endpoint snippet-cluster bootstrap replicates.
- F1 ties (`c=0`) and boundaries (`|c|=|b|`) are unavailable because logits
  were not stored.
- F3/F4 tie and boundary counts are retained separately in the CSV outputs.

## Manuscript actions

1. Table 5 must show D as `not measured`, not borrow D from another run.
2. Replace “same-environment grid noise floor” with “cross-environment
   reproducibility check”; retain the zero GPU-only additional effect.
3. Reconcile Section 6.1's 0.6%--7.1% scope against the complete 66-cell
   reaggregation (0.0%--8.31%) before publication.
4. Call F2 a “proxy-label agreement ceiling,” never “human accuracy,” and
   state that Java cross-dataset is undefined.
