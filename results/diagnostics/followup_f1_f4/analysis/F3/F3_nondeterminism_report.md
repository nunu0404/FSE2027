# F3 grid nondeterminism report

## Setting audit

The original grid and E4-A are not a same-environment repeat. Both model
manifests record Python 3.13.11 for the original grid and Python 3.10.12 for
E4-A. Prompt, model revision, protocol hash, rendering hash, BF16 dtype,
greedy decoding, seed, and GPU index are unchanged. The original run processed
72,000 calls per model over 12 conditions; E4-A processed 6,000 baseline calls
per model. Both runners issue one generation at a time, but the overall request
schedule and runtime environment differ. Package versions, KV-cache policy,
CUDA-graph mode, and deterministic-kernel flags were not fully captured in the
E4-A manifest, so they cannot be asserted equal post hoc.

| Setting | Original grid | E4-A | E4-B | Finding |
|---|---|---|---|---|
| Python | 3.13.11 | 3.10.12 | 3.10.12 | changed before E4-A |
| Calls/model | 72,000 | 6,000 | 6,000 | workload changed |
| Conditions | 12 | baseline only | baseline only | request schedule changed |
| GPU index | 0 | 0 | 1 | E4-B isolates physical GPU within repeat runtime |
| Engine | direct Transformers `generate` | same source runner | same source runner | no serving engine |
| Batching | one generation/call; two model processes concurrent | same | same | no continuous batching |
| TP/PP | none; one GPU/model process | same | same | unchanged |
| Seed | set once/process to 42 | same | same | process-level seed |
| Prompt/model/render | SHA/revision pinned | identical | identical | unchanged |
| KV cache/CUDA graph | not manifest-recorded | not manifest-recorded | not manifest-recorded | unverified |
| Deterministic kernels | no explicit deterministic-algorithm flag in runner | same source | same source | not enforced |
| Processor log | no fast-default warning recorded | Qwen logs a fast-processor default change; InternVL logs slow processor | same repeat stack | library behavior visibly differs |

E4-B uses the same Python 3.10.12 repeat setup as E4-A and changes the physical
GPU. E4-A and E4-B are decision-identical, which excludes the GPU UUID as an
additional observed source under that repeat environment.

## Mismatch characteristics

Across the six cells, verdict-mismatching calls have median original absolute
margin 0.1250--0.1250; matching calls
have median 0.8750--1.0000. Per-cell |c|,
|b|, ties, boundaries, and logit-difference distributions are in
`F3_mismatch_call_summary.csv`; every call is retained in
`F3_mismatch_call_detail.csv`.

## Cause statement

**Confirmed immediate cause of the apparent contradiction:** the experiment
label was wrong; E4-A changed the Python/runtime environment and workload
relative to the original grid, whereas the exact battery repeat retained
Python 3.13.11. The lower-level numerical mechanism is **not confirmed** from
stored metadata. Runtime/library and request-schedule changes can alter BF16
kernel execution near decision boundaries, but attributing the flips
specifically to continuous batching or floating-point accumulation would go
beyond the evidence: this runner is not a continuous-batching server.

## Excluded or unsupported hypotheses

- Physical GPU change: no added effect was observed between E4-A and E4-B.
- Sampling randomness: decoding was greedy with `do_sample=false`.
- Prompt/model/render changes: pinned hashes and model revisions match.
- Continuous batching: not used by the direct Transformers runner.
- Exact package/kernel cause: unresolved because package and kernel flags were
  not recorded in the E4-A manifest.

## Additional inference required, not executed

A true same-environment baseline repeat under the original pinned Python
3.13.11 environment requires 12,000 calls (2 models x 3 languages x 1,000
pairs x 2 orders). A controlled Python 3.10 versus 3.13 attribution would
require another 12,000 calls, for 24,000 total. No such inference was launched.

## English draft for Section 3.6 / Threats

Post hoc manifest auditing showed that the run initially labelled as a
same-environment grid repeat used Python 3.10.12, whereas the original grid
used Python 3.13.11, and also reduced the workload from the 12-condition grid
to the baseline condition. We therefore treat its 3.0--4.4% call disagreement
as cross-environment reproducibility, not a same-environment noise floor.
Changing only the physical GPU within the repeat environment produced no
additional decision disagreement. Because the repeat manifest did not capture
all package and kernel settings, the precise low-level numerical cause remains
unresolved; we do not attribute it specifically to continuous batching.
