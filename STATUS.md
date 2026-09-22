# Artifact status

We apply for **Available**, **Functional**, and **Reusable**.

## Available

The package is self-contained and intended for deposit in a public archive with
a DOI. It carries no author-identifying information, so it can be linked from a
double-blind submission.

*Outstanding:* `LICENSE` is still a placeholder, and
`data/benchmarks_raw/downloads/` redistributes third-party benchmark archives.
See README §10.

## Functional

Documented, consistent, complete, and exercisable:

* **Documented.** [`README.md`](README.md) gives the layout and the entry
  points. [`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md) maps every
  quantitative claim in the paper to a file and the script that produced it.
  [`REQUIREMENTS.md`](REQUIREMENTS.md) states hardware, software, and cost at
  four levels of engagement.
* **Consistent.** Every artifact here is referenced by the paper. Material from
  the same research line that the paper does not use is listed and justified in
  [`EXCLUSIONS.md`](EXCLUSIONS.md) rather than silently dropped.
* **Complete.** The full chain is present: published benchmark archives → 552
  standardized snippets → 9,000 pairs → 11,688 rendered PNGs → raw model calls
  with verdict-token logits → pair-level outcomes → figure and table data.
* **Exercisable.** Three commands reproduce the paper on a laptop with no GPU
  and no third-party packages, in under ten seconds of CPU time:

  ```bash
  python3 code/99_verification/recompute_headline_numbers.py   # RQ1
  python3 code/99_verification/recompute_rq2_rq3_rq4.py        # RQ2, RQ3, RQ4
  python3 code/99_verification/audit_parse_failures.py         # parse-failure counts
  ```

  The second runs ~90 checks, prints each recomputed value beside the
  manuscript value, and exits non-zero on any mismatch. All three expected
  outputs are checked in under `results/verified/`, so a reviewer can `diff`
  rather than read the numbers by eye; README §2.6 gives the exact commands.

### What we verified during assembly

Recomputed from the shipped raw data, not copied from the manuscript:

| Result | Outcome |
| --- | --- |
| Figure 6: per-language valid accuracy and `E`, 24 cells | match (3 cells differ by 0.1 pp through half-rounding, documented) |
| Figure 7: pooled `E`, `D`, `S` for all 8 judges | exact match |
| Figure 8(a): wrap and font contrasts, all 5 judges | exact match |
| Figure 8(b): indentation, all 6 cells incl. the 89-pair Dorn-only Java cell | exact match |
| Figure 8(c): both blur examples | exact match |
| Figure 9 / Section 5.2: held-out AUC range 0.52–0.63 | exact match |
| Figure 10: all 12 RQ3 contrast cells, plus per-language ranges | exact match |
| Table 2(a): GPT-5.4-mini, 4 conditions × 3 metrics | exact match |
| Table 2(b): 3 models × 3 languages × 3 metrics | exact match |
| Table 3(a): frozen-RF OCR substitution, 4 columns | exact match |
| Table 3(b): 3 pipelines × 3 languages, plus strict-rejection `E` | exact match |
| Table 3(c): 3 inputs × 4 metrics | match under the stated tie convention |
| Section 4.1: all pair-construction counts | exact match |
| Section 4.1: rater-split reference stability | exact match |
| Section 5.1: best source-feature baselines and the 67.9% pooled reference | exact match against the corrected file |
| Section 5.1: first-position selection rate 24.6–78.1% | exact match |
| Section 5.2: factorial wrap and font effects | exact match |
| Section 5.2: 22 of 66 contrasts significant after Holm | exact match |
| Section 5.2: indentation, 1.8–2.8 pp over 4 above-chance judges | exact match |
| Section 5.2: order-vs-rendering, all 15 groups and 6 ranges | exact match |
| Section 5.2: encoder top-one retrieval ranges | exact match (scope wording needs a fix: retrieval uses the 12-condition grid, not 17) |
| RQ1 raw call counts: 18,000 per judge × 8 judges | exact match |
| Section 5.2: reduced-grid parse-failure counts | Gemma-3-12B and InternVL3.5-8B exact; **Gemma4-12B's 33/21,000 unreported** |
| Section 5.3: "all 7,200 RQ3 calls parsed" | exact match |
| Section 5.4: "neither text condition has parsing failures" | exact match |

No reported value failed verification. Six smaller issues — one interval
computed against a superseded baseline, two unreported parse-failure facts,
three rounding or scope-wording details
— are reported in the "Known discrepancies" section of
`CLAIMS_TO_ARTIFACTS.md`. We report them rather than adjust the artifact.

A first pass of this verification *did* report a mismatch on the Python
baseline, because it used the convenient nine-predictor aggregation file, which
is stale for three of 9,000 pairs. That file is now flagged in
`CLAIMS_TO_ARTIFACTS.md` under "A trap for replicators", and
`recompute_headline_numbers.py` prints the stale and corrected columns side by
side so the next reader does not repeat it.

## Reusable

* **Beyond the paper's own results.** The 45,000-pair diagnostic set carries
  per-call verdict-token logits, so the position-aligned (`b`) and
  candidate-oriented (`c`) components can be recomputed under decision policies
  other than ours. Because changing only the tie-boundary convention moves
  Qwen2.5-VL-7B's strict-swap error by 6.78 pp, this is the part of the data we
  expect to be most reused.
* **Reusable inputs.** The 11,688 rendered PNGs span 18 conditions over a fixed
  snippet set, and are usable as a rendering-sensitivity benchmark for any
  image-input model, independent of readability.
* **Pinned and deterministic.** Exact Hugging Face revisions for all 12
  open-weight models (`environment/MODEL_REVISIONS.json`); greedy decoding,
  seed 42, single frozen prompt whose byte-identity across all runs is verified
  by SHA-256.
* **Honest about limits.** Pre-registered protocols, integrity gates, and
  negative results — including a blocked InternVL text+image arm that was not
  promoted past its 100-call gate — ship alongside the successful runs.

## Not claimed: Results Reproduced

Re-running inference requires roughly 250 GPU-hours on ~97 GB-class GPUs plus
gated model access, which exceeds a normal evaluation budget. Tiers 0–2 of
`REQUIREMENTS.md` let a reviewer confirm every reported number from the shipped
model outputs without any GPU.
