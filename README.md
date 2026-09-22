# Replication Package

## How Reliable Are VLM Judges of Rendered Code Readability?
### Rating Agreement, Order Consistency, and Rendering Sensitivity

Replication package for the paper submitted to **FSE 2027**.

This package contains the inputs, code, rendered images, raw model outputs,
derived results, and analysis artifacts behind **every number reported in the
paper**. A reviewer can confirm all reported values on a laptop, without a GPU,
without network access, and without installing a single third-party package.

> **Start here:** [§2 Getting Started](#2-getting-started-5-minutes-no-dependencies)
> reproduces the paper's headline numbers in under 10 seconds.

---

## Table of contents

1. [Purpose — badges claimed](#1-purpose--badges-claimed)
2. [Getting started (5 minutes, no dependencies)](#2-getting-started-5-minutes-no-dependencies)
3. [Step-by-step: reproducing every figure and table](#3-step-by-step-reproducing-every-figure-and-table)
4. [Provenance](#4-provenance)
5. [Data](#5-data)
6. [Setup — the four engagement tiers](#6-setup--the-four-engagement-tiers)
7. [Package structure](#7-package-structure)
8. [Reusability beyond this paper](#8-reusability-beyond-this-paper)
9. [Limitations and known discrepancies](#9-limitations-and-known-discrepancies)
10. [License and third-party terms](#10-license-and-third-party-terms)

---

## 1. Purpose — badges claimed

We apply for **Artifacts Available**, **Artifacts Evaluated — Functional**, and
**Artifacts Evaluated — Reusable**.

| Badge | Basis |
| --- | --- |
| **Available** | Self-contained, permanently archived, carries no author-identifying information ([`ANONYMIZATION.md`](ANONYMIZATION.md)). |
| **Functional** | *Documented* — this README plus [`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md), which maps every quantitative claim in the manuscript to a file and to the script that produced it. *Consistent* — every artifact here is used by the paper; related material the paper does **not** use is listed and justified in [`EXCLUSIONS.md`](EXCLUSIONS.md) rather than silently dropped. *Complete* — the full chain ships: published benchmark archives → 552 standardized snippets → 9,000 pairs → 11,688 rendered PNGs → raw model calls with verdict-token logits → pair-level outcomes → figure and table source data. *Exercisable* — see §2. |
| **Reusable** | Pinned model revisions, frozen prompt verified byte-identical by SHA-256, pre-registered protocols, per-call verdict-token logits enabling re-analysis under decision policies other than ours, and negative results retained alongside successful runs. See §8. |

**We do not claim *Results Reproduced*.** Re-running inference costs roughly
250 GPU-hours on 97 GB-class GPUs and needs gated model access, which exceeds a
normal evaluation budget. Tiers 0–2 in §6 let a reviewer confirm **every
reported number** from the shipped model outputs with no GPU at all.

### What the study does

Five open-weight vision-language models judge pairwise code readability from
screenshots alone, on 9,000 pairs built from 552 human-rated Java, Python, and
CUDA snippets. Every pair is judged twice with the candidate order reversed.

| RQ | Question | Results |
| --- | --- | --- |
| RQ1 | Can image-only VLM judges match source-feature predictors, and does agreement on order-consistent pairs survive counting every pair? | [`results/rq1_viability/`](results/rq1_viability/) |
| RQ2 | How does sensitivity to candidate order compare with sensitivity to rendering, packaging, and modality? | [`results/rq2_reliability/`](results/rq2_reliability/) |
| RQ3 | Do judges prefer original code rendered badly, or damaged code rendered cleanly? | [`results/rq3_content_vs_appearance/`](results/rq3_content_vs_appearance/) |
| RQ4 | With only images available, does direct VLM judging beat an OCR-assisted pipeline? | [`results/rq4_image_only/`](results/rq4_image_only/) |

---

## 2. Getting started (5 minutes, no dependencies)

**Requirements for this section:** Python 3.8+ (standard library only), ~3 GB
free disk. No GPU, no network, no `pip install`.

### 2.1 Obtain the package

```bash
git clone https://github.com/nunu0404/FSE2027.git
cd FSE2027
```

The repository is ~2.9 GB, so the clone takes a few minutes. A
[`.gitattributes`](.gitattributes) file disables all text normalization
(`* -text -diff`) so that every file is checked out **byte-for-byte** as it was
produced — this is what makes the SHA-256 manifest in §2.2 verifiable after a
clone on any platform.

### 2.2 Verify package integrity (optional, ~3 minutes)

```bash
bash code/99_verification/verify_manifest.sh
```

Expected final line:

```
OK: 14462 files verified
```

`MANIFEST.sha256` records a SHA-256 for every file in the package except itself
(and except `.gitattributes`, which was added for hosting and carries no data).

### 2.3 Reproduce the headline numbers (~1 second)

```bash
python3 code/99_verification/recompute_headline_numbers.py
```

This recomputes, **from the raw shipped model outputs**, the abstract's
71% / 46% / 60–65% figures, Figures 6 and 7, the best source-feature baselines,
the 67.9% pooled reference, and the Section 4.1 pair-construction counts. Each
recomputed value is printed next to the manuscript value.

### 2.4 Reproduce RQ2, RQ3 and RQ4 (~1 second)

```bash
python3 code/99_verification/recompute_rq2_rq3_rq4.py
```

This runs ~90 checks over Figures 8 and 10 and Tables 3(a)–(c). It prints each
recomputed value beside the manuscript value and **exits non-zero if any check
mismatches**, so it doubles as a smoke test. Expected final line:

```
All RQ2, RQ3 and RQ4 checks reproduce the manuscript.
```

### 2.5 Reproduce the parse-failure audit (~4 seconds)

```bash
python3 code/99_verification/audit_parse_failures.py
```

Recomputes every parse-failure count reported in Sections 5.2–5.4. The written
analysis is [`docs/PARSE_FAILURE_AUDIT.md`](docs/PARSE_FAILURE_AUDIT.md).

### 2.6 Confirm the outputs match byte-for-byte

All three scripts have their expected output checked in, so a reviewer does not
have to read the numbers by eye:

```bash
diff <(python3 code/99_verification/recompute_headline_numbers.py) \
     results/verified/EXPECTED_HEADLINE_OUTPUT.txt        && echo "RQ1       MATCH"
diff <(python3 code/99_verification/recompute_rq2_rq3_rq4.py) \
     results/verified/EXPECTED_RQ2_RQ3_RQ4_OUTPUT.txt     && echo "RQ2-RQ4   MATCH"
diff <(python3 code/99_verification/audit_parse_failures.py) \
     results/verified/EXPECTED_PARSE_FAILURE_AUDIT.txt    && echo "PARSE     MATCH"
```

Three `MATCH` lines and no `diff` output means every reported value in the paper
has been reproduced from the shipped data on your machine.

---

## 3. Step-by-step: reproducing every figure and table

Section 2 already covers all of these through the three verification scripts.
This section says **which file backs each paper artifact**, so a reviewer can
inspect the underlying data directly rather than trust a summary. The exhaustive
claim-by-claim map, including every in-text number, is
[`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md).

| Paper artifact | Source data | Verified by |
| --- | --- | --- |
| **Fig. 6** — RQ1 per-language `A_valid` / `E`, 8 judges | `figures_and_tables/fig06_rq1_by_language/fig06_data.csv` | §2.3 |
| **Fig. 7** — pooled `E`, `D`, `S`, policy gap `D − E` | `figures_and_tables/fig07_order_robustness/fig07_data.csv` | §2.3 |
| **Fig. 8(a)** — wrap and font contrasts at the Monokai baseline | `figures_and_tables/fig08_rendering_perturbation/A_GRID_RESULTS.csv` | §2.4 |
| **Fig. 8(b)** — indentation removed, order-consistent pairs | `figures_and_tables/fig08_rendering_perturbation/B_DORN_PRIMARY_RESULTS.csv` | §2.4 |
| **Fig. 8(c)** — strong blur, two examples | `figures_and_tables/fig08_rendering_perturbation/B_PERTURBATION_RESULTS.csv` | §2.4 |
| **Fig. 9** — logit-margin magnitude vs rating agreement, held-out AUC 0.52–0.63 | `figures_and_tables/fig09_logit_margin/fig09_data_5_diagnostic_judges.csv`, `isotonic_auc_and_trend_summary.csv` | §2.3 |
| **Fig. 10** — RQ3 source vs appearance preferences, 12 cells | `figures_and_tables/fig10_rq3_preferences/RQ3_IMAGE_ONLY_BY_CONTRAST.csv` | §2.4 |
| **Table 1** — experiment scope | Pre-registered protocols in `config/protocols/`; revisions in `environment/MODEL_REVISIONS.json` | descriptive |
| **Table 2(a)** — closed models, Java, 300 pairs/condition | `figures_and_tables/tab02_protocol_checks/E7_closed_model_pilot.csv` | §2.4 |
| **Table 2(b)** — open 24B–32B models, image only | `figures_and_tables/tab02_protocol_checks/large_open_java300_aggregate_metrics.csv` | §2.4 |
| **Table 3(a)** — frozen-RF OCR feature substitution (Java) | `figures_and_tables/tab03_rq4/table3a_frozen_rf_ocr_substitution.csv` | §2.4 |
| **Table 3(b)** — image-only pipelines, all-pair accuracy | `figures_and_tables/tab03_rq4/F5_deploy_debiased.csv`, `recomputed_intervals.csv` | §2.4 |
| **Table 3(c)** — Qwen2.5-VL-7B input ablation | `figures_and_tables/tab03_rq4/table3c_input_ablation_summary.csv` | §2.4 |
| **§4.1** — pair construction, strata, rater-split stability | `data/pairs/rq1_pairs_9000_seed42.csv`; `results/diagnostics/noise_ceiling/` | §2.3 |
| **§5.2** — order-vs-rendering, all 15 model×language groups | `results/rq2_reliability/verified_recomputation/rq2_order_vs_rendering.csv` | §2.4 |
| **§5.2** — 22 of 66 contrasts significant after Holm | `results/diagnostics/review_defense_e1_e8/analysis/E8/E8_multiplicity_families.csv` | §2.4 |
| **§5.2** — boundary-rule sensitivity, 45.78% → 52.56% | `code/99_verification/scratch_qwen_boundary.py` | §2.4 |
| **Fig. 3** — the judging prompt | `config/prompts/prompt_B_readability_judge.txt` (SHA-256 `1eb486ba…978ae1`) | byte-identical across all runs |

### 3.1 Re-deriving the pair-level files from the raw model calls

One level deeper than §2: rebuild the pair-level outcome files from the raw
JSONL in `results/*/raw_inference/` (18,000 calls per judge = 9,000 pairs × 2
candidate orders, with verdict-token logits). **Requires Tier 1 setup (§6).**

```bash
python3 code/05_analysis/rq1/analyze_full_results.py     # RQ1, 5-judge diagnostic battery
python3 code/05_analysis/rq1/analyze_primary.py          # RQ1, 3-judge extension
python3 code/05_analysis/rq2/analyze_grounded_ab.py      # RQ2 full grid  (2 judges × 17 conditions)
python3 code/05_analysis/rq2/analyze_rq2_reduced.py      # RQ2 reduced grid (3 judges × 7 conditions)
python3 code/05_analysis/rq3/analyze_rq3_image_only.py   # RQ3
python3 code/05_analysis/rq4/analyze_f5.py               # RQ4 image-only pipelines
```

Bootstrap intervals (10,000 snippet-cluster replicates, seed 42, product
snippet weights, common pairs within each replicate):

```bash
python3 code/05_analysis/rq1/generate_rq1_bundle.py
```

### 3.2 Re-rendering the images

The models consumed the PNGs in `images/`. They are **shipped rather than
regenerated**, so that a replication does not depend on your FreeType version.
To regenerate and compare (**Tier 2 setup**):

```bash
export PYTHONPATH="$PWD/code/01_rendering/vendor:$PYTHONPATH"
python3 code/00_data_preparation/prepare_grounded_rendering.py
python3 code/01_rendering/audit_grounded_rendering.py
```

Per-image SHA-256 digests for comparison are in
`data/render_metadata/grid_render_metadata.csv` and
`perturbation_render_metadata.csv`. Rasterization can differ across Pillow and
FreeType versions; the digests tell you exactly which images differ.

### 3.3 Re-running inference

**Tier 3 setup (§6) and ~250 GPU-hours.** All open-weight runs are
deterministic: BF16, `do_sample=false`, `temperature=0.0`, `top_p=1.0`, batch
size 1, `max_new_tokens=24`, `seed=42`. Exact Hugging Face revisions are pinned
in [`environment/MODEL_REVISIONS.json`](environment/MODEL_REVISIONS.json).

| Stage | Entry point |
| --- | --- |
| RQ1 primary battery | `code/02_inference/rq1/run_rq1_vlm.py`, `run_primary.py` |
| RQ2 grids | `code/02_inference/rq2/run_grounded_vlm.py`, `run_*_reduced.py` |
| RQ3 variants | `code/02_inference/rq3/run_rq3_vlm.py` |
| RQ4 deployment and text ablation | `code/02_inference/rq4/run_deployment.py`, `run_vlm_text_ablation.py` |
| Closed-model protocol checks | `code/02_inference/protocol_checks/run_openai_vlm_pilot.py` |

> **Recommended practice.** Given identical checkpoints and library versions,
> verdicts reproduce exactly. BF16 rounding across different CUDA or kernel
> versions can, however, move a small number of pairs that sit on an exact logit
> tie. Because changing only the tie-boundary convention shifts Qwen2.5-VL-7B's
> strict-swap error by 6.78 percentage points (§5.2 of the paper), **analyse the
> shipped verdict files rather than a fresh run when the goal is to confirm a
> reported value.**

---

## 4. Provenance

| | |
| --- | --- |
| **Paper** | *How Reliable Are VLM Judges of Rendered Code Readability? Rating Agreement, Order Consistency, and Rendering Sensitivity*, submitted to FSE 2027 |
| **This package** | https://github.com/nunu0404/FSE2027 |
| **Archival copy** | To be deposited with a DOI for the camera-ready version |
| **Author contact** | Withheld for double-blind review; correspondence via the FSE 2027 submission system |
| **Anonymization** | Absolute filesystem paths recorded by the original runs contained an account name and were rewritten in 389 text files. Method, scope, effect on legacy checksums, and the de-anonymization procedure: [`ANONYMIZATION.md`](ANONYMIZATION.md); full file list: [`ANONYMIZED_FILES.txt`](ANONYMIZED_FILES.txt) |
| **Integrity** | [`MANIFEST.sha256`](MANIFEST.sha256), generated **after** anonymization, is the authoritative record (14,462 entries). Several run directories also carry older, pre-anonymization checksum files; these are kept as provenance of the original runs, not as a check to run now — see [`ANONYMIZATION.md`](ANONYMIZATION.md) |
| **Scope honesty** | Material from the same research line that the paper does not use is enumerated with reasons in [`EXCLUSIONS.md`](EXCLUSIONS.md) |

---

## 5. Data

### 5.1 Source benchmarks

552 human-rated snippets from three published benchmarks, deduplicated, with
published mean ratings standardized within each benchmark:

| Benchmark | Language | Snippets |
| --- | --- | --- |
| Buse and Weimer (2010) | Java | 99 |
| Dorn (2012) | Java | 90 |
| Scalabrino et al. (2018) | Java | 124 |
| Dorn (2012) | Python | 119 |
| Dorn (2012) | CUDA | 120 |
| **Total** | | **552** |

Standardized snippets with ratings:
`data/snippets/snippets_552_with_human_ratings.csv`.
The upstream archives are in `data/benchmarks_raw/downloads/`, with
`SHA256SUMS` recorded so they can be replaced by download scripts if
redistribution terms require it (§10).

### 5.2 Pairs

`data/pairs/rq1_pairs_9000_seed42.csv` — 9,000 pairs, 3,000 per language, 1,000
per rating-gap stratum, seed 42. Strata: hard `0.2 ≤ |Δz| < 0.5`, medium
`0.5 ≤ |Δz| < 1.0`, easy `|Δz| ≥ 1.0`. The higher standardized mean defines the
reference direction. Of the 3,000 Java pairs, 1,004 are within-dataset and 1,996
cross datasets.

### 5.3 Rendered images (11,688 PNGs, the exact pixels the models consumed)

| Directory | Contents |
| --- | --- |
| `images/rq2_rendering_grid/` | 12 rendering conditions × 552 snippets — 3 themes (Monokai, Friendly, monochrome) × 2 font sizes (20/24) × 2 wrap widths (60/80) |
| `images/rq2_perturbation/` | 6 conditions × 552 — baseline, Gaussian blur σ ∈ {1, 2, 4}, indentation removed, blank lines removed |
| `images/rq3_variants/` | `golden` / `ugly_gold` / `beautiful_trash` / `ugly_trash` — the compound source × appearance intervention |
| `images/snippet_default_renders/` | Default renders used by RQ1 and RQ4 |

### 5.4 Model outputs

`results/*/raw_inference/` holds the raw calls as JSONL, with verdict-token
logits where the API exposed them: 18,000 calls per judge for RQ1 (8 judges),
7,200 for RQ3, and the RQ2 grids. Closed-model checks retain parsed verdicts
only, because the API exposes no logits.

### 5.5 Source-feature baselines

`data/features_and_baseline_predictions/` — extracted source features and
out-of-fold pair predictions for the nine supervised predictors (RF, MLP, SVR,
voting, linear and tree ensembles), under five-fold multiple-source grouped
cross-validation.

> ⚠ **A trap for replicators.** Use
> `data/features_and_baseline_predictions/java/table1_ml_replication/canonical_pairwise_summary.csv`
> as the source of record for baseline accuracies. The convenient
> nine-predictor aggregation file `ml_predictions_9models_9000.csv` is **stale
> for 3 of 9,000 pairs** and yields 64.60% instead of the reported 64.67% for
> Python. `recompute_headline_numbers.py` prints both columns side by side so
> the mistake is visible rather than silent. See "A trap for replicators" in
> [`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md).

---

## 6. Setup — the four engagement tiers

Full detail in [`INSTALL.md`](INSTALL.md) and [`REQUIREMENTS.md`](REQUIREMENTS.md).

### Tier 0 — confirm every reported number (what §2 does)

* Python 3.8+, standard library only. No GPU, no network, no `pip`.
* ~3 GB disk; under 10 seconds of CPU for all three scripts.

### Tier 1 — re-run the analyses from raw model outputs

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements_analysis.txt
```

* Python 3.10+; `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `PyYAML`.
* 16 GB RAM (peak is the 10,000-replicate bootstrap over 45,000 pairs).
* 30–60 minutes on a modern desktop CPU.

### Tier 2 — re-render the images

Tier 1, plus `Pillow` and `pygments` (both in the same requirements file). Use
the **vendored** tree-sitter grammars rather than reinstalling, so the pinned
versions are used:

```bash
export PYTHONPATH="$PWD/code/01_rendering/vendor:$PYTHONPATH"
```

The renderer font ships at `code/01_rendering/fonts/DejaVuSansMono.ttf`
(SHA-256 `39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b`).
A few minutes for all 18 conditions × 552 snippets.

### Tier 3 — re-run inference

Two environments are required, because Phi-4-multimodal does not run under the
base stack.

```bash
# Base environment — 7 of the 8 open-weight judges
python3.13 -m venv .venv-base && source .venv-base/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 transformers==5.9.0 \
            accelerate==1.14.0 huggingface-hub==1.5.0 safetensors==0.7.0 \
            tokenizers==0.22.2 pillow==12.1.1 pandas==3.0.1

# Phi-4-multimodal environment
python3.10 -m venv .venv-phi && source .venv-phi/bin/activate
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 \
            --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.48.2 accelerate==1.3.0 pillow==11.1.0
```

Model weights are **not redistributed**. Download at the pinned revisions:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Repeat for each entry in `environment/MODEL_REVISIONS.json`. The `google/gemma-*`
repositories are gated and require an accepted license on your account. Closed
models need `OPENAI_API_KEY`; **no key ships with this package**. For RQ4,
`pip install rapidocr-onnxruntime easyocr`.

**Hardware actually used** (captured in `environment/env_rq1_model_battery.json`):
2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB each), driver
580.95.05, CUDA 13.0. The 8B–12B primary judges fit in a single 48 GB GPU at
BF16, batch size 1.

**Cost:** ~250 GPU-hours total across all runs, plus roughly USD 25 of OpenAI
API usage for the closed-model checks. Per-run breakdown in
[`REQUIREMENTS.md`](REQUIREMENTS.md); per-condition token counts and costs in
`results/protocol_checks/closed_models_gpt/*/*_usage.json`.

---

## 7. Package structure

```
FSE2027/                          14,463 files · 2.9 GB
├── README.md                     ← you are here
├── CLAIMS_TO_ARTIFACTS.md        every paper number → file (+ producing script)
├── INSTALL.md                    environment setup, per tier
├── REQUIREMENTS.md               hardware, software, runtime, cost
├── STATUS.md                     badges claimed, and what was verified during assembly
├── EXCLUSIONS.md                 what is deliberately not here, and why
├── ANONYMIZATION.md              the double-blind path rewrite and its effect on checksums
├── ANONYMIZED_FILES.txt          the 389 rewritten files
├── MANIFEST.sha256               integrity record, 14,462 entries
├── LICENSE                       see §10
│
├── code/                221 files   the pipeline, in execution order
├── config/               12 files   frozen prompt and pre-registered protocols
├── data/                 62 files   benchmarks, snippets, pairs, features, render metadata
├── images/           11,689 files   the rendered PNGs the models actually consumed
├── results/           2,380 files   raw inference, pair-level outcomes, analyses
├── figures_and_tables/   51 files   per-figure and per-table source data
├── environment/          21 files   captured runtimes, pinned model revisions
└── docs/                 17 files   long-form experiment reports
```

### `code/` — stages in execution order

| Directory | Stage |
| --- | --- |
| `00_data_preparation/` | Parse the published benchmarks, standardize ratings, build pair sets and CV splits, extract source features, generate the RQ3 variants, prepare the rendering plan |
| `01_rendering/` | Token-preserving syntax-highlighting renderer, the 12-condition grid, the 5 perturbations, render audits. Ships `fonts/DejaVuSansMono.ttf` and pinned `vendor/tree_sitter*` grammars |
| `02_inference/` | Judge runners, one directory per RQ (`rq1/`, `rq2/`, `rq3/`, `rq4/`, `protocol_checks/`) |
| `03_source_feature_baselines/` | The nine supervised source-feature predictors |
| `04_ocr_pipelines/` | RapidOCR / EasyOCR / preprocessed-EasyOCR transcription and downstream feature substitution |
| `05_analysis/` | Metric computation, bootstrap intervals, McNemar tests, logit decomposition, encoder retrieval |
| `06_figures_tables/` | Figure and table source-data generation |
| `99_verification/` | Independent recomputation of the reported values — the three scripts in §2 |

### `results/` — one directory per research question

Plus `protocol_checks/` (closed GPT models, 24B–32B open models, reasoning
budget, max-token pilots), `diagnostics/` (review-defense E1–E8, follow-ups
F1–F4, noise ceiling, strict-swap-by-difficulty, |c| calibration), and
`verified/` (the expected outputs used in §2.6).

---

## 8. Reusability beyond this paper

* **Re-analysable under other decision policies.** The 45,000-pair diagnostic
  set carries per-call verdict-token logits, so the position-aligned (`b`) and
  candidate-oriented (`c`) components of Eq. 4 can be recomputed under
  acceptance rules other than ours. Because changing only the tie-boundary
  convention moves Qwen2.5-VL-7B's strict-swap error by 6.78 pp, this is the
  part of the data we expect to be most reused.
* **A rendering-sensitivity benchmark.** The 11,688 PNGs span 18 conditions over
  one fixed snippet set. They are usable to probe any image-input model's
  sensitivity to theme, font size, wrap width, blur, indentation, and blank
  lines — independent of readability.
* **Pinned and deterministic.** Exact Hugging Face revisions for all 12
  open-weight models; greedy decoding; seed 42; a single frozen prompt whose
  byte-identity across every run is verified by SHA-256.
* **Honest about limits.** Pre-registered protocols, integrity gates, and
  negative results ship alongside the successful runs — including an InternVL
  text+image arm that was blocked twice at its 100-call gate and never promoted
  to full inference.

---

## 9. Limitations and known discrepancies

We report these rather than adjust the artifact.

* **No *Results Reproduced* claim.** See §1.
* **Baselines are not fully endpoint-disjoint.** Source-feature predictors use
  snippet-level out-of-fold predictions under five-fold multiple-source grouped
  CV. Across folds an endpoint can train its partner's predictor, and score
  scales can differ. RQ4 inherits this. Stated in §4.2 of the paper.
* **Six smaller issues**, none of which changes a reported conclusion: one
  confidence interval computed against a superseded baseline value, two
  unreported parse-failure facts (including Gemma4-12B's 33/21,000), and three
  rounding or scope-wording details. All are enumerated under **"Known
  discrepancies"** in [`CLAIMS_TO_ARTIFACTS.md`](CLAIMS_TO_ARTIFACTS.md).
* **Three Figure 6 cells differ by 0.1 pp** through half-rounding; documented in
  the same place.
* **Legacy checksum files** inside some run directories predate anonymization
  and will not match for rewritten files. Use the root `MANIFEST.sha256`. See
  [`ANONYMIZATION.md`](ANONYMIZATION.md).
* **The stale nine-predictor aggregation file** — see the warning in §5.5.
* **`environment/Dockerfile.reference` and `requirements_lock_reference.txt`**
  are captures from the earlier project skeleton, kept as provenance. They refer
  to paths that do not exist in this package layout and are not runnable here;
  use §6 instead.

---

## 10. License and third-party terms

> ⚠ **`LICENSE` is currently a placeholder and must be replaced with the
> project-approved license before archival deposit.**

`data/benchmarks_raw/downloads/` redistributes the Buse and Weimer, Dorn, and
Scalabrino et al. benchmark archives. Before public deposit, confirm that their
terms permit redistribution, or replace the archives with download scripts plus
the SHA-256 sums already recorded in
`data/benchmarks_raw/downloads/SHA256SUMS`.

Model weights are not redistributed. The `google/gemma-*` checkpoints are gated
and carry their own license terms.

The vendored tree-sitter grammars under `code/01_rendering/vendor/` and the
DejaVu Sans Mono font under `code/01_rendering/fonts/` retain their upstream
licenses.
