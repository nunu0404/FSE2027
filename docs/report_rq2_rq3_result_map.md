# Grounded Three-Language Results Package

Updated: 2026-07-23

This directory is the self-contained package for the corrected Java, Python,
and CUDA rendered-code experiments. The final machine gate is `PASS`.

## Read First

1. `FINAL_GROUNDED_3LANG_REPORT.md`
   - Integrated methods, conditions, all aggregate A-D results, hypotheses,
     limitations, and interpretation.
2. `analysis/FINAL_VALIDATION.json`
   - Machine-readable final integrity gate.
3. `EXPERIMENT_CONDITION_REGISTER.md`
   - Frozen definitions, statistical rationale, and literature grounding.
4. `ARTIFACT_SHA256.csv`
   - Paths, sizes, and SHA-256 hashes for 127 tracked artifacts.

## Completion Status

| Component | Status | Size |
|---|---|---:|
| A: 12-condition rendering grid | Complete | 144,000 calls |
| B: baseline plus five perturbations | Complete | 72,000 calls |
| A+B pair-level analysis | Complete | 216,000 calls / 108,000 rows |
| RQ3 image-only, two models | Complete | 7,200 calls / 3,600 contrasts |
| RQ3 Qwen text+image | Complete, supplementary | 3,600 calls |
| RQ3 InternVL text+image | Blocked at GA0 | 0 full calls |
| D: vision embeddings, two models | Complete | 18,768 image passes |
| AB/BA logit decomposition | Complete | 108,000 rows |

InternVL text+image failed the same 100-call GA0 twice, with one malformed
verdict/argmax mismatch in each run. It was not promoted to full inference.

## Result Map

### A. Rendering Grid

- Main aggregate table: `analysis/ab/A_GRID_RESULTS.csv`
- Baseline-versus-condition tests: `analysis/ab/A_GRID_MCNEMAR.csv`
- Complete-set sensitivity: `analysis/ab/COCHRAN_Q.csv`
- Figure: `analysis/figures/a_grid_overall.png`

### B. Perturbations and Language Effects

- Main aggregate table: `analysis/ab/B_PERTURBATION_RESULTS.csv`
- Paired tests: `analysis/ab/B_PERTURBATION_MCNEMAR.csv`
- Dorn-only primary summaries: `analysis/ab/B_DORN_PRIMARY_RESULTS.csv`
- Dorn-only primary interactions:
  `analysis/ab/B_LANGUAGE_INTERACTIONS_DORN_PRIMARY.csv`
- Pooled-Java secondary interactions:
  `analysis/ab/B_LANGUAGE_INTERACTIONS_POOLED_SECONDARY.csv`
- Cue-present sensitivity: `analysis/ab/B_CUE_PRESENT_SENSITIVITY.csv`
- Difficulty breakdown: `analysis/ab/BY_DIFFICULTY.csv`
- Figure: `analysis/figures/b_perturbation_overall.png`

### Shared A/B Data

- Pair-condition rows: `analysis/ab/A_B_PAIR_LEVEL.csv`
- Overall model-condition summaries: `analysis/ab/OVERALL_RESULTS.csv`
- Allocation gate: `analysis/ab/INTEGRITY.json`

The SHA-verified A-grid baseline is
`monokai_dark__fs20__wrap80__lnon`. It is exactly identical to the B baseline
for all 6,000 model-language-pair rows, including margins and outcomes.

### C. RQ3 Semantic-Visual Conflict

Image-only primary results:

- Report: `rq3/analysis/image_only/RQ3_IMAGE_ONLY_REPORT.md`
- Overall: `rq3/analysis/image_only/RQ3_IMAGE_ONLY_OVERALL.csv`
- Six contrasts: `rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv`
- Language x contrast:
  `rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv`
- Stratum x contrast:
  `rq3/analysis/image_only/RQ3_IMAGE_ONLY_BY_STRATUM_CONTRAST.csv`
- Pair-level: `rq3/analysis/image_only/RQ3_IMAGE_ONLY_PAIR_LEVEL.csv`

Qwen text-plus-image supplementary results:

- `rq3/analysis/qwen_text_plus_image/`

Arm status and historical comparison:

- `rq3/analysis/historical_alignment/RQ3_ARM_STATUS.csv`
- `rq3/analysis/historical_alignment/RQ3_ORDER_ALIGNMENT.csv`
- `rq3/analysis/historical_alignment/LIMITATIONS.md`
- Figure: `analysis/figures/c_rq3_primary_conflict.png`

RQ3 reports directional target preference, not independent-gold accuracy.

### Logit Decomposition

- Pair-level b/c values: `analysis/logit_decomposition/logit_decomposition.csv`
- Accuracy/reliability summary:
  `analysis/logit_decomposition/debiased_accuracy_summary.csv`
- Valid-versus-swap tests:
  `analysis/logit_decomposition/VALID_INVALID_EFFECT_TESTS.csv`
- Content-margin calibration:
  `analysis/logit_decomposition/ABS_C_CALIBRATION.csv`
- Interpretation: `analysis/logit_decomposition/LOGIT_INTERPRETATION.md`
- Figures:
  - `analysis/logit_decomposition/b_c_scatter.png`
  - `analysis/logit_decomposition/content_margin_calibration.png`

There are zero `|c|>|b|` identity violations away from exact BF16 logit-tie
boundaries. Boundary rows are reported separately.

### D. Vision Representation Stability

- Main table: `embedding/analysis/d_embedding_stability.csv`
- Retrieval by condition: `embedding/analysis/d_retrieval_by_condition.csv`
- Perturbation summary: `embedding/analysis/d_perturbation_distance.csv`
- Snippet-level distances:
  - `embedding/analysis/d_intra_snippet_distances.csv`
  - `embedding/analysis/d_perturbation_snippet_distances.csv`
- Integrity: `embedding/analysis/D_INTEGRITY.json`
- Figures:
  - `embedding/analysis/d_intra_distance_distributions.png`
  - `embedding/analysis/d_blur_layout_distances.png`

Raw embedding arrays and per-image metadata are in `embedding/raw/`.

### Rendering and OCR Audits

- Render gate: `audit/GROUNDED_RENDER_AUDIT.json`
- OCR manifest: `audit/BLUR_OCR_MANIFEST.json`
- OCR summary: `audit/blur_ocr_summary.csv`
- OCR pairwise degradation: `audit/blur_ocr_pairwise.csv`
- All-image blur similarity: `audit/blur_ssim_all.csv`
- Contact sheets: `audit/contact_*.png`

The corrected rendered images are under `rendered/`; RQ3 images are under
`rq3/rendered/`. Per-image hashes are stored in their metadata tables.

## Reproducibility Map

- Frozen configuration: `config/protocol_grounded.json`
- Frozen prompt: `config/prompt_B_template.txt`
- Fixed pairs: `data/pairs_seed42_grounded.csv`
- Rendering conditions: `data/render_conditions.csv`
- Analysis and runner code: `code/`
- Full raw A/B inference:
  - `inference/grid/raw/`
  - `inference/perturbation/raw/`
- Full raw RQ3 inference: `rq3/inference/raw/`
- Model and pipeline logs: `logs/`

The final package can be checked with:

```bash
python results/grounded_protocol_3lang_20260721/code/validate_final_outputs.py
```

## Interpretation Boundaries

- H-B1 is not supported.
- H-B2 is not proven as equivalence; nonsignificance is not equivalence.
- Fixed representation-ratio cutoffs are not used.
- Historical Swiss/Elo values are provenance evidence, not directly pooled
  effect sizes.
- Earlier malformed-render outputs are not regression oracles for this package.
