# Code

Numbered by execution order. Scripts are preserved as run; paths inside them
were rewritten for anonymization only (see `../ANONYMIZATION.md`).

| Stage | Contents |
| --- | --- |
| `00_data_preparation/` | `prepare_datasets.py` parses the three benchmark archives and standardizes ratings. `extract_features.py` computes the 27 source features. `create_splits.py` builds the five-fold multiple-source grouped CV. `build_pair_sets.py` samples the balanced pair sets under seed 42. `prepare_rq3_variants.py` generates the AST-guided RQ3 transformations. `build_prereg_assets.py` freezes the RQ1 pre-registration assets |
| `01_rendering/` | `token_preserving_renderer.py` is the renderer. `prepare_grounded_rendering.py` drives the 12-condition grid and 5 perturbations; `render_code_images*.py` the default renders; `render_combined_pair_images.py` the combined composites. `audit_*.py` verify pixel-level integrity. Ships `fonts/DejaVuSansMono.ttf` and `vendor/` tree-sitter grammars at their pinned versions |
| `02_inference/rq1/` | `run_rq1_vlm.py` (5-judge battery) and `run_primary.py` (3-judge extension). `materialize_effective_logits.py` extracts verdict-step logits; `validate_primary_run.py` checks generated verdicts against logit signs |
| `02_inference/rq2/` | `run_grounded_vlm.py` (full grid), `run_{gemma,internvl,gemma4}_reduced.py` (reduced grid), smoke tests |
| `02_inference/rq3/` | `run_rq3_vlm.py` and its pipeline drivers |
| `02_inference/rq4/` | `run_f5_inference.py` (deployment routes), `run_vlm_text_ablation.py` (Table 3c), `run_ocr_text_llm_baseline.py` (Qwen2.5-Coder on OCR text), `create_execution_lock.py` (pre-registration lock) |
| `02_inference/protocol_checks/` | `run_openai_vlm_pilot.py` (closed models), `run_inference.py` / `run_multilang_inference.py` (24B–32B open models), reasoning-budget drivers |
| `03_source_feature_baselines/` | `train_classical_baselines.py`, `compute_classical_pairwise.py`, `run_canonical_baselines.py`, plus the `*.yaml` configs naming the nine predictors |
| `04_ocr_pipelines/` | `run_clean_multi_ocr_experiment.py` (frozen-RF substitution, Table 3a), `run_python_cuda_multi_ocr.py`, `run_easyocr_preprocessed_shard.py`, `run_blur_ocr_audit.py` |
| `05_analysis/` | Per-RQ metric computation, bootstrap intervals, McNemar with Holm, logit decomposition, encoder retrieval. `rq1/generate_rq1_bundle.py` produces the 10,000-replicate intervals (L140–235 is the bootstrap loop that also yields `D`). `shared/estimate_noise_ceiling.py` produces the rater-split stability table. `review_followups/` holds the E1–E8 and F1–F4 analyses |
| `06_figures_tables/` | `generate_fig_numbers.py` and the `task*.py` scripts that produced the figure and table source data, including `task1_boundary_sensitivity.py` (the 6.78 pp boundary-rule result) and `task9_figure5_closed_bootstrap.py` |
| `99_verification/` | `recompute_headline_numbers.py` (RQ1) and `recompute_rq2_rq3_rq4.py` (~90 checks over RQ2–RQ4) reproduce the paper from the shipped result files with no dependencies; `audit_parse_failures.py` audits every experiment for unparseable verdicts (see `docs/PARSE_FAILURE_AUDIT.md`); `verify_manifest.sh` checks `MANIFEST.sha256`. Also the `scratch_*.py` scripts used for the independent 2026-09-17 → 2026-09-20 recomputations. `scratch_calc_flip_6groups.py` and `scratch_calc_flip_7conds.py` produced the 15-group order-vs-rendering table; `scratch_check_baseline_dup.py` proves the duplicated 18th condition; `scratch_compute_ci.py` recomputed the RQ4 intervals |

## Running order

```
00 → 01 → {02 (needs GPU), 03, 04} → 05 → 06
```

Stages 03–06 run on CPU from the shipped inputs. Stage 02 is the only one that
requires model weights.
