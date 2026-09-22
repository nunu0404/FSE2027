# Deliberate exclusions

The research line behind this paper produced more material than the paper uses.
Everything omitted from this package is listed here with a reason, so that a
reviewer can tell the difference between "not reported" and "not kept".

## Excluded because it is derivable, not because it is missing

| Item | Size | Why, and how to get it |
| --- | --- | --- |
| Combined per-pair composite PNGs | ~2.7 GB | Used only by the "combined image" packaging conditions of the closed-model and open-model checks. They are deterministic side-by-side concatenations of the shipped per-snippet renders. Regenerate with `code/01_rendering/render_combined_pair_images.py`; per-image SHA-256 digests are preserved in `results/protocol_checks/large_open_models/*/image_checksums.json` and in the `render_metadata` tables |
| `runs/grounded_protocol_3lang_20260721/` inside the review-defense package | ~1.7 GB | A byte-identical working copy of the main RQ2/RQ3 package, staged for re-analysis. The original is shipped at `results/rq2_reliability/` and `results/rq3_content_vs_appearance/` |
| `audit/`, `config/`, `data/`, `rendered/` inside the E4 repeat runs | ~2.2 GB | These were symlinks to the shared grounded package, not separate data. `results/diagnostics/review_defense_e1_e8/runs/*/SYMLINKS_RESOLVED.md` maps each one to its in-package location |
| Model checkpoints | ~400 GB | Not redistributable. Exact revisions are pinned in `environment/MODEL_REVISIONS.json` |

## Excluded because the paper does not use it

| Item | Why |
| --- | --- |
| DeepSeek-VL-7B extension run (9,000 pairs, with logits) | A ninth judge evaluated after the fact. The paper reports five primary judges plus three earlier checkpoints; DeepSeek appears in none of them |
| Sergeyuk 2024 readability corpus (120 snippets) | Part of a separate dataset-extension study. The paper's 552 snippets are Buse and Weimer, Dorn, and Scalabrino et al. only. The archive stays in `data/benchmarks_raw/downloads/` because the shipped Python and CUDA source-feature predictions were fit within a pipeline that also processed it; the Sergeyuk rows are never joined into the paper's pair sets |
| Elo/Swiss tournament, triview-rank, CARR-judge, and other pre-FSE prototypes | Earlier designs superseded by the pairwise strict-swap protocol. They contributed no reported number |
| Behavioral and counterfactual maintainability benchmarks | Separate projects sharing the same repository |
| `psc2code`-style natural screenshot captures | Section 8 names these as future validation; none were run |
| `text_ablation_20260920/backup_buggy_lang_20260920/` | The pre-fix RQ4 text-ablation run. A language-label bug affected the Python and CUDA calls; the whole condition was rerun on 2026-09-20 and Table 3(c) uses the rerun. The backup's only purpose was to confirm the rerun's scope (Java unchanged, 10,341 OCR and 10,735 source calls changed), which is recorded in `results/rq4_image_only/text_input_ablation/README.md` |

## Excluded for safety or hygiene

| Item | Why |
| --- | --- |
| API keys, `.rq1_secrets/` | Credentials. `INSTALL.md` says which variables to set |
| Hugging Face caches, virtualenvs, `__pycache__` | Reconstructible from `environment/` |
| Author-identifying paths, hostnames, and notes | Removed for double-blind review |

## Kept although the paper reports it as a limitation

These are negative or partial results. They are retained because the paper
relies on them.

| Item | Location |
| --- | --- |
| RQ3 InternVL text+image arm, blocked twice at its 100-call GA0 gate with a malformed verdict/argmax mismatch, never promoted to full inference | `results/rq3_content_vs_appearance/analysis/historical_alignment/RQ3_ARM_STATUS.csv`, `docs/report_rq2_rq3_result_map.md` |
| Large open-model checks with no verdict-token logits (hooks disabled), 2,700 verdict-only rows | `results/protocol_checks/large_open_models/` |
| Closed-model checks with no API-exposed logits | `results/protocol_checks/closed_models_gpt/` |
| Excluded pilots: 90-pair reduced-budget runs and one aborted 122-pair GPT-5.5 run (Section 4.4.1) | `results/protocol_checks/reasoning_budget/`, `results/protocol_checks/closed_models_gpt/gpt54_and_gpt55_20260708/` |
| The duplicated 18th RQ2 condition label (`baseline` ≡ `monokai_dark__fs20__wrap80__lnon`) | Kept in `A_B_PAIR_LEVEL.csv` so the 108,000 shipped rows are complete; the duplication proof is `code/99_verification/scratch_check_baseline_dup.py` |
