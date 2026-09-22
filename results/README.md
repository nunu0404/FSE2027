# Results

One directory per research question, plus protocol checks and diagnostics.
For the claim-by-claim map, see `../CLAIMS_TO_ARTIFACTS.md`.

## Shared column vocabulary

Pair-level files use a common schema. `X`/`Y` identify snippets; `A`/`B` label
displayed positions.

| Column | Meaning |
| --- | --- |
| `pair_id` | Joins across every file in the package |
| `valid` / `strict_valid` | Both orders parsed and selected the same snippet |
| `correct` / `strict_correct` | Valid **and** matching the rating-derived reference |
| `margin_ab`, `margin_ba` | First-position-oriented verdict-token logit margins |
| `position_margin` / `b` | `(margin_ab + margin_ba) / 2` — position-aligned component |
| `content_margin` / `c` | `(margin_ab − margin_ba) / 2` — candidate-oriented component |
| `debiased_correct` | Two-order averaged decision by the sign of `c`; ties are errors |
| `content_tie` | `c == 0` in exact BF16 |
| `boundary` | `\|c\| == \|b\|` |
| `parse_failure_calls` | Calls whose output did not yield `FINAL_VERDICT: A\|B` |

Derived metrics: `A_valid = N_correct,valid / N_valid`,
`E = N_correct,valid / N`, `S = 1 − N_valid / N`, and `D` = share of all pairs
with `debiased_correct`. Identity: `E = A_valid (1 − S)`.

## `rq1_viability/`

| Path | Contents |
| --- | --- |
| `pair_level/pair_level_5model_battery_45000.csv` | Qwen2.5-VL-7B, Gemma-3-12B, InternVL3-8B, Ministral-3-8B, Phi-4-multimodal — 9,000 pairs each. This is also the **logit-margin diagnostic set** of Section 4.4.2 |
| `pair_level/pair_level_3model_extension_27000.csv` | Qwen3-VL-8B, InternVL3.5-8B, Gemma4-12B — 9,000 pairs each |
| `raw_inference/<judge>/raw.jsonl` | 18,000 calls per judge (9,000 pairs × AB/BA), with prompt, raw output, parsed verdict, verdict-token ids, and per-token logits. Verified: 18,000 lines for all eight judges |
| `raw_inference/<judge>/effective_logits.jsonl` | Extension judges only; materialized verdict-step logits |
| `bootstrap_intervals/` | 10,000-replicate snippet-cluster bootstrap (seed 42). `rq1_intervals.csv` has 128 rows = 8 judges × 4 scopes × 4 metrics. `fig_data/` holds the Figure 6 and Figure 7 inputs; `tab/` the LaTeX tables |
| `analysis_5model_battery/`, `analysis_3model_extension/` | Metric tables, F6/F7 confirmations, GA0 gate, ML decision-sign correlation |
| `audit/` | Execution environment, 552-image integrity, ML pair-id coverage, pre-registration asset audit |

The two pair-level files split the eight judges only because they were run in
two batteries; they are concatenated for every pooled number in the paper.

## `rq2_reliability/`

| Path | Contents |
| --- | --- |
| `full_grid_2models/A_B_PAIR_LEVEL.csv` | 108,000 rows = 2 judges × 3 languages × 1,000 pairs × 18 condition labels. **17 conditions are unique**: the label `baseline` duplicates `monokai_dark__fs20__wrap80__lnon` exactly, including margins. The paper's 102,000 observations are the deduplicated count |
| `full_grid_2models/A_GRID_*.csv`, `B_PERTURBATION_*.csv` | Aggregates and exact McNemar tests with Holm correction |
| `reduced_grid_3models/<judge>/` | Gemma-3-12B, InternVL3.5-8B, Gemma4-12B; 21,000 pair rows each (7 conditions × 3,000), raw calls, matched contrasts with cluster-robust SE |
| `reduced_grid_3models/anchor_qwen/` | Qwen2.5-VL-7B baseline anchor, for comparability with the full grid |
| `logit_margin_diagnostics/` | `b`/`c` decomposition, `\|c\|` calibration, valid-vs-swap tests, identity-violation check |
| `encoder_retrieval/` | Vision-encoder diagnostics. Embeddings cover all 17 conditions (`D_INTEGRITY.json`: 9,384 rows = 552 × 17 per judge); `d_embedding_stability.csv` and `d_retrieval_by_condition.csv` report top-one retrieval over the **12-condition rendering grid**, and `d_perturbation_distance.csv` covers the 5 perturbations by cosine distance |
| `packaging_and_modality/` | 8 conditions on the same 3,000 Java pairs: 2 judges × {separate, combined} images × {image only, +source text}. Backs the packaging and modality paragraph of Section 5.2 |
| `verified_recomputation/rq2_order_vs_rendering.csv` | **The Section 5.2 headline table.** All 15 model×language groups with median swap error, median rendering-flip rate, valid-both flip rate, and `order_exceeds_rendering` |
| `audit/` | Render gate, blur OCR manifest and SSIM, contact sheets |

## `rq3_content_vs_appearance/`

| Path | Contents |
| --- | --- |
| `analysis/image_only/RQ3_IMAGE_ONLY_BY_CONTRAST.csv` | **Figure 10.** 12 rows = 2 judges × 6 contrasts |
| `analysis/image_only/RQ3_IMAGE_ONLY_PAIR_LEVEL.csv` | 3,600 rows = 2 judges × 300 snippets × 6 contrasts |
| `analysis/image_only/RQ3_IMAGE_ONLY_BY_LANGUAGE_CONTRAST.csv` | The 66–91% / 89–94% per-language conflict ranges |
| `analysis/qwen_text_plus_image/` | Supplementary text+image arm |
| `analysis/historical_alignment/RQ3_ARM_STATUS.csv` | Arm status, including the InternVL text+image arm that failed its gate twice and was not promoted |
| `raw_inference/raw/` | All 7,200 calls; all parsed |

## `rq4_image_only/`

| Path | Contents |
| --- | --- |
| `ocr_substitution_frozen_rf/` | **Table 3(a).** Java RF held fixed while source features are replaced by RapidOCR / EasyOCR / preprocessed-EasyOCR features. `python_cuda/` holds the Python and CUDA equivalents used for Table 3(b) |
| `image_only_pipelines/` | **Table 3(b).** `Table5_deploy_v2.csv` (per-language `E`, `A_valid`, `S`, and four `D` conventions), `F5_deploy_debiased.csv` (paired bootstrap CIs vs OCR+ML), `recomputed_intervals_20260920.csv` (the CUDA interval, recomputed against Table 3(b)'s EasyOCR+LR comparator) |
| `text_input_ablation/` | **Table 3(c).** Source text / OCR text / images for Qwen2.5-VL-7B |
| `ocr_transcription_quality/` | Token F1 per engine on 313 Java, 119 Python, 120 CUDA snippets, plus the OCR text itself |
| `ocr_text_llm_judge/` | Qwen2.5-Coder-7B-Instruct on source and OCR text |
| `raw_inference/` | Deploy-v2, text-ablation, and OCR-text-LLM calls |

`Table5_deploy_v2.csv` reports `D_main` (ties counted as errors, the
manuscript's convention), `D_excl` (ties excluded), and `D_half` (half credit).
Table 3(c)'s `debiased_accuracy` column uses the tie-excluded denominator; see
the note in `CLAIMS_TO_ARTIFACTS.md` for the conversion.

## `protocol_checks/`

| Path | Contents |
| --- | --- |
| `closed_models_gpt/` | **Table 2(a).** GPT-5.4-mini, GPT-5.4, GPT-5.5; 300 Java pairs × {image, text+image, combined image, combined+text}. Raw JSONL, per-condition summaries, token usage and cost. No API logits |
| `large_open_models/multilang_300_20260910/` | **Table 2(b).** Qwen2.5-VL-32B, Mistral-Small-3.1-24B, Gemma-3-27B; 300 pairs per language. Verdicts only — logit hooks disabled |
| `large_open_models/java_300_20260910/` | The earlier Java-only run of the same three models |
| `reasoning_budget/` | GPT-5.4 low vs high reasoning at 4,096 tokens, 300 Java pairs, both orders; includes the excluded 90-pair pilots |
| `max_token_pilot/` | Output-length sensitivity check behind the 24-token limit |

## `diagnostics/`

| Path | Backs |
| --- | --- |
| `review_defense_e1_e8/analysis/E1/` | RQ1 bootstrap CIs against RF and the best per-language source baseline (Section 5.1) |
| `.../E4/` and `runs/repeat_*_env/` | Same-environment and changed-environment repeat runs |
| `.../E5/` | RQ3 text+image; InternVL combined-image packaging (Section 5.2) |
| `.../E7/` | Table 2(a) consolidation; the GPT-5.4 `S` 11.33% → 29.00% text check |
| `.../E8/` | The 66 rendering and 30 perturbation multiplicity families; rendering decomposition |
| `followup_f1_f4/` | Cross-run determinism (F3), both-valid flip floor (F4), noise-ceiling table (F2) |
| `noise_ceiling/` | Section 4.1 rater-split reference stability (98.8 / 98.7 / 80.5%) |
| `reinforcement_abcd/B_content_calibration/` | **Figure 9** and the 0.52–0.63 held-out AUC |
| `reinforcement_abcd/A_preference_flip/` | Condition-pair flip matrices |
| `strict_swap_by_difficulty/` | Swap error by rating gap, with 10,000 bootstrap replicates |
| `verified_recomputation_20260919/` | Independent 2026-09-19/20 recomputations |
| `vlm_claims.csv` | 100+ individually sourced claim values, including the 5,577/45,000 boundary count |

## `verified/`

`EXPECTED_HEADLINE_OUTPUT.txt` — the reference output of
`code/99_verification/recompute_headline_numbers.py`. `diff` against it to
confirm the package reproduces the paper's headline numbers.
