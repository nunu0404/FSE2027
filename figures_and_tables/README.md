# Figures and tables

Numeric source data for every figure and table in the paper. Figure rendering
source (TikZ/PGF) lives with the manuscript, not here.

| Directory | Paper item | Key file |
| --- | --- | --- |
| `fig06_rq1_by_language/` | Figure 6 — valid and effective accuracy per language, 8 judges | `fig06_data.csv` (24 rows), `rq1_intervals_all_scopes.csv` (128 rows) |
| `fig07_order_robustness/` | Figure 7 — pooled `E`, `D`, `S` and the `D − E` policy gap | `fig07_data.csv` (8 rows), `vlm_tab_rq1_intervals.tex` |
| `fig08_rendering_perturbation/` | Figure 8 — rendering and perturbation effects | **panel (a):** `*_matched_contrasts.csv` for the three reduced-grid judges, and `A_GRID_RESULTS.csv` / `A_B_PAIR_LEVEL.csv` for the two full-grid judges. **panel (b):** `B_DORN_PRIMARY_RESULTS.csv`. **panel (c):** `B_PERTURBATION_RESULTS.csv`. `*_fig6_input.csv` and `rq2_order_vs_rendering.csv` hold the 15-group order-vs-rendering table used in the Section 5.2 text, not panel (a) |
| `fig09_logit_margin/` | Figure 9 — `\|c\|` decile curves, 5 diagnostic judges | `fig09_data_5_diagnostic_judges.csv` (50 rows = 5 judges × 10 deciles), `isotonic_auc_and_trend_summary.csv` (the 0.52–0.63 AUC range) |
| `fig10_rq3_preferences/` | Figure 10 — source vs appearance preferences | `RQ3_IMAGE_ONLY_BY_CONTRAST.csv` (12 rows) |
| `tab02_protocol_checks/` | Table 2 — closed and large-open protocol checks | `E7_closed_model_pilot.csv`, `aggregate_metrics_multilang.csv`, `fig05_closed_model_bootstrap.csv` |
| `tab03_rq4/` | Table 3 — OCR substitution, image-only pipelines, input ablation | `table3a_frozen_rf_ocr_substitution.csv`, `Table5_deploy_v2.csv`, `F5_deploy_debiased.csv`, `table3c_input_ablation_summary.csv`, `recomputed_intervals.csv` |

Table 1 is descriptive; its scope is fixed by the protocols in `../config/protocols/`
and the run manifests listed in `../CLAIMS_TO_ARTIFACTS.md` §4.2.

## `FIGURE_NUMBER_HISTORY.md`

The figure numbering changed between the 2026-09-12 analysis bundle and the
submitted manuscript. This file records the mapping and, for three Figure 6
cells, the exact unrounded ratios behind a 0.1 pp half-rounding difference.

| 2026-09-12 bundle | Submitted manuscript |
| --- | --- |
| Figure 3 / Table 1 | Figure 6 |
| Figure 4 / Table 2 | Figure 7 |
| Figure 6(a) / Table 3 | Figure 8(a) |

Filenames inside `fig06_*` and `fig07_*` that still read `fig3_data`/`fig4_data`
in the original bundle have been renamed here to the submitted numbering.

## Panel (b) uses the Dorn-only Java subset

Figure 8(b) reads `valid_accuracy` from `B_DORN_PRIMARY_RESULTS.csv`, not from
the pooled `B_PERTURBATION_RESULTS.csv`. Python and CUDA are 1,000 pairs either
way, but **Java is the Dorn-only 89-pair subset** — the cell the paper singles
out in Section 5.2. The protocol requires this: language interactions use the
Dorn-only three-language subset, because pooled Java mixes three rater panels.

Using pooled Java instead gives −2.36 pp for Qwen and −0.64 pp for InternVL3,
against the −0.03 and −2.90 the figure reports. All six cells reproduce exactly
from the Dorn-only file.

## Sign convention in `*_matched_contrasts.csv`

These files store each contrast as `baseline = monokai_dark__fs20__wrap80__lnon`
against `condition = monokai_dark__fs20__wrap60__lnon`, i.e. **wrap 80 → wrap 60**.
Figure 8(a) reports **wrap 60 → wrap 80**, so its values are the negation of
`delta_effective_accuracy` and `delta_strict_swap_error`. Verified for all three
reduced-grid judges: Gemma-3-12B ΔE +0.7…+1.6, Gemma4-12B −0.0…+1.2,
InternVL3.5-8B +4.0…+7.3.

The two full-grid judges are not in these files; their panel-(a) values are
recomputed from `A_B_PAIR_LEVEL.csv` by differencing the two single conditions
(verified: InternVL3-8B ΔE +0.9…+4.2, Qwen2.5-VL-7B ΔE −2.3…+1.3).

## `fig09_logit_margin/supplementary_abs_c_bins_3_extension_judges.csv`

A `|c|`-binned curve for Qwen3-VL-8B, InternVL3.5-8B, and Gemma4-12B. These
three judges are **not** in Figure 9, which covers the five diagnostic judges.
Kept as a supplementary check that the same monotone pattern holds for the newer
checkpoints.
