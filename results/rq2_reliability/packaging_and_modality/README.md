# Packaging and modality (Section 5.2)

Eight conditions on the same 3,000 Java pairs: 2 judges × {separate images,
combined image} × {image only, matching source text added}.

`packaging_modality_summary_8conditions.csv` is the summary table. The
packaging claim in Section 5.2 — "Combining Java snippets raises both open
models' swap error" — reads off its `swap_error_rate` column:

| Judge | separate images | combined image |
| --- | --- | --- |
| Qwen2.5-VL-7B | 37.77% | 51.40% |
| InternVL3-8B | 54.80% | 60.07% |

The same table also carries the modality contrast (adding source text lowers
swap error for both judges here, the opposite direction from the closed-model
checks in Table 2(a)) and the RF-failure / VLM-rescue diagnostics.

Per-condition directories hold the pair-level predictions, visual and source
feature tables, and the rescue-model outputs.

A pre-correction run of the same analysis exists in the source tree
(`results/complementarity_mechanism/`, **not shipped**). It is excluded because
its own `DEPRECATED_CORRUPTED_RENDER_INPUTS.md` marks its rendered inputs as
corrupted and superseded by these `clean_*` runs.

The per-condition `README.md` files in each subdirectory are the original run
READMEs. Their "Required inputs" and "Command order" paths point at the source
tree under `/ANON/experiment_root/`, not at this package; the inputs they name
are shipped here as `data/features_and_baseline_predictions/java/features_313.csv`,
`data/render_metadata/`, and the judge outputs under
`results/rq1_viability/raw_inference/`.
