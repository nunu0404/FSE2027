# Rendered images

These are the PNGs the judges actually consumed. 11,688 files, 1.3 GB.

| Directory | Files | Contents |
| --- | --- | --- |
| `rq2_rendering_grid/` | 12 × 552 = 6,624 | The 12-condition rendering grid: 3 themes (`monokai_dark`, `friendly_light`, `mono_light`) × 2 font sizes (20, 24) × 2 wrap widths (60, 80), line numbers always on |
| `rq2_perturbation/` | 6 × 552 = 3,312 | `baseline`, `no_indent`, `no_blank_lines`, `gaussian_sigma_1`, `gaussian_sigma_2`, `gaussian_sigma_4` |
| `rq3_variants/` | 1,200 | The four RQ3 variants for 300 base snippets |
| `snippet_default_renders/` | 552 | The default single-snippet renders consumed by the OCR pipelines (`java/` 313, `python_cuda/` 239) |

## Which condition is which

* **RQ1** and the **RQ4 direct route** use `rq2_rendering_grid/monokai_dark__fs20__wrap80__lnon/`
  — the baseline condition. `data/pairs/rq1_pairs_9000_seed42.csv` names these
  files explicitly in `image_i_path` / `image_j_path`, with per-image SHA-256.
* **RQ2 full grid** uses all 12 grid conditions plus the 5 perturbations.
  `rq2_perturbation/baseline/` is byte-identical to
  `rq2_rendering_grid/monokai_dark__fs20__wrap80__lnon/`; this is the duplicated
  18th condition label discussed in `results/README.md`.
* **RQ2 reduced grid** uses 7 of those conditions: the baseline, wrap 60,
  font 24, one theme change (`mono_light__fs20__wrap80__lnon`), `no_indent`,
  `no_blank_lines`, and `gaussian_sigma_4`.
* **RQ3** uses `rq3_variants/`. Clean rendering = Monokai, line numbers,
  font 20, wrap 80. Degraded = monochrome, no line numbers, font 18, wrap 60,
  no displayed indentation, no blank lines.

## Integrity and regeneration

Per-image SHA-256 digests: `data/render_metadata/grid_render_metadata.csv`,
`perturbation_render_metadata.csv`, `data/rq3_variants/rq3_render_metadata.csv`.

```bash
python3 code/01_rendering/prepare_grounded_rendering.py
python3 code/01_rendering/audit_grounded_rendering.py
```

The renderer is token-preserving: it lexes first, then soft-wraps, so wrapping
never splits a token and continuation rows carry no source line number. Font,
tab width (4), padding (10 px), and line gap (6 px) are fixed across every
condition; only the named factor varies.

Images are shipped rather than regenerated because PNG bytes depend on the
Pillow and FreeType versions used for rasterization.

## Not included

Per-pair combined composites (~2.7 GB) for the "combined image" packaging
conditions. They are deterministic concatenations of the renders here;
regenerate with `code/01_rendering/render_combined_pair_images.py`. Their
digests are preserved in `results/protocol_checks/large_open_models/*/image_checksums.json`.
