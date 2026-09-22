# RQ3 Readiness Record

Status: automatic preparation PASS; author GC1 pending; inference not authorized.

## Frozen design

- Bases: 300 total, 100 each for Java, Python, and CUDA.
- Human-score strata per language: 34 low, 33 mid, 33 high.
- Cells per base: golden, ugly_gold, beautiful_trash, ugly_trash.
- Contrasts: all six unordered cell pairs, 600 per language and 1,800 total.
- Orders: AB and BA for every contrast.
- Models: pinned Qwen2.5-VL-7B-Instruct and InternVL3-8B revisions.
- Modalities: image_only and text_plus_image, analyzed separately.
- Prompt/decoding: frozen Prompt B, BF16, greedy, temperature 0, inactive top-p 1.0, max 24 tokens, seed 42.
- Output: direct pairwise preference plus verdict-token A/B logits. RQ3 outcomes are preference rates, not accuracy.

The full plan is 3,600 calls per model-modality, 7,200 per model, and 14,400 calls total. Each model-modality first runs a distinct 100-call GA0 pilot; pilot rows are never resumed into full outputs.

## Passed automatic checks

The independent audit in `audit/RQ3_AUTOMATIC_AUDIT.json` verifies 300 bases, 600 sources, 1,200 images, 1,800 contrasts, complete six-contrast coverage, file/hash/dimension/row accounting, nonblank images, no source-hash drift within clean/ugly pairs, no parser-error increase, no original identifier cue shared by gold and trash, and no nonopaque comment left in trash.

All three final contact sheets were visually inspected. No code-row overlap, clipping, blank render, or failure to distinguish clean from ugly was observed. The GC1 HTML references 240 distinct images and all links resolve.

## Blocking author gate

Review `review/GC1_REVIEW.html` and complete all six decision columns for every row in `review/GC1_REVIEW_FORM.csv`. Then run:

```bash
/ANON/home/miniconda3/bin/python \
  results/grounded_protocol_3lang_20260721/code/validate_rq3_gc1.py
```

The validator requires all 60 stratified items to pass every check. It writes `review/GC1_APPROVAL.json`; the real inference runner refuses to execute without this approval. Four model-modality dry-run manifests already pass with `gc1_authorized=false` under `inference/raw/`.

After approval and GPU availability, launch one tmux session per model with `code/run_rq3_model_pipeline.sh`. Both modalities execute sequentially within each model process to avoid duplicate model residency.
