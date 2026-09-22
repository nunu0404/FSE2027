# Requirements

## Tier 0 — Verify the reported numbers (no GPU, no network)

Everything in §3 of the README runs here.

* Python 3.8 or newer, standard library only
* ~3 GB free disk for the unpacked package
* Under a minute of CPU time

```bash
python3 code/99_verification/recompute_headline_numbers.py
```

## Tier 1 — Re-run the analyses from raw model outputs (no GPU)

Re-derives the pair-level outcome files, bootstrap intervals, McNemar tests and
logit decompositions from the shipped `raw_inference/` JSONL.

* Python 3.10+
* `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `PyYAML`
* 16 GB RAM (the 10,000-replicate snippet-cluster bootstrap over 45,000 pairs is
  the peak)
* Roughly 30–60 minutes total on a modern desktop CPU

See [`INSTALL.md`](INSTALL.md).

## Tier 2 — Re-render the images (no GPU)

* `Pillow`, `pygments`
* `tree_sitter` with the Java, Python and C++ grammars — pinned copies ship in
  `code/01_rendering/vendor/`
* `code/01_rendering/fonts/DejaVuSansMono.ttf` (shipped; SHA-256
  `39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b`)
* A few minutes for all 18 conditions × 552 snippets

Rasterization can differ across Pillow and FreeType versions. Compare against
the per-image SHA-256 digests in `data/render_metadata/`.

## Tier 3 — Re-run inference

### Hardware actually used

Recorded in `environment/env_rq1_model_battery.json`:

* 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each
* Driver 580.95.05, CUDA 13.0

The largest checkpoints in the package (Qwen2.5-VL-32B, Gemma-3-27B,
Mistral-Small-3.1-24B) were only used for the 300-pair protocol checks. The 8B–12B
primary judges fit in a single 48 GB GPU at BF16, batch size 1.

### Software

Base environment (`environment/env_rq1_model_battery.json`):

| Package | Version |
| --- | --- |
| python | 3.13.11 |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| transformers | 5.9.0 |
| accelerate | 1.14.0 |
| pillow | 12.1.1 |
| pandas | 3.0.1 |
| huggingface-hub | 1.5.0 |
| safetensors | 0.7.0 |
| tokenizers | 0.22.2 |

Phi-4-multimodal needed a separate, older environment (python 3.10.12,
torch 2.9.1+cu130, transformers 4.48.2); its full capture is in the same file
under `phi_environment`, together with the attention-backend setting and the
Ministral regex fix that were required.

### Model weights

Not redistributed. Download from Hugging Face at the exact revisions pinned in
`environment/MODEL_REVISIONS.json`. Gated repositories (`google/gemma-*`) need
an accepted license on your account.

### Cost

| Run | Calls | Wall clock |
| --- | --- | --- |
| One judge, RQ1, 3 languages | 18,000 | ~6–8 GPU-hours |
| RQ1 primary battery, 5 judges | 90,000 | ~30–40 GPU-hours |
| RQ2 full grid, 2 judges × 17 conditions | 216,000 | ~70–90 GPU-hours |
| RQ2 reduced grid, 3 judges × 7 conditions | 126,000 | ~45 GPU-hours |
| RQ3, 2 judges × 6 contrasts | 7,200 | ~3 GPU-hours |
| RQ4 deployment and text ablation | 54,000 | ~18 GPU-hours |

Closed-model checks additionally cost roughly USD 25 in OpenAI API usage across
all conditions; per-condition token counts and costs are recorded in the
`*_usage.json` files under `results/protocol_checks/closed_models_gpt/`.

## Network access

Tiers 0–2 need none. Tier 3 needs Hugging Face (weights) and, for the closed
models, an OpenAI API key. No API key is included in this package.
