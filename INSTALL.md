# Installation

## Verify the package first

```bash
cd FSE2027
bash code/99_verification/verify_manifest.sh
```

This checks every shipped file against `MANIFEST.sha256`. On a healthy package
it prints `OK: <n> files verified`.

## Tier 0 — reported numbers, no dependencies

```bash
diff <(python3 code/99_verification/recompute_headline_numbers.py) \
     results/verified/EXPECTED_HEADLINE_OUTPUT.txt && echo "RQ1 MATCH"
diff <(python3 code/99_verification/recompute_rq2_rq3_rq4.py) \
     results/verified/EXPECTED_RQ2_RQ3_RQ4_OUTPUT.txt && echo "RQ2-RQ4 MATCH"
```

The second script also exits non-zero by itself if any check mismatches, so it
can be used as a smoke test without the `diff`.

## Tier 1 and 2 — analysis and rendering

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements_analysis.txt
```

The tree-sitter grammars used by the renderer and by the RQ3 variant generator
ship in `code/01_rendering/vendor/`. Put them on the path rather than
reinstalling, so that the pinned grammar versions are used:

```bash
export PYTHONPATH="$PWD/code/01_rendering/vendor:$PYTHONPATH"
```

## Tier 3 — inference

Two environments are needed, because Phi-4-multimodal does not run under the
base stack.

### Base environment (7 of 8 open-weight judges)

```bash
python3.13 -m venv .venv-base
source .venv-base/bin/activate
pip install \
  torch==2.11.0 torchvision==0.26.0 \
  transformers==5.9.0 accelerate==1.14.0 \
  huggingface-hub==1.5.0 safetensors==0.7.0 tokenizers==0.22.2 \
  pillow==12.1.1 pandas==3.0.1
```

### Phi-4-multimodal environment

```bash
python3.10 -m venv .venv-phi
source .venv-phi/bin/activate
pip install \
  torch==2.9.1+cu130 torchvision==0.24.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.48.2 accelerate==1.3.0 pillow==11.1.0
```

The attention backend and the Ministral tokenizer-regex fix that the original
runs applied are recorded in `environment/env_rq1_model_battery.json` under
`phi_attention_backend` and `ministral_fix_mistral_regex`. Install logs are in
`environment/install_logs/`.

### Model weights

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Repeat for each entry in `environment/MODEL_REVISIONS.json`. The `google/gemma-*`
repositories are gated and require an accepted license.

### Closed models

```bash
export OPENAI_API_KEY=...
python3 code/02_inference/protocol_checks/run_openai_vlm_pilot.py --help
```

No key ships with this package.

## OCR engines (RQ4 only)

```bash
pip install rapidocr-onnxruntime easyocr
```

The EasyOCR weights that the original run used are preserved alongside the
outputs; see `results/rq4_image_only/ocr_substitution_frozen_rf/run_manifest.json`
for the exact engine versions and settings.
