# Adapter Attempt Ledger

This ledger distinguishes adapter/environment failures from GA0 outcome runs. No
failed adapter output is included in GA0 or full-run results.

| Model | Attempt | Status | Finding and permitted repair |
|---|---:|---|---|
| Qwen2.5-VL-7B | 0 | pass | Existing grounded adapter; exact output and verdict logits verified. |
| InternVL3-8B | 0 | fail before inference | New `HF_HOME` omitted the existing remote-code module cache. |
| InternVL3-8B | 1 | fail before inference | Snapshot-directory reference produced an incomplete dynamic-module namespace. |
| InternVL3-8B | 2 | pass | Used the pinned model ID/main-ref with the existing remote-code cache. |
| Gemma-3-12B | 0 | pass | Native Transformers processor/chat template. |
| Ministral-3-8B | 0 | superseded | Transformers warned that the tokenizer regex required correction. |
| Ministral-3-8B | 1 | pass | Set documented `fix_mistral_regex=True`; semantic prompt unchanged. |
| Phi-4 multimodal | 0 | fail before inference | First-party PyTorch 2.6/cu124 lacks Blackwell `sm_120` kernels. |
| Phi-4 multimodal | 1 | pass, superseded | PyTorch 2.9.1/cu130 with eager attention proved hardware compatibility. |
| Phi-4 multimodal | 2 | pass | Frozen SDPA backend, which is supported by pinned Phi custom code. |

Phi retains Transformers 4.48.2 and Accelerate 1.3.0. PyTorch was changed from the
card's 2.6.0 example to 2.9.1+cu130 solely because the former cannot execute on the
available Blackwell GPU. Flash Attention was not selected because the host has no
CUDA compiler (`nvcc`); SDPA is built into the pinned PyTorch runtime and explicitly
supported by the pinned Phi implementation.

The exact semantic Prompt B, image bytes, AB/BA order, BF16 dtype, greedy decoding,
24-token budget, and seed 42 were unchanged in every inference-producing attempt.
