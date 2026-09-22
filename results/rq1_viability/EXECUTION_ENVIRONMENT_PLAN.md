# RQ1 Execution Environment Plan

## Hardware Policy

The main run uses one model process per physical GPU. Two large model processes are not
co-located on one GPU even if static weights fit, because image-dependent activation
peaks and kernel scheduling can change close logits. GPU UUID, driver, CUDA runtime,
free memory before load, and competing processes are recorded.

At preparation time:

- GPU 0: RTX PRO 6000 Blackwell, 97,251 MiB free, idle.
- GPU 1: RTX PRO 6000 Blackwell, 38,552 MiB free, 100% utilized by an unrelated
  58,690 MiB Python process.

GPU 1 is not used while that process remains. At execution time the scheduler checks
again: if both GPUs are idle, one model runs on each GPU; otherwise models run
sequentially on GPU 0. All jobs run in named tmux sessions.

## Storage and Downloads

At preparation time the filesystem has 168 GiB free and the Hugging Face cache uses
47 GiB. New BF16 parameter payloads are approximately 11.1 GB for Phi-4, 24.4 GB for
Gemma, and 17.8 GB for Ministral before auxiliary files and temporary download space.
Downloads are serialized and verified by revision. The process stops rather than
starting a download if the projected temporary footprint would leave less than 50 GiB
free.

Gemma is manually gated. No Hugging Face token is currently configured. Its license
must be accepted and a token supplied without writing the token into logs or manifests.

## Software Isolation

Qwen and InternVL reuse the validated grounded runner environment where compatible.
New models use isolated pinned environments because their native implementations have
different requirements:

- Phi-4: isolated Python 3.10 environment; first-party card recommends
  Transformers 4.48.2, PyTorch 2.6.0, and Accelerate 1.3.0. Pinned custom code is
  reviewed before `trust_remote_code=True`.
- Gemma 3: isolated environment with a Gemma-3-capable Transformers release and the
  frozen model revision.
- Ministral 3: isolated environment using its supported Mistral/vLLM stack. The adapter
  must expose emitted-token log probabilities and two separate image inputs.

Different native processors are model implementation details, not experimental
treatments. The exact same PNG bytes and semantic prompt are used. Environment
manifests record package lock files and processor configurations.

## Adapter Gate

Before GA0, each adapter is tested without reading RQ1 outcomes:

1. Load the pinned revision in BF16 without quantization.
2. Confirm two distinct images are represented in the serialized request.
3. Confirm AB/BA image order changes only the image order.
4. Confirm the final A/B token position and recover both token logits.
5. Confirm generated text obeys the frozen verdict contract.
6. Record peak GPU memory and calls per minute.

Adapter fixes cannot alter Prompt B or add model-specific demonstrations.
