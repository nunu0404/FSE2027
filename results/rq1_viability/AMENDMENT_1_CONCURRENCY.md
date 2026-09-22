# Amendment 1: Two-Model GPU Concurrency

Approved at: 2026-07-23T06:28:00Z

## Author Instruction

The author explicitly instructed the agent to stop the sequential execution and run
two models at a time because the projected sequential duration was too long.

## Change

The original hardware policy used one model process per physical GPU. The amended
execution uses physical GPU 0 as follows:

1. Qwen2.5-VL-7B and InternVL3-8B concurrently.
2. Gemma-3-12B and Ministral-3-8B concurrently after both models in group 1 finish.
3. Phi-4 multimodal alone after both models in group 2 finish.

GPU 1 remains excluded while an unrelated user's process occupies it.

No model, revision, pair, image, prompt, order, seed, dtype, decoding parameter,
output budget, or metric changes. Each model still processes 18,000 calls.

## Handling Existing Output

This amendment was made after Qwen had produced 3,296 calls under sequential
execution. Mixing those calls with concurrent calls would create a within-model
environment change. Therefore:

- the partial output is archived as `qwen_attempt0_sequential_partial`;
- it is excluded from the amended main result;
- Qwen restarts from zero concurrently with InternVL.

No other model had begun full inference.

## Interpretation

The amendment is a post-output operational change and must be disclosed. Co-location
can alter scheduling and latency and may affect very close BF16 logits. It does not
change the semantic evaluation contract. Main-result manifests identify the amended
execution group so the concurrency condition is auditable.
