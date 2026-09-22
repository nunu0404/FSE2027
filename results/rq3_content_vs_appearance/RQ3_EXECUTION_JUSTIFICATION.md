# RQ3 Execution Justification

Decision: authorized for execution after automatic preparation PASS and author GC1 approval on 2026-07-21.

## Compatibility with the running A/B protocol

The RQ3 dry-run manifests were compared field by field with the currently running A-grid manifests. Both models match on exact Hugging Face revision, Prompt B hash, BF16 precision, greedy decoding, temperature 0, inactive top-p 1.0, 24 generated-token budget, seed 42, separate ordered images, strict AB/BA presentation, and verdict-token logit capture. The same conda interpreter and inference implementation are used.

RQ3 intentionally adds a `text_plus_image` arm beside `image_only`; this is an input ablation required by the RQ3 protocol, not an accidental environment mismatch. It uses the exact source associated with each displayed variant and is analyzed separately from image-only. No results may be pooled across modalities.

## Evidence-backed choices

- **AB/BA order swap and strict-swap reporting:** Wang et al. demonstrate that pairwise LLM evaluators change judgments when response order is reversed and define a conflict rate from the two orders. This directly supports measuring both AB and BA rather than relying on one presentation: [Wang et al., ACL 2024](https://aclanthology.org/2024.acl-long.511/).
- **Qwen2.5-VL:** the official report describes native image/document/layout understanding and dynamic-resolution visual processing, making it an appropriate open VLM family for rendered-code inputs: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923).
- **InternVL3:** the official report describes native multimodal pretraining over visual and textual data, providing a distinct open VLM family rather than a second checkpoint from Qwen: [InternVL3 Technical Report](https://arxiv.org/abs/2504.10479).
- **Controlled semantic destruction:** mutation testing provides the established basis for operator-level, recorded source perturbations. RQ3 adapts this principle to construct controlled semantic-integrity cells; it does not claim that the exact opaque-name recipe is a literature standard: [Jia and Harman, IEEE TSE 2011](https://doi.org/10.1109/TSE.2010.62).
- **Four cells and six contrasts:** semantic integrity x visual quality is a complete 2 x 2 factorial manipulation. Testing all `4 choose 2 = 6` unordered contrasts avoids post-result contrast selection.

## Controlled study decisions, not claimed as literature constants

- **100 bases per language:** user-specified balanced scope with exact-binomial 80% power sufficient to detect roughly a 14.5 percentage-point deviation from 50%. Smaller effects are reported as inconclusive.
- **Seed 42:** an arbitrary fixed reproducibility identifier. Greedy decoding means it is not treated as an experimental factor.
- **24 output tokens:** output-contract headroom, not a quality hyperparameter. Earlier GA0 generations required at most 7 tokens; every model-modality arm must independently pass a fresh 100-call GA0 before its full run.
- **Prompt B:** exact replication of the study's frozen readability rubric. It is not tuned using RQ3 outcomes.
- **BF16:** an operationally frozen precision on the recorded GPU. Close-logit sensitivity is retained in the raw output rather than hidden.

## Remaining construct-validity limits

The trash variants are parser-audited fragments, not compilation-equivalent programs. Their intended unintelligibility is supported by the author GC1 review, not asserted from parsing alone. RQ3 has no independent human pairwise gold label, so outputs are preference rates and strict-swap reliability, never accuracy. These limitations must remain explicit in the paper.
