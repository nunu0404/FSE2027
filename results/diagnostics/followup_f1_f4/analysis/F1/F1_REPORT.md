# F1 result and manuscript draft

## Audit conclusion

The stored deployment runs contain generated AB/BA verdicts but not the
verdict-token logits required to compute c=(m_AB-m_BA)/2. Consequently, D is
not measured for Direct Qwen image-only, source-text Qwen2.5-Coder, or
RapidOCR-text Qwen2.5-Coder in any language. Runs with logits elsewhere in the
repository use different rendering or execution protocols and cannot be
merged with the deployment run.

## Required answer

Whether adding the debiased column changes the strongest screenshot-only
system is **not identifiable from the stored deployment assets**. The existing
effective-accuracy result remains: RapidOCR+ML is 52.97% on Java, 59.53% on
Python, and 60.73% on CUDA. It must not be compared against a D value imported
from another run.

## English draft

The deployment runs preserved both order-specific verdicts but did not retain
the verdict-token logits required for logit-level order debiasing. We therefore
report the debiased column as not measured, rather than importing estimates
from a different rendering or execution run. Under the available end-to-end
effective metric, RapidOCR plus the best supervised regressor remains the
strongest screenshot-only pipeline in each language. Establishing whether a
two-call, logit-accessible zero-shot VLM changes that conclusion requires a
separately preregistered deployment rerun; it cannot be inferred from these
stored outputs.

Such a rerun would require 54,000 calls (three neural rows x three languages x
3,000 pairs x two orders), preserve the two-call cost and open-model/logit
access caveats, and retain the supervised OCR+ML versus zero-shot VLM
asymmetry.
