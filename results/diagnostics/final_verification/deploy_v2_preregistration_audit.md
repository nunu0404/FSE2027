# F7 deploy-v2 Preregistration Audit

## Verdict

Use **"prespecified before the rerun"**, not the stronger unqualified **"preregistered rerun"**. The local evidence is internally strong: a protocol and cryptographic execution lock precede inference, and their current hashes match. However, the repository has no Git history and no external/public immutable timestamp or registry was found, so the local filesystem timestamps alone do not establish conventional preregistration.

## Evidence and timeline

- Protocol: `/ANON/scratch_rq1/deploy_v2_logit_20260731/config/F5_PREREGISTRATION.md`
- Protocol birth/mtime: 2026-07-31 01:58:54.097405482 UTC.
- Current SHA-256: `c008a88fe2f1033a55bd9e8927b111e267d5f6a591d54c48354f3566247f34a7`.
- Execution lock: `/ANON/scratch_rq1/deploy_v2_logit_20260731/config/F5_EXECUTION_LOCK.json`
- Lock timestamp: 2026-07-31 02:01:13.544493 UTC.
- The lock records the same protocol SHA-256 and frozen-input/runner hashes.
- First full-run call: 2026-07-31 02:01:53.953237 UTC in `inference/direct_qwen_image_only.jsonl`, approximately 40.4 seconds after the lock.
- Final call: 2026-07-31 06:44:49.179210 UTC.

## Prespecified fields

The protocol fixes all requested fields before inference:

- Three pipelines and their roles.
- Three languages x 3,000 pairs x AB/BA x three systems = 54,000 calls.
- `c=(m_AB-m_BA)/2`, main D=`sign(c)`, and AB-only/E/V/S definitions.
- Main tie rule (`c=0` incorrect), exclude-tie and half-credit sensitivities, and separate boundary count for `|c|=|b|` with `c!=0`.
- Language-specific primary comparator `D(Direct Qwen)-E(best RapidOCR+ML)` and eight secondary row-language comparisons.
- 10,000 two-endpoint snippet-cluster bootstrap replicates.
- Exact McNemar tests and Holm correction over nine comparisons.
- Outcome categories (a)/(b)/(c).
- Frozen model revisions, BF16 greedy settings, batch size one, seed 42, max 24 tokens, prompt, inputs, hashes, stop gates, and no-update rule.

The resulting 18,000-row files per system and manifests are consistent with the 54,000-call plan. No protocol/hash mismatch was detected. The execution lock has `deterministic_algorithms_enabled=false`; this is disclosed rather than treated as a protocol mismatch.

## Post-run modification check

The protocol's current hash equals `preregistration_sha256` in the execution lock. The frozen inventory's current hash also equals the lock. This detects content modification relative to the lock, but it does not make the timestamp independently immutable.

## Replication-package status

The protocol, lock, frozen prompt, inventories, runner, and analysis are technically includable in an anonymous package after copying them from `/data/tmp/...` into the package and removing machine-specific absolute paths where anonymity requires it. They have not been proven to be in an existing public or externally timestamped package. Preserve original hashes and add a package-level hash manifest.

## Paper-ready text

"Before the deploy-v2 rerun, we fixed the 54,000-call protocol, model revisions, inputs, decision and tie rules, primary comparator, 10,000-replicate endpoint-cluster bootstrap, exact McNemar tests, Holm family, and outcome categories in a SHA-256-locked local protocol. We therefore describe this analysis as prespecified before rerun; no external registration timestamp is available."

