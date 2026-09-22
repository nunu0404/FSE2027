# Configuration

## `prompts/`

One frozen judging instruction, used by every run. See `prompts/README.md`.

## `protocols/`

| File | Scope |
| --- | --- |
| `protocol_rq1_model_battery.json` | RQ1 5-judge battery: models and revisions, 9,000 pairs, image-only, both orders, decoding settings, rendering condition, analysis endpoints |
| `protocol_rq1_latest_vlm_extension.json` | RQ1 3-judge extension |
| `protocol_rq2_rq3_grounded_3lang.json` | RQ2 rendering grid and perturbations, RQ3 variants, and the statistical plan (cluster-robust primary inference, McNemar and Cochran Q sensitivity, Holm within family, Wilson intervals) |
| `protocol_rq4_deploy_execution_lock.json` | RQ4 deployment routes, locked before execution |
| `rq4_deploy_preregistration.md` | The RQ4 pre-registration text |
| `rq2_reduced_manifests/*.json` | Per-judge reduced-grid run manifests: 7 conditions, revisions, decoding |

These are pre-registration documents. Where a protocol records a plan that the
runs did not reach — for example the RQ3 InternVL text+image arm — the outcome
is recorded in the corresponding results directory rather than by editing the
protocol.
