# Environment

| File | Contents |
| --- | --- |
| `MODEL_REVISIONS.json` | Exact Hugging Face revisions for all 12 open-weight models, the closed-model snapshots, and the renderer settings |
| `env_rq1_model_battery.json` | The capture from the RQ1 battery: GPUs, driver, CUDA, and two package environments (base, and the older stack Phi-4-multimodal required), plus the attention-backend setting, the Ministral tokenizer-regex fix, runner SHA-256, and per-model snapshot paths |
| `latest_vlm_extension/*.json` | Per-model captures for the three extension judges, smoke and full |
| `env_review_defense.json` | The environment for the E1–E8 review-defense runs |
| `install_logs/` | Installation logs for the Phi-4 environment |
| `rq2_reduced_run_manifests/` | Per-judge reduced-grid manifests, including revisions and decoding settings |
| `requirements_base.txt` | The project's loose requirements, as used during development |
| `requirements_analysis.txt` | Pinned-enough set to re-run analysis and rendering on CPU |
| `requirements_lock_reference.txt`, `Dockerfile.reference` | Reference artifacts from the earlier project skeleton |

## Determinism

Every open-weight run: BF16, `do_sample=false`, `temperature=0.0`, `top_p=1.0`,
batch size 1, `max_new_tokens=24`, `seed=42`. Given the same checkpoint revision
and library versions, verdicts reproduce exactly.

BF16 rounding differences across CUDA, cuDNN, or transformers versions can move
pairs that sit on an exact logit tie. That matters here: discarding every
boundary pair (`|c| = |b|`) instead of accepting agreeing ones moves
Qwen2.5-VL-7B's strict-swap error from 45.78% to 52.56%. To confirm a reported
number, analyse the shipped verdict files rather than a fresh run. The
same-environment and changed-environment repeat runs in
`../results/diagnostics/review_defense_e1_e8/analysis/E4/` quantify this
directly.
