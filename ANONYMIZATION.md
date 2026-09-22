# Anonymization

The paper is under double-blind review. Absolute filesystem paths recorded by
the original runs contained the author's account name, so those paths were
rewritten before this package was assembled.

## What was rewritten

Applied to **389 text files**; no binary file contained any of these strings.

| Original | Replacement |
| --- | --- |
| `/home/<author>/experiment_26_v1` | `/ANON/experiment_root` |
| `/data/tmp/<author>-rq1` | `/ANON/scratch_rq1` |
| `/data/tmp/<author>_hf_cache` | `/ANON/hf_cache` |
| `/home/<author>` | `/ANON/home` |
| remaining bare `<author>` | `anon` |

The full list of rewritten files is `ANONYMIZED_FILES.txt`.

Nothing else was altered. No numeric value, verdict, logit, checksum field, or
code path was touched — only the account-name component of absolute paths.

## Effect on the checksums

`MANIFEST.sha256` at the package root is the authoritative integrity record and
was generated **after** anonymization. Use it:

```bash
bash code/99_verification/verify_manifest.sh
```

Several run directories also carry their own older checksum files, written by
the original pipelines before anonymization:

| Legacy manifest | Entries whose file was rewritten |
| --- | --- |
| `results/diagnostics/review_defense_e1_e8/SHA256SUMS` | up to 67 |
| `results/rq2_reliability/ARTIFACT_SHA256.csv` | up to 27 |
| `results/rq4_image_only/image_only_pipelines/SHA256_INVENTORY.csv` | up to 22 |
| `results/rq1_viability/analysis_5model_battery/table1_ml_replication/SHA256SUMS` | up to 16 |
| `data/features_and_baseline_predictions/java/table1_ml_replication/SHA256SUMS` | up to 16 |
| `results/diagnostics/final_verification/SHA256SUMS` | up to 15 |
| `results/diagnostics/followup_f1_f4/inventory/SHA256_INVENTORY.{csv,md}` | up to 6 |
| `results/diagnostics/reinforcement_abcd/ARTIFACT_SHA256.csv` | up to 2 |

For a file that was rewritten, these legacy digests describe the
pre-anonymization bytes and will not match. They are kept as provenance of the
original runs, not as a check to run now. Every entry in them for a file that
was *not* rewritten — which is the overwhelming majority, including all raw
inference JSONL, pair-level CSVs, and rendered PNGs — still verifies.

## De-anonymizing for camera-ready

```bash
grep -rlI '/ANON/' . | xargs sed -i \
  -e 's|/ANON/experiment_root|<your path>|g' \
  -e 's|/ANON/scratch_rq1|<your path>|g' \
  -e 's|/ANON/hf_cache|<your path>|g' \
  -e 's|/ANON/home|<your path>|g'
```

Regenerate `MANIFEST.sha256` afterwards:

```bash
find . -type f ! -name MANIFEST.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > MANIFEST.sha256
```

## Also checked

* No API keys, tokens, private keys, or credential files. `/ANON/scratch_rq1/hf/token`
  appears as a *path* in a cache configuration; no token value is present.
* The `@gmail.com` and `@github.com` addresses that appear in
  `results/*/audit/blur_ocr_raw.csv`, `rq3_mutation_log.csv`, and the OCR text
  dumps belong to third-party contributors of the tree-sitter grammars and
  benchmark snippets. They are part of the analyzed source code, not author
  contact information, and were left intact.
* Korean-language working notes under `docs/` are preserved as run provenance and
  carry no author identification after the path rewrite.
