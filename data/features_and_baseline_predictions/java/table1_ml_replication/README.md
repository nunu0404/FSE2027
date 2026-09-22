# Table 1 ML baseline evidence

This directory is a derived, auditable slice of the frozen RQ1 battery assets. It contains no newly fitted model results.

## Files

- `canonical_pairwise_summary.csv`: the nine Table 1 cells, including exact numerators and denominators.
- `canonical_pairwise_predictions.csv`: all 27,000 model-language-pair decisions used to aggregate the table.
- `canonical_oof_snippet_predictions.csv`: snippet-level OOF scores and fold IDs from which pair decisions were made.
- `fold_summary.csv`: fold-by-fold OOF coverage for every model-language cell.
- `legacy_reported_values_audit.csv`: requested legacy values versus pair-ID-corrected values and their deltas.
- `pair_id_reuse_audit.csv`: the three reused IDs with their old and frozen endpoint snippets.
- `run_manifest.json`: pair-set identity, CV protocols, model settings, source paths, and source hashes.
- `SHA256SUMS`: hashes for every artifact in this directory.
- `generate.log`: validation and generation summary.

## Evaluation contract

Each language has one frozen 3,000-pair set (1,000 easy, 1,000 medium, 1,000 hard), and all three models use exactly that language's same pair IDs. Each snippet score is produced out of fold under a fixed five-fold protocol with seed 42. MLP and Voting compare OOF readable-class probabilities; SVR compares OOF predicted readability z-scores.

These baselines are deterministic pair scorers rather than two-call judges. Every numeric prediction is valid, swapping the displayed order cannot change the preferred snippet, and therefore valid accuracy equals effective accuracy and strict-swap error is zero. Exact score ties are retained as valid conservative errors; their counts are reported separately.

The earlier battery aggregation reused three pair IDs after the Python/CUDA clean-pair rebuild even though their endpoint snippets changed (one CUDA pair and two Python pairs). The corrected summary re-scores the frozen clean pairs from the stored snippet-level OOF predictions. The legacy audit file preserves the originally reported values and quantifies the correction.

## Rebuild

From the repository root:

```bash
python results/rq1_model_battery_3lang_20260723/code/build_table1_ml_replication.py
```
