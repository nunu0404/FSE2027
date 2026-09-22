# FSE 2027 Reinforcement Analyses A-D

## Scope and integrity

- New model calls: **0**.
- Analysis A uses only `grounded_protocol_3lang_20260721` grid full runs.
- Analyses B-D use only `rq1_model_battery_3lang_20260723` complete five-model full runs.
- The runs are never pooled or compared as matched cells.
- GA0 passed on **234,000 raw calls**: 144,000 grid calls and 90,000 battery calls. Duplicate keys, incomplete AB/BA groups, parse failures, verdict/logit-argmax mismatches, non-finite logits, and image-mapping failures were all zero.

## A. Cross-rendering preference flips

### Method

For each model-language cell, the same 1,000 pairs were compared under all 66 unordered pairs of the 12 rendering conditions. AB and BA decisions were separately normalized to snippet identity. The both-valid analysis retained only pairs that were strict-swap valid under both rendering conditions. Exact 95% binomial intervals accompany every estimate. A one-sided exact test against a 0.5 chance flip rate was Holm-corrected within model-language-metric families.

The order baseline is the within-condition strict-swap error. No repeated full run of the same condition exists, so the deterministic repeat-run preference-flip baseline is **not measurable**; no new run was created.

### Main results

| Model         | Lang.   | AB med.   | AB max   | BA med.   | BA max   | Both-valid med.   | Both-valid max   | Order baseline med.   |
|:--------------|:--------|:----------|:---------|:----------|:---------|:------------------|:-----------------|:----------------------|
| InternVL3-8B  | cuda    | 9.2%      | 14.2%    | 11.8%     | 16.9%    | 1.1%              | 3.9%             | 69.3%                 |
| InternVL3-8B  | java    | 25.4%     | 30.3%    | 16.6%     | 21.3%    | 7.1%              | 11.1%            | 41.4%                 |
| InternVL3-8B  | python  | 12.6%     | 17.4%    | 15.2%     | 17.5%    | 2.5%              | 6.4%             | 59.9%                 |
| Qwen2.5-VL-7B | cuda    | 10.8%     | 14.4%    | 9.7%      | 13.2%    | 0.7%              | 1.6%             | 38.6%                 |
| Qwen2.5-VL-7B | java    | 14.9%     | 20.0%    | 11.1%     | 15.2%    | 2.1%              | 4.8%             | 49.3%                 |
| Qwen2.5-VL-7B | python  | 12.0%     | 15.9%    | 10.2%     | 12.7%    | 0.6%              | 2.8%             | 57.0%                 |

Across model-language cells, median fixed-order preference-flip rates ranged from 9.2% to 25.4%, and the largest individual fixed-order flip rate was 30.3%. When both conditions were strict-valid, median flip rates fell to 0.6%-7.1%. The median within-condition order baseline was much larger, 38.6%-69.3%, showing that order was the larger disturbance in this run. No condition pair exceeded a 0.5 chance flip rate after Holm correction (0/1,188 metric rows). The largest factor-level mean was wrap_only for OpenGVLab/InternVL3-8B/java/fixed_order_AB (26.9%).

Interpretation is limited to **preference change**, not correctness. Rendering changes can reverse a fixed-order preference for unchanged code content, but most such reversals disappear when requiring strict validity in both conditions.

## B. |c| decile calibration

### Method

Within each battery model, nonzero-|c| pairs were sorted into ten equal-count deciles with deterministic pair-ID tie breaking. Exact `c=0` pairs were assigned to D1 and counted as failures in primary debiased metrics; all non-tie sensitivity columns are retained. Decile valid accuracy conditions on strict-valid pairs, whereas effective accuracy counts invalid pairs as failures.

Top-minus-bottom differences request 10,000 two-way snippet-cluster bootstrap replicates. Replicates with a zero denominator in either compared decile are omitted; each `*_bootstrap_reps_used` column records the effective count. Calibration partitions snippets, stratified by language, into deterministic SHA-256 50/50 train/test halves. Isotonic regression is fitted only to train-train pairs and evaluated only on test-test pairs; cross-half pairs are excluded from calibration only. Every model-scope split has zero train/test snippet overlap. ECE uses ten equal-frequency probability bins, and AUROC targets debiased correctness.

### Overall results

| Model          |   rho(valid) |   rho(effective) |   rho(debiased) | D10-D1 effective [95% CI]   | ECE   |   AUROC |
|:---------------|-------------:|-----------------:|----------------:|:----------------------------|:------|--------:|
| Gemma-3-12B    |         0.95 |             0.99 |            0.97 | 70.5% [62.5%, 77.7%]        | 3.8%  |    0.58 |
| InternVL3-8B   |        -0.05 |             0.96 |           -0.19 | 40.6% [30.0%, 52.1%]        | 5.1%  |    0.52 |
| Ministral-3-8B |         0.45 |             0.99 |            0.74 | 61.6% [52.7%, 70.2%]        | 2.8%  |    0.54 |
| Phi-4-MM       |        -0.45 |             0.73 |           -0.22 | 26.9% [16.7%, 38.7%]        | 3.9%  |    0.63 |
| Qwen2.5-VL-7B  |         0.94 |             1    |            1    | 83.4% [76.4%, 89.3%]        | 6.8%  |    0.62 |

Effective accuracy increased strongly with |c| for every model (decile rho 0.73-1.00), largely because |c| also predicts strict validity. Correctness *within valid pairs* was model-dependent: Qwen and Gemma were strongly positive, Ministral was moderate, InternVL was flat, and Phi was negative. Debiased correctness showed the same caution: Qwen/Gemma were monotone, Ministral was weaker, and InternVL/Phi were non-monotone or inverse. Thus |c| is a useful failure/coverage signal across the battery, but it is not a universally calibrated content-correctness confidence score.

Phi CUDA and Python have no strict-valid observations in D1, so their D10-D1 valid-accuracy differences are undefined; effective and debiased differences remain defined. Their non-tie D1 cells are also empty, making the corresponding non-tie differences undefined. These denominator failures are retained as `NA`.

## C. Joint (|b|, |c|) distribution

The battery contributes 45,000 pair rows. The identity `|c|>|b|` matched strict validity with **0 non-boundary violations**. There were 5,577 exact `|c|=|b|` boundaries; parsed validity is not algebraically determined on those boundaries. Under a conservative boundary-as-nonvalid rule, 2,576 boundary cases differ from parsed validity and are reported separately, not counted as non-boundary violations.

The full pair-level coordinates are in `C_bc_joint.csv`; the five-panel density figure draws `|c|=|b|`. For readability, only the plotted axes are clipped at each model's 99.5th percentile; CSVs and summary statistics retain every observation. `C_grid_condition_trajectory.csv` separately records condition-wise median coordinates within the grid run.

## D. First-position choice rate

The battery was decomposed into 15 model-language cells, each with 3,000 pairs and 6,000 calls. Rates and 95% intervals use 10,000 pair-cluster bootstrap replicates. The observed language-specific range was 24.6%-78.1%. Difficulty-specific rows are retained for the appendix.

## Tie and denominator policy

- B: `c=0` is assigned to D1 and is incorrect in primary debiased metrics; non-tie sensitivity estimates are provided.
- C: exact `|c|=|b|` boundaries are reported separately because the strict inequality cannot classify them.
- Every rate includes `n_pairs`, `n_calls`, `valid_pairs`, or the applicable denominator in its CSV.

## Interpretation cautions

1. Preference flip is not an error rate.
2. Both-valid flip estimates condition on a selected subset and must be shown with their variable denominator.
3. The absent repeated-run baseline is not assumed to be zero; it is marked unmeasurable.
4. Isotonic ECE/AUROC use a snippet-disjoint subset, not all 9,000 pairs.
5. The grid and battery generations remain analytically separate.
