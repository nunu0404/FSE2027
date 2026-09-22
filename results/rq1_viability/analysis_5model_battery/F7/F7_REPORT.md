# F7 baseline confirmation and E1 correction

No new model inference was performed. The frozen clean pair endpoints were rescored from stored five-fold OOF snippet predictions.

## Decision

Voting (LR+NB+RF) on Python is 1,940/3,000 = 64.6667% (64.67%), not 1,938/3,000 = 64.60%.
The old battery ML aggregation joined pre-clean predictions to reused pair IDs after three endpoint pairs changed (one CUDA and two Python).

## Requested cells

| model                     | language   |   correct_pairs |   num_pairs |   effective_accuracy |   confirmed_value_percent_2dp |
|:--------------------------|:-----------|----------------:|------------:|---------------------:|------------------------------:|
| Voting (LR+NB+RF)         | Python     |            1940 |        3000 |             0.646667 |                         64.67 |
| Multilayer Perceptron     | CUDA       |            2069 |        3000 |             0.689667 |                         68.97 |
| Multilayer Perceptron     | Python     |            1888 |        3000 |             0.629333 |                         62.93 |
| Support Vector Regression | Python     |            1786 |        3000 |             0.595333 |                         59.53 |

## Corrected E1: debiased minus language-best

| model_key   | language   |   debiased_accuracy | baseline_model             |   baseline_accuracy |   diff_debiased_minus_baseline |   snippet_cluster_ci_low |   snippet_cluster_ci_high |   mcnemar_exact_p |   holm_p_30_comparisons |
|:------------|:-----------|--------------------:|:---------------------------|--------------------:|-------------------------------:|-------------------------:|--------------------------:|------------------:|------------------------:|
| qwen        | java       |            0.585333 | Multilayer Perceptron      |            0.631667 |                     -0.0463333 |               -0.101248  |                0.00989662 |      0.000118039  |            0.00094431   |
| internvl    | java       |            0.462667 | Multilayer Perceptron      |            0.631667 |                     -0.169     |               -0.231444  |               -0.105583   |      3.80952e-36  |            7.61904e-35  |
| gemma       | java       |            0.607667 | Multilayer Perceptron      |            0.631667 |                     -0.024     |               -0.0776577 |                0.0307904  |      0.0461227    |            0.230613     |
| ministral   | java       |            0.54     | Multilayer Perceptron      |            0.631667 |                     -0.0916667 |               -0.153717  |               -0.02884    |      4.33424e-12  |            4.76766e-11  |
| phi         | java       |            0.378667 | Multilayer Perceptron      |            0.631667 |                     -0.253     |               -0.318398  |               -0.185109   |      5.71587e-73  |            1.54328e-71  |
| qwen        | cuda       |            0.671667 | svr                        |            0.758    |                     -0.0863333 |               -0.14966   |               -0.0245475  |      8.16103e-17  |            1.06093e-15  |
| internvl    | cuda       |            0.557667 | svr                        |            0.758    |                     -0.200333  |               -0.273041  |               -0.124665   |      1.12905e-59  |            2.70973e-58  |
| gemma       | cuda       |            0.629333 | svr                        |            0.758    |                     -0.128667  |               -0.198938  |               -0.0590811  |      4.11557e-27  |            7.40803e-26  |
| ministral   | cuda       |            0.564667 | svr                        |            0.758    |                     -0.193333  |               -0.263802  |               -0.120161   |      5.06639e-54  |            1.16527e-52  |
| phi         | cuda       |            0.378333 | svr                        |            0.758    |                     -0.379667  |               -0.453308  |               -0.304585   |      2.27936e-171 |            6.83809e-170 |
| qwen        | python     |            0.667667 | Voting ensemble (LR+NB+RF) |            0.646667 |                      0.021     |               -0.0619165 |                0.105194   |      0.0588113    |            0.230613     |
| internvl    | python     |            0.509667 | Voting ensemble (LR+NB+RF) |            0.646667 |                     -0.137     |               -0.240321  |               -0.0275363  |      4.62698e-23  |            7.86587e-22  |
| gemma       | python     |            0.623333 | Voting ensemble (LR+NB+RF) |            0.646667 |                     -0.0233333 |               -0.121661  |                0.0728661  |      0.0539243    |            0.230613     |
| ministral   | python     |            0.524333 | Voting ensemble (LR+NB+RF) |            0.646667 |                     -0.122333  |               -0.230025  |               -0.0135079  |      2.48416e-19  |            3.97465e-18  |
| phi         | python     |            0.378    | Voting ensemble (LR+NB+RF) |            0.646667 |                     -0.268667  |               -0.375906  |               -0.153427   |      3.62055e-74  |            1.01375e-72  |

## Method

The CI uses 10,000 percentile bootstrap replicates with seed 42. Unique snippets are sampled with replacement within each language; a pair receives the product of its two endpoint multiplicities. Exact McNemar uses matched pair outcomes. Holm adjustment is recomputed over the prespecified 30 E1 comparisons (5 VLMs x 3 languages x RF/language-best).
