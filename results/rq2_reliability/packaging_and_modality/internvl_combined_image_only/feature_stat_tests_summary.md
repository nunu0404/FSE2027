# Feature Statistical Tests Summary

Mann-Whitney U tests with Cliff's delta effect sizes and Benjamini-Hochberg FDR correction were used. The table below keeps FDR < 0.05 and |Cliff's delta| >= 0.147 where available.

| comparison                      | feature                                 |   n_a |   n_b |   mean_a |   mean_b |   median_a |   median_b |   mean_diff_a_minus_b |   median_diff_a_minus_b |   mannwhitney_p |   cliffs_delta |   fdr_p |
|:--------------------------------|:----------------------------------------|------:|------:|---------:|---------:|-----------:|-----------:|----------------------:|------------------------:|----------------:|---------------:|--------:|
| RFwrong_VLMrescue_vs_VLMinvalid | visual_non_background_pixel_density_max |   248 |   667 |   0.9969 |   0.9964 |     0.9976 |     0.9971 |             0.0004607 |               0.0005117 |       0.0001049 |         0.1667 | 0.02854 |
