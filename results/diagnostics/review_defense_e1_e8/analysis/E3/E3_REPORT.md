# E3 Numerical corrections

## (a) OCR cost

The Java source RF has 1870/3000
correct pairs (62.33%). The best RapidOCR+ML row
is svr, with 1589/3000
(52.97%). The exact loss is
9.37 percentage points, not 10.5. The 10.5 value is the loss from
source RF to RapidOCR+RF (10.47 pp)
after rounding; it mixes a fixed RF comparison with a table row labeled as the
best OCR+ML system. Python+CUDA remains 11.90 pp.

## (b) Battery recovery

Across all five preregistered models, debiasing recovers
6.60-26.27 pp. The published
11.7-25.6 pp range excludes both Phi (the lower endpoint) and InternVL (the upper
endpoint). It exactly corresponds, up to rounding, to the post-hoc subset with
positive pooled rho and debiased accuracy above chance: Qwen, Gemma, and
Ministral. Because that alignment rule was not preregistered, the defensible main
statement is the five-model range; the restricted range may only be labeled
descriptive.

## (c) Ties and boundaries

The 108,000 observations comprise 72,000 grid observations (2 models x 3
languages x 12 conditions x 1,000 pairs) plus 36,000 perturbation observations
(2 x 3 x 6 x 1,000). There are 2,946 ties (`c=0`) and 8,168 boundaries
(`|c|=|b|`). Therefore 99,832 is the non-boundary count, not the non-tie count.
The reported 1.6-5.4% range is correctly the cell-level tie range over all 108
cells. The manuscript should use the same two explicit definitions everywhere.

## (d) Odds ratios

The source code defines the response as `invalid = ~is_valid_strict_swap`; it
does not model correct verdicts. OR<1 therefore means that larger score gaps
reduce strict-swap invalidity, which is directionally consistent with improved
reliability. The quoted medium/easy ORs are per +1.0 z, an extrapolation wider
than their respective bins. Per +0.1 z, the ORs are
0.942 and
0.989. Replace “for a correct verdict” with
“for a strict-swap failure” and report the +0.1 scale.
