# F2 proxy-label agreement ceiling

## Definition

This is not human accuracy. It is the probability that a resampled individual
rating (`k=1`) or a bootstrap panel mean at each snippet's original panel size
(`k=panel`) agrees with the direction of the stored mean-score proxy label.
Ratings for the two snippets are sampled independently from their observed
rating distributions because the full pair set is not restricted to common
raters. Draw ties receive 0.5 credit. Confidence intervals use 10,000
two-endpoint snippet-cluster bootstrap replicates.

## Coverage and mapping caveat

Python and CUDA cover all 3,000 pairs per language. Java covers the 1,004
within-dataset pairs benchmark by benchmark; the 1,996 cross-dataset pairs are
undefined and are not interpolated. The public Dorn Java matrix reproduces the
stored proxy means for 89/90 selected snippets using its distributed column
layout. It has no column that reproduces `rq0_0099` (snippet 101), so the nine
within-Dorn pairs containing that snippet are reported unavailable rather than
assigned an invented rating vector.

## Sanity check

For the five-model battery, the highest easy-bin valid accuracy is
84.44% on Python and
87.99% on CUDA. The corresponding
bootstrap-panel proxy-agreement ceilings are
100.00% and
100.00%. Any model value above the
estimated ceiling should be interpreted as alignment with this fixed proxy
sample, not as exceeding human performance.

## English Threats draft

We estimated a proxy-label agreement ceiling by resampling the released rater
matrices. For k=1, we computed the probability that one independently sampled
rating per snippet reproduced the direction of the benchmark mean-score
label; for k=panel, we bootstrap-resampled each snippet at its original panel
size and compared the resulting means. Ties received half credit, and 95%
intervals used 10,000 two-endpoint snippet-cluster bootstrap replicates. These
quantities are agreement probabilities with the proxy label, not estimates of
human accuracy. Java cross-benchmark pairs have no coherent shared-panel
ceiling and were left undefined. In addition, one selected Dorn Java snippet
could not be mapped to a released rating column that reproduced its stored
mean, so its nine within-Dorn pairs were explicitly excluded.
