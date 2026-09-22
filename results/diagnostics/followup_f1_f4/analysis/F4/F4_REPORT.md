# F4 both-valid floor

E4-A cannot supply the requested same-environment floor: the original grid
manifest records Python 3.13.11, whereas E4-A records Python 3.10.12. The
corresponding cross-environment both-valid disagreement ranges from
0.000% to
0.466% across six cells. E4-B produces
the same values relative to the original because E4-A and E4-B are exactly
identical at the both-valid decision level (E4-A versus E4-B floor:
0.000% to
0.000%).

Across the 11 nonbaseline rendering conditions, the directly recomputed
both-valid flip range is 0.000% to
8.311%. The detailed file retains every
condition and denominator.

## English draft

Among pairs that were strict-valid in both rendering conditions, decision
flips ranged from 0.0% to
8.3% across model-language-condition cells.
The previously labelled same-environment grid repeat was subsequently found
to use a different Python runtime, so it cannot define a same-environment
noise floor. Its cross-environment both-valid disagreement was at most
0.5%, while changing only the
physical GPU within the repeat environment added no decision disagreement.
We therefore avoid claiming a zero same-environment floor until the baseline
is repeated under the original pinned runtime.
