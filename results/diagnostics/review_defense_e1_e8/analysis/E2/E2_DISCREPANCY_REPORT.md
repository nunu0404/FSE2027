# E2 Run-discrepancy diagnosis

## Determination

The manuscript values are from separate runs and must not be merged. `battery`,
`clean_default`, `grid`, `rq3`, and `deploy` are the fixed table labels.

The 18.9-point InternVL/Java strict-swap discrepancy is not caused by pair
sampling: battery and clean_default contain the same 3,000 Java pair IDs
(intersection 3000/3,000). The runs use different
baseline renderings. clean_default uses the corrected default PNGs, whereas
battery uses corrected grounded PNGs rendered with Monokai Dark, font size 20,
wrap width 80, and line numbers on. The battery run also captures verdict logits
under a pinned checkpoint/reproducibility protocol that the earlier clean run
did not record. The state (selected snippet or invalid) changes on
1205/3,000 pairs
(40.17%). Among pairs valid in both runs, the
selected snippet changes on 89/
1081 (8.23%).
Thus the discrepancy is a cross-protocol rendering/run effect, not a sampling
effect.

The completed E4 control resolves the environment
hypothesis. Same-environment pair-state disagreement is
5.80%-
8.70%; changing only the physical GPU UUID
produces 5.80%-
8.70%. The six E4-B cell results are
identical to their E4-A counterparts, including call verdicts, pair states, and
stored logits. Thus the physical GPU change adds no measured disagreement and
cannot explain the 18.9-point InternVL discrepancy. The approximately 7%
same-environment noise floor and the rendering/protocol difference must be
reported separately.

The earlier clean_default run did not store verdict logits. Consequently, a
two-sided cross-run verdict-margin median is not measurable. The agreement CSV
reports the median using only logit-bearing sides and marks this limitation
explicitly; it must not be presented as the old Section 3.6 two-run statistic.

## Manuscript insertion draft

Results in the battery, rendering-grid, and deployment analyses come from
separate runs and are not pooled. We label them `battery`, `grid`, and `deploy`
throughout; the earlier Java packaging audit is labeled `clean_default`.
Although `battery` and `clean_default` use the same 3,000 Java pairs, they use
different corrected renderings and inference records. For InternVL, the
strict-swap error is 35.94% in `battery` and 54.80% in `clean_default`; paired
inspection confirms that this difference is not due to pair sampling. We
therefore interpret cross-run differences as protocol sensitivity rather than
controlled treatment effects and report the controlled same-environment and
single-hardware-change repeats separately.
