# E8 Restructuring inputs

## Contrast inventory and multiplicity

All 66 rendering-grid and 30 perturbation contrasts are retained in
`E8_contrast_inventory.csv`, including effect size, cluster-robust 95% CI,
exact McNemar p, Holm-adjusted p, and direction. Holm adjustment was performed
within six model-language families (11 grid or 5 perturbation comparisons per
family), not as one 96-test family.

## RQ3 compact view

The six direct contrasts reduce to four conceptual rows without discarding either
model: aligned semantics+appearance, semantic-only under ugly rendering,
semantic/appearance conflict, and appearance-only. In the appearance-only rows,
Qwen selects golden over ugly_gold effectively on 56.0% and beautiful_trash over
ugly_trash on 8.3% (25/300 valid); InternVL gives 57.3% and 60.3%.

## Rendering decomposition

The exact product decomposition has maximum numerical residual
9.19e-17. The valid-coverage /
consistency component is larger in absolute value for
70/96
condition cells; valid-only accuracy dominates in
26/96. Thus the
consistency hypothesis is evaluated cell by cell rather than asserted globally.

## Label robustness

The easy slice contains 1000 pairs per
model-language cell. The Java within-dataset slice contains
1004 pairs, after excluding 1,996
cross-dataset pairs. `E8_label_robustness.csv` reports every VLM and all nine ML
baselines so model ranks and the baseline comparison can be audited directly.
