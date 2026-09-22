# Paper Insert Draft (English)

## Analysis A: compact table caption

**Table X. Cross-rendering preference flips for fixed code pairs.** Each cell summarizes all 66 unordered pairs of the 12 rendering conditions within one model-language run. AB and BA columns hold presentation order fixed and report snippet-identity preference changes over 1,000 pairs per condition pair; both-valid columns restrict the denominator to pairs that are strict-swap valid under both conditions. The order baseline is the median within-condition strict-swap error across the 12 renderings. Full condition-pair estimates, exact 95% intervals, denominators, and Holm-adjusted tests are in the replication package.

## Analysis A: body text

Holding presentation order fixed, changing only the rendering reversed the selected snippet for a median 9.2%-25.4% of pairs across model-language cells, with individual condition pairs reaching 30.3%. Requiring strict-swap validity under both renderings reduced the median reversal rate to 0.6%-7.1%, indicating that many apparent rendering reversals interact with position instability. The within-rendering order baseline was substantially larger (38.6%-69.3% median strict-swap error), so order was the dominant disturbance in this run. We call these events *preference flips*, not errors, because this analysis measures whether a preference changes rather than whether either choice is correct.

## Analysis B: figure caption

**Figure X. Verdict content-margin calibration in the five-model battery.** Pairs are grouped by within-model deciles of `|c|`, where `c=(m_AB-m_BA)/2`; exact `c=0` ties are conservatively assigned to D1. Lines show strict-valid accuracy, effective accuracy (invalid swaps count as failures), and logit-debiased accuracy over 9,000 pairs per model. Full language-specific deciles, non-tie sensitivity estimates, 10,000-replicate two-way snippet-cluster bootstrap intervals, and snippet-disjoint isotonic ECE/AUROC are provided in the replication package.

## Analysis B: body text

Effective accuracy increased monotonically with `|c|` for all five models (decile Spearman rho 0.73-1.00), confirming that the stored verdict margin is informative about strict-swap reliability. This pattern did not imply universal content-correctness calibration: valid-only and debiased accuracy rose strongly for Qwen and Gemma, more weakly for Ministral, but were flat or inverse for InternVL and Phi. On snippet-disjoint test pairs, isotonic ECE ranged from 2.8% to 6.8%, while AUROC ranged from 0.52 to 0.63. Verdict logits should therefore be retained as a diagnostic of coverage and position instability, but `|c|` should not be treated as a model-agnostic probability of correctness.

## Analysis C: supplementary figure caption

**Figure Sx. Joint position and content components of verdict margins.** Hexagonal density plots summarize all 9,000 battery pairs per model in (`|c|`, `|b|`) space; axes are clipped at each model's 99.5th percentile for display only, while all observations remain in the data and statistics. The white diagonal marks `|c|=|b|`. Away from exact boundaries, all 39,423 observations obeyed the algebraic partition `|c|>|b|` iff strict-swap valid (zero violations). The 5,577 exact boundaries are reported separately because parsed validity is not determined by the strict inequality.

## Analysis D: table column description

`First-position rate` is the fraction of all AB/BA calls in which the model selected the first displayed snippet. `95% CI` is a percentile interval from 10,000 pair-cluster bootstrap replicates, preserving the dependence between the two orders of each pair. Each language row contains 3,000 pairs and 6,000 calls.
