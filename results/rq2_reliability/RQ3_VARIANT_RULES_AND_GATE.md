# RQ3 Variant Rules and Author Gate

Status: generation and automatic audit passed; variant inference blocked pending author GC1 approval.

## Base allocation

Use the frozen 300-row file `rq3/data/rq3_grounded_bases_seed42.csv`: 100 unique bases per language, allocated 34/33/33 across low/mid/high human-score strata. A base produces four cells and all six unordered contrasts, therefore 600 unique contrast pairs per language. AB/BA, model, and modality multiply calls, not unique pairs.

## Two-factor cells

| Cell | Semantic integrity | Visual quality |
|---|---|---|
| golden | preserved | clean |
| ugly_gold | preserved | degraded |
| beautiful_trash | destroyed | clean |
| ugly_trash | destroyed | degraded |

The clean render uses the corrected baseline condition. The ugly transform is applied only at render time after lexing. `golden` and `ugly_gold` must share exactly the same source hash; `beautiful_trash` and `ugly_trash` must also share a source hash. This makes the 2 x 2 manipulation auditable.

## Semantic-destruction rule

The newer study requirement is stronger than a single subtle mutation: a human reader should not be able to recover a coherent local intent from a trash variant. A trash generator must therefore apply a recorded, language-aware set of syntax-preserving destructive operators rather than one unlogged string edit.

Required operator families:

1. Mask all semantic identifier cues with deterministic opaque names, then fragment alternating occurrences so declarations and references no longer expose a coherent data flow where syntax permits.
2. Reverse, permute, or replace relational/arithmetic/logical operators.
3. Replace boundary literals and branch constants.
4. Permute independent-looking statement payloads or call arguments while preserving parseability.
5. Remove semantic naming cues by consistent neutral renaming only as a secondary operator; renaming alone is not enough to qualify as trash.

Each item records operator, changed source spans, original/replacement text, parser result, and source hashes. Because the benchmark contains code fragments rather than guaranteed standalone compilation units, Java, Python, and CUDA are all validated with pinned tree-sitter grammars. A trash fragment may not contain more parser error/missing nodes than its own gold fragment. This is a fragment-level syntax gate, not a compilation claim.

## Automatic gates

1. All four variants render as nonblank, unclipped images.
2. Golden and ugly_gold token/source hashes are identical.
3. Beautiful_trash and ugly_trash token/source hashes are identical.
4. Trash source has no more parser error/missing nodes than its corresponding gold fragment and differs at every recorded mutation span.
5. At least two recorded semantic-destruction mechanisms are present per trash item. Opaque consistent renaming alone never qualifies; alternating identity fragmentation is required so declaration/reference coherence is also removed.
6. Operator-family frequencies are reported by language so one mutation does not dominate.
7. Clean/ugly rendering changes pixels but does not change the associated source hash.

## Human gate GC1

A seed-frozen, score-stratified 20-item sample per language (20%) must be reviewed before any C inference. Reviewers answer separately:

- Is golden locally coherent and faithful to the selected source?
- Is trash syntactically code-like but semantically incoherent enough that local intent cannot reasonably be recovered?
- Is the clean image legible and visually organized?
- Is the ugly image visibly degraded without overlap, clipping, or raster corruption?
- Do the paired clean/ugly images contain the same source tokens?

Any failed item is repaired under the same documented rules; the complete language batch is re-audited, and the 20% sample is reviewed again. No preference result may be inspected before GC1 passes.

## Statistical wording

There is no human gold for generated variants. Report `semantic-preserving preference rate` for contrasts crossing semantic integrity and `clean-presentation preference rate` when semantics match. Use Wilson 95% intervals and exact binomial sensitivity tests versus 50%. At N=100 bases, effects smaller than roughly 14.5 percentage points from 50% are below the precomputed 80%-power MDE and must be called inconclusive, not absent.
