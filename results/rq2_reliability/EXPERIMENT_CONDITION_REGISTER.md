# Experiment Condition Register

Protocol: `grounded-3lang-20260721-v2`  
Frozen on: 2026-07-21  
Scope: A rendering grid, B visual perturbations, C RQ3 variants, D representation analysis, and shared VLM inference/evaluation.

## 1. Evidence labels

Not every numeric setting can or should be presented as a universal literature standard. This register uses five labels.

| Label | Meaning |
|---|---|
| external | A prior study or public standard directly motivates the factor or procedure. |
| mathematical | The setting follows from the estimand, test, or power calculation. |
| replication | The value is frozen to reproduce the project's earlier experiment. |
| calibrated | The value is an explicit manipulation level checked by a pilot or manipulation audit. |
| reproducibility | An otherwise arbitrary constant is frozen to make reruns deterministic. |

Calling a calibrated or reproducibility value a literature standard is prohibited.

## 2. Data and pair construction

| Condition | Frozen decision | Basis and rationale |
|---|---|---|
| Human reference | Mean human readability rating supplied by Buse, Dorn, or Scalabrino | **external**. These benchmarks operationalize code readability through human judgments. Buse and Weimer learn from human ratings; Dorn extends readability features to additional languages and visual/spatial properties. |
| Languages | Java, Python, CUDA | **external + scope control**. Dorn explicitly motivates a language-general readability model. The three-language extension tests whether Java-only findings generalize. |
| Pair count | 1,000 fixed pairs per language | **calibrated/design**. Equal allocation prevents a language from dominating descriptive summaries. It is not a literature constant. |
| Gold normalization | `z=(score-dataset mean)/(dataset SD)` before cross-dataset Java pairing | **mathematical**. It places benchmark-specific rating scales on a common standardized scale. Raw ratings remain archived. |
| Primary difficulty | Continuous `abs(z_i-z_j)` | **mathematical**. It retains all observed information and avoids arbitrary cut points. |
| Difficulty labels | Within-language rank tertiles: near/middle/far | **descriptive**. Tertiles provide balanced tables only. They are not claims that 0.2, 0.5, or 1.0 are validated readability thresholds. |
| Legacy 0.2/0.5/1.0 thresholds | Reproducibility-only column | **replication**. The old pool was sampled with these project-defined thresholds. It must not support a threshold-based scientific claim. |
| Shared snippets | Retained and explicitly modeled | Pair observations are not independent when the same snippet appears repeatedly. Primary uncertainty uses crossed clustering by `snippet_i` and `snippet_j`; naive pair tests are sensitivity analyses. |
| Language interaction population | Dorn-only Java/Python/CUDA | **design correction**. Pooled Java additionally contains Buse and Scalabrino, so a pooled three-language interaction confounds language and benchmark. Pooled Java is secondary. |

Primary sources: [Buse and Weimer, Learning a Metric for Code Readability](https://web.eecs.umich.edu/~weimerw/p/weimer-issta2008-readability.pdf); [Dorn, A General Software Readability Model](https://web.eecs.umich.edu/~weimerw/students/dorn-mcs-paper.pdf).

## 3. Rendering grid A

| Factor | Frozen levels | Basis and permitted interpretation |
|---|---|---|
| Display mode | monochrome light, Friendly light, Monokai dark | **external + calibrated**. Syntax coloring and visual presentation can affect code reading, but these named styles jointly change palette/background. Analyze them as display modes, not as a pure causal color effect. |
| Font size | 20 px, 24 px | **calibrated**. These are two legible rasterization levels for the fixed DejaVu Sans Mono font. They are not literature-standard font sizes. The renderer audit confirms nonblank, unclipped output. |
| Wrap column | 60, 80 | **external + calibrated**. 80 is close to established style guidance (PEP 8 uses 79); 60 is a preregistered narrow stress condition. Neither is universal across languages. |
| Line numbers | On and fixed | **replication/control**. The corrected 12-cell design is 3 display modes x 2 font sizes x 2 wrap widths. Line numbers are no longer varied, so no line-number effect will be claimed. |
| Baseline | Monokai, 20 px, wrap 80, line numbers on | **external + replication**. It preserves the prior display/font while replacing the unusually narrow wrap-60 default with the conventional 80-column level. |
| Wrapping | Lex source first, then visual soft-wrap token runs | **measurement validity**. This preserves the token stream and syntax colors. Continuation rows have a blank gutter and do not masquerade as source lines. |
| Dimensions | Fixed within each rendering condition | **measurement validity**. Stable widths prevent content-dependent horizontal scaling from becoming an uncontrolled cue. Height follows required visual rows. |

Relevant sources: [PEP 8 maximum line length](https://peps.python.org/pep-0008/); [Google Java Style, column limit and whitespace](https://google.github.io/styleguide/javaguide.html); [Sarkar, The Impact of Syntax Colouring on Program Comprehension](https://www.ppig.org/files/2015-PPIG-26th-Sarkar1.pdf).

## 4. Visual perturbations B

| Condition | Frozen implementation | Basis and correction |
|---|---|---|
| Baseline | Corrected A baseline pixels | Makes A/B directly comparable. Pixel equality is an audit gate. |
| No indentation | Remove only rendered leading horizontal offset after lexing | **external**. Indentation is a recognized readability/structure cue. It does not alter source or tokenization. For Python, this visually suppresses a syntax-bearing cue and is not called presentation-only. |
| No blank lines | Suppress rendered blank logical rows after lexing | **external**. Blank lines are features in established readability models and style guidance. Analyze cue-present pairs separately because some snippets contain no removable blank line. |
| Gaussian blur | Pillow Gaussian blur with radius/sigma 1, 2, and 4 pixels | **external + calibrated**. Gaussian blur is a standard corruption family in ImageNet-C. Pillow defines GaussianBlur radius as standard deviation. The exact 1/2/4 subset is this study's calibrated severity grid, not ImageNet-C's complete severity scale. Name levels by sigma, not low/medium/high. |
| Application | Same perturbation on both snippets; blur only after rasterization | Controls the pairwise treatment and prevents the blur operation from changing source layout or tokenization. |
| Manipulation checks | Pixel hash, dimensions, gradient energy, SSIM, and two-engine OCR | **calibrated**. Pixel/gradient checks run on all 552 snippets. OCR uses a deterministic 10-length-quantile sample per language and compares each engine's blurred output with its own clean OCR. |

Indentation evidence: [Hanenberg et al., controlled experiment on indentation and comprehensibility](https://link.springer.com/article/10.1007/s10664-024-10531-y). Corruption evidence: [Hendrycks and Dietterich, Benchmarking Neural Network Robustness to Common Corruptions](https://openreview.net/pdf?id=HJz6tiCqYm); [official ImageNet-C implementation](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py); [Pillow GaussianBlur definition](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html).

Hypothesis corrections:

- H-B1 is tested as a language x no-indent interaction on the Dorn-only subset. Python is expected to be more affected, but this is an empirical directional hypothesis.
- H-B2 is tested primarily as a language x categorical blur-condition interaction because OCR can reach a floor at sigma 4; an ordinal severity trend is secondary. A nonsignificant interaction is not evidence of equivalence. Equivalence requires preregistered bounds, which are not available here.
- Primary B tables include all pairs; cue-present-only tables are required sensitivity analyses for no-indent and no-blank-lines.

## 5. RQ3 variants C

| Condition | Frozen decision | Basis and correction |
|---|---|---|
| Base sample | 100 snippets per language, score-stratified | **user-specified + calibrated**. This is not a literature standard. Exact one-sample power at alpha .05 and 80% power detects approximately 14.5 percentage points from 50%. |
| Four cells | golden, ugly_gold, beautiful_trash, ugly_trash | **factorial design**: semantic integrity x visual quality. |
| Six contrasts | Every unordered pair among four cells | **mathematical** completeness; four cells produce `4 choose 2 = 6` contrasts, or 600 unique base-contrast pairs per language. |
| Trash mutation | Syntactically renderable but intentionally semantically incoherent; mutation type and diff recorded | **external + user requirement**. Mutation testing motivates controlled operator-based semantic faults. “Unintelligible” is verified by author review, not assumed from an automatic transform. |
| Ugly transform | Render-only indentation/blank-line/alignment degradation | Separates presentation from stored source. Python source remains parsable because the degradation is applied at render time. |
| Author gate | Stratified random 20% per language before inference | **quality-control rule**, not an inferential standard. C inference remains blocked until approval. |
| Outcome wording | Preference rate, never accuracy | There is no independent human gold label for generated variants. The primary conflict is ugly_gold vs beautiful_trash. |
| C modalities | image_only and text_plus_image, each run separately | **requested input ablation**. Both use the same two labeled PNGs; text_plus_image additionally includes the exact source associated with each displayed variant. Results are never pooled across modalities. |

Mutation-testing background: [Jia and Harman, An Analysis and Survey of the Development of Mutation Testing](https://doi.org/10.1109/TSE.2010.62).

## 6. Models and inference

| Condition | Frozen value | Basis |
|---|---|---|
| Models | Qwen2.5-VL-7B-Instruct revision `cc5948...`; InternVL3-8B revision `853e3a...` | **external + replication**. Two open-weight VLM families with image input support; exact local revisions prevent model drift. |
| Modality | image-only, two separately labeled images | **construct validity**. The experiment asks whether the visible rendering drives the judgment. Text-plus-image is a separate input ablation and must not be mixed into A/B. |
| Prompt | Existing code-level Prompt B, frozen in `config/prompt_B_template.txt` | **replication**. Reuses the actually executed three-criterion rubric and exact `FINAL_VERDICT` contract; no prompt tuning on outcomes. The previously referenced paper-reconstructed file was different and has been removed from this protocol. |
| Pair direction | AB and BA for every condition | **external**. Pairwise LLM judges exhibit position bias; swapped presentation measures order consistency. |
| Strict swap | Valid only if both orders select the same underlying snippet | **measurement rule**. Disagreement is retained as a reliability failure, not silently resolved. |
| Decoding | Greedy, `do_sample=false`, temperature 0 | **replication/control**. Removes sampling as a source of order flips. Hardware/kernel nondeterminism is still audited. |
| `top_p` | 1.0, inactive | **reproducibility**. It has no effect when sampling is disabled and is not an experimental factor. |
| `max_new_tokens` | 24 | **replication + output-contract headroom**. Prior complete outputs used 7 generated tokens, so 24 is sufficient for the required one-line verdict. It is not a model-quality setting here. |
| Seed | 42 | **reproducibility**. Arbitrary fixed seed; no scientific meaning is claimed. |
| Precision | BF16 | **operational**. Supported inference precision on the recorded GPUs; exact precision is frozen because numerical changes can alter close logits. |
| Logits | Capture A/B logits at the emitted verdict token | Enables position/content decomposition. A 100-call gate requires parse success and parsed-verdict/argmax agreement of 100% before a full run. |

Model sources: [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923); [InternVL3 technical report](https://arxiv.org/abs/2504.10479). Position-bias sources: [Wang et al., Large Language Models are not Fair Evaluators, ACL 2024](https://aclanthology.org/2024.acl-long.511/); [Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685); [Shi et al., systematic position-bias study](https://arxiv.org/abs/2406.07791).

## 7. Outcomes and inference

| Item | Frozen definition and rationale |
|---|---|
| Effective accuracy | Correct strict-valid pairs divided by all pairs. Primary ability-plus-reliability endpoint. Invalid swaps count as failures. |
| Strict-swap error | Fraction whose AB/BA calls do not choose the same underlying snippet. Primary reliability endpoint. |
| Valid accuracy | Accuracy among strict-valid pairs. Secondary because conditioning on validity can select easier pairs. |
| Spearman valid-only | Spearman correlation between valid pair sign and human z difference. Secondary and reported with valid coverage. |
| Confidence intervals | Wilson 95% for proportions; bootstrap intervals for continuous/derived effects. |
| Primary paired inference | Regression/estimating equations with crossed snippet clustering. Condition and language interactions are tested directly. |
| Sensitivity tests | Exact McNemar for two matched conditions and Cochran Q for complete matched condition sets. Their naive pair-independence limitation is stated. |
| Multiplicity | Holm family-wise correction within each prespecified model-language-endpoint family. |
| Power | Exact binomial/McNemar tables are stored in `audit/`. McNemar MDE is indexed by discordant probability; it cannot be inferred from total N alone. |

Statistical sources: [McNemar power depends on discordant pairs](https://pmc.ncbi.nlm.nih.gov/articles/PMC3898531/); [Cameron, Gelbach, and Miller, multi-way cluster-robust inference](https://www.nber.org/papers/t0327); [Wilson interval review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2706447/).

## 8. Representation analysis D

- Extract the last vision-encoder patch embeddings and mean-pool them. A CLS-like vector is secondary only if the model architecture explicitly defines one; do not invent a universal CLS token.
- Report intra-snippet cosine distance, inter-snippet distance, their continuous ratio, and same-snippet retrieval.
- Remove the old `R <= 0.5` and `R >= 0.8` labels. They have no validated domain threshold. Use bootstrap confidence intervals and a permutation null.
- Reuse cached embeddings for all analyses; generation is not involved.

## 9. Non-negotiable reporting constraints

1. Do not label project-chosen numeric levels as field standards.
2. Do not infer equivalence from a nonsignificant difference.
3. Do not call RQ3 preference an accuracy without independent human gold.
4. Do not pool heterogeneous Java benchmarks into the primary language interaction.
5. Report coverage with every valid-only metric.
6. Record model revision, prompt/config/image hashes, software versions, GPU, precision, and resume state in every run manifest.
