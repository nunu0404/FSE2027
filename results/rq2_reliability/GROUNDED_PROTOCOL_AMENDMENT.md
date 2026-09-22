# Grounded Protocol Amendment and Readiness Record

## Why an amendment was required

The inherited v3 instructions mixed literature-motivated factors with project-specific values. In particular, the old difficulty cut points, categorical representation-ratio thresholds, and an N-only McNemar MDE were not defensible as established standards. The previous rendering implementation also wrapped already formatted text in a way that could overlap tokens and create false source-line numbers.

This amendment freezes the corrected interpretation before new A/B inference. Existing outputs remain untouched and are not silently mixed with the corrected run.

## Corrections made

1. Replaced threshold difficulty as the primary analysis with continuous absolute z-score difference. Balanced rank tertiles are descriptive only.
2. Changed the baseline wrap from 60 to 80 columns. Wrap 60 remains the narrow stress level.
3. Fixed line numbers on. The 12 cells now form a complete 3 display x 2 font x 2 wrap grid; there is no underpowered or aliased line-number factor.
4. Reimplemented rendering as lex-first, token-preserving visual soft wrap. Layout ablations occur after lexing and blur occurs after rasterization.
5. Renamed blur levels by exact sigma (1, 2, 4 px). Removed unsupported low/medium/high semantics.
6. Restricted primary cross-language interactions to the Dorn-only subset; pooled Java is secondary.
7. Added crossed snippet-clustered uncertainty as primary inference because 3,000 pairs reuse 552 snippets.
8. Removed fixed representation ratio labels (0.5/0.8). Continuous estimates, bootstrap intervals, and permutation nulls replace them.
9. Recomputed exact power. McNemar sensitivity is conditional on the expected discordant fraction.
10. Pinned both model revisions and added prompt, protocol, render-audit, and image provenance checks to the runner.
11. Replaced the incorrect pointer to a paper-reconstructed prompt with the exact Prompt B template used by the existing inference implementation; the runner now fails if the strings drift.

## Rendering evidence

All corrected assets are under `rendered/`; no old image was modified.

| Audit | Result |
|---|---:|
| Source snippets | 552 |
| A-grid images | 6,624 / 6,624 |
| B-perturbation images | 3,312 / 3,312 |
| Missing, blank, or invalid-dimension images | 0 |
| Source-hash mismatches | 0 |
| Visual-row accounting errors | 0 |
| Variable widths within a condition | 0 |
| B baseline exact pixel matches to A baseline | 552 / 552 |
| Blur dimension preservation | 552 / 552 snippets x 3 levels |
| Monotonic gradient-energy reduction across blur levels | 552 / 552 |
| Distinct blur hashes at all levels | 552 / 552 |
| Overall render gate | PASS |

The complete machine-readable result is `audit/GROUNDED_RENDER_AUDIT.json`; all-snippet SSIM is in `audit/blur_ssim_all.csv`; 12 language/length contact sheets are in `audit/`.

## Pair and perturbation coverage

The fixed pair list has 1,000 pairs each for Java, Python, and CUDA. No-indent changes at least one side in 1,000/1,000 Java, 996/1,000 Python, and 998/1,000 CUDA pairs. No-blank-lines changes at least one side in 875/1,000 Java, 1,000/1,000 Python, and 978/1,000 CUDA pairs. All-pair estimates remain primary, with cue-present-only sensitivity tables required.

## Power record

At two-sided alpha .05 and 80% exact-binomial power, the minimum detectable deviation from 50% is approximately 14.52 percentage points at N=100, 8.23 at N=300, 6.37 at N=500, and 4.48 at N=1,000. For McNemar at N=1,000, the marginal MDE varies from about 2.91 to 6.37 points as the discordant fraction varies from 0.1 to 0.5. Exact tables are in `audit/power_*.csv`.

## Execution gates

| Gate | Requirement | Current state |
|---|---|---|
| G-render | Complete hashes, dimensions, row accounting, token provenance, blur checks | PASS |
| G-OCR | RapidOCR and EasyOCR characterize baseline/sigma 1/2/4 on 30 stratified snippets | PASS: 240 / 240 tasks, no runtime error |
| G-dry | Both models x A/B complete path/manifests validate without loading GPU | PASS: 4 / 4 model-experiment plans |
| GA0 | 100 calls: no parse failure and A/B logit argmax equals parsed verdict | Required immediately before full inference |
| GC1 | Author approves stratified 20% of generated RQ3 variants per language | BLOCKS C inference |

GA1's old requirement of exact equality with earlier default outputs is retired for this corrected renderer: the pixels and baseline width intentionally changed. Historical outputs may be shown descriptively, but they are not a valid bit-for-bit regression oracle for the new images.

The OCR manipulation audit confirms ordered degradation. Across languages, RapidOCR median CER rises from 0.1168 at sigma 1 to 0.3730 at sigma 2 and 1.0000 at sigma 4; median token F1 falls from 0.8746 to 0.6022 and 0.0000. EasyOCR median CER is 0.2055, 0.6691, and 1.0000; median token F1 is 0.8353, 0.1461, and 0.0000. Sigma 4 reaches an OCR floor (RapidOCR 0/30 nonempty, EasyOCR 5/30 nonempty), so it is retained explicitly as an extreme stress condition. Blur is categorical in the primary model; a monotonic trend is secondary. These scores compare each engine's blurred OCR to its own baseline OCR, not to source text.

## Frozen call plan

Per model, A uses 3,000 pairs x 12 conditions x 2 orders = 72,000 calls. B uses 3,000 x 6 x 2 = 36,000 calls. Across two models the full A+B plan is 216,000 calls. Both use image-only Prompt B, separate A/B images, BF16 greedy decoding, max 24 generated tokens, seed 42, and verdict-logit capture.

`code/run_grounded_vlm.py --dry-run` validates the complete plan and writes immutable manifests. A pilot must use a distinct `--output-tag`; pilot rows must never be resumed into a full output.

## Readiness boundary

A and B rendering/data preparation is complete. Full inference is intentionally not started by this preparation task. RQ3 variant generation is a separate author-gated stage: 100 bases yield 600 unique six-contrast pairs per language, not 1,000. D can begin after the A image/model hook gate, and does not require text generation.
