# F6 reference-direction placement verification

- Source: `data/rq1_pairs_9000.csv`
- Source SHA-256: `e41fc9896dc927b98d35eb57e44a4f7139d2013353d061fbac854470d3733183`
- Population: 9 language-difficulty cells, exactly 1,000 pairs per cell
- Reference direction: the endpoint with the larger within-benchmark standardized mean rating
- First position: `snippet_i` in the frozen pair metadata
- Check: `human_preference` disagreed with the score-derived reference direction in 0/9,000 pairs
- Check: score ties were present in 0/9,000 pairs
- Calculation: `100 * count(human_preference == snippet_i) / 1,000`, separately per cell
- Rounding: one decimal place using round-half-up. Because every cell has denominator 1,000, each percentage is already an exact multiple of 0.1 percentage points; no halfway case occurs.

The observed cell-wise range is **48.2% to 51.5%**. Therefore, the manuscript text `48.2-51.8%` is incorrect and must be changed to `48.2-51.5%`.

The previously reported 51.5% uses the requested cell-wise definition and is correct. A value of 51.8% cannot be reproduced from this frozen battery pair list under either the metadata `human_preference` direction or an independently reconstructed score direction; it therefore came from a different pair asset or aggregation, not from rounding or truncation of these nine cells.
