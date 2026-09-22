#!/usr/bin/env python3
"""Reproducible exact-binomial power tables used by the grounded protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/grounded_protocol_3lang_20260721"
ALPHA = 0.05


def rejection_region(n: int) -> np.ndarray:
    k = np.arange(n + 1)
    lower = binom.cdf(k, n, 0.5)
    upper = binom.sf(k - 1, n, 0.5)
    return 2 * np.minimum(lower, upper) <= ALPHA


def power(n: int, p: float) -> float:
    reject = rejection_region(n)
    return float(binom.pmf(np.arange(n + 1), n, p)[reject].sum())


def minimum_delta(n: int, transform) -> float:
    for delta in np.linspace(0, 0.5, 5001):
        trials, p = transform(float(delta))
        if trials > 0 and p <= 1 and power(trials, p) >= 0.80:
            return float(delta)
    return float("nan")


def main() -> int:
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    one_sample = []
    for n in [100, 150, 200, 300, 500, 1000]:
        delta = minimum_delta(n, lambda d, n=n: (n, 0.5 + d))
        one_sample.append({"n": n, "alpha_two_sided": ALPHA, "target_power": 0.8, "mde_percentage_points": 100 * delta})
    paired = []
    for n in [500, 1000]:
        for discordant_fraction in [0.10, 0.20, 0.30, 0.50]:
            m = int(round(n * discordant_fraction))
            delta = minimum_delta(m, lambda d, m=m, q=discordant_fraction: (m, 0.5 + d / (2 * q)))
            paired.append({
                "total_pairs": n, "assumed_discordant_fraction": discordant_fraction,
                "discordant_pairs": m, "alpha_two_sided": ALPHA, "target_power": 0.8,
                "mde_marginal_percentage_points": 100 * delta,
            })
    pd.DataFrame(one_sample).to_csv(OUT / "audit/power_one_sample_binomial.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "audit/power_mcnemar_conditional.csv", index=False)
    print(pd.DataFrame(one_sample).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
