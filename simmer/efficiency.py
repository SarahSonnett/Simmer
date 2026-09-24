"""Detection efficiency function eta(D) = P(detected | diameter), with Wilson
confidence intervals.

This is the core scientific product of a Simmer run: the fraction of synthetic
objects of each diameter that the survey detects. Applied to the real observed
counts it debiases the underlying size-frequency distribution,
``N_true(D) = N_obs(D) / eta(D)``.

Each synthetic object of size D is an independent Bernoulli detection trial with
success probability eta(D), so the count detected in a diameter bin is
Binomial(N_bin, eta) and the uncertainty on eta is the **Wilson score interval**
-- which stays inside [0, 1] and behaves correctly as eta -> 0 or 1, unlike the
normal approximation. The error bar is set by ``N_bin`` alone: multiple runs of a
population add nothing that a single larger run does not, since they only
increase the total object count.

To measure eta(D) cleanly across all sizes, sample the input population roughly
uniform in log D (the steep real SFD leaves the large-D bins sparse -- though
those are the easy eta ~ 1 bins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_interval(k, n, z: float = 1.0):
    """Wilson score interval for a binomial proportion ``k/n`` at ``z`` sigma.

    ``z=1`` gives the 1-sigma (68.3%) interval; ``z=1.96`` the 95% interval.
    Returns ``(p, lo, hi)`` with ``p = k/n`` and the interval clipped to [0, 1].
    """
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def efficiency_function(diam_km, detected, bins=None, n_bins: int = 20,
                        z: float = 1.0) -> pd.DataFrame:
    """Detection efficiency ``eta`` vs diameter with Wilson intervals.

    ``diam_km`` and ``detected`` are aligned per-object arrays. ``bins`` overrides
    the default log-spaced edges (``n_bins`` of them). Returns one row per
    non-empty bin: edges, geometric-mean centre, input/detected counts, ``eta``
    and its z-sigma Wilson interval ``[eta_lo, eta_hi]``.
    """
    diam = np.asarray(diam_km, float)
    det = np.asarray(detected).astype(bool)
    if bins is None:
        pos = diam[diam > 0]
        d_min, d_max = pos.min(), diam.max()
        if d_min == d_max:                       # all one size -> a single bin
            d_min, d_max = d_min * 0.999, d_max * 1.001
        bins = np.geomspace(d_min, d_max, n_bins + 1)
    bins = np.asarray(bins, float)
    idx = np.digitize(diam, bins) - 1

    rows = []
    for i in range(len(bins) - 1):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            continue
        k = int(det[m].sum())
        p, lo, hi = wilson_interval(k, n, z)
        rows.append(dict(d_lo=bins[i], d_hi=bins[i + 1],
                         d_mid=float(np.sqrt(bins[i] * bins[i + 1])),
                         n_input=n, n_detected=k,
                         eta=float(p), eta_lo=float(lo), eta_hi=float(hi)))
    return pd.DataFrame(rows, columns=["d_lo", "d_hi", "d_mid", "n_input",
                                       "n_detected", "eta", "eta_lo", "eta_hi"])
