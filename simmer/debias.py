"""Debias an observed size-frequency distribution with the efficiency function.

Step 2 of the science chain. Given the detection efficiency ``eta(D)`` measured
by :mod:`simmer.efficiency` and the *real* survey's observed counts per diameter
bin, recover the underlying (debiased) size-frequency distribution

    N_true(D) = N_obs(D) / eta(D).

Two independent uncertainties propagate into ``N_true``:

* the **observed counts** are a Poisson process -- for a bin containing ``k``
  detections the count uncertainty is the (asymmetric) Poisson confidence
  interval, which matters in the sparse large-D bins where ``k`` is small
  (:func:`poisson_interval`, Gehrels 1986);
* the **efficiency** carries the Wilson interval ``[eta_lo, eta_hi]`` from the
  simulation (:mod:`simmer.efficiency`).

Because ``N_true = N_obs / eta`` is exact in ``eta``, the efficiency term is
propagated by evaluating the inversion at the Wilson endpoints rather than
linearising; the two one-sided deviations are then combined in quadrature. Where
``eta`` is small the correction ``1/eta`` blows up and becomes unreliable, so
bins with ``eta < eta_min`` (or ``eta_lo = 0``) are flagged ``reliable = False``
and their upper bound is left unbounded rather than quietly reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def poisson_interval(k, z: float = 1.0):
    """Approximate Poisson confidence limits on a count ``k`` at ``z`` sigma.

    Wilson-Hilferty cube-root approximation to the chi-square (Poisson) limits
    (Gehrels 1986, ApJ 303, 336); ``z=1`` is the ~68% (1-sigma) interval. Unlike
    ``sqrt(k)`` the interval is asymmetric and well behaved for small ``k`` (for
    ``k=0`` the lower limit is 0 and the upper ~1.84 at 1 sigma). Returns
    ``(k, lo, hi)``.
    """
    k = np.asarray(k, float)
    hi = (k + 1.0) * (1.0 - 1.0 / (9.0 * (k + 1.0))
                      + z / (3.0 * np.sqrt(k + 1.0))) ** 3
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = k * (1.0 - 1.0 / (9.0 * k) - z / (3.0 * np.sqrt(k))) ** 3
    lo = np.clip(np.where(k > 0, lo, 0.0), 0.0, None)
    return k, lo, hi


def bin_observed(diam_obs, eff: pd.DataFrame) -> np.ndarray:
    """Histogram observed diameters onto the efficiency table's bin edges.

    ``eff`` is a table from :func:`simmer.efficiency.efficiency_function`. Its
    rows must be contiguous in diameter (they are, unless empty simulation bins
    were dropped); pass matched counts directly to :func:`debias` otherwise.
    """
    d_lo = eff["d_lo"].to_numpy(float)
    d_hi = eff["d_hi"].to_numpy(float)
    if not np.allclose(d_lo[1:], d_hi[:-1]):
        raise ValueError("efficiency bins are not contiguous (empty bins were "
                         "dropped); bin the observed counts on the same edges "
                         "and pass them to debias() directly")
    edges = np.append(d_lo, d_hi[-1])
    return np.histogram(np.asarray(diam_obs, float), bins=edges)[0]


def debias(eff: pd.DataFrame, n_observed, eta_min: float = 0.05,
           z: float = 1.0) -> pd.DataFrame:
    """Recover ``N_true(D) = N_obs(D) / eta(D)`` with propagated uncertainties.

    ``eff`` is the efficiency table from
    :func:`simmer.efficiency.efficiency_function`; ``n_observed`` is the real
    survey's detected count per bin, aligned to ``eff``'s rows (see
    :func:`bin_observed`). ``eta_min`` flags bins whose efficiency is too small
    to invert reliably. Returns one row per bin with the diameter edges/centre,
    the observed counts and their Poisson interval, the efficiency and its Wilson
    interval, and ``N_true`` with its ``[n_true_lo, n_true_hi]`` band plus a
    ``reliable`` flag.
    """
    eff = eff.reset_index(drop=True)
    n_obs = np.asarray(n_observed, float)
    if n_obs.shape[0] != len(eff):
        raise ValueError(f"n_observed has {n_obs.shape[0]} bins but eff has "
                         f"{len(eff)}; align them (see bin_observed)")

    eta = eff["eta"].to_numpy(float)
    eta_lo = eff["eta_lo"].to_numpy(float)
    eta_hi = eff["eta_hi"].to_numpy(float)
    _, nobs_lo, nobs_hi = poisson_interval(n_obs, z)

    reliable = (eta >= eta_min) & (eta_lo > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_true = np.where(eta > 0, n_obs / eta, np.nan)
        # count term: Poisson spread carried through the 1/eta factor
        d_obs_hi = (nobs_hi - n_obs) / eta
        d_obs_lo = (n_obs - nobs_lo) / eta
        # efficiency term: exact inversion at the Wilson endpoints
        d_eta_hi = n_obs * (1.0 / eta_lo - 1.0 / eta)   # -> inf if eta_lo = 0
        d_eta_lo = n_obs * (1.0 / eta - 1.0 / eta_hi)
        n_true_hi = n_true + np.sqrt(d_obs_hi ** 2 + d_eta_hi ** 2)
        n_true_lo = n_true - np.sqrt(d_obs_lo ** 2 + d_eta_lo ** 2)

    n_true_lo = np.clip(n_true_lo, 0.0, None)

    return pd.DataFrame({
        "d_lo": eff["d_lo"], "d_hi": eff["d_hi"], "d_mid": eff["d_mid"],
        "n_obs": n_obs.astype(int), "n_obs_lo": nobs_lo, "n_obs_hi": nobs_hi,
        "eta": eta, "eta_lo": eta_lo, "eta_hi": eta_hi,
        "n_true": n_true, "n_true_lo": n_true_lo, "n_true_hi": n_true_hi,
        "reliable": reliable,
    })
