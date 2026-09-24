"""Fit a (broken) power law to a debiased size-frequency distribution.

The physical payoff of a debiased SFD is its shape: the power-law slope(s) and
the location of any break(s). Slopes encode the collisional state (Dohnanyi
equilibrium is a cumulative slope ~2.5); breaks and "waves" encode material
strength, the strength->gravity transition, and post-impact evolution.

**Convention.** We fit the per-dex differential density q(D) = dN/dlog10(D),
i.e. the counts in each bin divided by the bin's log10 width, as a straight line
(or broken line) in log10 q vs log10 D. The fitted log-log slope ``s`` maps to
the **cumulative** slope ``alpha`` of N(>D) ~ D**-alpha as ``alpha = -s`` (for
equal-log bins q ~ D**-alpha). The differential slope of dN/dD is ``alpha + 1``.
Every result reports the cumulative ``alpha`` per segment.

For a fixed set of break positions the model is *linear* in its coefficients
(hinge basis), so each candidate is a closed-form weighted least squares; we grid
over break positions and pick the model (0, 1 or 2 breaks) with the lowest BIC,
so a break is only reported when the data justify the extra parameters.

TODO (offer both fit spaces): keep this DIFFERENTIAL per-dex fit as the default --
it is the statistically clean choice for this project because the bins are
independent -- but also add an optional CUMULATIVE N(>D) fit whose reported slopes
match a cumulative-SFD plot bin-for-bin, for direct comparison with published
studies (e.g. Vavra & Broz 2026, Bottke et al. 2026). Note the two agree for a
clean power law but diverge across a turnover, and cumulative bins are correlated
(so a cumulative fit's error bars/BIC need care, not naive independent-bin stats).
"""

from __future__ import annotations

from itertools import combinations
import numpy as np

LN10 = np.log(10.0)


def _log_density(d_lo, d_hi, n):
    """u = log10(d_mid), y = log10(dN/dlog10 D) for the per-dex differential SFD."""
    d_lo = np.asarray(d_lo, float)
    d_hi = np.asarray(d_hi, float)
    n = np.asarray(n, float)
    dlog = np.log10(d_hi) - np.log10(d_lo)
    d_mid = np.sqrt(d_lo * d_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = n / dlog
        return np.log10(d_mid), np.log10(q), d_mid   # log10(0) -> -inf, masked by n>0


def _sigma_y(n, n_lo, n_hi):
    """1-sigma on log10(count). Uses the supplied band, else Poisson sqrt(n)."""
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if n_lo is None or n_hi is None:
            sy = 1.0 / (np.sqrt(n) * LN10)                 # Poisson
        else:
            sy = (np.asarray(n_hi, float) - np.asarray(n_lo, float)) / (2.0 * n * LN10)
    return sy


def fit_sfd_mc(eff, n_observed, max_breaks: int = 2, n_draws: int = 500,
               eta_min: float = 0.05, d_min: float = 0.0, seed: int = 0) -> dict:
    """Fit the debiased SFD with Monte-Carlo parameter CIs from a combined
    efficiency table + real observed counts.

    Works from a saved ``*_efficiency_ensemble.csv`` (needs ``d_lo``, ``d_hi``,
    ``eta`` and the total band ``eta_tot_lo``/``eta_tot_hi``). The central fit uses
    the central eta; the CIs come from ``n_draws`` Monte-Carlo realizations that
    **combine both independent error sources** -- eta sampled from its total
    (statistical (+) systematic) band, and N_obs resampled by Poisson -- refit with
    the central model's number of breaks. Returns the central fit plus 16/84
    percentiles of the cumulative slopes and break diameters.

    ``d_min`` restricts the *fit* to bins with centre >= d_min (the H&M
    completeness floor); the debiased SFD is still produced for all bins.
    """
    eff = eff.reset_index(drop=True)
    d_lo = eff["d_lo"].to_numpy(float)
    d_hi = eff["d_hi"].to_numpy(float)
    d_mid = np.sqrt(d_lo * d_hi)
    eta = eff["eta"].to_numpy(float)
    tot_sig = ((eff["eta_tot_hi"] - eff["eta_tot_lo"]) / 2.0).to_numpy(float)
    tot_sig = np.where(np.isfinite(tot_sig), tot_sig, 0.0)
    n_obs = np.asarray(n_observed, float)
    rel = np.isfinite(eta) & (eta >= eta_min) & (d_mid >= d_min)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_true_c = np.where(eta > 0, n_obs / eta, np.nan)
    central = fit_sfd(d_lo, d_hi, n_true_c, reliable=rel, max_breaks=max_breaks)["best"]
    k = central["n_breaks"]

    rng = np.random.default_rng(seed)
    alphas, breaks = [], []
    for _ in range(n_draws):
        eta_d = np.clip(eta + rng.normal(0.0, tot_sig), 1e-3, 1.0)
        nobs_d = rng.poisson(np.clip(n_obs, 0.0, None)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            nt = np.where(eta_d > 0, nobs_d / eta_d, np.nan)
        try:
            f = fit_sfd(d_lo, d_hi, nt, reliable=rel & (nt > 0),
                        force_breaks=k, max_breaks=max_breaks)["best"]
        except ValueError:
            continue
        alphas.append(f["alpha"])
        breaks.append(f["break_diam_km"])

    alphas = np.array(alphas)
    pct = lambda a, p: np.nanpercentile(a, p, axis=0).tolist()
    res = {"central": central, "n_breaks": k, "n_draws": len(alphas),
           "d_min": float(d_min),
           "alpha_median": pct(alphas, 50),
           "alpha_lo": pct(alphas, 16), "alpha_hi": pct(alphas, 84)}
    if k > 0 and breaks:
        b = np.array(breaks)
        res.update(break_diam_median=pct(b, 50),
                   break_diam_lo=pct(b, 16), break_diam_hi=pct(b, 84))
    return res


def predict_density(best: dict, d):
    """Fitted per-dex density dN/dlog10(D) at diameters ``d`` (to draw the fit).

    ``best`` is a model dict from :func:`fit_sfd` (the ``best`` entry or any
    candidate). Multiply by a bin's log10 width to get its predicted count.
    """
    d = np.asarray(d, float)
    u = np.log10(d)
    slopes = [-a for a in best["alpha"]]                   # log-log slope = -alpha
    y = best["log10_A"] + slopes[0] * u
    for bu, s_prev, s_next in zip((np.log10(b) for b in best["break_diam_km"]),
                                  slopes[:-1], slopes[1:]):
        y = y + (s_next - s_prev) * np.maximum(0.0, u - bu)
    return 10.0 ** y


def _design(u, breaks):
    """Hinge design matrix for a continuous piecewise-linear model in ``u``."""
    cols = [np.ones_like(u), u]
    for b in breaks:
        cols.append(np.maximum(0.0, u - b))
    return np.column_stack(cols)


def _fit_fixed(u, y, sy, breaks):
    """Weighted LSQ for fixed break positions. Returns (slopes, D_breaks-less)."""
    X = _design(u, breaks)
    w = 1.0 / sy
    Xw, yw = X * w[:, None], y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = (X @ beta - y) / sy
    chi2 = float(resid @ resid)
    cov = np.linalg.inv(Xw.T @ Xw)
    # segment log-log slopes: s0 = beta[1]; each hinge adds its coefficient
    slopes = beta[1] + np.concatenate([[0.0], np.cumsum(beta[2:])])
    slope_err = np.sqrt(np.diag(cov))[1:]                  # crude per-coef errors
    return beta, cov, chi2, slopes, slope_err


def _bic(chi2, n_pts, n_par):
    return chi2 + n_par * np.log(n_pts)


def fit_sfd(d_lo, d_hi, n, n_lo=None, n_hi=None, reliable=None,
            max_breaks: int = 2, min_bins_per_segment: int = 3,
            force_breaks: int | None = None) -> dict:
    """Fit 0..``max_breaks`` break power laws; return the BIC-preferred model.

    ``n`` is the (debiased) count per diameter bin with edges ``d_lo``/``d_hi``;
    ``n_lo``/``n_hi`` its uncertainty band (else Poisson). ``reliable`` masks bins
    to fit. ``force_breaks`` fits exactly that many breaks (skipping model
    selection) -- used to fit every ensemble member with the same structure.
    Returns a dict with the chosen model plus every candidate, each giving the
    cumulative ``alpha`` per segment, the break diameter(s), chi2/dof and BIC.
    """
    d_lo = np.asarray(d_lo, float)
    d_hi = np.asarray(d_hi, float)
    n = np.asarray(n, float)
    m = np.isfinite(n) & (n > 0)
    if reliable is not None:
        m &= np.asarray(reliable, bool)
    if n_lo is not None:
        n_lo = np.asarray(n_lo, float)
    if n_hi is not None:
        n_hi = np.asarray(n_hi, float)

    u_all, y_all, d_mid_all = _log_density(d_lo, d_hi, n)
    u, y = u_all[m], y_all[m]
    sy = _sigma_y(n[m], None if n_lo is None else n_lo[m],
                  None if n_hi is None else n_hi[m])
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, np.nanmedian(sy[np.isfinite(sy)]))
    npts = u.size
    if npts < 2:
        raise ValueError("need >= 2 usable bins to fit an SFD")

    # candidate break positions: interior bin centres, keeping each segment >= min bins
    order = np.argsort(u)
    u_s, y_s, sy_s = u[order], y[order], sy[order]
    cand = u_s[min_bins_per_segment - 1: npts - (min_bins_per_segment - 1)] \
        if npts >= 2 * min_bins_per_segment else np.array([])

    def _package(breaks):
        beta, cov, chi2, slopes, slope_err = _fit_fixed(u_s, y_s, sy_s, breaks)
        n_par = 2 + len(breaks)
        dof = max(npts - n_par, 1)
        return {
            "n_breaks": len(breaks),
            "alpha": (-slopes).tolist(),                    # cumulative slope per segment
            "alpha_err": slope_err.tolist(),
            "break_diam_km": [float(10 ** b) for b in breaks],
            "log10_A": float(beta[0]),                      # intercept -> lets you draw the fit
            "chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof,
            "bic": _bic(chi2, npts, n_par), "n_par": n_par,
        }

    def _best_k(k):
        if k == 0:
            return _package([])
        if cand.size < k:
            return None
        best = None
        for combo in combinations(cand, k):
            b = sorted(combo)
            seg_edges = [u_s.min() - 1] + list(b) + [u_s.max() + 1]
            if any(((u_s > lo) & (u_s <= hi)).sum() < min_bins_per_segment
                   for lo, hi in zip(seg_edges[:-1], seg_edges[1:])):
                continue
            cf = _package(b)
            if best is None or cf["bic"] < best["bic"]:
                best = cf
        return best

    ks = [force_breaks] if force_breaks is not None else range(0, max_breaks + 1)
    candidates = [c for c in (_best_k(k) for k in ks) if c is not None]
    if not candidates:
        raise ValueError("no valid segmentation for the given bins/constraints")
    best = min(candidates, key=lambda c: c["bic"])
    return {"best": best, "candidates": candidates,
            "n_bins": npts, "convention": "alpha = cumulative slope of N(>D) ~ D**-alpha"}
