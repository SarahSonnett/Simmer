"""All-surveys extension: push the NEOWISE-debiased SFD below the completeness
limit using the optically-discovered known population.

The NEOWISE debiasing is complete and reliable only above each subpopulation's
completeness limit ``D_complete``. Below it the NEOWISE sample runs out, but the
*optically discovered* known population continues to smaller sizes. This module
bridges the two via the **catalog completeness**

    C(D) = N_known(>D) / N_true(>D),

which is *measured* in the overlap above ``D_complete`` (there N_true is the NEOWISE
debiased count) and then *extrapolated* below it, so the known counts can be corrected
down to ~1 km:

    N_true(>D) = N_known(>D) / C(D),      D < D_complete.

Two extrapolation forms are provided and **compared** (their spread is reported as a
systematic band; Cibulková et al. 2014 use IR-D where available + H->D for the rest):

  (1) :func:`hm_rollover_completeness` -- an optical H-completeness rollover using the
      per-subpopulation H_lim (Hendler & Malhotra 2020 convention) folded with the
      family albedo distribution. Because a magnitude-limited survey finds a
      high-albedo object at fixed D before a dark one, folding the *distribution*
      (rather than a single mean albedo) into the completeness is the albedo-selection
      correction of Jedicke & Metcalfe (1998) -- it makes C(D) roll off at larger D for
      darker families. This is ``jedicke=True`` (default); ``jedicke=False`` collapses
      the albedo distribution to its median (the biased single-albedo treatment) and is
      provided only for sensitivity tests.

  (2) :func:`fit_logistic_completeness` -- an empirical logistic in log D fit to the
      measured overlap C(D) and extrapolated. Assumption-light but pure extrapolation.

The extension is **data-driven**: C(D) is applied directly to the (binned/cumulative)
known counts; no power-law slope is imposed on the sub-completeness SFD, because the
small main-belt SFD is not a single power law (bump near D~3-4 km; Terai & Yoshida
2021). The known count is a **hard lower bound** on N_true (known ⊆ true), which both
validates the correction in the overlap and floors the extension.

References: Hendler & Malhotra 2020; Jedicke & Metcalfe 1998 (Icarus 131, 245);
Cibulková et al. 2014 (A&A 570, A126); Terai & Yoshida 2021.
"""

from __future__ import annotations

import numpy as np

from . import optical


def hm_h_limit(H_values, bin_width: float = 0.5, h_range=(10.0, 20.0)):
    """Optical catalog completeness H_lim = peak of the known-population H histogram.

    The Hendler & Malhotra (2020) convention: the differential count-vs-H of the *known*
    (all-surveys) members turns over at the catalog's completeness magnitude. This is the
    **optical** catalog limit — distinct from, and fainter than, the NEOWISE thermal
    completeness — and is the H_lim to pass to :func:`hm_rollover_completeness` /
    :func:`extend_sfd`. Returns the center of the fullest 0.5-mag bin.
    """
    H = np.asarray(H_values, float)
    H = H[np.isfinite(H)]
    if H.size == 0:
        return float("nan")
    edges = np.arange(h_range[0], h_range[1] + bin_width, bin_width)
    cnt, _ = np.histogram(H, bins=edges)
    k = int(cnt.argmax())
    return float(0.5 * (edges[k] + edges[k + 1]))


def measured_completeness(d_grid, n_known_cum, n_true_cum, d_complete):
    """Catalog completeness C(D)=N_known(>D)/N_true(>D) on ``d_grid`` in the overlap.

    Evaluated only where ``d_grid >= d_complete`` and ``n_true_cum > 0``; elsewhere
    NaN. Clipped to (0, 1] (statistical noise can push the raw ratio slightly above 1).
    """
    d_grid = np.asarray(d_grid, float)
    nk = np.asarray(n_known_cum, float)
    nt = np.asarray(n_true_cum, float)
    C = np.full(d_grid.shape, np.nan)
    ok = (d_grid >= d_complete) & (nt > 0) & np.isfinite(nt) & np.isfinite(nk)
    C[ok] = np.clip(nk[ok] / nt[ok], 1e-6, 1.0)
    return C


def jedicke_bright_fraction(f_true, R, size_index):
    """Observed bright-complex fraction from the true one (Jedicke & Metcalfe 1998).

    f' = f R^(a/2) / [1 + f (R^(a/2) - 1)], with ``R`` the bright:dark albedo ratio and
    ``a`` the size index (dN/dD ~ D^-(a+1)). A magnitude-limited survey over-represents
    the bright complex, so f' >= f. Provided as a documented reference/inverse helper;
    the completeness rollover applies the same physics by folding the full measured
    albedo distribution (see :func:`hm_rollover_completeness`).
    """
    f = np.asarray(f_true, float)
    r = R ** (size_index / 2.0)
    return f * r / (1.0 + f * (r - 1.0))


def hm_rollover_completeness(d_grid, H_lim, albedo_model, jedicke: bool = True):
    """Optical catalog completeness C(D) from an H_lim rollover + the albedo mix.

    An object of true diameter D and albedo pV has H = H(D, pV) and is in the optical
    catalog iff H <= ``H_lim`` <=> D >= ``optical.diameter_km(H_lim, pV)``. So the
    completeness at D is the fraction of the family's albedo distribution light enough
    to have been found:

        C(D) = P_pV( D >= diameter_km(H_lim, pV) ).

    With ``jedicke=True`` (default) the *full* measured albedo distribution is used, so
    darker families roll off at larger D (the Jedicke & Metcalfe 1998 albedo-selection
    effect); with ``jedicke=False`` the distribution is collapsed to its median (the
    biased single-albedo treatment). C is clipped to (0, 1] and is monotonically
    non-decreasing in D by construction.
    """
    d_grid = np.asarray(d_grid, float)
    if jedicke:
        pv = np.asarray(albedo_model.pv, float)
    else:
        pv = np.array([albedo_model.median()], float)
    d_lim = optical.diameter_km(H_lim, pv)                     # per-albedo discovery limit
    # fraction of albedos whose discovery-limit diameter is <= D
    C = np.array([(d_lim <= D).mean() for D in d_grid], float)
    return np.clip(C, 1e-6, 1.0)


def fit_logistic_completeness(d_overlap, c_overlap):
    """Fit C(D) = 1/(1+exp(-k(logD - logD50))) to the measured overlap; return callable.

    A logistic in log10 D. Falls back to a monotone-clipped linear-in-logD fit if the
    nonlinear fit fails (e.g. too few points). The returned callable maps D (km) -> C in
    (0, 1], usable for extrapolation below the overlap.
    """
    d = np.asarray(d_overlap, float)
    c = np.asarray(c_overlap, float)
    m = np.isfinite(d) & np.isfinite(c) & (d > 0) & (c > 0)
    x, y = np.log10(d[m]), np.clip(c[m], 1e-4, 1 - 1e-4)

    def _logistic(x_, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x_ - x0)))

    if x.size >= 3:
        try:
            from scipy.optimize import curve_fit
            p, _ = curve_fit(_logistic, x, y, p0=[5.0, x.min()], maxfev=10000)
            k, x0 = float(p[0]), float(p[1])
            if k > 0:
                return lambda D: np.clip(_logistic(np.log10(np.asarray(D, float)), k, x0),
                                         1e-6, 1.0)
        except Exception:
            pass
    # fallback: logit-linear least squares
    logit = np.log(y / (1 - y))
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, logit, rcond=None)[0]
    k = max(k, 1e-3)
    return lambda D: np.clip(1.0 / (1.0 + np.exp(-(k * np.log10(np.asarray(D, float)) + b))),
                             1e-6, 1.0)


def extend_sfd(d_grid, n_known_cum, n_true_cum, d_complete, H_lim, albedo_model,
               n_known_lo=None, n_known_hi=None, jedicke: bool = True,
               c_floor: float = 0.05, C_reshape=None):
    """Extend N_true(>D) below ``d_complete`` from the known counts and C(D).

    Inputs are cumulative counts already evaluated on ``d_grid`` (e.g. from
    :func:`optical.cumulative_known` for the known counts and the debiased SFD
    interpolated onto the same grid for ``n_true_cum``). ``H_lim`` must be the **optical
    catalog** completeness magnitude (from :func:`hm_h_limit` on the known members) — the
    depth to which the all-surveys catalog is complete — NOT the NEOWISE thermal
    completeness, which is much brighter. ``c_floor`` is the completeness below which the
    N_known/C correction is deemed unreliable (dividing a sparse known count by a tiny
    completeness is unstable and over-corrects); such bins are flagged and excluded from
    the reliable range rather than reported as a hard number (Cibulková et al. 2014: no
    correction below a population-dependent limit). Returns a dict:

    - ``C_measured``      : measured overlap completeness (NaN below d_complete),
    - ``C_hm``, ``C_log`` : the two extrapolated completeness curves on the full grid,
    - ``n_true_hm``, ``n_true_log`` : N_true(>D) = N_known/C for each form (above
      d_complete the NEOWISE debiased ``n_true_cum`` is used unchanged),
    - ``n_true``          : per-bin mean of the two forms (the reported extension),
    - ``band_lo``/``band_hi`` : envelope combining the two-form spread with the known
      MC band (if ``n_known_lo``/``hi`` given), floored at the known count,
    - ``lower_bound``     : the known count itself (hard floor; known ⊆ true),
    - ``reliable``        : bool mask (C above ``c_floor`` for both forms, or in overlap),
    - ``d_reliable_min``  : smallest D the extension is trustworthy to.
    """
    d_grid = np.asarray(d_grid, float)
    nk = np.asarray(n_known_cum, float)
    nt = np.asarray(n_true_cum, float)

    C_meas = measured_completeness(d_grid, nk, nt, d_complete)
    C_hm = hm_rollover_completeness(d_grid, H_lim, albedo_model, jedicke=jedicke)
    # Anchor the H&M rollover to the MEASURED completeness at the boundary. The optical
    # H_lim model reads ~100% complete at D_complete, but the data (N_known/N_debiased)
    # is only ~0.8-0.9 there, so an un-anchored C_hm is discontinuous with the debiased
    # SFD and over-optimistic below it. Rescaling by the measured value at the boundary
    # preserves the albedo-rollover SHAPE while making both extrapolation forms agree
    # with the data at D_complete (continuity with the debiased SFD).
    near = np.isfinite(C_meas) & (d_grid <= d_complete * 1.4)
    j = int(np.argmin(np.abs(d_grid - d_complete)))
    anchor = float(np.nanmedian(C_meas[near])) if near.any() else (
        C_meas[j] if np.isfinite(C_meas[j]) else 1.0)
    if np.isfinite(anchor) and anchor > 0 and C_hm[j] > 0:
        C_hm = np.clip(C_hm * anchor / C_hm[j], 1e-6, 1.0)
    C_log_fn = fit_logistic_completeness(d_grid[np.isfinite(C_meas)],
                                         C_meas[np.isfinite(C_meas)])
    C_log = C_log_fn(d_grid)
    if C_reshape is not None:
        # Census-measured SHAPE correction (docs/phase3_rerun_design.md): both
        # extrapolation forms are multiplied by the population's i/albedo-
        # resolved relative completeness decline pk(D)/pk(D_complete) from the
        # blind census. The boundary anchor (which absorbs chain-acceptance
        # losses the census does not measure) is preserved; only the SHAPE of
        # the decline below D_complete is corrected. Below the census's flux
        # floor the ratio is flat, so the H&M/logistic small-D plunge (and the
        # c_floor reliability gates) still govern there.
        ratio = np.clip(np.asarray(C_reshape, float), 0.05, 1.0)
        C_hm = np.clip(C_hm * ratio, 1e-6, 1.0)
        C_log = np.clip(C_log * ratio, 1e-6, 1.0)

    below = d_grid < d_complete
    # N_true = NEOWISE debiased above completeness; N_known / C_ext below it.
    def _combine(C_ext):
        out = nt.copy()
        out[below] = nk[below] / np.clip(C_ext[below], c_floor, 1.0)
        return np.maximum(out, nk)                    # hard lower bound: known <= true
    n_true_hm = _combine(C_hm)
    n_true_log = _combine(C_log)
    n_true = 0.5 * (n_true_hm + n_true_log)

    # reliability: above completeness always; below, only where BOTH completeness forms
    # exceed the floor (so N_known/C is not a divide-by-~0 extrapolation).
    reliable = (~below) | ((C_hm >= c_floor) & (C_log >= c_floor))
    rel_d = d_grid[reliable]
    d_reliable_min = float(rel_d.min()) if rel_d.size else float(d_complete)

    lo_forms = np.minimum(n_true_hm, n_true_log)
    hi_forms = np.maximum(n_true_hm, n_true_log)
    if n_known_lo is not None and n_known_hi is not None:
        klo, khi = np.asarray(n_known_lo, float), np.asarray(n_known_hi, float)
        # propagate the known MC band through the same C division below completeness
        band_lo = lo_forms.copy(); band_hi = hi_forms.copy()
        scale_lo = np.where(nk > 0, klo / np.maximum(nk, 1e-9), 1.0)
        scale_hi = np.where(nk > 0, khi / np.maximum(nk, 1e-9), 1.0)
        band_lo[below] = lo_forms[below] * scale_lo[below]
        band_hi[below] = hi_forms[below] * scale_hi[below]
    else:
        band_lo, band_hi = lo_forms, hi_forms
    band_lo = np.maximum(band_lo, nk)                 # never below the known floor

    return dict(d_grid=d_grid, C_measured=C_meas, C_hm=C_hm, C_log=C_log,
                n_true_hm=n_true_hm, n_true_log=n_true_log, n_true=n_true,
                band_lo=band_lo, band_hi=band_hi, lower_bound=nk,
                reliable=reliable, d_reliable_min=d_reliable_min)
