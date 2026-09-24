"""Re-derive known-population diameters for the small-size SFD extension.

Below each subpopulation's NEOWISE completeness limit, the debiased SFD is extended
using the *all-known* (optical + IR) member population, whose diameters must be
re-derived from catalog absolute magnitudes H via

    D [km] = 1329 / sqrt(pV) * 10^(-H/5)          (Bowell et al. 1989; Harris 1998)

with an albedo pV assigned **per subpopulation**: the MEASURED NEOWISE albedo where
the object has one, otherwise a Monte-Carlo draw from the subpopulation's NEOWISE
albedo distribution (which is strongly bright/dark bimodal; Masiero et al. 2011).
Diameter uncertainty is propagated from the H uncertainty and the albedo spread by
Monte Carlo. Catalog H additionally carries a magnitude-dependent systematic (Vereš
et al. 2015) that is corrected before conversion.

This module builds the re-derived diameter distribution; the all-surveys efficiency
ratio and its extrapolation live in a separate step. See
docs/methodology_decisions.md Secs. 3-6 for the method, references, and caveats.

STATUS (first cut): the physics, the per-subpopulation albedo model, and the
Monte-Carlo uncertainty propagation are implemented and tested. Two seams are left
open, clearly, for decisions that need external data / calibration:
  * H for PROVISIONAL members needs a current MPCORB (only catalog with H keyed by
    packed designation) -- ``member_H`` returns NaN + a coverage flag for them;
  * the Vereš H-correction is implemented but DISABLED by default until its exact
    magnitude/sign are set from the paper (``h_correction``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import designations

H_TO_D_KM = 1329.0                                  # D = H_TO_D_KM / sqrt(pV) * 10^(-H/5)


def diameter_km(H, pV):
    """Diameter (km) from absolute magnitude H and visual albedo pV (vectorized)."""
    H = np.asarray(H, float)
    pV = np.asarray(pV, float)
    return H_TO_D_KM / np.sqrt(pV) * 10.0 ** (-H / 5.0)


class AlbedoModel:
    """Per-subpopulation visual-albedo distribution for Monte-Carlo assignment.

    Built from a subpopulation's MEASURED NEOWISE albedos. Objects lacking a measured
    albedo are assigned one by **bootstrap-resampling** this distribution, which
    reproduces the belt's bright/dark bimodality (Masiero et al. 2011) with no
    parametric assumption. CAVEAT: the measured NEOWISE albedo distribution is itself
    survey-biased (Masiero 2011) and should be debiased before use as a true sampling
    distribution -- a TODO (docs/methodology_decisions.md Sec. 3).
    """

    def __init__(self, pv):
        pv = np.asarray(pv, float)
        self.pv = pv[np.isfinite(pv) & (pv > 0)]
        if self.pv.size == 0:
            raise ValueError("AlbedoModel needs at least one positive measured albedo")

    @property
    def n(self):
        return self.pv.size

    def frac_bright(self, split: float = 0.10) -> float:
        """Fraction of the bright complex (pV >= split)."""
        return float((self.pv >= split).mean())

    def median(self) -> float:
        return float(np.median(self.pv))

    def sample(self, size, rng) -> np.ndarray:
        """Bootstrap-draw albedos (reproduces the measured bimodal distribution)."""
        return rng.choice(self.pv, size=size, replace=True)


def family_albedo_model(member_ids, neowise_pv, neowise_D, d_complete,
                        min_n: int = 30) -> "AlbedoModel":
    """Build the family albedo model from members ABOVE the completeness limit.

    A thermal survey preferentially detects dark objects at a fixed magnitude, so the
    albedos of the small/faint detected members are survey-skewed. Above the
    completeness limit, though, NEOWISE caught essentially every known member
    regardless of albedo, so those albedos are an unbiased estimate of the family's
    albedo distribution (empirically ~size-independent). Falls back to all members
    with a measured albedo if fewer than ``min_n`` sit above ``d_complete``.
    """
    above = [neowise_pv[i] for i in member_ids
             if i in neowise_pv and neowise_D.get(i, 0.0) >= d_complete]
    if len(above) >= min_n:
        return AlbedoModel(np.asarray(above, float))
    allpv = [neowise_pv[i] for i in member_ids if i in neowise_pv]
    return AlbedoModel(np.asarray(allpv, float))


def h_correction(H, enabled: bool = False, peak: float = 0.35, center: float = 14.0,
                 width: float = 1.7):
    """Magnitude-dependent additive H-correction (Vereš et al. 2015), to ADD to H.

    Vereš et al. 2015 (Icarus 261, 34) re-derived H for ~250k asteroids from uniform
    Pan-STARRS PS1 photometry and found MPC catalog H is systematically **too bright**
    (too small): their PS1 H is fainter than the MPC value, by a mean of +0.22 mag
    (Bowell G) and up to a **peak of ~0.35 mag at H ~ 14**, with agreement <0.1 mag at
    H < 11 and H > 19. So the correction ADDED to catalog H is **positive** (makes it
    fainter), which makes the derived diameter **smaller** (D ~ 10^(-H/5)).

    They give no formula, so this is a Gaussian fit to their Fig. 8 bump (peak +0.35
    at H=14, ~0.07 at H=11, ~0 at H=19 and at the bright end). **DISABLED by default**
    (returns 0) -- enabling it is a deliberate modelling choice, and it should be
    applied only to OPTICAL-ONLY objects (an object with a NEOWISE thermal diameter
    already has a self-consistent H/pV/D and should use that D directly). See
    docs/methodology_decisions.md Sec. 4.
    """
    H = np.asarray(H, float)
    if not enabled:
        return np.zeros_like(H)
    return peak * np.exp(-0.5 * ((H - center) / width) ** 2)


def sample_diameters(H, measured_pv, albedo: AlbedoModel, rng, n_mc: int = 200,
                     h_sigma: float = 0.3, h_correct=None) -> np.ndarray:
    """Monte-Carlo diameters per object from H + albedo, with uncertainty.

    For each of ``n_mc`` realizations: perturb the (optionally corrected) catalog H by
    N(0, ``h_sigma``), take the measured albedo where finite else a bootstrap draw from
    ``albedo``, and convert to D. Returns an ``(n_obj, n_mc)`` diameter array.
    ``h_correct`` is a callable H -> delta (e.g. :func:`h_correction`); None = no
    correction. ``h_sigma`` is the assumed catalog H uncertainty (mag).
    """
    H = np.asarray(H, float)
    measured_pv = np.asarray(measured_pv, float)
    n = H.size
    Hc = H + (h_correct(H) if h_correct is not None else 0.0)
    have_pv = np.isfinite(measured_pv) & (measured_pv > 0)
    D = np.empty((n, n_mc))
    for j in range(n_mc):
        Hj = Hc + rng.normal(0.0, h_sigma, n)
        pvj = np.where(have_pv, measured_pv, albedo.sample(n, rng))
        D[:, j] = diameter_km(Hj, pvj)
    return D


def eta_from_efficiency(d_mid, eta):
    """Build an interpolator D(km) -> detection efficiency from an efficiency table.

    Log-D linear interpolation, clipped to [0, 1]; extrapolates to 0 below the
    smallest bin and to the plateau above the largest. Used to condition the
    optical-only albedo assignment on NEOWISE non-detection (see
    :func:`sample_conditional_albedo`).
    """
    d_mid = np.asarray(d_mid, float)
    eta = np.asarray(eta, float)
    m = np.isfinite(d_mid) & np.isfinite(eta) & (d_mid > 0)
    ld = np.log10(d_mid[m])
    e = np.clip(eta[m], 0.0, 1.0)
    order = np.argsort(ld)
    ld, e = ld[order], e[order]

    def f(D):
        return np.clip(np.interp(np.log10(np.asarray(D, float)), ld, e,
                                 left=0.0, right=e[-1]), 0.0, 1.0)
    return f


def sample_conditional_albedo(H, pv_pool, eta_of_D, rng, n_mc: int = 200,
                              max_pool: int = 200):
    """Albedos for optical-only objects, conditioned on NEOWISE non-detection.

    Each object with catalog magnitude ``H`` draws its albedo from the family pool
    ``pv_pool`` reweighted by the NEOWISE NON-detection probability 1 - eta(D(H,pV)).
    Because D = 1329/sqrt(pV) * 10^(-H/5), at a fixed H a dark object is large (eta~1,
    it would have been detected -> down-weighted) and a bright object is small
    (eta<1 -> up-weighted), so the optical-only members are placed at their correct
    small diameters. ``eta_of_D`` maps diameter (km) -> efficiency (e.g. from
    :func:`eta_from_efficiency`). Returns an ``(n_obj, n_mc)`` albedo array.
    """
    H = np.asarray(H, float)
    pv_pool = np.asarray(pv_pool, float)
    pv_pool = pv_pool[np.isfinite(pv_pool) & (pv_pool > 0)]
    if pv_pool.size > max_pool:                          # cap for speed; pool only sets albedo support
        pv_pool = rng.choice(pv_pool, max_pool, replace=False)
    n, K = H.size, pv_pool.size
    D = diameter_km(H[:, None], pv_pool[None, :])        # (n, K)
    w = np.clip(1.0 - np.asarray(eta_of_D(D), float), 0.0, None)
    rs = w.sum(axis=1, keepdims=True)
    w = np.where(rs > 0, w / rs, 1.0 / K)                # fully-detected row -> uniform fallback
    cdf = np.cumsum(w, axis=1)
    out = np.empty((n, n_mc))
    for j in range(n_mc):
        u = rng.random((n, 1))
        idx = np.clip((cdf < u).sum(axis=1), 0, K - 1)   # per-row inverse-CDF sample
        out[:, j] = pv_pool[idx]
    return out


def known_diameters(member_ids, proper_catalog, neowise_pv, neowise_D, albedo,
                    rng, n_mc: int = 200, h_sigma: float = 0.3,
                    apply_h_correction: bool = True, thermal_d_sigma: float = 0.10,
                    eta_of_D=None, disc_year=None, cond_cut_year: int = 2010):
    """All-known member diameters ``(n_obj, n_mc)`` for the SFD extension.

    Uses each member's NEOWISE **thermal diameter** where it has one (a direct
    measurement; given a small ``thermal_d_sigma`` fractional scatter), and otherwise
    re-derives an OPTICAL-ONLY diameter from catalog H + a sampled albedo, with the
    Vereš H-correction applied **on that branch only** (``apply_h_correction``, default
    True -- NEOWISE objects already have a self-consistent H/pV/D). When ``eta_of_D`` is
    given, the optical-only albedo is drawn conditioned on NEOWISE **non-detection**
    (weighted by 1 - eta(D); see :func:`sample_conditional_albedo`) -- the correct
    treatment, since at a fixed H the NEOWISE-missed members are the small/bright ones;
    otherwise the unconditioned family distribution is used. Members with neither a
    thermal diameter nor a catalog H are left NaN. "IR diameter where available, H->D
    for the rest" (Cibulková 2014, Ryan 2012).
    """
    H = np.array([proper_catalog["H"].get(m, np.nan) for m in member_ids], float)
    thermalD = np.array([neowise_D.get(m, np.nan) for m in member_ids], float)
    _ = neowise_pv                                       # measured pV -> thermal D branch below
    n = len(member_ids)
    has_thermal = np.isfinite(thermalD) & (thermalD > 0)
    opt = (~has_thermal) & np.isfinite(H)
    D = np.full((n, n_mc), np.nan)
    if has_thermal.any():
        base = thermalD[has_thermal][:, None]
        D[has_thermal] = base * (1.0 + rng.normal(0.0, thermal_d_sigma,
                                                  (int(has_thermal.sum()), n_mc)))
    if opt.any():
        # Epoch-aware conditioning (Phase-3 audit, docs/phase3_rerun_design.md):
        # "no NEOWISE entry" conflates (a) WISE truly missed it with (b) it was
        # not yet DISCOVERED in 2010, so no association could exist. The
        # non-detection conditioning (dark-draw suppression) is only valid for
        # (a); members first observed after cond_cut_year get the UNCONDITIONED
        # pool — WISE plausibly saw them, their missing entry carries no albedo
        # information. disc_year: array-like of first-observation years aligned
        # with member_ids (unknown year -> treated as post-cut, the safe side).
        idx_opt = np.where(opt)[0]
        if disc_year is not None and eta_of_D is not None:
            yrs = np.asarray(disc_year, float)[idx_opt]
            cond = np.isfinite(yrs) & (yrs <= cond_cut_year)
        else:
            cond = np.ones(idx_opt.size, bool) if eta_of_D is not None                 else np.zeros(idx_opt.size, bool)
        known_diameters.last_audit = (int(idx_opt.size), int((~cond).sum()))
        for sel_c, use_cond in ((cond, True), (~cond, False)):
            if not sel_c.any():
                continue
            rows = idx_opt[sel_c]
            Hopt = H[rows] + (h_correction(H[rows], enabled=True)
                              if apply_h_correction else 0.0)
            if use_cond:
                pvj = sample_conditional_albedo(Hopt, albedo.pv, eta_of_D, rng,
                                                n_mc=n_mc)
                Hpert = Hopt[:, None] + rng.normal(0.0, h_sigma, (Hopt.size, n_mc))
                D[rows] = diameter_km(Hpert, pvj)
            else:
                D[rows] = sample_diameters(Hopt, np.full(rows.size, np.nan),
                                           albedo, rng, n_mc=n_mc,
                                           h_sigma=h_sigma, h_correct=None)
    return D


def cumulative_known(D_samples, d_grid):
    """Median cumulative N(>D) + 16/84 band over the MC diameter realizations.

    ``D_samples`` is ``(n_obj, n_mc)`` (from :func:`sample_diameters` or
    :func:`known_diameters`); ``d_grid`` the diameters at which to report N(>D).
    NaN rows (unassigned members) are ignored. Returns ``(median, lo, hi)`` arrays.
    """
    d_grid = np.asarray(d_grid, float)
    n_mc = D_samples.shape[1]
    counts = np.empty((n_mc, d_grid.size))
    for j in range(n_mc):
        col = D_samples[:, j]
        ds = np.sort(col[np.isfinite(col)])
        counts[j] = ds.size - np.searchsorted(ds, d_grid, side="right")   # N(> d)
    return (np.median(counts, axis=0),
            np.percentile(counts, 16, axis=0),
            np.percentile(counts, 84, axis=0))


def neowise_diameters(neowise_file, neo_number_col: int = 2, neo_d_col: int = 11):
    """Median measured NEOWISE thermal diameter (km) per canonical object id."""
    neo = pd.read_csv(neowise_file, header=None, usecols=[neo_number_col, neo_d_col],
                      names=["desig", "D"], dtype={neo_number_col: str})
    cid = neo["desig"].map(designations.canonical)
    d = pd.to_numeric(neo["D"], errors="coerce")
    ok = cid.notna() & d.notna() & (d > 0)
    return (pd.DataFrame({"id": cid[ok], "D": d[ok]})
            .groupby("id")["D"].median().to_dict())


# --- catalog loaders --------------------------------------------------------

def load_proper_catalog(path):
    """AstDyS proper-element catalog (`proper_catalog24.dat`): H + proper a/e/sinI.

    Whitespace-delimited, 12 columns: proper a, e, sin(i) at positions 0/2/4, H at 8,
    and the designation at position 11 -- a plain NUMBER for numbered objects OR a
    PROVISIONAL designation for multi-opposition unnumbered objects (the catalog holds
    both: ~623k numbered + ~626k provisional). Read as strings and selected by position
    (passing ``usecols`` to the C parser trips on the mixed-type packed column across
    chunks). Indexed by CANONICAL id (int or provisional str; see
    :mod:`simmer.designations`), so numbered AND provisional members can be looked up.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    out = df.iloc[:, [0, 2, 4, 8, 11]].copy()
    out.columns = ["a", "e", "sinI", "H", "desig"]
    for c in ("a", "e", "sinI", "H"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["id"] = out["desig"].map(designations.canonical)
    out = out.dropna(subset=["id", "H"]).drop(columns="desig")
    return out[~out["id"].duplicated()].set_index("id")


def neowise_albedos(neowise_file, neo_number_col: int = 2, neo_pv_col: int = 13):
    """Median measured NEOWISE albedo per canonical object id (dict id -> pV).

    Canonicalizes the designation column (see :mod:`simmer.designations`) so numbered
    (incl. >=100000 packed) and provisional objects are all keyed consistently.
    """
    neo = pd.read_csv(neowise_file, header=None, usecols=[neo_number_col, neo_pv_col],
                      names=["desig", "pV"], dtype={neo_number_col: str})
    cid = neo["desig"].map(designations.canonical)
    pv = pd.to_numeric(neo["pV"], errors="coerce")
    ok = cid.notna() & pv.notna() & (pv > 0)
    return (pd.DataFrame({"id": cid[ok], "pV": pv[ok]})
            .groupby("id")["pV"].median().to_dict())


def member_H(member_ids, proper_catalog, neowise_pv=None):
    """Per-member catalog H and measured albedo, keyed by canonical id.

    Returns a DataFrame indexed by member id with columns ``H`` (from the proper-element
    catalog -- covers numbered AND provisional/multi-opposition members), ``pV`` (the
    measured NEOWISE albedo if available, else NaN), and ``numbered`` (bool). Members
    absent from the catalog (e.g. single-opposition, no proper elements) get H=NaN.
    """
    neowise_pv = neowise_pv or {}
    H_by_id = proper_catalog["H"]
    rows = []
    for mid in member_ids:
        rows.append((mid, float(H_by_id.get(mid, np.nan)),
                     float(neowise_pv.get(mid, np.nan)),
                     isinstance(mid, (int, np.integer))))
    return pd.DataFrame(rows, columns=["id", "H", "pV", "numbered"]).set_index("id")
