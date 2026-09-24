"""WISE bandpass response and the color-corrected in-band flux convention.

Loads the WISE Relative System Response (RSR) curves (Wright et al. 2010; files
``data/RSR-W{n}.txt``) and turns a source spectrum into the flux density WISE
would *report* -- the monochromatic F_nu at the band isophotal wavelength under
WISE's F_nu ~ nu^-2 (f_lambda = const) calibration reference. That is the system
the 5-sigma sensitivity limits are quoted in, so it is what the detection stage
must compare against.

Convention (All-Sky Explanatory Supplement IV.4.h). The RSR is tabulated **per
photon**, so a source's detected signal is

    S(F) = INT F_nu(lam) R(lam) dlam / lam .

The pipeline reports flux assuming the reference spectrum F_nu ~ nu^-2, i.e.
F_nu(lam) / F_nu(lam_iso) = (lam / lam_iso)^2. Matching signals gives the
reported in-band flux density

    F_rep = [ INT F_nu^src R dlam/lam ] / [ INT (lam/lam_iso)^2 R dlam/lam ] ,

which equals F_nu(lam_iso) for the reference spectrum (color correction = 1 by
construction) and applies the WISE color correction automatically otherwise.

Because the temperature dependence separates out, we precompute the 1-D lookup
``P(T) = INT B_nu(lam, T) R dlam/lam`` once per band; the per-object flux is then
just a surface integral over ``P(T(psi, phi))`` -- no per-wavelength work at
evaluation time.

CONVENTION NOTE (please sanity-check): the color correction is derived here from
the *stated* WISE reference spectrum and per-photon RSR weighting; the exact
equation in the Explanatory Supplement is an image we could not transcribe.
:func:`color_correction_blackbody` prints f_c for blackbodies so the values can
be checked against Explanatory Supplement Table 2.

References
----------
- Wright, E. L., et al. 2010, AJ, 140, 1868 -- WISE mission; relative system
  response (RSR) curves and photon-counting definition (RSR-W{n}.txt).
- Jarrett, T. H., et al. 2011, ApJ, 735, 112 -- WISE absolute photometric
  calibration and color corrections.
- Cutri, R. M., et al. 2012, "Explanatory Supplement to the WISE All-Sky Data
  Release Products", Sec. IV.4.h -- reference spectrum (F_nu ~ nu^-2), isophotal
  wavelengths, zero-magnitude flux densities (Table 1), and the color-correction
  table (Table 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

# numpy >= 2.0 renamed trapz -> trapezoid; support both so the environment's
# numpy pin (e.g. tensorflow requires numpy < 2) cannot break the pipeline.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_DATA = Path(__file__).resolve().parent.parent / "data"

# Isophotal wavelength (m) and zero-magnitude flux density (Jy): All-Sky
# Explanatory Supplement IV.4.h (Vega basis; W4 includes the +2.7% MSX offset).
ISO_WAVELENGTH_M = {"W1": 3.35e-6, "W2": 4.60e-6, "W3": 11.56e-6, "W4": 22.08e-6}
ZERO_MAG_JY = {"W1": 309.540, "W2": 171.787, "W3": 31.674, "W4": 8.363}

_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def planck_nu(lam_m: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Planck B_nu (W m^-2 Hz^-1 sr^-1) on a (T,) x (lam,) grid -> (nT, nlam)."""
    nu = _C / lam_m
    T = np.atleast_1d(np.asarray(T, float))
    out = np.zeros((T.size, lam_m.size))
    pos = T > 0
    if pos.any():
        x = _H * nu[None, :] / (_KB * T[pos][:, None])
        with np.errstate(over="ignore"):
            out[pos] = (2 * _H * nu[None, :] ** 3 / _C ** 2) / np.expm1(x)
    return out


@dataclass
class Bandpass:
    band: str
    lam_m: np.ndarray          # wavelength grid (m), trimmed to the RSR support
    R: np.ndarray              # relative system response, per photon
    lam_iso_m: float
    fnu_zero_jy: float
    ref_norm: float            # INT (lam/lam_iso)^2 R dlam/lam
    _T_grid: np.ndarray
    _P_grid: np.ndarray        # P(T) = INT B_nu(lam,T) R dlam/lam, tabulated

    def P(self, T) -> np.ndarray:
        """Band-integrated Planck-through-RSR ``INT B_nu R dlam/lam`` at T (K)."""
        return np.interp(np.asarray(T, float), self._T_grid, self._P_grid, left=0.0)


@lru_cache(maxsize=None)
def get_bandpass(band: str, t_max: float = 700.0, n_T: int = 1400) -> Bandpass:
    """Load a band's RSR and precompute its reference norm and P(T) table."""
    d = np.genfromtxt(_DATA / f"RSR-{band}.txt", comments="#")
    lam, R = d[:, 0] * 1e-6, d[:, 1]
    keep = R > 1e-4 * R.max()               # trim the near-zero wings
    lam, R = lam[keep], R[keep]
    order = np.argsort(lam)
    lam, R = lam[order], R[order]

    lam_iso = ISO_WAVELENGTH_M[band]
    w = R / lam                              # so INT g R dlam/lam = trapz(g * w, lam)
    ref_norm = float(_trapezoid((lam / lam_iso) ** 2 * w, lam))

    T_grid = np.linspace(1.0, t_max, n_T)
    B = planck_nu(lam, T_grid)               # (n_T, nlam)
    P_grid = _trapezoid(B * w[None, :], lam, axis=1)
    return Bandpass(band, lam, R, lam_iso, ZERO_MAG_JY[band], ref_norm, T_grid, P_grid)


def color_correction_blackbody(T, band: str) -> np.ndarray:
    """Color-correction factor f_c for an isothermal blackbody at temperature T.

    ``f_c = F_nu(lam_iso) / F_rep`` = ``B_nu(lam_iso, T) * ref_norm / P(T)``.
    Compare against Explanatory Supplement Table 2 to validate the convention.
    """
    bp = get_bandpass(band)
    T = np.atleast_1d(np.asarray(T, float))
    b_iso = planck_nu(np.array([bp.lam_iso_m]), T)[:, 0]
    return b_iso * bp.ref_norm / bp.P(T)
