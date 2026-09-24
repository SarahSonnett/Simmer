"""NEATM thermal flux (Harris 1998).

The Near-Earth Asteroid Thermal Model is the Standard Thermal Model with a
beaming parameter ``eta`` that absorbs the non-standard surface temperature
distribution (roughness/thermal inertia). The subsolar temperature is

    T_ss = [ (1 - A) * S0 / r_au**2 / (eta * eps * sigma) ] ** 0.25

with Bond albedo ``A = pV * q``, ``q = 0.290 + 0.684*G`` the phase integral, and
``S0`` the solar constant at 1 AU. Dayside temperature falls off as
``T = T_ss * cos(gamma)**0.25`` with ``gamma`` the angular distance from the
subsolar point; the nightside is taken as non-emitting.

The observed monochromatic flux density is the Planck function integrated over
the surface that is both sunlit and visible to the observer (separated by the
phase angle ``alpha``):

    F_nu = eps * (R/Delta)**2 *
           int int B_nu(T) * [cos(psi) cos(phi - alpha)]_+ * cos(psi) dpsi dphi

evaluated by direct quadrature. This is exact-NEATM (no thermal-IR phase
approximation); for large catalogs precompute a (T_ss, alpha) lookup table --
see the note in ``flux_band``.

References
----------
- Harris, A. W. 1998, Icarus, 131, 291 -- the Near-Earth Asteroid Thermal Model
  (NEATM), introducing the fitted beaming parameter eta.
- Lebofsky, L. A., et al. 1986, Icarus, 68, 239 -- the refined Standard Thermal
  Model that NEATM generalises (emissivity, phase-integral / Bond albedo).
- Mainzer, A., et al. 2011, ApJ, 736, 100 -- thermal-model (NEATM) calibration
  for minor planets observed with WISE/NEOWISE (the application this reproduces).
- Masiero, J. R., et al. 2011, ApJ, 741, 68 -- Main-Belt asteroid diameters and
  albedos from WISE/NEOWISE (the pV/eta/D distributions SynthPop samples).
- Mainzer, A., et al. 2011, ApJ, 731, 53 -- NEOWISE thermal fits of near-Earth
  objects.
For the RSR convolution and color correction applied to these fluxes, see
:mod:`simmer.bandpass`.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from . import bandpass

# Physical constants (SI).
_H = 6.62607015e-34       # Planck
_C = 2.99792458e8         # speed of light
_KB = 1.380649e-23        # Boltzmann
_SIGMA = 5.670374419e-8   # Stefan-Boltzmann
_S0 = 1361.0              # solar constant at 1 AU, W/m^2
_AU_M = 1.495978707e11    # AU in metres
_KM_M = 1.0e3

# Band effective wavelengths (metres). Monochromatic first cut; replace with
# RSR-convolved, color-corrected in-band fluxes in a survey backend if needed.
BAND_WAVELENGTH_M = {
    "W1": 3.3526e-6,
    "W2": 4.6028e-6,
    "W3": 11.5608e-6,
    "W4": 22.0883e-6,
}


def bond_albedo(pV: np.ndarray, G: float = 0.15) -> np.ndarray:
    """Bond albedo from visible geometric albedo via the H-G phase integral."""
    q = 0.290 + 0.684 * G
    return pV * q


def subsolar_temperature(pV, r_au, eta, G=0.15, emissivity=0.9) -> np.ndarray:
    """NEATM subsolar temperature (K)."""
    A = bond_albedo(np.asarray(pV, float), G)
    flux = (1 - A) * _S0 / np.asarray(r_au, float) ** 2
    return (flux / (np.asarray(eta, float) * emissivity * _SIGMA)) ** 0.25


def planck_nu(wavelength_m: float, T: np.ndarray) -> np.ndarray:
    """Planck spectral radiance B_nu (W m^-2 Hz^-1 sr^-1) at fixed wavelength."""
    nu = _C / wavelength_m
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    pos = T > 0
    x = _H * nu / (_KB * T[pos])
    # Cold facets give huge x -> expm1 overflows to +inf -> radiance 0 (correct).
    with np.errstate(over="ignore"):
        out[pos] = (2 * _H * nu ** 3 / _C ** 2) / np.expm1(x)
    return out


def flux_band(diam_km, pV, eta, r_au, delta_au, alpha_deg, band,
              G=0.15, emissivity=0.9, n_psi=30, n_phi=60) -> np.ndarray:
    """In-band NEATM flux density (Jy) for arrays of objects.

    Direct surface quadrature; vectorized over objects with a fixed grid. For
    very large catalogs, build a 2-D interpolant of the dimensionless integral
    over (T_ss-scaled) and alpha rather than calling this per detection.
    """
    wavelength = BAND_WAVELENGTH_M[band]
    nu = _C / wavelength

    diam = np.atleast_1d(np.asarray(diam_km, float))
    pV = np.atleast_1d(np.asarray(pV, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    r = np.atleast_1d(np.asarray(r_au, float))
    delta = np.atleast_1d(np.asarray(delta_au, float))
    alpha = np.radians(np.atleast_1d(np.asarray(alpha_deg, float)))

    T_ss = subsolar_temperature(pV, r, eta, G, emissivity)        # (N,)

    # Surface grid (subsolar-centred). psi=latitude, phi=longitude from subsolar.
    psi = np.linspace(-np.pi / 2, np.pi / 2, n_psi)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    dpsi = psi[1] - psi[0]
    dphi = phi[1] - phi[0]
    PSI, PHI = np.meshgrid(psi, phi, indexing="ij")               # (n_psi, n_phi)

    cos_gamma = np.cos(PSI) * np.cos(PHI)                          # subsolar angle
    lit = np.clip(cos_gamma, 0, None)                             # T ~ lit**0.25

    # Broadcast: (N, n_psi, n_phi).
    Tgrid = T_ss[:, None, None] * lit[None, :, :] ** 0.25
    B = planck_nu(wavelength, Tgrid.ravel()).reshape(Tgrid.shape)

    visible = np.clip(np.cos(PHI[None, :, :] - alpha[:, None, None]), 0, None)
    weight = visible * np.cos(PSI)[None, :, :]                     # (N, n_psi, n_phi)
    integral = np.sum(B * weight, axis=(1, 2)) * dpsi * dphi       # (N,) sr

    R_m = 0.5 * diam * _KM_M
    delta_m = delta * _AU_M
    F_nu = emissivity * (R_m / delta_m) ** 2 * integral           # W m^-2 Hz^-1
    return F_nu * 1.0e26                                          # -> Jy


def flux_band_wise(diam_km, pV, eta, r_au, delta_au, alpha_deg, band,
                   G=0.15, emissivity=0.9, n_psi=30, n_phi=60) -> np.ndarray:
    """WISE-*reported* color-corrected in-band flux density (Jy).

    Same NEATM surface quadrature as :func:`flux_band`, but the monochromatic
    Planck function is replaced by the RSR-band-integrated ``P(T)`` and the
    result divided by the reference-spectrum norm, giving the flux density WISE
    would report under its F_nu ~ nu^-2 calibration (see :mod:`simmer.bandpass`).
    This is what the 5-sigma sensitivity limits are quoted in. The color
    correction is carried automatically. Faster than the monochromatic form,
    since ``P(T)`` is a precomputed 1-D lookup (no per-wavelength work here).
    """
    bp = bandpass.get_bandpass(band)

    diam = np.atleast_1d(np.asarray(diam_km, float))
    pV = np.atleast_1d(np.asarray(pV, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    r = np.atleast_1d(np.asarray(r_au, float))
    delta = np.atleast_1d(np.asarray(delta_au, float))
    alpha = np.radians(np.atleast_1d(np.asarray(alpha_deg, float)))

    T_ss = subsolar_temperature(pV, r, eta, G, emissivity)        # (N,)

    psi = np.linspace(-np.pi / 2, np.pi / 2, n_psi)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    dpsi, dphi = psi[1] - psi[0], phi[1] - phi[0]
    PSI, PHI = np.meshgrid(psi, phi, indexing="ij")

    lit = np.clip(np.cos(PSI) * np.cos(PHI), 0, None)
    Tgrid = T_ss[:, None, None] * lit[None, :, :] ** 0.25
    Pgrid = bp.P(Tgrid.ravel()).reshape(Tgrid.shape)              # band-integrated

    visible = np.clip(np.cos(PHI[None, :, :] - alpha[:, None, None]), 0, None)
    weight = visible * np.cos(PSI)[None, :, :]
    integral = np.sum(Pgrid * weight, axis=(1, 2)) * dpsi * dphi

    R_m = 0.5 * diam * _KM_M
    delta_m = delta * _AU_M
    F_rep = emissivity * (R_m / delta_m) ** 2 * integral / bp.ref_norm
    return F_rep * 1.0e26                                         # -> Jy


@lru_cache(maxsize=None)
def _surface_lut(band, n_Tss=140, n_alpha=61, Tss_min=80.0, Tss_max=650.0,
                 alpha_max_deg=120.0, n_psi=30, n_phi=60):
    """Precompute the NEATM surface integral S(T_ss, alpha) and an interpolant.

    ``S = INT INT P_band(T_ss * lit^0.25) * vis(alpha) * cos(psi) dpsi dphi`` --
    the dimensionless band-integrated surface sum that :func:`flux_band_wise`
    computes per detection. It depends only on subsolar temperature and phase
    angle, so tabulating it once (same psi/phi quadrature as the direct form, so
    only the T_ss/alpha grid introduces error) lets the flux be an O(1) lookup.
    Cached per band.
    """
    from scipy.interpolate import RegularGridInterpolator
    bp = bandpass.get_bandpass(band)

    psi = np.linspace(-np.pi / 2, np.pi / 2, n_psi)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    dpsi, dphi = psi[1] - psi[0], phi[1] - phi[0]
    PSI, PHI = np.meshgrid(psi, phi, indexing="ij")
    lit = np.clip(np.cos(PSI) * np.cos(PHI), 0, None)

    Tss = np.linspace(Tss_min, Tss_max, n_Tss)
    alpha = np.radians(np.linspace(0.0, alpha_max_deg, n_alpha))
    Tgrid = Tss[:, None, None] * lit[None, :, :] ** 0.25
    P = bp.P(Tgrid.ravel()).reshape(Tgrid.shape)                 # (nT, npsi, nphi)
    vis = np.clip(np.cos(PHI[None, :, :] - alpha[:, None, None]), 0, None)
    weight = vis * np.cos(PSI)[None, :, :]                       # (nA, npsi, nphi)
    S = np.einsum("ikl,jkl->ij", P, weight) * dpsi * dphi        # (nT, nA)
    interp = RegularGridInterpolator((Tss, np.degrees(alpha)), S,
                                     bounds_error=False, fill_value=None)
    return interp


def flux_band_wise_lut(diam_km, pV, eta, r_au, delta_au, alpha_deg, band,
                       G=0.15, emissivity=0.9) -> np.ndarray:
    """WISE-reported in-band flux (Jy) via the precomputed (T_ss, alpha) table.

    Identical to :func:`flux_band_wise` up to interpolation accuracy (<~0.1% over
    MBA temperatures/phases), but O(1) per detection instead of a per-detection
    surface quadrature -- the fast path for large catalogs. ``emissivity``/``G``
    only enter through T_ss, so the same table serves any of their values.
    """
    bp = bandpass.get_bandpass(band)
    interp = _surface_lut(band)

    diam = np.atleast_1d(np.asarray(diam_km, float))
    pV = np.atleast_1d(np.asarray(pV, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    r = np.atleast_1d(np.asarray(r_au, float))
    delta = np.atleast_1d(np.asarray(delta_au, float))
    alpha = np.abs(np.atleast_1d(np.asarray(alpha_deg, float)))

    T_ss = subsolar_temperature(pV, r, eta, G, emissivity)
    S = interp(np.column_stack([T_ss, alpha]))
    R_m = 0.5 * diam * _KM_M
    delta_m = delta * _AU_M
    return emissivity * (R_m / delta_m) ** 2 * S / bp.ref_norm * 1.0e26
