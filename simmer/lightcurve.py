"""Rotational lightcurve modulation from a rotating ellipsoid.

The flux is modulated by the projected area of a prolate ellipsoid (long axis
``a``, short axes ``b = c``) rotating about a short axis, viewed at aspect angle
``xi`` between the spin pole and the line of sight (Surdej & Surdej 1978). With
``a = 1`` and ``b = b_over_a`` and rotation angle ``theta = 2*pi*phase``,

    A(theta) = sqrt( 1 - k^2 cos^2(theta) ),   k^2 = (1 - b^2) sin^2(xi) .

The flux multiplier is ``A(theta) / <A>`` where ``<A> = (2/pi) E(k^2)`` is the
rotation-averaged area (E = complete elliptic integral of the 2nd kind), so the
mean flux is preserved. This is intrinsically double-peaked (two maxima per
rotation) and reproduces the observed **broad maxima / narrow minima** shape --
a real improvement over a symmetric sinusoid, at no runtime cost.

Its peak-to-peak amplitude ``2.5 log10(A_max/A_min)`` reduces *exactly* to the
Sheppard & Jewitt (2004) relation (see :func:`amplitude_mag`), so amplitudes are
unchanged; only the curve *shape* is now physical. The aspect angle comes from
the spin pole (ecliptic ``beta``/``lambda``) vs. the line of sight, and each
object's rotation period + phase make repeated detections temporally coherent.

References
----------
- Surdej, J., & Surdej, A. 1978, A&A, 66, 31 -- lightcurve of a rotating
  three-axis ellipsoid (the projected-area model used here).
- Sheppard, S. S., & Jewitt, D. C. 2004, AJ, 127, 3023 -- the equivalent
  peak-to-peak amplitude / aspect-angle relation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import seeding


def pole_unit_vector(beta_deg: np.ndarray, lambda_deg: np.ndarray) -> np.ndarray:
    """Spin-pole unit vector(s) in ecliptic Cartesian from (beta, lambda)."""
    b = np.radians(np.asarray(beta_deg, float))
    l = np.radians(np.asarray(lambda_deg, float))
    return np.column_stack([np.cos(b) * np.cos(l),
                            np.cos(b) * np.sin(l),
                            np.sin(b)])


def aspect_angle_deg(beta_deg, lambda_deg, los_unit: np.ndarray) -> np.ndarray:
    """Aspect angle theta (deg) between the spin pole and the line of sight.

    Folded into [0, 90]: an amplitude is symmetric about an equator-on view, so
    the pole's hemisphere sign does not matter.
    """
    pole = pole_unit_vector(beta_deg, lambda_deg)
    cos_t = np.abs(np.sum(pole * los_unit, axis=1))
    return np.degrees(np.arccos(np.clip(cos_t, 0, 1)))


def amplitude_mag(b_over_a: np.ndarray, aspect_deg: np.ndarray) -> np.ndarray:
    """Peak-to-peak lightcurve amplitude dm (mag), Sheppard & Jewitt (2004).

    The S&J equation is written with theta measured from the spin *equator*
    (theta=0 -> equator-on -> maximum amplitude):

        dm = 2.5 log10(a:b) - 1.25 log10[((a:b)^2 - 1) sin^2(theta) + 1]

    Here ``aspect_deg`` is the standard aspect angle (pole vs. line of sight, in
    [0, 90]) returned by :func:`aspect_angle_deg`, which is the *complement* of
    S&J's theta -- so ``sin^2(theta) -> cos^2(aspect)``. The result is then 0
    pole-on (aspect=0) and maximal equator-on (aspect=90), as physical.
    """
    ab = 1.0 / np.asarray(b_over_a, float)                 # a:b elongation >= 1
    cos2 = np.cos(np.radians(np.asarray(aspect_deg, float))) ** 2
    return 2.5 * np.log10(ab) - 1.25 * np.log10((ab ** 2 - 1) * cos2 + 1.0)


@dataclass
class Rotation:
    """Per-object rotation state, fixed for the whole survey."""
    period_hr: np.ndarray
    phase0: np.ndarray


def assign_rotation(n: int, cfg) -> Rotation:
    """Assign each of ``n`` objects a rotation period and initial phase.

    The period is drawn log-uniformly over ``[cfg.rot_period_min_hr,
    cfg.rot_period_max_hr]`` and the initial phase uniformly. These are fixed
    per object so that repeated detections across the cadence are temporally
    coherent -- which is what makes a double-peaked curve meaningful (drawing an
    independent phase per detection would wash out to the same distribution for
    any harmonic).

    Draws are keyed by *global object index* via :mod:`simmer.seeding`, so they
    are reproducible and identical regardless of how objects are batched across
    parallel workers.
    """
    idx = np.arange(n)
    u_period = seeding.hash01(cfg.seed, seeding.STREAM_ROT_PERIOD, idx)
    phase0 = seeding.hash01(cfg.seed, seeding.STREAM_ROT_PHASE, idx)
    lo, hi = np.log(cfg.rot_period_min_hr), np.log(cfg.rot_period_max_hr)
    period = np.exp(lo + u_period * (hi - lo))
    return Rotation(period_hr=period, phase0=phase0)


def rotational_phase(mjd, epoch_mjd, period_hr, phase0) -> np.ndarray:
    """Rotational phase in [0, 1) at each observation time ``mjd``."""
    period_day = np.asarray(period_hr, float) / 24.0
    return ((np.asarray(mjd, float) - epoch_mjd) / period_day
            + np.asarray(phase0, float)) % 1.0


def projected_area(b_over_a, aspect_deg, phase) -> np.ndarray:
    """Relative projected area ``A(theta)/a`` of a prolate ellipsoid.

    ``a = 1``, ``b = c = b_over_a``; rotation angle ``theta = 2*pi*phase``, aspect
    angle ``aspect_deg`` (pole vs. line of sight). Surdej & Surdej (1978).
    """
    b = np.asarray(b_over_a, float)
    xi = np.radians(np.asarray(aspect_deg, float))
    theta = 2.0 * np.pi * np.asarray(phase, float)
    k2 = (1.0 - b ** 2) * np.sin(xi) ** 2
    return np.sqrt(np.clip(1.0 - k2 * np.cos(theta) ** 2, 0.0, None))


def _mean_area(b_over_a, aspect_deg) -> np.ndarray:
    """Rotation-averaged relative area ``<A>/a = (2/pi) E(k^2)``."""
    from scipy.special import ellipe
    b = np.asarray(b_over_a, float)
    xi = np.radians(np.asarray(aspect_deg, float))
    k2 = (1.0 - b ** 2) * np.sin(xi) ** 2
    return (2.0 / np.pi) * ellipe(k2)


def modulate_flux(flux, b_over_a, aspect_deg, phase) -> np.ndarray:
    """Modulate ``flux`` by the prolate-ellipsoid projected-area lightcurve.

    Multiplies by ``A(theta)/<A>`` (:func:`projected_area` normalized by the
    rotation-averaged area), preserving the mean flux. Intrinsically
    double-peaked with physical broad-max / narrow-min shape; its amplitude
    equals the Sheppard & Jewitt value (Surdej & Surdej 1978).
    """
    A = projected_area(b_over_a, aspect_deg, phase)
    return np.asarray(flux, float) * A / _mean_area(b_over_a, aspect_deg)
