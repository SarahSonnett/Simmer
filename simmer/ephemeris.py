"""Vectorized two-body ephemeris: synthetic elements -> sky positions & geometry.

For the purpose of *detection* (matching to ~47' frames over a multi-year
survey), heliocentric two-body propagation is more than accurate enough --
planetary perturbations move an MBA far less than a frame width on these
timescales. We solve Kepler's equation across all objects at once, build
heliocentric ecliptic positions, then subtract Earth's heliocentric position
(astropy, low overhead, cached by the caller) to get the geocentric vector.

WISE's low-Earth-orbit offset from the geocenter is ~4e-5 AU and is ignored.

Conventions: angles in degrees on input, ecliptic J2000 Cartesian internally
(AU). Outputs RA/Dec (equatorial J2000, degrees), heliocentric distance ``r``,
observer distance ``delta`` (AU), and solar phase angle ``alpha`` (degrees).

References
----------
- Murray, C. D., & Dermott, S. F. 1999, Solar System Dynamics (Cambridge Univ.
  Press) -- two-body propagation via Kepler's equation, orbital elements to
  Cartesian position, and the transformation to sky coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Obliquity of the ecliptic (J2000), radians.
_OBLIQUITY = np.radians(23.43928)
# Gaussian gravitational constant -> mean motion n = K * a**-1.5 (rad/day),
# heliocentric, mass-less test particle. K = 0.01720209895 rad/day.
_K_GAUSS = 0.01720209895


@dataclass
class Geometry:
    """Per-(object, time) observing geometry."""
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    ecl_lon_deg: np.ndarray
    ecl_lat_deg: np.ndarray
    r_au: np.ndarray         # heliocentric distance
    delta_au: np.ndarray     # observer (Earth) distance
    alpha_deg: np.ndarray    # solar phase angle
    helio_xyz: np.ndarray    # (N, 3) heliocentric ecliptic position, AU
    los_unit: np.ndarray     # (N, 3) observer->object unit vector, ecliptic frame


def solve_kepler(M: np.ndarray, e: np.ndarray, tol: float = 1e-10,
                 max_iter: int = 50) -> np.ndarray:
    """Solve M = E - e sin E for eccentric anomaly E (radians), vectorized."""
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi          # wrap to [-pi, pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))  # starting guess
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def elements_to_helio_xyz(elements: dict, epoch_jd: float, t_jd: float) -> np.ndarray:
    """Propagate orbital elements to heliocentric ecliptic Cartesian (AU) at t_jd.

    ``elements`` holds equal-length arrays: a_au, e, i_deg, node_deg,
    argperi_deg, M_deg (mean anomaly at ``epoch_jd``).
    """
    a = elements["a_au"]
    e = elements["e"]
    inc = np.radians(elements["i_deg"])
    node = np.radians(elements["node_deg"])
    argp = np.radians(elements["argperi_deg"])
    M0 = np.radians(elements["M_deg"])

    n = _K_GAUSS * a ** -1.5                          # mean motion, rad/day
    M = M0 + n * (t_jd - epoch_jd)
    E = solve_kepler(M, e)

    # Position in the orbital plane.
    cosE, sinE = np.cos(E), np.sin(E)
    x_orb = a * (cosE - e)
    y_orb = a * np.sqrt(1 - e * e) * sinE

    # Rotate orbital plane -> ecliptic: Rz(node) Rx(inc) Rz(argp).
    cn, sn = np.cos(node), np.sin(node)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    p11 = cn * cw - sn * sw * ci
    p12 = -cn * sw - sn * cw * ci
    p21 = sn * cw + cn * sw * ci
    p22 = -sn * sw + cn * cw * ci
    p31 = sw * si
    p32 = cw * si

    x = p11 * x_orb + p12 * y_orb
    y = p21 * x_orb + p22 * y_orb
    z = p31 * x_orb + p32 * y_orb
    return np.column_stack([x, y, z])


def _ecliptic_to_equatorial(xyz: np.ndarray) -> np.ndarray:
    """Rotate ecliptic Cartesian -> equatorial Cartesian (Rx by +obliquity)."""
    c, s = np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.column_stack([x, c * y - s * z, s * y + c * z])


def earth_helio_xyz(t_jd) -> np.ndarray:
    """Earth's heliocentric *ecliptic* position (AU), via astropy (lazy import).

    Accepts a scalar (returns shape ``(3,)``) or an array of JD times (returns
    ``(M, 3)``). astropy is imported here so the rest of the package works
    without it; duplicate times cost nothing extra if the caller de-duplicates.
    """
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris

    scalar = np.ndim(t_jd) == 0
    t_arr = np.atleast_1d(np.asarray(t_jd, dtype=float))
    with solar_system_ephemeris.set("builtin"):
        t = Time(t_arr, format="jd", scale="tdb")
        earth = get_body_barycentric("earth", t)
        sun = get_body_barycentric("sun", t)
    # Heliocentric = barycentric(earth) - barycentric(sun), equatorial AU, (M,3).
    helio_eq = np.stack([
        (earth.x - sun.x).to("AU").value,
        (earth.y - sun.y).to("AU").value,
        (earth.z - sun.z).to("AU").value,
    ], axis=-1)
    # Rotate equatorial -> ecliptic (Rx by -obliquity).
    c, s = np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)
    x, y, z = helio_eq[:, 0], helio_eq[:, 1], helio_eq[:, 2]
    helio_ecl = np.stack([x, c * y + s * z, -s * y + c * z], axis=-1)
    return helio_ecl[0] if scalar else helio_ecl


def geocentric_unit(elements: dict, epoch_jd: float, t_jd,
                    earth_xyz: np.ndarray) -> np.ndarray:
    """Equatorial geocentric *unit vectors* only -- the cheap coarse-match path.

    No distances, phase, or LOS: just where each object is on the sky, which is
    all the KD-tree needs. ``t_jd`` may be scalar or per-object; ``earth_xyz`` is
    the matching ecliptic Earth position(s).
    """
    obj = elements_to_helio_xyz(elements, epoch_jd, t_jd)        # (N, 3) ecliptic
    geo = obj - np.atleast_2d(earth_xyz)
    geo_eq = _ecliptic_to_equatorial(geo)
    return geo_eq / np.linalg.norm(geo_eq, axis=1)[:, None]


def geometry_from_positions(obj_xyz: np.ndarray, earth_xyz: np.ndarray) -> Geometry:
    """Full observing geometry from precomputed ecliptic positions.

    ``obj_xyz`` and ``earth_xyz`` are both ``(M, 3)`` heliocentric ecliptic (AU).
    Shared by the single-time :func:`observe` and the per-candidate refine path.
    """
    geo = obj_xyz - earth_xyz                                   # observer->object
    delta = np.linalg.norm(geo, axis=1)
    r = np.linalg.norm(obj_xyz, axis=1)
    los = geo / delta[:, None]

    to_sun = -obj_xyz
    to_obs = -geo
    cos_alpha = np.sum(to_sun * to_obs, axis=1) / (r * delta)
    alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))

    ecl_lon = np.degrees(np.arctan2(geo[:, 1], geo[:, 0])) % 360.0
    ecl_lat = np.degrees(np.arcsin(np.clip(geo[:, 2] / delta, -1, 1)))

    geo_eq = _ecliptic_to_equatorial(geo)
    ra = np.degrees(np.arctan2(geo_eq[:, 1], geo_eq[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(geo_eq[:, 2] / delta, -1, 1)))

    return Geometry(ra_deg=ra, dec_deg=dec, ecl_lon_deg=ecl_lon, ecl_lat_deg=ecl_lat,
                    r_au=r, delta_au=delta, alpha_deg=alpha, helio_xyz=obj_xyz,
                    los_unit=los)


def observe(elements: dict, epoch_jd: float, t_jd: float,
            earth_xyz: np.ndarray | None = None) -> Geometry:
    """Full observing geometry for all objects at a single time ``t_jd``."""
    obj = elements_to_helio_xyz(elements, epoch_jd, t_jd)        # (N, 3) ecliptic
    if earth_xyz is None:
        earth_xyz = earth_helio_xyz(t_jd)
    return geometry_from_positions(obj, np.broadcast_to(earth_xyz, obj.shape))
