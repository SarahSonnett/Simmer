"""Field-of-view cross-match: which objects fell in which frames.

The expensive stage, done as **coarse match -> fine refine**:

1. **Ecliptic-latitude prefilter** (survey backend / config): only frames with
   |ecliptic latitude| < ``max_ecl_lat_deg`` are kept.
2. **Coarse match** (positions only): propagate objects to each coarse time-bin
   centre, and KD-tree match frame boresights against object sky positions with a
   *generous* radius (frame half-diagonal + max intra-bin sky motion) so no true
   detection is dropped. Assembled into flat candidate arrays -- no per-frame
   Python object construction.
3. **Fine refine**: for each surviving candidate, recompute the object's exact
   position at the *true frame timestamp* and test it against the real frame
   footprint -- a point-in-quadrilateral test using the per-band frame corners,
   inset by ``edge_reject_pixels`` (compromised-photometry edge zone). Full
   observing geometry (r, Delta, alpha, LOS) is computed only for survivors.

If the frames lack corner columns (e.g. offline ``synthetic_frames``), the fine
test falls back to a circular footprint of radius ``fov_match_radius_deg``.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import ephemeris

_ELEMENT_KEYS = ("a_au", "e", "i_deg", "node_deg", "argperi_deg", "M_deg")
_CORNER_COLS = ("ra1", "dec1", "ra2", "dec2", "ra3", "dec3", "ra4", "dec4")
OUT_COLUMNS = ["id", "obj_idx", "frame_idx", "mjd", "band", "ra_deg", "dec_deg",
               "ecl_lat_deg", "r_au", "delta_au", "alpha_deg",
               "los_x", "los_y", "los_z"]


def _radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def _tangent_plane(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic standard coordinates (deg) about tangent point(s) (ra0, dec0).

    Broadcasts, so points can be (M,) about (M,) centres, or (M, 4) corners about
    (M, 1) centres. Exact enough over the 47' WISE field.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cosc
    return np.degrees(xi), np.degrees(eta)


def _point_in_quad(px, py, qx, qy):
    """Vectorized point-in-convex-quadrilateral. ``qx``/``qy`` are (M, 4), in
    perimeter order; ``px``/``py`` are (M,). Inside iff on the same side of all
    four edges (sign consistent for either winding)."""
    pos = np.ones(px.shape[0], dtype=bool)
    neg = np.ones(px.shape[0], dtype=bool)
    for k in range(4):
        k2 = (k + 1) % 4
        ex, ey = qx[:, k2] - qx[:, k], qy[:, k2] - qy[:, k]
        cx, cy = px - qx[:, k], py - qy[:, k]
        cross = ex * cy - ey * cx
        pos &= cross >= 0
        neg &= cross <= 0
    return pos | neg


def _inset_scale(band: np.ndarray, cfg) -> np.ndarray:
    """Per-candidate quad shrink factor = 1 - edge_px / (frame_px / 2), by band."""
    scale = np.ones(band.shape[0])
    for b in np.unique(band):
        e = cfg.edge_reject_pixels.get(b, 0)
        n = cfg.band_frame_pixels.get(b, 1016)
        scale[band == b] = 1.0 - e / (n / 2.0)
    return scale


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OUT_COLUMNS)


def _refine(cand_obj, cand_frame, jd, frames, elements, obj_ids,
            has_corners, fr_ra, fr_dec, cfg):
    """Fine footprint test on coarse candidates -> survivor associations.

    Operates on one batch of candidates (a single coarse time-bin's worth), so
    the large per-candidate arrays never span the whole survey at once. Returns
    the survivor rows (unsorted) or ``None`` if the batch has no survivors.
    """
    t_cand = jd[cand_frame]
    el_cand = {k: v[cand_obj] for k, v in elements.items()}
    obj_xyz = ephemeris.elements_to_helio_xyz(el_cand, cfg.epoch_jd, t_cand)
    ut, inv = np.unique(t_cand, return_inverse=True)
    earth_cand = ephemeris.earth_helio_xyz(ut)[inv]
    geom = ephemeris.geometry_from_positions(obj_xyz, earth_cand)

    ra0, dec0 = fr_ra[cand_frame], fr_dec[cand_frame]
    if has_corners:
        qra = np.stack([frames[f"ra{k}"].to_numpy()[cand_frame] for k in (1, 2, 3, 4)], axis=1)
        qdec = np.stack([frames[f"dec{k}"].to_numpy()[cand_frame] for k in (1, 2, 3, 4)], axis=1)
        px, py = _tangent_plane(geom.ra_deg, geom.dec_deg, ra0, dec0)
        qx, qy = _tangent_plane(qra, qdec, ra0[:, None], dec0[:, None])
        scale = _inset_scale(frames["band"].to_numpy()[cand_frame], cfg)
        cx, cy = qx.mean(axis=1, keepdims=True), qy.mean(axis=1, keepdims=True)
        qx = cx + scale[:, None] * (qx - cx)
        qy = cy + scale[:, None] * (qy - cy)
        inside = _point_in_quad(px, py, qx, qy)
    else:
        u_obj = _radec_to_unit(geom.ra_deg, geom.dec_deg)
        u_bore = _radec_to_unit(ra0, dec0)
        sep = np.degrees(np.arccos(np.clip(np.sum(u_obj * u_bore, axis=1), -1, 1)))
        inside = sep < cfg.fov_match_radius_deg

    if not inside.any():
        return None

    ci_o, ci_f = cand_obj[inside], cand_frame[inside]
    return {                                          # dict of arrays; framed once at the end
        "id": obj_ids[ci_o], "obj_idx": ci_o, "frame_idx": ci_f,
        "mjd": frames["mjd"].to_numpy()[ci_f], "band": frames["band"].to_numpy()[ci_f],
        "ra_deg": geom.ra_deg[inside], "dec_deg": geom.dec_deg[inside],
        "ecl_lat_deg": geom.ecl_lat_deg[inside],
        "r_au": geom.r_au[inside], "delta_au": geom.delta_au[inside],
        "alpha_deg": geom.alpha_deg[inside],
        "los_x": geom.los_unit[inside, 0], "los_y": geom.los_unit[inside, 1],
        "los_z": geom.los_unit[inside, 2],
    }


def match_frames(catalog: pd.DataFrame, frames: pd.DataFrame, cfg) -> pd.DataFrame:
    """Associate synthetic objects with survey frames (coarse match -> refine).

    ``frames`` columns: ``mjd``, ``ra``, ``dec`` (deg), ``band``, and optionally
    the per-band footprint corners ``ra1..ra4``/``dec1..dec4``. Returns one row
    per surviving (object, frame) association with the observing geometry.
    """
    if len(catalog) == 0 or len(frames) == 0:
        return _empty()

    frames = frames.reset_index(drop=True)            # row position = frame_idx
    elements = {k: catalog[k].to_numpy() for k in _ELEMENT_KEYS}
    obj_ids = catalog["id"].to_numpy()
    has_corners = all(c in frames.columns for c in _CORNER_COLS)

    # Coverage guard: an object reaches |ecliptic latitude| up to ~its inclination, so
    # if the population is more inclined than the frame-list latitude cut, its
    # high-latitude apparitions fall outside a latitude-clipped frame list and are
    # silently under-counted. Fires only when the cut is below the inclinations (e.g.
    # a high-i family on the near-ecliptic list) -- use the all-sky list or raise
    # cfg.max_ecl_lat_deg. See efficiency_systematics.md sec. 10.
    i_max = float(np.nanmax(elements["i_deg"])) if len(catalog) else 0.0
    if i_max > cfg.max_ecl_lat_deg:
        import warnings
        warnings.warn(
            "match_frames: catalog max inclination %.1f deg exceeds max_ecl_lat_deg "
            "%.1f deg; members above the latitude cut may be under-covered if the frame "
            "list is latitude-clipped (use the all-sky pointings or raise max_ecl_lat_deg)."
            % (i_max, cfg.max_ecl_lat_deg), stacklevel=2)

    jd = frames["mjd"].to_numpy() + 2400000.5
    fr_ra, fr_dec = frames["ra"].to_numpy(), frames["dec"].to_numpy()
    fr_units = _radec_to_unit(fr_ra, fr_dec)

    # --- coarse match (positions only) -----------------------------------
    t0 = jd.min()
    bin_idx = np.floor((jd - t0) / cfg.coarse_grid_days).astype(int)
    coarse_chord = 2.0 * np.sin(np.radians(cfg.coarse_radius_deg()) / 2.0)

    # Coarse-match and fine-refine one time-bin at a time, keeping only the
    # survivors of each bin. The big per-candidate arrays are thus bounded by a
    # single bin's candidates, not the whole survey -- flat memory in object
    # count -- while the final sort keeps the result batch-order-invariant.
    survivors = []
    for b in np.unique(bin_idx):
        sel = np.where(bin_idx == b)[0]               # global frame indices
        bin_jd = t0 + (b + 0.5) * cfg.coarse_grid_days
        earth = ephemeris.earth_helio_xyz(bin_jd)
        obj_units = ephemeris.geocentric_unit(elements, cfg.epoch_jd, bin_jd, earth)
        tree = cKDTree(obj_units)
        neigh = tree.query_ball_point(fr_units[sel], r=coarse_chord)
        counts = np.fromiter((len(x) for x in neigh), dtype=int, count=sel.size)
        total = int(counts.sum())
        if total == 0:
            continue
        cand_frame = np.repeat(sel, counts)
        cand_obj = np.fromiter(itertools.chain.from_iterable(neigh),
                               dtype=int, count=total)
        surv = _refine(cand_obj, cand_frame, jd, frames, elements, obj_ids,
                       has_corners, fr_ra, fr_dec, cfg)
        if surv is not None:
            survivors.append(surv)

    if not survivors:
        return _empty()
    out = pd.DataFrame({k: np.concatenate([s[k] for s in survivors])
                        for k in survivors[0]})
    # Stable order -> reproducible regardless of bin iteration / batching.
    return out.sort_values(["obj_idx", "frame_idx"], kind="stable").reset_index(drop=True)
