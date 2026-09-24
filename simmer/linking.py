"""Moving-object tracklet linking -- the step that turns per-frame detections
into a *linked* detection eligible for the observed catalog.

A single-exposure detection above the sensitivity limit is not enough to enter a
moving-object catalog. A real survey must *link* several detections of the same
object, close in time, into a tracklet, then report it with a pipeline efficiency
below unity. Requiring only >=1 detection (or even >=N detections spread across
the whole survey) makes the efficiency function far too optimistic at small
sizes, because a faint object is easy to catch *once* but hard to catch enough
times within a single apparition to be linked.

We model linking with two ingredients:

1. **Tracklet window.** The object must be detected at least ``min_detections``
   times within a sliding time window of ``window_hr`` hours -- one apparition.
   (A near-ecliptic field is revisited many times over roughly a day as the
   survey scans past, so an object's detections cluster in time; the window
   isolates one such cluster.) Because each individual detection must clear the
   per-frame S/N cut, the number that land within one window falls steeply as an
   object fades -- so this requirement, not the per-frame limit alone, sets the
   small-size roll-off. The magnitude dependence is therefore *emergent*, not a
   separately tuned curve.
2. **Flat linking efficiency.** Objects that clear the tracklet requirement are
   kept with probability ``link_efficiency`` -- via a deterministic per-object
   draw. This models ONLY the residual WMOPS automated-processing loss (linker
   association, track-validation/quality cuts, de-duplication, region masking),
   *conditional* on a valid tracklet already existing. It deliberately does NOT
   re-apply flux/extraction losses: single-frame detectability (photometric noise
   + the bad-pixel/blend decimation) is handled upstream in :mod:`simmer.detect`,
   and the tracklet-formation loss is ingredient 1 above. Folding any of those
   into this factor would double-count them.

Published basis (no mission-internal sources): the WISE All-Sky Release
Explanatory Supplement, Sec. 4.5 (WMOPS), states a tracklet required "a minimum
of 5 detections from different scans" above a S/N threshold of ~4.5, and reports
90% automated-processing completeness (92% in pipeline V3.5) "for the objects
that met the original criteria"; see also Mainzer et al. 2011 (ApJ 731, 53;
ApJ 743, 156). The literature gives this efficiency vs *flux*, not vs apparent
motion rate (rate is a top-hat acceptance window), so ``link_efficiency`` is a
rate-independent constant. Those numbers motivate the ``min_detections`` and
``link_efficiency`` defaults but are supplied via ``SimConfig``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import seeding

# Distinct hashing stream so the link-keep draw is uncorrelated with the
# rotation / bad-pixel / photometric-noise draws (see simmer.seeding).
STREAM_LINK = np.uint64(5)


def max_in_window(times_days: np.ndarray, window_days: float) -> int:
    """Largest number of points falling within any window of the given width.

    Two-pointer scan over the sorted times (input need not be sorted). O(n log n)
    from the sort. A window is closed on both ends: points exactly ``window_days``
    apart count together.
    """
    t = np.sort(np.asarray(times_days, dtype=float))
    n = t.size
    if n == 0:
        return 0
    best = 1
    j = 0
    for i in range(n):
        while t[i] - t[j] > window_days:
            j += 1
        best = max(best, i - j + 1)
    return best


def tracklet_sizes(detected: pd.DataFrame, window_hr: float) -> pd.Series:
    """Per-object max detections within any ``window_hr`` window.

    ``detected`` must already be filtered to detected frames (and to the linking
    band, if any) and carry ``id`` and ``mjd`` columns. Returns a Series indexed
    by ``id``. Only objects present in ``detected`` appear.
    """
    window_days = window_hr / 24.0
    if detected.empty:
        return pd.Series(dtype=int)
    return detected.groupby("id")["mjd"].apply(
        lambda m: max_in_window(m.to_numpy(), window_days)).astype(int)


def _great_circle_deg(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Angular separation (deg) between sky positions, via the haversine form."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dr, dd = r2 - r1, d2 - d1
    a = np.sin(dd / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin(dr / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def apparent_rates(detected: pd.DataFrame, max_dt_days: float) -> pd.Series:
    """Per-object apparent sky-motion rate (deg/day), robust to survey gaps.

    For each object, sort detections by time and take the median rate over adjacent
    pairs separated by less than ``max_dt_days`` -- i.e. within one apparition, so
    the estimate is the true instantaneous rate and not aliased by the object's
    motion across a months-long gap between apparitions. Needs ``id``, ``mjd``,
    ``ra_deg`` and ``dec_deg`` columns; returns an empty Series (gate skipped) if
    any are absent. Objects with no close pair get NaN.
    """
    need = {"id", "mjd", "ra_deg", "dec_deg"}
    if detected.empty or not need.issubset(detected.columns):
        return pd.Series(dtype=float)

    def _rate(g: pd.DataFrame) -> float:
        g = g.sort_values("mjd")
        t = g["mjd"].to_numpy()
        dt = np.diff(t)
        if dt.size == 0:
            return np.nan
        sep = _great_circle_deg(g["ra_deg"].to_numpy()[:-1], g["dec_deg"].to_numpy()[:-1],
                                g["ra_deg"].to_numpy()[1:], g["dec_deg"].to_numpy()[1:])
        close = (dt > 0) & (dt <= max_dt_days)
        return float(np.median(sep[close] / dt[close])) if close.any() else np.nan

    return detected.groupby("id")[["mjd", "ra_deg", "dec_deg"]].apply(_rate)


def _int_key(ids: np.ndarray) -> np.ndarray:
    """Stable non-negative integer key per object id for the hashing PRNG."""
    a = np.asarray(ids)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.uint64)
    codes, _ = pd.factorize(a)                     # deterministic for a fixed id set
    return codes.astype(np.uint64)


def link_keep(ids: np.ndarray, link_efficiency: float, seed: int) -> np.ndarray:
    """Boolean keep-mask applying a flat ``link_efficiency`` per object.

    Deterministic and keyed by object identity (not draw order), so the result is
    independent of how a run is batched across workers.
    """
    if link_efficiency >= 1.0:
        return np.ones(len(ids), dtype=bool)
    u = seeding.hash01(seed, STREAM_LINK, _int_key(ids))
    return u < link_efficiency
