"""Fully-cryogenic WISE survey backend (2010), W3/W4 thermal bands.

This is the primary, fully-wired backend. The 12 um (W3) and 22 um (W4) bands
only exist during the fully-cryogenic phase (2010 Jan-Aug); the post-2013
NEOWISE *reactivation* has only W1/W2 and is a separate backend to add later.

Frame metadata comes from the IRSA WISE "Scan/Frame Metadata" tables, queried
once over TAP and cached locally (``cfg.frames_cache``). Network access is only
attempted when ``cfg.fetch_frames`` is True; otherwise an existing cache is
required. For tests and offline development, :func:`synthetic_frames` produces a
small near-ecliptic frame set with the same schema.

Sensitivity defaults are approximate single-exposure 5-sigma point-source limits
(WISE Explanatory Supplement, cryo phase); refine per your analysis.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .base import Survey, register

# IRSA TAP service and the cryo 4-band frame metadata table.
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
FRAME_TABLE = "allsky_4band_p1bs_frm"   # per-frame (level-1b) metadata, cryo 4-band


@register("neowise_cryo")
class CryoWISE(Survey):
    default_bands = ("W3", "W4")
    # Single-EXPOSURE 5-sigma point-source sensitivity (Jy).
    #
    # "Single exposure" = ONE L1b frame: a single ~8.8 s (W3/W4) integration read
    # out by sampling up the ramp -- the 9 non-destructive reads are combined by
    # weights (-4..+4) into one on-board slope estimate = that exposure's flux.
    # That is NOT a coadd of sub-exposures; it is the optimal readout of one
    # integration. Asteroids are detected in these single frames (WMOPS; Mainzer
    # et al. 2011a) and CANNOT be stacked -- they move ~arcmin between frames --
    # so the single-frame limit is the right one, NOT the deeper catalog/coadd
    # depth (the ~8-frame coadd used for static sources).
    #
    # W3/W4 single-EXPOSURE 5-sigma depth. Moving asteroids are detected in single
    # exposures and do NOT co-add across frames, so the relevant limit is the
    # single-exposure depth = (measured coadd depth) x sqrt(N_coadd), undoing the
    # coaddition. We anchor on the MEASURED near-ecliptic coadd depths from the WISE
    # All-Sky Release Explanatory Supplement Sec. 6.3a -- W3 = 0.86 mJy, W4 = 5.4 mJy
    # -- at their measured depth-of-coverage N ~ 11 (these are ecliptic, high-zodiacal
    # values, i.e. the shallow regime where main-belt objects are observed). That
    # gives single-exposure W3 ~2.85 mJy, W4 ~17.9 mJy. This is ~1.25x shallower than
    # the earlier back-projection from Wright et al. 2010's idealized 8-frame coadd
    # (44/93/800/5500 uJy x sqrt(8) -> W3 2.26, W4 15.6 mJy): the ecliptic achieved
    # ~11 coverages (not 8), and the measured coadd fell slightly short of ideal
    # sqrt(N) scaling. Uncertain at the ~1.2x level (coaddition efficiency). Refs:
    # Wright et al. 2010 (AJ 140, 1868); WISE All-Sky Expl. Supp. Sec. 6.3a & 2.2.
    # W1/W2 retained on the Wright 8-coadd basis (not used for the W3/W4 thermal fit).
    # Values vary with ecliptic latitude; override via SimConfig.sensitivity_jy.
    default_sensitivity_jy = {
        "W1": 44e-6 * 8 ** 0.5,      # ~1.24e-4 Jy
        "W2": 93e-6 * 8 ** 0.5,      # ~2.63e-4 Jy
        "W3": 0.86e-3 * 11 ** 0.5,   # ~2.85e-3 Jy (measured coadd x sqrt(N~11))
        "W4": 5.4e-3 * 11 ** 0.5,    # ~1.79e-2 Jy
    }

    def load_frames(self, cfg) -> pd.DataFrame:
        """Read the local pointings file (built once by scripts/build_pointings.py).

        Network access is only attempted as a fallback when ``cfg.fetch_frames``
        is True and the cache is missing; the recommended path is to build the
        local file once and ingest it on every run.
        """
        cache = Path(cfg.frames_cache) if cfg.frames_cache else None
        if cache and cache.exists():
            frames = _read_cache(cache)
        elif cfg.fetch_frames:
            frames = query_irsa(self.bands(cfg), lat_max=cfg.max_ecl_lat_deg)
            if cache:
                _write_cache(frames, cache)
        else:
            raise FileNotFoundError(
                "no local pointings file. Build it once with\n"
                "  python scripts/build_pointings.py --survey neowise_cryo "
                "--out data/cryo_pointings.parquet\n"
                "then point cfg.frames_cache at it. (Or set fetch_frames=True to "
                "pull inline, or use neowise.synthetic_frames(...) for offline tests.)"
            )
        return _prefilter(frames, cfg)


# Per-band "was this band processed on this frame?" flags (0/1). A WISE frame
# images all four bands at ONE pointing/time, so bands share ra/dec/mjd and we
# just attach a band label per processed band. (Verified against the live IRSA
# TAP schema 2026-07-14; qual_frame takes values {0, 5, 10}.)
_BAND_RUN = {"W1": "w1bandrun", "W2": "w2bandrun", "W3": "w3bandrun", "W4": "w4bandrun"}


def _corner_cols(band: str):
    """IRSA per-band footprint-corner column names for a band, e.g. W3 ->
    (w3ra1..w3ra4), (w3dec1..w3dec4). Corners are in perimeter order."""
    n = band[1]
    return [f"w{n}ra{i}" for i in (1, 2, 3, 4)], [f"w{n}dec{i}" for i in (1, 2, 3, 4)]


def query_irsa(bands=("W3", "W4"), lat_max=None, qual_min=5, top=None,
               maxrec=5_000_000) -> pd.DataFrame:
    """Query IRSA TAP for cryo 4-band frame pointings, timestamps and footprints.

    Dependency-light (requests + the TAP sync endpoint). Selects frame boresight
    ra/dec, mjd and the per-band footprint corners for good-quality frames
    (``qual_frame >= qual_min``), and explodes to one row per (frame, band) for
    the requested bands that were processed (``w{n}bandrun == 1``). An optional
    server-side near-ecliptic cut (``|w1elat| < lat_max``, degrees) shrinks the
    ~1.5M-frame download. Returns the engine schema: ``mjd``, ``ra``, ``dec``,
    ``band``, and footprint corners ``ra1..ra4``/``dec1..dec4``.
    """
    import io
    import requests

    select = ["ra", "dec", "mjd"]
    for b in bands:
        rc, dc = _corner_cols(b)
        select += [_BAND_RUN[b], *rc, *dc]
    where = [f"qual_frame >= {qual_min}"]
    if lat_max is not None:
        where.append(f"w1elat BETWEEN {-abs(lat_max)} AND {abs(lat_max)}")
    top_clause = f"TOP {int(top)} " if top else ""
    adql = (f"SELECT {top_clause}{', '.join(select)} "
            f"FROM {FRAME_TABLE} WHERE " + " AND ".join(where))
    resp = requests.get(f"{IRSA_TAP}/sync", params={
        "QUERY": adql, "FORMAT": "CSV", "LANG": "ADQL", "MAXREC": maxrec},
        timeout=1800)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))

    out = []
    for band in bands:
        rc, dc = _corner_cols(band)
        m = raw[_BAND_RUN[band]] == 1
        sub = pd.DataFrame({
            "mjd": raw.loc[m, "mjd"].to_numpy(),
            "ra": raw.loc[m, "ra"].to_numpy(),
            "dec": raw.loc[m, "dec"].to_numpy(),
            "band": band,
        })
        for i, col in enumerate(rc, start=1):
            sub[f"ra{i}"] = raw.loc[m, col].to_numpy()
        for i, col in enumerate(dc, start=1):
            sub[f"dec{i}"] = raw.loc[m, col].to_numpy()
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def build_pointings(out_path, bands=("W3", "W4"), lat_max: float = 35.0,
                    qual_min: int = 5):
    """Pull frame pointings from IRSA once and write a local pointings file.

    The near-ecliptic cut (|ecliptic latitude| < ``lat_max``, a little wider than
    the MBA cut so runtime ``max_ecl_lat_deg`` can still vary) is applied
    server-side to keep the download small. Returns the written path.
    """
    out_path = Path(out_path)
    frames = query_irsa(bands, lat_max=lat_max, qual_min=qual_min)
    _write_cache(frames, out_path)
    return out_path


def synthetic_frames(n_scans: int = 20000, bands=("W3", "W4"), seed: int = 0,
                     mjd_start: float = 55197.0, days: float = 200.0,
                     lat_max: float = 29.0) -> pd.DataFrame:
    """Toy near-ecliptic frame set with the real schema, for offline use/tests.

    This is a *coverage* toy, NOT a faithful cadence model: it densely tiles the
    near-ecliptic band (full ecliptic longitude, |lat| < ``lat_max``) at random
    times so that synthetic objects get caught and stages 2-4 are exercised. The
    real WISE scan geometry/cadence comes from IRSA frame metadata; do not use
    this for science.
    """
    rng = np.random.default_rng(seed)
    mjd = mjd_start + rng.random(n_scans) * days
    ecl_lon = rng.uniform(0.0, 360.0, n_scans)
    ecl_lat = rng.uniform(-lat_max, lat_max, n_scans)

    # Ecliptic lon/lat -> equatorial ra/dec.
    obl = np.radians(23.43928)
    lam, bet = np.radians(ecl_lon), np.radians(ecl_lat)
    x = np.cos(bet) * np.cos(lam)
    y = np.cos(bet) * np.sin(lam)
    z = np.sin(bet)
    ye = np.cos(obl) * y - np.sin(obl) * z
    ze = np.sin(obl) * y + np.cos(obl) * z
    ra = np.degrees(np.arctan2(ye, x)) % 360.0
    dec = np.degrees(np.arcsin(ze))

    rows = []
    for band in bands:
        rows.append(pd.DataFrame({"mjd": mjd, "ra": ra, "dec": dec, "band": band}))
    return pd.concat(rows, ignore_index=True)


def _ecliptic_latitude(ra_deg, dec_deg) -> np.ndarray:
    """Ecliptic latitude (deg) from equatorial ra/dec (deg)."""
    obl = np.radians(23.43928)
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    sin_b = np.cos(obl) * np.sin(dec) - np.sin(obl) * np.cos(dec) * np.sin(ra)
    return np.degrees(np.arcsin(np.clip(sin_b, -1, 1)))


def _prefilter(frames: pd.DataFrame, cfg) -> pd.DataFrame:
    """Keep only near-ecliptic frames and the requested bands."""
    ecl_lat = _ecliptic_latitude(frames["ra"].to_numpy(), frames["dec"].to_numpy())
    near = np.abs(ecl_lat) < cfg.max_ecl_lat_deg
    want = frames["band"].isin(cfg.bands if cfg.bands else CryoWISE.default_bands)
    return frames[near & want].reset_index(drop=True)


def _read_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".feather", ".ft"):
        return pd.read_feather(path)
    return pd.read_csv(path)


def _write_cache(frames: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frames.to_parquet(path, index=False)
    elif path.suffix in (".feather", ".ft"):
        frames.to_feather(path)
    else:
        frames.to_csv(path, index=False)
