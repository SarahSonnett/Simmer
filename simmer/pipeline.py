"""``run_simmer`` -- the end-to-end driver, mirroring SynthPop's ``run_synthpop``.

Stages, in order:

1. **Field of view** -- load survey frames (near-ecliptic prefiltered), propagate
   objects on a coarse grid, KD-tree cross-match (:mod:`simmer.geometry`).
2. **Thermal flux** -- NEATM in-band flux per (object, frame) hit
   (:mod:`simmer.thermal`).
3. **Lightcurve** -- modulate by a random-phase rotational lightcurve whose
   amplitude follows Sheppard & Jewitt (2004) (:mod:`simmer.lightcurve`).
4. **Detection** -- sensitivity cut + 1% bad-pixel decimation
   (:mod:`simmer.detect`).
5. **Write** -- per-detection table, per-object detection summary, the detection
   efficiency eta(D) with Wilson intervals (:mod:`simmer.efficiency`), and
   provenance.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import SimConfig
from . import catalog_io, geometry, thermal, lightcurve, detect, efficiency, linking
from .surveys.base import get_survey


def simulate(cfg: SimConfig, frames: pd.DataFrame | None = None,
             verbose: bool = True, catalog: pd.DataFrame | None = None):
    """Run the compute stages (FOV -> flux -> lightcurve -> detection).

    Returns ``(catalog, hits, sensitivity)`` in memory without writing anything.
    ``run_simmer`` wraps this then writes; the ensemble driver
    (:mod:`simmer.ensemble`) uses it to get per-object detections for many member
    catalogs without emitting M sets of output files. Pass ``catalog`` directly
    (already loaded/perturbed) to bypass loading from ``cfg.catalog`` -- the
    ensemble's science-floor uses this to shift the beaming column per member.
    """
    survey = get_survey(cfg.survey)

    def log(msg):
        if verbose:
            print(f"[simmer:{cfg.name}] {msg}")

    # Ingest --------------------------------------------------------------
    if catalog is None:
        log(f"loading catalog {cfg.catalog}")
        catalog = catalog_io.load_catalog(cfg.catalog)
    log(f"{len(catalog)} synthetic objects")

    if frames is None:
        log(f"loading {cfg.survey} frames")
        frames = survey.load_frames(cfg)
    bands = survey.bands(cfg)
    sensitivity = survey.sensitivity(cfg)
    log(f"{len(frames)} near-ecliptic frames; bands={bands}")

    # 1. field of view ---------------------------------------------------
    log("cross-matching objects to frames")
    hits = geometry.match_frames(catalog, frames, cfg)
    log(f"{len(hits)} (object, frame) associations")

    if hits.empty:
        return catalog, hits, sensitivity

    # 2. thermal flux -----------------------------------------------------
    flux_fn = thermal.flux_band_wise_lut if cfg.use_flux_lut else thermal.flux_band_wise
    log("computing NEATM RSR-convolved, color-corrected in-band fluxes"
        + (" (lookup table)" if cfg.use_flux_lut else ""))
    obj = catalog.iloc[hits["obj_idx"].to_numpy()]
    flux = np.zeros(len(hits))
    for band in bands:
        m = (hits["band"] == band).to_numpy()
        if not m.any():
            continue
        flux[m] = flux_fn(
            obj["diam_km"].to_numpy()[m], obj["pV"].to_numpy()[m],
            obj["eta"].to_numpy()[m], hits["r_au"].to_numpy()[m],
            hits["delta_au"].to_numpy()[m], hits["alpha_deg"].to_numpy()[m],
            band, G=cfg.slope_G, emissivity=cfg.emissivity)
    hits = hits.assign(flux_thermal_jy=flux)

    # 3. lightcurve -------------------------------------------------------
    log("applying rotational lightcurve modulation (rotating ellipsoid)"
        if cfg.apply_lightcurve else "spherical objects (no lightcurve)")
    los = hits[["los_x", "los_y", "los_z"]].to_numpy()
    b_over_a = obj["b_over_a"].to_numpy()
    theta = lightcurve.aspect_angle_deg(
        obj["pole_beta_deg"].to_numpy(), obj["pole_lambda_deg"].to_numpy(), los)
    amp = lightcurve.amplitude_mag(b_over_a, theta)

    # Per-object rotation, fixed across the survey, then phase at each frame time.
    rot = lightcurve.assign_rotation(len(catalog), cfg)
    oi = hits["obj_idx"].to_numpy()
    epoch_mjd = cfg.epoch_jd - 2400000.5
    phase = lightcurve.rotational_phase(
        hits["mjd"].to_numpy(), epoch_mjd, rot.period_hr[oi], rot.phase0[oi])

    modulated = (lightcurve.modulate_flux(flux, b_over_a, theta, phase)
                 if cfg.apply_lightcurve else flux)
    hits = hits.assign(
        aspect_deg=theta,
        lc_amp_mag=amp,
        rot_period_hr=rot.period_hr[oi],
        rot_phase=phase,
        flux_jy=modulated,
    )

    # 4. detection --------------------------------------------------------
    log("applying photometric noise + sensitivity cut + bad-pixel decimation"
        if cfg.photometric_noise else "applying sensitivity cut + bad-pixel decimation")
    hits = detect.apply_detection(hits, sensitivity, cfg)
    n_det = int(hits["detected"].sum())
    log(f"{n_det} detections of {hits['id'].nunique()} distinct objects")

    return catalog, hits, sensitivity


def object_summary(catalog: pd.DataFrame, hits: pd.DataFrame,
                   min_detections: int = 1, detection_band=None,
                   tracklet_window_hr=None, link_efficiency: float = 1.0,
                   seed: int = 0, min_motion_deg_day=None,
                   max_motion_deg_day=None) -> pd.DataFrame:
    """Per-object detection summary: ``id``, ``n_detections``, ``tracklet_len``,
    ``detected``.

    ``n_detections`` counts detected frames in ``detection_band`` (any band if
    None). ``tracklet_len`` is the most detections landing within any window of
    ``tracklet_window_hr`` hours -- one apparition -- or, when that is None, just
    ``n_detections`` (the looser survey-wide count). An object is ``detected``
    (i.e. enters the observed sample) when ``tracklet_len >= min_detections``, its
    apparent sky-motion is within ``[min_motion_deg_day, max_motion_deg_day]`` (gate
    skipped when either bound is None or sky positions are absent), and it survives a
    flat ``link_efficiency`` keep. See :mod:`simmer.linking` and the corresponding
    ``SimConfig`` fields.
    """
    detected = hits[hits["detected"]] if "detected" in hits and not hits.empty \
        else hits.iloc[0:0]
    if detection_band is not None and not detected.empty:
        detected = detected[detected["band"] == detection_band]
    counts = detected.groupby("id").size() if not detected.empty else pd.Series(dtype=int)
    summary = catalog[["id"]].copy()
    summary["n_detections"] = summary["id"].map(counts).fillna(0).astype(int)

    if tracklet_window_hr is None:
        summary["tracklet_len"] = summary["n_detections"]
    else:
        sizes = linking.tracklet_sizes(detected, tracklet_window_hr)
        summary["tracklet_len"] = summary["id"].map(sizes).fillna(0).astype(int)

    linked = summary["tracklet_len"].to_numpy() >= min_detections
    if min_motion_deg_day is not None or max_motion_deg_day is not None:
        dt_days = (tracklet_window_hr or 24.0) / 24.0
        rates = summary["id"].map(linking.apparent_rates(detected, dt_days))
        lo = -np.inf if min_motion_deg_day is None else min_motion_deg_day
        hi = np.inf if max_motion_deg_day is None else max_motion_deg_day
        # NaN rate (no close pair) -> do not reject; those objects fail on tracklet_len anyway
        linked &= ((rates >= lo) & (rates <= hi)).fillna(True).to_numpy()
    linked &= linking.link_keep(summary["id"].to_numpy(), link_efficiency, seed)
    summary["detected"] = linked
    return summary


def run_simmer(cfg: SimConfig, frames: pd.DataFrame | None = None,
               verbose: bool = True) -> Path:
    """Replay a survey against a SynthPop catalog; write detections to disk.

    Pass ``frames`` directly to bypass the survey backend's loader (useful for
    tests / offline runs with ``surveys.neowise.synthetic_frames``).
    """
    def log(msg):
        if verbose:
            print(f"[simmer:{cfg.name}] {msg}")

    catalog, hits, sensitivity = simulate(cfg, frames, verbose=verbose)
    return _write(cfg, hits, catalog, sensitivity, log)


def _write(cfg, hits, catalog, sensitivity, log) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / cfg.name

    # Per-detection table.
    if cfg.out_format == "parquet":
        det_path = stem.with_name(f"{cfg.name}_detections.parquet")
        hits.to_parquet(det_path, index=False)
    else:
        det_path = stem.with_name(f"{cfg.name}_detections.csv")
        hits.to_csv(det_path, index=False, float_format="%.6g")

    # Per-object detection summary (how many times each object was detected).
    summary = object_summary(catalog, hits, cfg.min_detections, cfg.detection_band,
                             cfg.tracklet_window_hr, cfg.link_efficiency, cfg.seed,
                             cfg.min_motion_deg_day, cfg.max_motion_deg_day)
    summary.to_csv(stem.with_name(f"{cfg.name}_object_summary.csv"), index=False)

    # Detection efficiency eta(D) with Wilson intervals -- the debiasing product.
    eff = efficiency.efficiency_function(
        catalog["diam_km"].to_numpy(), summary["detected"].to_numpy(),
        n_bins=cfg.efficiency_n_bins)
    eff.to_csv(stem.with_name(f"{cfg.name}_efficiency.csv"), index=False)

    prov = {
        "name": cfg.name, "survey": cfg.survey, "seed": cfg.seed,
        "epoch_jd": cfg.epoch_jd, "sensitivity_jy": sensitivity,
        "n_objects": int(len(catalog)),
        "n_associations": int(len(hits)),
        "n_detections": int(hits["detected"].sum()) if "detected" in hits else 0,
        "n_objects_detected": int(summary["detected"].sum()),
    }
    # namespaced to avoid clobbering SynthPop sidecars (SynthPop note 2026-08-23)
    (stem.with_name(f"{cfg.name}_simmer_provenance.json")).write_text(
        json.dumps(prov, indent=2, default=str))
    cfg.to_json(stem.with_name(f"{cfg.name}_simmer_config.json"))

    log(f"wrote detections -> {det_path}")
    return det_path
