"""Configuration for the Simmer survey-detection pipeline.

``SimConfig`` plays the same role here that ``SynthConfig`` plays in SynthPop and
``AnalysisConfig`` plays in PyLEADER: a single, documented, serialisable object
that replaces a notebook "top cell". One ``SimConfig`` fully specifies one
survey-replay run against one synthetic catalog.

Nothing in this module does science; it only declares *what* to simulate and
*where the inputs live*. The stage modules (``geometry``, ``thermal``,
``lightcurve``, ``detect``) and the survey backends consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence
import json


@dataclass
class SimConfig:
    """Everything needed to replay one survey against one synthetic catalog."""

    # --- identity / input -------------------------------------------------
    name: str                                     # run name, e.g. "Hygiea_neowise"
    catalog: Path = Path("")                      # <name>_synthpop.csv from SynthPop

    # --- survey selection -------------------------------------------------
    # Key into the survey registry (see simmer.surveys). The cryo WISE survey
    # is the only fully-implemented backend; panstarrs/atlas are stubs.
    survey: str = "neowise_cryo"
    # Which bands to simulate. Defaults are set per-survey when left None;
    # for neowise_cryo that is ("W3", "W4").
    bands: Optional[Sequence[str]] = None

    # --- ephemeris time anchor -------------------------------------------
    # SynthPop draws mean anomaly uniformly with NO real epoch, so we must
    # anchor the elements at a chosen reference epoch (TDB JD). The orbits are
    # statistical, so this only fixes the rotational/orbital phase. Default is
    # the start of the cryo WISE survey.
    epoch_jd: float = 2455197.5                   # 2010-01-01 00:00 TDB

    # --- field-of-view stage ---------------------------------------------
    max_ecl_lat_deg: float = 30.0                 # drop frames above this |ecl lat|
    coarse_grid_days: float = 1.0                 # coarse-match propagation grid (days)
    # Coarse pre-filter net = frame half-diagonal + intra-bin sky motion. Must be
    # generous enough that no true detection is dropped before the fine refine.
    frame_half_diag_deg: float = 0.56             # ~half-diagonal of the 47' WISE frame
    max_sky_motion_deg_per_day: float = 1.0       # generous MBA apparent rate for the pad
    fov_match_radius_deg: float = 0.40            # fallback circular footprint if no corners
    # Edge exclusion: reject objects within this many pixels of the frame edge
    # (flagged as compromised photometry in the single-exposure source catalog).
    # PER-BAND, because plate scales differ: W1-W3 = 2.75"/px on a 1016-px
    # detector, W4 = 5.5"/px on 508 px. So 9 W3-px = 24.75" while 5 W4-px = 27.5".
    # Already correct should W4 ever become a primary analysis band.
    edge_reject_pixels: Optional[dict] = None     # default set in __post_init__
    band_frame_pixels: Optional[dict] = None       # detector size (px) per band

    # --- detection stage --------------------------------------------------
    snr_threshold: float = 5.0                    # detection significance
    # An object counts as "detected" (available in the observed sample) only if
    # it is caught in at least this many frames in `detection_band`. The real
    # NEOWISE observed sample (the .obs files / PDS SBN thermal-fit catalog that
    # N_obs is drawn from) requires >= 5 quality single-exposure detections to be
    # linked and thermally fit (Mainzer et al. 2011, ApJ 736, 100; matches the
    # upstream PyLEADER ObsBuildConfig.min_obs = 5, W3 filterpriority). Using 1
    # here makes eta far too optimistic at small sizes, since a single-frame
    # detection is much easier than making the catalog.
    min_detections: int = 5
    detection_band: Optional[str] = "W3"          # band the min_detections count is in (None = any)
    # Tracklet linking (WMOPS-style). A detection enters the observed catalog only
    # if the object forms a linkable tracklet: >= min_detections detections within
    # ONE apparition -- a sliding window of `tracklet_window_hr` hours -- not merely
    # min_detections spread over the whole survey. This is what makes eta roll off
    # steeply at small sizes (a faint object is caught once easily but rarely enough
    # times within a single window to link). Set tracklet_window_hr=None to fall
    # back to the looser survey-wide count.
    #
    # `link_efficiency` is the RESIDUAL WMOPS automated-processing completeness for
    # objects that already meet the tracklet criteria -- the linker/track-validation
    # loss ONLY (velocity-space association, quality/linearity cuts, de-duplication,
    # region masking). It is deliberately NOT the flux/extraction loss: single-frame
    # detectability is modeled upstream in detect.py (the smooth SNR curve + the
    # `bad_pixel_fraction` decimation, which already covers bad pixels/cosmic rays/
    # blends). Keeping the flux loss out of this factor avoids double-penalizing it.
    # Published basis: WISE All-Sky Explanatory Supplement Sec. 4.5 -- WMOPS reached
    # 90% automated-processing completeness, raised to 92% in pipeline V3.5, "for the
    # objects that met the original criteria" (>=5 detections from different scans,
    # S/N > 4.5); Mainzer et al. 2011 (ApJ 731, 53; ApJ 743, 156). NOTE: the
    # literature does NOT resolve this efficiency vs apparent motion rate -- rate
    # enters only as the top-hat window below -- so this stays a flat (rate-
    # independent) factor rather than a curve. Slow MBAs (~0.2 deg/day, 10-12
    # detections/apparition) clear the tracklet step with margin, so the processing
    # residual is the operative loss and 0.90-0.92 is the published range.
    tracklet_window_hr: Optional[float] = 36.0    # one apparition; None = whole survey
    link_efficiency: float = 0.92                 # WMOPS V3.5 processing completeness (residual only)
    # Tracklet sky-motion gate (deg/day): an object must move within this apparent-
    # rate band to be linked -- too slow reads as an inertial (stationary) source,
    # too fast trails out of the tracklet. WMOPS was sensitive to 0.06-3.2 deg/day
    # (Mainzer et al. 2011, ApJ 743, 156). Rarely limiting for main-belt rates
    # (~0.2-0.3 deg/day); set either bound to None to disable that side of the gate.
    min_motion_deg_day: Optional[float] = 0.06
    max_motion_deg_day: Optional[float] = 3.2
    bad_pixel_fraction: float = 0.01              # drop this fraction: compromised photometry (bad pixels, blends)
    # Per-band 5-sigma single-frame limits (Jy). None -> survey backend default.
    sensitivity_jy: Optional[dict] = None
    # Photometric noise: scatter each measured flux by sigma(F) and detect on the
    # scattered value, so completeness near the limit is a smooth error function
    # rather than a hard cut. sigma(F) = sqrt(sigma_floor^2 + (frac_err*F)^2), with
    # sigma_floor = (5-sigma limit)/5 the single-frame noise floor.
    photometric_noise: bool = True
    photometric_frac_err: float = 0.03            # bright-end fractional error floor

    # --- frame metadata cache --------------------------------------------
    # Local cache of survey frame pointings/timestamps (parquet/feather/csv).
    # If absent and fetch_frames is True, the backend retrieves it from IRSA.
    frames_cache: Optional[Path] = None
    fetch_frames: bool = False                    # allow network retrieval if cache missing

    # --- thermal model ----------------------------------------------------
    emissivity: float = 0.9                       # IR emissivity for NEATM
    slope_G: float = 0.15                         # H-G phase slope -> Bond albedo phase integral
    # Use the precomputed (T_ss, alpha) surface-integral lookup table for the
    # in-band flux (O(1)/detection); set False for the exact per-detection
    # quadrature (identical to ~0.1%).
    use_flux_lut: bool = True

    # --- rotational lightcurve -------------------------------------------
    # Each object is assigned a rotation period (log-uniform over this range)
    # and a fixed initial phase, so detections are phase-linked across the
    # cadence. The lightcurve shape is the rotating-ellipsoid projected area
    # (Surdej & Surdej 1978) -- intrinsically double-peaked, no shape knob.
    rot_period_min_hr: float = 2.2
    rot_period_max_hr: float = 20.0
    # Set False to treat objects as uniform spheres (no rotational variation):
    # flux = the NEATM mean, no lightcurve. Useful for isolating the effect of
    # the lightcurve assumption on the detected-object distribution.
    apply_lightcurve: bool = True

    # --- output -----------------------------------------------------------
    out_dir: Path = Path("simmer_out")
    out_format: str = "csv"                       # "csv" or "parquet"
    seed: int = 0
    efficiency_n_bins: int = 20                    # diameter bins for eta(D)

    def __post_init__(self) -> None:
        for p in ("catalog", "frames_cache", "out_dir"):
            v = getattr(self, p)
            if v is not None and v != "":
                setattr(self, p, Path(v))
        if self.bands is not None:
            self.bands = list(self.bands)
        if self.edge_reject_pixels is None:
            self.edge_reject_pixels = {"W1": 9, "W2": 9, "W3": 9, "W4": 5}
        if self.band_frame_pixels is None:
            self.band_frame_pixels = {"W1": 1016, "W2": 1016, "W3": 1016, "W4": 508}

    def coarse_radius_deg(self) -> float:
        """Angular radius of the coarse pre-filter net (deg)."""
        return self.frame_half_diag_deg + \
            self.max_sky_motion_deg_per_day * self.coarse_grid_days / 2.0

    # --- (de)serialisation -------------------------------------------------
    def to_json(self, path: Path) -> None:
        d = asdict(self)
        d = json.loads(json.dumps(d, default=str))  # Path -> str
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "SimConfig":
        return cls(**json.loads(Path(path).read_text()))
