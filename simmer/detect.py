"""Detection filter: photometric noise, sensitivity cut, bad-pixel decimation.

A frame "detects" an object when its measured flux exceeds the per-band
``snr_threshold``-sigma single-frame limit. Two effects around that limit:

* **Photometric noise** (``cfg.photometric_noise``): the measured flux is the
  true flux plus Gaussian noise of width ``sigma(F) = sqrt(sigma_floor^2 +
  (frac_err*F)^2)``, where ``sigma_floor = (5-sigma limit)/5`` is the
  single-frame noise floor (background/read-noise limited) and ``frac_err`` is
  the bright-end systematic floor. Detection is then ``F_meas >= snr * sigma``,
  which turns the hard threshold into a smooth completeness curve -- a source at
  the limit is detected ~50% of the time. ``sigma_floor`` follows directly from
  the quoted 5-sigma sensitivities (Wright et al. 2010; single-exposure source
  photometry).

* **Compromised-photometry decimation** (``bad_pixel_fraction``, default 1%): a
  separate, small fraction of the above-threshold detections is dropped to
  approximate photometry lost to bad pixels, cosmic rays, or blending with
  background sources.

Both the noise draw and the bad-pixel draw are keyed by (object index, frame
index) via :mod:`simmer.seeding`, so they are deterministic per detection and
identical no matter how the run is batched across workers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import seeding


def apply_detection(hits: pd.DataFrame, sensitivity_jy: dict, cfg) -> pd.DataFrame:
    """Return ``hits`` annotated with ``flux_meas_jy``, ``above_threshold`` and
    ``detected``.

    ``hits`` must carry ``band``, ``flux_jy``, ``obj_idx`` and ``frame_idx``.
    ``sensitivity_jy`` maps band -> 5-sigma single-frame limit (Jy).
    """
    if hits.empty:
        return hits.assign(flux_meas_jy=pd.Series(dtype=float),
                           above_threshold=pd.Series(dtype=bool),
                           detected=pd.Series(dtype=bool))

    sens = hits["band"].map(sensitivity_jy).to_numpy(dtype=float)   # 5-sigma limit
    flux = hits["flux_jy"].to_numpy()
    obj_idx = hits["obj_idx"].to_numpy()
    frame_idx = hits["frame_idx"].to_numpy()

    if cfg.photometric_noise:
        from scipy.special import ndtri
        sigma_floor = sens / 5.0
        sigma = np.sqrt(sigma_floor ** 2 + (cfg.photometric_frac_err * flux) ** 2)
        u = np.clip(seeding.hash01(cfg.seed, seeding.STREAM_PHOT_NOISE, obj_idx, frame_idx),
                    1e-9, 1 - 1e-9)
        flux_meas = flux + ndtri(u) * sigma
        above = flux_meas >= cfg.snr_threshold * sigma
    else:
        flux_meas = flux
        above = flux >= sens * (cfg.snr_threshold / 5.0)

    keep = above.copy()
    if cfg.bad_pixel_fraction > 0:
        u = seeding.hash01(cfg.seed, seeding.STREAM_BAD_PIXEL, obj_idx, frame_idx)
        keep &= u >= cfg.bad_pixel_fraction

    out = hits.copy()
    out["flux_meas_jy"] = flux_meas
    out["above_threshold"] = above
    out["detected"] = keep
    return out
