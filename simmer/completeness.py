"""Observational completeness limit via the Hendler & Malhotra (2020) convention.

H&M (2020, PSJ 1, 75) define the completeness limit as the **center of the bin
with the greatest number of objects** -- the peak of the differential (incremental)
distribution -- deliberately agnostic about the underlying size distribution, with
0.25 mag bins and an uncertainty sigma = (bin width) / sqrt(n_peak).

Here it is applied to a family's **NEOWISE-observed** members (per the user's
choice: do not constrain sizes below what the same systematic IR survey actually
detected). We report the peak in absolute magnitude H (H&M's native variable) and
-- because the SFD fit lives in diameter -- also the peak of the observed diameter
distribution (their method in 0.05 dex log-D bins, = 0.25 mag), which is the floor
the SFD fit is restricted to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import designations, observed
from .observed import family_member_ids


def hm_peak(values, bin_width: float, log: bool = False):
    """H&M peak: center of the fullest bin, with sigma = bin_width / sqrt(n_peak).

    ``log=True`` bins in log10(values) (for diameters; use bin_width in dex).
    Returns ``(limit, sigma, n_peak)`` in the *original* units.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    x = np.log10(v[v > 0]) if log else v
    edges = np.arange(np.floor(x.min() / bin_width) * bin_width,
                      x.max() + bin_width, bin_width)
    cnt, _ = np.histogram(x, bins=edges)
    i = int(np.argmax(cnt))
    center = 0.5 * (edges[i] + edges[i + 1])
    limit = 10 ** center if log else center
    sigma = bin_width / np.sqrt(max(int(cnt[i]), 1))
    return limit, sigma, int(cnt[i])


def observed_completeness(family_file, neowise_file, id_col: int = 0,
                          neo_number_col: int = 2, neo_h_col: int = 3,
                          neo_d_col: int = 11, neo_pv_col: int = 13,
                          neo_epoch_col: int = 5, cryo_jd=observed.CRYO_JD,
                          h_bin: float = 0.25, d_dex: float = 0.05) -> dict:
    """H&M completeness of a family's NEOWISE-detected members.

    Returns ``H_complete`` (+ sigma) from the observed H-distribution, the median
    NEOWISE albedo, ``D_from_H`` (H_complete converted with that median albedo),
    and ``D_complete`` (H&M's peak applied directly to the observed diameters --
    the SFD-fit floor), plus the observed member count.
    """
    members = family_member_ids(family_file, id_col)
    cols = [neo_number_col, neo_h_col, neo_d_col, neo_pv_col] + ([neo_epoch_col] if cryo_jd else [])
    neo = pd.read_csv(neowise_file, header=None, usecols=cols, dtype={neo_number_col: str})
    cid = neo[neo_number_col].map(designations.canonical)  # numbered + provisional
    H = pd.to_numeric(neo[neo_h_col], errors="coerce")
    D = pd.to_numeric(neo[neo_d_col], errors="coerce")
    pV = pd.to_numeric(neo[neo_pv_col], errors="coerce")
    obs = cid.notna() & cid.isin(members)
    if cryo_jd is not None:                                # cryo-phase detections only
        ep = pd.to_numeric(neo[neo_epoch_col], errors="coerce")
        obs &= (ep >= cryo_jd[0]) & (ep <= cryo_jd[1])
    # Collapse to one row per UNIQUE object before finding the H&M peak (see
    # observed.observed_sfd): the catalog's size-correlated row multiplicity would
    # otherwise inflate the fuller (larger-object) bins and bias the completeness peak.
    per_obj = (pd.DataFrame({"num": cid[obs], "H": H[obs], "D": D[obs], "pV": pV[obs]})
               .groupby("num").median(numeric_only=True))
    Hm = per_obj["H"].dropna().to_numpy()
    Dm = per_obj["D"][per_obj["D"] > 0].dropna().to_numpy()
    pv_med = float(per_obj["pV"][per_obj["pV"] > 0].median())

    H_complete, sig_H, _ = hm_peak(Hm, h_bin)
    D_from_H = 1329.0 / np.sqrt(pv_med) * 10 ** (-H_complete / 5.0)
    D_complete, _, _ = hm_peak(Dm, d_dex, log=True)
    return {"H_complete": float(H_complete), "sigma_H": float(sig_H),
            "pV_median": pv_med, "D_from_H": float(D_from_H),
            "D_complete": float(D_complete), "n_observed": int(len(per_obj))}
