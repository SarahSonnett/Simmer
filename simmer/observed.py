"""Assemble the *real* observed size-frequency distribution of a family.

The debiasing needs N_obs(D) -- the count of real NEOWISE-detected family members
per diameter bin -- which is then divided by the simulated efficiency eta(D). This
module builds N_obs(D) by cross-matching a family's member list (AFP proper-element
file) to the NEOWISE diameter catalog and histogramming the matched diameters onto
the efficiency bin grid.

CAVEAT (flag to the user): dividing by eta corrects for NEOWISE's *detection*
selection, but NOT for incompleteness in the family *membership identification*
itself (the AFP list is drawn from a proper-element catalog biased toward larger,
numbered, multi-opposition objects). So the debiased SFD is corrected for the
survey, not for how the family was defined -- reliable above the membership
completeness limit, increasingly uncertain below it.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from . import designations


def family_member_ids(family_file, id_col: int = 0) -> set:
    """Canonical ids of an AFP family's members (numbered AND provisional).

    Each member designation is reduced to a canonical key by
    :func:`simmer.designations.canonical` -- an int for numbered objects, a
    normalized provisional string otherwise -- so members join to the NEOWISE
    catalog regardless of how either side writes the designation (plain number,
    MPC-packed number, or provisional). This keeps the ~34-43% of AFP members that
    are provisional (small, recently-discovered objects), which matter for the
    sub-completeness / small-size part of the SFD. Unparseable entries are dropped.
    See docs/methodology_decisions.md Sec. 1.
    """
    ids = np.genfromtxt(family_file, usecols=[id_col], dtype=str)
    out = {designations.canonical(x) for x in np.atleast_1d(ids)}
    out.discard(None)
    return out


# Cryogenic phase of the WISE mission (2010-01-07 .. 2010-08-06), as a fit-epoch JD
# window. The NEOWISE diameter catalog is a MULTI-YEAR compilation (cryo 2010 +
# reactivation 2013-2017; col 20 references Mas11 vs Nug15/16/Mas17), so N_obs must be
# restricted to the cryo phase to be consistent with Simmer's cryo-only survey
# simulation -- otherwise multi-year counts are divided by cryo efficiency.
CRYO_JD = (2455203.5, 2455415.5)


def observed_sfd(family_file, neowise_file, bins, id_col: int = 0,
                 neo_number_col: int = 2, neo_d_col: int = 11,
                 neo_epoch_col: int = 5, cryo_jd=CRYO_JD):
    """Observed counts per diameter bin = NEOWISE diameters of family members.

    Returns ``(n_obs, diam_obs, n_matched)``: the per-bin count array aligned to
    ``bins``, the matched diameters, and how many unique family members had a
    NEOWISE diameter. Designations are canonicalized (:mod:`simmer.designations`), so
    numbered objects >= 100000 (MPC-packed) and provisional objects match. Detections
    are restricted to the ``cryo_jd`` fit-epoch window (the cryogenic phase, to match
    the cryo survey simulation); pass ``cryo_jd=None`` for the full multi-year catalog.
    """
    members = family_member_ids(family_file, id_col)
    cols = [neo_number_col, neo_d_col] + ([neo_epoch_col] if cryo_jd else [])
    neo = pd.read_csv(neowise_file, header=None, usecols=cols,
                      dtype={neo_number_col: str})
    cid = neo[neo_number_col].map(designations.canonical)
    d = pd.to_numeric(neo[neo_d_col], errors="coerce")
    keep = cid.notna() & d.notna() & (d > 0) & cid.isin(members)
    if cryo_jd is not None:
        ep = pd.to_numeric(neo[neo_epoch_col], errors="coerce")
        keep &= (ep >= cryo_jd[0]) & (ep <= cryo_jd[1])
    # One diameter per UNIQUE object. The NEOWISE catalog has multiple rows per
    # object (separate per-band/per-apparition fits), and the row multiplicity rises
    # with diameter, so histogramming rows would inflate large-D bins relative to
    # small-D bins and tilt the SFD slope toward zero/negative. Collapse by object
    # id first (within-object diameter spread is ~0, so the median is safe).
    diam_obs = (pd.DataFrame({"num": cid[keep], "D": d[keep]})
                .groupby("num")["D"].median().to_numpy())
    n_obs, _ = np.histogram(diam_obs, bins=np.asarray(bins, float))
    return n_obs, diam_obs, diam_obs.size
