#!/usr/bin/env python
"""Run Simmer on every SynthPop family catalog in a belt-zone directory.

Given a directory laid out as ``<zone>/<Family>/<Family>_synthpop.csv`` (the
SynthPop ``synthpop_runs`` convention), this runs ``run_simmer`` on each family
in order of increasing median semi-major axis (inner belt -> outer belt) and
writes each family's outputs back into its own folder:

    <Family>_efficiency.csv      detection efficiency eta(D) with Wilson intervals
    <Family>_object_summary.csv  per-object detected flag
    <Family>_detections.csv      one row per (object, frame) detection
    <Family>_config.json         the full effective SimConfig (all settings)
    <Family>_provenance.json     run provenance

Settings: everything is the ``SimConfig`` default (survey=neowise_cryo -> W3/W4,
snr_threshold=5, photometric_noise on with frac_err=0.03, bad_pixel_fraction=0.01,
apply_lightcurve on, use_flux_lut on, emissivity=0.9, slope_G=0.15, cryo epoch)
except ``name``/``catalog``/``out_dir`` and ``--seed``. Override any of them by
editing the ``SimConfig(...)`` call below; each run records its own settings to
``<Family>_config.json``.

The survey frames (~182 MB) are loaded once and reused, so the whole zone runs in
a single ~1.3 GB process -- serial, ~20 s per 10k-object family. Example:

    python scripts/run_families.py ~/synthpop_runs/IB
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simmer import SimConfig, run_simmer                          # noqa: E402

DEFAULT_FRAMES = (Path(__file__).resolve().parent.parent
                  / "data/cryo_4band_pointings_near_ecliptic.csv")


def _plateau_and_d50(eff: pd.DataFrame):
    """Plateau efficiency (mean of the largest-D bins) and the half-plateau D."""
    plateau = float(eff["eta"].iloc[-5:].mean())
    e = eff[eff["n_input"] >= 30]
    half = 0.5 * plateau
    eta, d_mid = e["eta"].to_numpy(), e["d_mid"].to_numpy()
    cr = np.where((eta[:-1] < half) & (eta[1:] >= half))[0]
    d50 = (float(np.interp(half, eta[cr[0]:cr[0] + 2], d_mid[cr[0]:cr[0] + 2]))
           if len(cr) else float("nan"))
    return plateau, d50


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zone_dir", type=Path,
                   help="directory of <Family>/<Family>_synthpop.csv catalogs")
    p.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                   help="survey frame pointings CSV (default: near-ecliptic cryo cadence)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    catalogs = sorted(args.zone_dir.glob("*/*_synthpop.csv"))
    if not catalogs:
        p.error(f"no <Family>/<Family>_synthpop.csv under {args.zone_dir}")

    # order inner -> outer by median semi-major axis
    fams = [(pd.read_csv(c, usecols=["a_au"])["a_au"].median(), c.parent.name, c)
            for c in catalogs]
    fams.sort()

    print(f"loading frames {args.frames.name} ...")
    frames = pd.read_csv(args.frames)
    print(f"{len(frames):,} frames; running {len(fams)} families inner -> outer\n")

    t_all = time.perf_counter()
    for a, name, cat in fams:
        t0 = time.perf_counter()
        run_simmer(SimConfig(name=name, catalog=cat, out_dir=cat.parent,
                             seed=args.seed), frames=frames, verbose=False)
        eff = pd.read_csv(cat.parent / f"{name}_efficiency.csv")
        summ = pd.read_csv(cat.parent / f"{name}_object_summary.csv")
        plateau, d50 = _plateau_and_d50(eff)
        print(f"  {name:14s} a={a:.3f}  {time.perf_counter()-t0:4.0f}s  "
              f"detected {int(summ['detected'].sum()):6d}/{len(summ)}  "
              f"plateau eta={plateau:.2f}  D50={d50:.1f} km")

    print(f"\nTOTAL: {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
