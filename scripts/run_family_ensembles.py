#!/usr/bin/env python
"""Run Simmer's *ensemble* on every SynthPop family ensemble in a zone directory.

Given a zone of SynthPop ensembles laid out as
``<zone>/<Family>_ensemble/<Family>_ensemble_manifest.json`` (the
``synthpop_runs/ensembles`` convention), this runs Simmer's parallel ensemble on
each family in order of increasing median semi-major axis (inner -> outer) and
writes, per family, into ``<out_root>/<zone_name>/<Family>/``:

    <Family>_efficiency_ensemble.csv   eta(D) with statistical + systematic + total bands
    <Family>_efficiency.png            the efficiency plot (nested stat / total bands)

(The debiased SFD + power-law fit are written too, but only if you pass real
observed counts -- see ``EnsembleResult.save``; those need the real family SFD,
not the synthetic populations, so they are omitted here.)

Members run in parallel (``--n-workers``); each is an independent ~1.3-2 GB
run_simmer, so budget n_workers * ~2 GB of RAM. Example:

    python scripts/run_family_ensembles.py ~/synthpop_runs/ensembles/IB \\
        --out-root ~/simmer_runs --n-workers 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simmer import SimConfig, plotting                            # noqa: E402
from simmer.ensemble import run_ensemble, ScienceFloor            # noqa: E402

DEFAULT_FRAMES = (Path(__file__).resolve().parent.parent
                  / "data/cryo_4band_pointings_near_ecliptic.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zone_dir", type=Path,
                   help="dir of <Family>_ensemble/<Family>_ensemble_manifest.json")
    p.add_argument("--out-root", type=Path, default=Path.home() / "simmer_runs")
    p.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    p.add_argument("--n-workers", type=int, default=10)
    p.add_argument("--n-bins", type=int, default=20,
                   help="diameter bins for eta(D) (debias reuses these)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--science-floor", action="store_true",
                   help="add model/calibration systematics (sensitivity + beaming) "
                        "on top of the SynthPop data bootstrap")
    p.add_argument("--sens-frac", type=float, default=0.15,
                   help="1σ sensitivity-limit scaling (flux calibration + depth)")
    p.add_argument("--beaming-sigma", type=float, default=0.04,
                   help="1σ additive shift of the beaming parameter η_IR")
    args = p.parse_args()

    floor = (ScienceFloor(sensitivity_frac=args.sens_frac,
                          beaming_sigma=args.beaming_sigma)
             if args.science_floor else None)

    manifests = sorted(args.zone_dir.glob("*/*_ensemble_manifest.json"))
    if not manifests:
        p.error(f"no <Family>_ensemble/*_ensemble_manifest.json under {args.zone_dir}")
    zone_name = args.zone_dir.name

    # order inner -> outer by median semi-major axis (from each ensemble's member 0)
    fams = []
    for mp in manifests:
        m = json.loads(mp.read_text())
        cat0 = mp.parent / m["members"][0]["catalog"]
        a = pd.read_csv(cat0, usecols=["a_au"])["a_au"].median()
        fams.append((a, m["name"], mp))
    fams.sort()

    print(f"{len(fams)} ensembles in zone {zone_name!r}; "
          f"{args.n_workers} workers; out -> {args.out_root / zone_name}\n")

    t_all = time.perf_counter()
    for a, name, mp in fams:
        outdir = args.out_root / zone_name / name
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        res = run_ensemble(mp, SimConfig(name=name, seed=args.seed),
                           frames_path=args.frames, n_workers=args.n_workers,
                           n_bins=args.n_bins, floor=floor, verbose=False)
        res.save(outdir, name)                          # efficiency_ensemble.csv
        comb = res.combine()
        plotting.plot_efficiency(comb, outdir / f"{name}_efficiency.png",
                                 title=f"{name}  (η vs diameter)")
        # headline: plateau and typical systematic vs statistical band width
        e = comb[np.isfinite(comb["eta"])]
        plateau = float(e["eta"].iloc[-5:].mean())
        # max (not median) systematic: for well-measured families the systematic
        # is below the statistical noise in most bins, so the median reads ~0 --
        # the peak per-bin value is the informative headline.
        sys_w = float(((e["eta_sys_hi"] - e["eta_sys_lo"]) / 2).max())
        stat_w = float(((e["eta_stat_hi"] - e["eta_stat_lo"]) / 2).median())
        print(f"  {name:12s} a={a:.3f}  {time.perf_counter()-t0:5.0f}s  "
              f"plateau η={plateau:.2f}  σ_stat={stat_w:.3f}  σ_sys(max)={sys_w:.3f}  "
              f"-> {outdir}")

    print(f"\nTOTAL: {(time.perf_counter()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
