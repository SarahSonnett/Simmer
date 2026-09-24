#!/usr/bin/env python
"""CLI for the Simmer survey-detection pipeline.

Two ways to run:

  # From a saved JSON config
  python scripts/run_simmer.py --config myrun.json

  # Built from command-line flags
  python scripts/run_simmer.py \
      --name Hygiea_neowise --survey neowise_cryo \
      --catalog ~/Projects/SynthPop/synthpop_out/Hygiea_synthpop.csv \
      --frames ~/Projects/Simmer/data/cryo_4band_pointings_near_ecliptic.csv \
      --out simmer_out

  # Offline smoke test with synthetic frames (no IRSA, no real catalog needed)
  python scripts/run_simmer.py --name demo --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simmer import SimConfig, run_simmer            # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="load a saved SimConfig JSON (overrides flags)")
    p.add_argument("--name")
    p.add_argument("--survey", default="neowise_cryo")
    p.add_argument("--catalog", type=Path, help="<name>_synthpop.csv from SynthPop")
    p.add_argument("--frames", type=Path, help="cached frame metadata (parquet/feather/csv)")
    p.add_argument("--fetch-frames", action="store_true", help="pull frames from IRSA if no cache")
    p.add_argument("--epoch-jd", type=float, default=2455197.5)
    p.add_argument("--max-ecl-lat", type=float, default=30.0)
    p.add_argument("--snr", type=float, default=5.0, dest="snr_threshold")
    p.add_argument("--bad-pixel-frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("simmer_out"))
    p.add_argument("--format", choices=["csv", "parquet"], default="csv", dest="out_format")
    p.add_argument("--demo", action="store_true",
                   help="run offline against synthetic frames + a tiny built-in catalog")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.config:
        cfg = SimConfig.from_json(args.config)
        run_simmer(cfg)
        return

    if args.demo:
        _run_demo(args)
        return

    if not (args.name and args.catalog):
        build_parser().error("--name and --catalog are required unless --config or --demo")
    cfg = SimConfig(
        name=args.name, survey=args.survey, catalog=args.catalog,
        frames_cache=args.frames, fetch_frames=args.fetch_frames,
        epoch_jd=args.epoch_jd, max_ecl_lat_deg=args.max_ecl_lat,
        snr_threshold=args.snr_threshold, bad_pixel_fraction=args.bad_pixel_frac,
        seed=args.seed, out_dir=args.out, out_format=args.out_format,
    )
    run_simmer(cfg)


def _run_demo(args):
    """Self-contained offline demo: tiny catalog + synthetic frames."""
    import numpy as np
    import pandas as pd
    from simmer.surveys import neowise

    rng = np.random.default_rng(0)
    n = 50
    cat = pd.DataFrame({
        "id": [f"demo_{k:04d}" for k in range(n)],
        "diam_km": rng.uniform(5, 80, n),
        "b_over_a": rng.uniform(0.4, 1.0, n),
        "pole_beta_deg": rng.uniform(-90, 90, n),
        "pole_lambda_deg": rng.uniform(0, 360, n),
        "a_au": rng.uniform(2.2, 3.2, n),
        "e": rng.uniform(0, 0.2, n),
        "i_deg": rng.uniform(0, 20, n),
        "node_deg": rng.uniform(0, 360, n),
        "argperi_deg": rng.uniform(0, 360, n),
        "M_deg": rng.uniform(0, 360, n),
        "pV": rng.uniform(0.03, 0.3, n),
        "eta": rng.uniform(0.8, 1.2, n),
        "H": rng.uniform(9, 15, n),
    })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat_path = out / "demo_synthpop.csv"
    cat.to_csv(cat_path, index=False)

    cfg = SimConfig(name=args.name or "demo", catalog=cat_path, out_dir=out, seed=0)
    frames = neowise.synthetic_frames(seed=1)
    run_simmer(cfg, frames=frames)


if __name__ == "__main__":
    main()
