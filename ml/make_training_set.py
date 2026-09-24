"""Generate the surrogate-model training set by sweeping the Simmer physics model.

The ML project's goal is a learned surrogate ("emulator") of Simmer's expensive
detection-efficiency calculation. Surrogate modeling starts by using the
simulator itself as the data generator: sample the input space broadly, run the
physics forward, and record (features -> outcome) pairs.

Design choices (recorded here because they ARE the methodology):

- PER-OBJECT rows, not per-population eta curves. Each row is one synthetic
  asteroid with its physical/orbital parameters and the simulator's verdict
  (detected or not, and how many raw detections). This supports either framing
  downstream: a classifier P(detect | features), whose averaged probabilities
  reproduce eta(D) for any population mix, or binned eta regression.
- RANDOM (not gridded) sampling. Tree ensembles do not need a lattice, and
  random sampling covers interactions (e.g. the a-dependent sky-coverage
  plateau x diameter threshold) without a combinatorial grid.
- RANGES bracket the production populations with margin, so the surrogate is
  interpolating, never extrapolating, on real use cases:
    diam_km      log-uniform 0.5 - 40      (production fits use ~1-30 km)
    a_au         uniform 1.75 - 3.70       (Hungaria region through outer belt)
    e            uniform 0.00 - 0.35
    i_deg        uniform 0 - 35
    pV           log-uniform 0.02 - 0.45   (darkest C to brightest S/V)
    eta (NEATM beaming) uniform 0.7 - 1.5  (spans all-rows AND fitted vintages)
    b_over_a     uniform 0.55 - 1.00       (sphere to 1.8:1 ellipsoid)
    pole         isotropic (beta sin-uniform, lambda uniform)
    rot_period_h log-uniform 2.2 - 24
    angles (node, argperi, M) uniform      (nuisance orbital phase)
    H            derived: H = 5 log10(1329 / (D sqrt(pV)))  (consistency with
                 the SynthPop convention, verified against production catalogs)
- Everything else is the PRODUCTION SimConfig default (neowise_cryo W3/W4,
  snr 5, lightcurve modulation on, bad-pixel 1%, linking >=5 dets/apparition),
  so the surrogate emulates the exact physics chain used for science.

Output: ml/data/eta_training.csv (+ PROVENANCE.md). Regenerable; gitignored.

Run:  /usr/local/bin/python ml/make_training_set.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simmer import SimConfig, run_simmer  # noqa: E402

SEED = 42
N_TOTAL = 150_000
BATCH = 10_000
FRAMES = ROOT / "data/cryo_4band_pointings_near_ecliptic.csv"
OUT = ROOT / "ml/data"
TMP = OUT / "_sweep_tmp"


def sample_batch(rng: np.random.Generator, n: int, offset: int) -> pd.DataFrame:
    D = 10 ** rng.uniform(np.log10(0.5), np.log10(40.0), n)
    pV = 10 ** rng.uniform(np.log10(0.02), np.log10(0.45), n)
    df = pd.DataFrame({
        "id": [f"sw_{offset + k:06d}" for k in range(n)],
        "diam_km": D,
        "b_over_a": rng.uniform(0.55, 1.00, n),
        "pole_beta_deg": np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n))),
        "pole_lambda_deg": rng.uniform(0.0, 360.0, n),
        "rot_period_h": 10 ** rng.uniform(np.log10(2.2), np.log10(24.0), n),
        "a_au": rng.uniform(1.75, 3.70, n),
        "e": rng.uniform(0.00, 0.35, n),
        "i_deg": rng.uniform(0.0, 35.0, n),
        "node_deg": rng.uniform(0.0, 360.0, n),
        "argperi_deg": rng.uniform(0.0, 360.0, n),
        "M_deg": rng.uniform(0.0, 360.0, n),
        "pV": pV,
        "eta": rng.uniform(0.7, 1.5, n),
    })
    df["H"] = 5.0 * np.log10(1329.0 / (df.diam_km * np.sqrt(df.pV)))
    return df


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    TMP.mkdir(parents=True, exist_ok=True)
    print(f"loading frames {FRAMES.name} ...", flush=True)
    frames = pd.read_csv(FRAMES)
    print(f"{len(frames):,} frames loaded", flush=True)

    pieces = []
    for b in range(N_TOTAL // BATCH):
        feats = sample_batch(rng, BATCH, b * BATCH)
        cat = TMP / f"sweep{b:02d}_synthpop.csv"
        feats.to_csv(cat, index=False)
        cfg = SimConfig(name=f"sweep{b:02d}", catalog=cat, out_dir=TMP,
                        seed=SEED + b)
        run_simmer(cfg, frames=frames, verbose=False)
        obj = pd.read_csv(TMP / f"sweep{b:02d}_object_summary.csv")
        merged = feats.merge(obj, on="id", validate="1:1")
        pieces.append(merged)
        det = merged.detected.mean()
        print(f"batch {b:02d}: {BATCH} objects, detected fraction "
              f"{det:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    full = pd.concat(pieces, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT / "eta_training.csv", index=False)

    commit = subprocess.run(["/usr/local/bin/git", "-C", str(ROOT), "rev-parse",
                             "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    prov = f"""# eta_training.csv provenance

Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by ml/make_training_set.py
(seed {SEED}) at Simmer commit {commit}.

- {len(full):,} synthetic objects, random-sampled over the ranges documented
  in the script docstring; per-object detected flag + n_detections from the
  full production physics chain (SimConfig defaults: neowise_cryo W3/W4,
  snr 5, lightcurve modulation, 1% bad-pixel, >=5-detection linking).
- Frames: {FRAMES.name} ({len(frames):,} rows).
- Overall detected fraction: {full.detected.mean():.4f}.
- Wall time: {time.time()-t0:.0f} s (single process, frames loaded once) —
  this is the "expensive physics" baseline the surrogate is judged against.
- Regenerate with the command in the script header; byte-reproducible for a
  fixed seed and commit.
"""
    (OUT / "PROVENANCE.md").write_text(prov)
    for f in TMP.iterdir():
        f.unlink()
    TMP.rmdir()
    print(f"\nwrote ml/data/eta_training.csv ({len(full):,} rows) + "
          f"PROVENANCE.md in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
