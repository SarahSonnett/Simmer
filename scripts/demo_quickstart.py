"""Self-contained quickstart: simulate a survey against a synthetic population.

Runs entirely from this repository — no external data needed. Builds a small
random asteroid population, replays a synthetic WISE-like cadence against it
with the full physics chain (ephemeris -> field-of-view -> NEATM thermal flux
-> rotational lightcurve -> noise -> detection -> tracklet linking), and
prints the resulting detection-efficiency curve eta(D).

    python scripts/demo_quickstart.py        # ~1 minute
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simmer import SimConfig, run_simmer
from simmer.surveys import neowise

N = 4000
rng = np.random.default_rng(7)
D = 10 ** rng.uniform(np.log10(0.7), np.log10(30.0), N)
pV = 10 ** rng.uniform(np.log10(0.03), np.log10(0.35), N)
cat = pd.DataFrame({
    "id": [f"demo_{k:05d}" for k in range(N)],
    "diam_km": D,
    "b_over_a": rng.uniform(0.6, 1.0, N),
    "pole_beta_deg": np.degrees(np.arcsin(rng.uniform(-1, 1, N))),
    "pole_lambda_deg": rng.uniform(0, 360, N),
    "rot_period_h": 10 ** rng.uniform(np.log10(2.5), np.log10(20), N),
    "a_au": rng.uniform(2.1, 3.3, N),
    "e": rng.uniform(0.0, 0.25, N),
    "i_deg": rng.uniform(0.0, 2.5, N),
    "node_deg": rng.uniform(0, 360, N),
    "argperi_deg": rng.uniform(0, 360, N),
    "M_deg": rng.uniform(0, 360, N),
    "pV": pV,
    "eta": rng.uniform(0.9, 1.3, N),
})
cat["H"] = 5 * np.log10(1329.0 / (cat.diam_km * np.sqrt(cat.pV)))

out = Path(tempfile.mkdtemp())
cat.to_csv(out / "demo_synthpop.csv", index=False)
# Toy cadence: a densely-tiled low-latitude strip over a 2-day window (see
# its docstring — a coverage toy, not the real scan geometry; science runs
# use the real 1.24M-frame IRSA table). The short window keeps each object's
# detections inside one apparition so tracklet linking behaves like the real
# survey's; the low-i population stays inside the strip.
frames = neowise.synthetic_frames(n_scans=60_000, seed=1, days=2.0,
                                  lat_max=3.0)
print(f"{N} synthetic asteroids vs {len(frames):,} synthetic frames ...")
run_simmer(SimConfig(name="demo", catalog=out / "demo_synthpop.csv",
                     out_dir=out, seed=0), frames=frames, verbose=False)

obj = pd.read_csv(out / "demo_object_summary.csv").merge(cat, on="id")
edges = np.logspace(np.log10(0.7), np.log10(30), 11)
print("\n  D range (km)    eta = detected fraction")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (obj.diam_km >= lo) & (obj.diam_km < hi)
    bar = "#" * int(40 * obj[m].detected.mean())
    print(f"  {lo:5.1f} - {hi:5.1f}   {obj[m].detected.mean():5.3f}  {bar}")
print("\nThe rise-to-plateau shape is the survey selection function this "
      "pipeline\nexists to measure — and that the ml/ surrogate learns to "
      "emulate.")
