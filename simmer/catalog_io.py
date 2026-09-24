"""Ingest the SynthPop synthetic-asteroid catalog -- the pipeline contract.

The catalog is whatever ``synthpop.assemble.build_catalog`` wrote, i.e. one row
per synthetic body with the columns in :data:`REQUIRED_COLUMNS`. We validate the
contract loudly here so downstream stages can assume clean inputs.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# The exact columns SynthPop emits (see synthpop/assemble.py COLUMNS).
REQUIRED_COLUMNS = [
    "id",
    "diam_km",          # diameter (km)
    "b_over_a",         # shape elongation b/a in (0, 1]
    "pole_beta_deg",    # spin-pole ecliptic latitude (signed)
    "pole_lambda_deg",  # spin-pole ecliptic longitude [0, 360)
    "a_au", "e", "i_deg", "node_deg", "argperi_deg", "M_deg",  # orbit at cfg.epoch_jd
    "pV",               # visible albedo
    "eta",              # NEATM beaming parameter
    "H",                # absolute magnitude
]


def load_catalog(path: Path) -> pd.DataFrame:
    """Read a ``<name>_synthpop.csv`` (or ``.parquet``) catalog and validate it."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing SynthPop catalog columns {missing}; "
            f"got {list(df.columns)}"
        )
    if (df["b_over_a"] <= 0).any() or (df["b_over_a"] > 1).any():
        raise ValueError("b_over_a must lie in (0, 1] (it is b/a, the inverse of a:b)")
    if (df["diam_km"] <= 0).any():
        raise ValueError("diam_km must be positive")
    return df.reset_index(drop=True)
