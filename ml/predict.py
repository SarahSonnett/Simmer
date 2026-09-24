"""Public API for the Simmer detection-efficiency emulator.

    from ml import predict_eta, eta_curve

    p = predict_eta(catalog)          # per-object P(detect), catalog = a
                                      # SynthPop-style DataFrame (any extra
                                      # columns are ignored)
    d, eta = eta_curve(catalog)       # binned eta(D), the familiar curve

The model is a gradient-boosted classifier trained on 150k objects swept
through the full Simmer physics chain (see ml/data/PROVENANCE.md), with a
monotonic-in-diameter constraint (session-4 verdict: aggregate accuracy is
marginally lower than unconstrained, but slices are guaranteed physical).
Test-set eta-MAE 0.016; ~5,000x faster than the physics.

Trust boundary (the fig-6 lesson, enforced in code): predictions are only
valid INSIDE the training ranges. Out-of-range rows get NaN by default
(``clip="nan"``); pass ``clip="edge"`` to knowingly accept edge-value
extrapolation, e.g. for D > 40 km where eta is a flat plateau.

Model artifacts are regenerable, not committed: if ml/models/production.joblib
is missing, run ml/make_training_set.py then the build snippet in
ml/metrics_production.json's git history (or ml/train_model.py + session-3/5
scripts).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
FEATURES = ["diam_km", "a_au", "e", "i_deg", "pV", "eta", "b_over_a",
            "pole_beta_deg", "rot_period_h"]
# training ranges from make_training_set.py (the interpolation box)
RANGES = {"diam_km": (0.5, 40.0), "a_au": (1.75, 3.70), "e": (0.0, 0.35),
          "i_deg": (0.0, 35.0), "pV": (0.02, 0.45), "eta": (0.7, 1.5),
          "b_over_a": (0.55, 1.0), "pole_beta_deg": (-90.0, 90.0),
          "rot_period_h": (2.2, 24.0)}
_model = None


def _load():
    global _model
    if _model is None:
        path = _HERE / "models" / "production.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — model artifacts are regenerable, not "
                "committed; see ml/predict.py docstring.")
        _model = joblib.load(path)
    return _model


def in_range(catalog: pd.DataFrame) -> np.ndarray:
    """Boolean mask: rows fully inside the training (interpolation) box."""
    ok = np.ones(len(catalog), bool)
    for f, (lo, hi) in RANGES.items():
        ok &= (catalog[f].values >= lo) & (catalog[f].values <= hi)
    return ok


def predict_eta(catalog: pd.DataFrame, clip: str = "nan",
                model=None) -> np.ndarray:
    """Per-object detection probability for a SynthPop-style catalog.

    clip="nan" (default): out-of-range rows -> NaN (honest refusal).
    clip="edge": out-of-range rows are clamped to the training box and
    predicted at the edge (deliberate, documented extrapolation).
    """
    missing = [f for f in FEATURES if f not in catalog.columns]
    if missing:
        raise ValueError(f"catalog missing feature columns: {missing}")
    m = model or _load()
    X = catalog[FEATURES].copy()
    ok = in_range(catalog)
    if clip == "edge":
        for f, (lo, hi) in RANGES.items():
            X[f] = X[f].clip(lo, hi)
        return m.predict_proba(X)[:, 1]
    p = np.full(len(X), np.nan)
    if ok.any():
        p[ok] = m.predict_proba(X[ok])[:, 1]
    return p


def eta_curve(catalog: pd.DataFrame, n_bins: int = 24, clip: str = "nan",
              model=None):
    """Binned eta(D): (bin centers, mean predicted P(detect) per bin)."""
    p = predict_eta(catalog, clip=clip, model=model)
    edges = np.logspace(np.log10(0.5), np.log10(40.0), n_bins + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    idx = np.digitize(catalog["diam_km"].values, edges) - 1
    eta = np.array([np.nanmean(p[idx == b]) if (idx == b).sum() else np.nan
                    for b in range(n_bins)])
    return mid, eta
