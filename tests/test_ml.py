"""Tests for the ml/ emulator subpackage. Runnable with pytest or directly:

    /usr/local/bin/python tests/test_ml.py

Designed to pass WITHOUT the 20 MB training file or saved model artifacts:
a tiny model is trained on the tracked 5k-row sample, so these tests (and
CI) exercise the API contract, the range guard, and the monotonicity
property end-to-end.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from ml import FEATURES, RANGES, eta_curve, in_range, predict_eta  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "ml/data/eta_training_sample.csv"


def _tiny_model(df):
    mono = [1 if f == "diam_km" else 0 for f in FEATURES]
    return HistGradientBoostingClassifier(
        max_iter=60, monotonic_cst=mono, random_state=0
    ).fit(df[FEATURES], df.detected)


def test_sample_schema():
    df = pd.read_csv(SAMPLE)
    assert set(FEATURES) <= set(df.columns)
    assert df.detected.dtype == bool or set(df.detected.unique()) <= {0, 1}
    for f, (lo, hi) in RANGES.items():
        assert df[f].min() >= lo - 1e-9 and df[f].max() <= hi + 1e-9


def test_predict_contract_and_range_guard():
    df = pd.read_csv(SAMPLE)
    m = _tiny_model(df)
    p = predict_eta(df.head(100), model=m)
    assert p.shape == (100,) and np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    # out-of-range rows: NaN by default, finite with clip="edge"
    bad = df.head(5).copy()
    bad.loc[bad.index[0], "diam_km"] = 100.0
    p = predict_eta(bad, model=m)
    assert np.isnan(p[0]) and np.isfinite(p[1:]).all()
    p = predict_eta(bad, clip="edge", model=m)
    assert np.isfinite(p).all()
    assert not in_range(bad)[0]


def test_monotonic_in_diameter():
    df = pd.read_csv(SAMPLE)
    m = _tiny_model(df)
    row = df.iloc[[0]][FEATURES]
    q = pd.concat([row] * 50, ignore_index=True)
    q["diam_km"] = np.logspace(np.log10(0.5), np.log10(40), 50)
    p = m.predict_proba(q)[:, 1]
    assert (np.diff(p) >= -1e-9).all(), "P(detect) must not decrease with D"


def test_eta_curve_shape():
    df = pd.read_csv(SAMPLE)
    m = _tiny_model(df)
    mid, eta = eta_curve(df, model=m)
    assert len(mid) == len(eta) == 24
    ok = np.isfinite(eta)
    # large-D bins must sit well above small-D bins (the threshold rise)
    assert np.nanmean(eta[ok][-5:]) > np.nanmean(eta[ok][:5]) + 0.3


def test_missing_column_raises():
    df = pd.read_csv(SAMPLE).drop(columns=["pV"])
    try:
        predict_eta(df, model=object())
    except ValueError as e:
        assert "pV" in str(e)
    else:
        raise AssertionError("expected ValueError for missing column")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} OK")
    print("all ml tests passed")
