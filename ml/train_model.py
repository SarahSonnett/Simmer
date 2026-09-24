"""Session 2: baseline + gradient-boosted surrogate for Simmer's eta physics.

The ladder (each rung exists to justify the next):
  1. baseline_logD   — logistic regression on log10(D) alone. The dumbest
                       defensible model: eta is mostly a function of size.
                       Everything else must beat THIS, or it isn't a result.
  2. logistic_all    — logistic regression on all 9 features (standardized,
                       logs where dynamic range demands). Shows how far a
                       LINEAR model gets: it can use every feature but cannot
                       express interactions (D x albedo at threshold, a x
                       synodic-window geometry) unless we hand-engineer them.
  3. gbm             — gradient-boosted trees on all 9 raw features. Learns
                       thresholds and interactions itself; the actual
                       surrogate candidate.
  4. gbm_mono        — same, plus injected physics: P(detect) must be
                       monotonically NON-DECREASING in D (all else fixed).
                       Inductive bias; should cost ~nothing and kill any
                       unphysical wiggles at sparse large D.

Split discipline: train 100k / val 25k / test 25k, fixed seed. The TEST rows
are written to disk and NOT TOUCHED here — they are session 3's final exam.
All numbers reported below are computed on VAL.

Metrics: log-loss (the loss being optimized), ROC AUC (ranking), Brier score
(calibration-sensitive), and — the physically meaningful one — eta-space MAE:
bin val objects by D, compare the model's mean predicted probability to the
actual detected fraction per bin. That is literally "how wrong is the
emulated efficiency curve".

Run:  /usr/local/bin/python ml/train_model.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FEATS = ["diam_km", "a_au", "e", "i_deg", "pV", "eta", "b_over_a",
         "pole_beta_deg", "rot_period_h"]
SEED = 42
PHYSICS_S_PER_OBJ = 578.0 / 150_000        # measured sweep wall time


def eta_space_mae(d_km, y_true, p_pred, n_bins=24):
    """MAE between predicted and actual detected fraction, in log-D bins."""
    edges = np.logspace(np.log10(0.5), np.log10(40), n_bins + 1)
    idx = np.digitize(d_km, edges) - 1
    errs = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 200:
            errs.append(abs(p_pred[m].mean() - y_true[m].mean()))
    return float(np.mean(errs)), len(errs)


def main():
    df = pd.read_csv(HERE / "data/eta_training.csv")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    tr, va, te = order[:100_000], order[100_000:125_000], order[125_000:]
    pd.Series(te, name="test_row").to_csv(HERE / "data/test_indices.csv",
                                          index=False)
    dtr, dva = df.iloc[tr], df.iloc[va]
    ytr, yva = dtr.detected.values, dva.detected.values
    print(f"train {len(dtr):,} / val {len(dva):,} / test {len(te):,} "
          f"(test written to disk, untouched)")

    def lin_feats(d):
        out = d[FEATS].copy()
        out["diam_km"] = np.log10(out["diam_km"])
        out["pV"] = np.log10(out["pV"])
        out["rot_period_h"] = np.log10(out["rot_period_h"])
        return out.values

    models = {}
    t = time.time()
    models["baseline_logD"] = make_pipeline(
        StandardScaler(), LogisticRegression()).fit(
        np.log10(dtr[["diam_km"]].values), ytr)
    models["logistic_all"] = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)).fit(
        lin_feats(dtr), ytr)
    models["gbm"] = HistGradientBoostingClassifier(
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        random_state=SEED).fit(dtr[FEATS], ytr)
    mono = [1 if f == "diam_km" else 0 for f in FEATS]
    models["gbm_mono"] = HistGradientBoostingClassifier(
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        monotonic_cst=mono, random_state=SEED).fit(dtr[FEATS], ytr)
    print(f"all four fits: {time.time()-t:.1f} s")

    results = {}
    for name, m in models.items():
        if name == "baseline_logD":
            p = m.predict_proba(np.log10(dva[["diam_km"]].values))[:, 1]
        elif name == "logistic_all":
            p = m.predict_proba(lin_feats(dva))[:, 1]
        else:
            p = m.predict_proba(dva[FEATS])[:, 1]
        mae, nb = eta_space_mae(dva.diam_km.values, yva, p)
        results[name] = dict(
            logloss=round(float(log_loss(yva, p)), 4),
            auc=round(float(roc_auc_score(yva, p)), 4),
            brier=round(float(brier_score_loss(yva, p)), 4),
            eta_mae=round(mae, 4), eta_bins=nb)

    # runtime: the headline ratio
    m = models["gbm_mono"]
    t = time.time(); _ = m.predict_proba(dva[FEATS]); dt = time.time() - t
    per_obj = dt / len(dva)
    results["runtime"] = dict(
        physics_s_per_object=round(PHYSICS_S_PER_OBJ, 5),
        surrogate_s_per_object=round(per_obj, 8),
        speedup=round(PHYSICS_S_PER_OBJ / per_obj, 0))

    (HERE / "models").mkdir(exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, HERE / "models" / f"{name}.joblib")
    (HERE / "metrics_session2.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
