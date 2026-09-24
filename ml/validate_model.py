"""Session 3: hyperparameter tuning + grouped validation + the final exams.

Four escalating tests, in order of honesty:

  1. TUNING (train+val only; test never touched). Small grid over
     learning_rate x max_depth. The selection metric is val log-loss; we
     report eta-MAE alongside. Tuning on the same split you report on is
     mild leakage — so the tuned model's headline numbers come from the
     tests below, not from here.
  2. GROUPED CV BY REGION. Random CV lets the model fill gaps between
     near-identical neighbors; grouped CV holds out entire semimajor-axis
     bands, so each fold predicts a REGION it never saw — the sweep-data
     analog of leave-family-out, and partly near-extrapolation at the outer
     edges (fig-6 lesson, live). The random-vs-grouped gap measures how much
     of the random score was interpolation comfort.
  3. FINAL EXAM: the 25k test rows saved (and untouched) since session 2,
     evaluated exactly once with the tuned model.
  4. REAL POPULATIONS: predict per-family eta(D) for eight production
     families from their real SynthPop catalogs (fitted-beaming vintage =
     the vintage the physics measurements used) and compare to Simmer's
     measured efficiency curves. Correlated, realistic feature mixes the
     i.i.d. sweep never showed it. Comparison restricted to D <= 40 km
     (the training range: beyond it the model extrapolates by design).

Run:  /usr/local/bin/python ml/validate_model.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
FEATS = ["diam_km", "a_au", "e", "i_deg", "pV", "eta", "b_over_a",
         "pole_beta_deg", "rot_period_h"]
SEED = 42

FAMILIES = [("IB", "Vesta"), ("IB", "Eulalia"), ("MB", "Eunomia"),
            ("MB", "Hansa"), ("OB", "Themis"), ("OB", "Eos"),
            ("PB", "Koronis-2"), ("HU", "Hungaria")]


def eta_mae(d_km, y, p, n_bins=24):
    edges = np.logspace(np.log10(0.5), np.log10(40), n_bins + 1)
    idx = np.digitize(d_km, edges) - 1
    errs = [abs(p[idx == b].mean() - y[idx == b].mean())
            for b in range(n_bins) if (idx == b).sum() > 200]
    return float(np.mean(errs))


def fit(params, X, y):
    return HistGradientBoostingClassifier(
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        random_state=SEED, **params).fit(X, y)


def main():
    df = pd.read_csv(HERE / "data/eta_training.csv")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    tr, va = order[:100_000], order[100_000:125_000]
    te = pd.read_csv(HERE / "data/test_indices.csv").test_row.values
    dtr, dva, dte = df.iloc[tr], df.iloc[va], df.iloc[te]
    out = {}

    # ---- 1. tuning grid --------------------------------------------------
    print("== tuning (9 fits) ==")
    grid = []
    for lr in (0.03, 0.1, 0.3):
        for depth in (3, 6, None):
            m = fit(dict(learning_rate=lr, max_depth=depth),
                    dtr[FEATS], dtr.detected)
            p = m.predict_proba(dva[FEATS])[:, 1]
            row = dict(lr=lr, depth=str(depth),
                       val_logloss=round(float(log_loss(dva.detected, p)), 4),
                       val_eta_mae=round(eta_mae(dva.diam_km.values,
                                                 dva.detected.values, p), 4),
                       n_trees=int(m.n_iter_))
            grid.append(row)
            print(f"  lr={lr:<5} depth={str(depth):<5} "
                  f"logloss={row['val_logloss']} eta_mae={row['val_eta_mae']} "
                  f"trees={row['n_trees']}")
    best = min(grid, key=lambda r: r["val_logloss"])
    out["tuning_grid"] = grid
    out["best_params"] = best
    bp = dict(learning_rate=best["lr"],
              max_depth=None if best["depth"] == "None" else int(best["depth"]))

    # ---- 2. grouped CV by a-band vs random CV ----------------------------
    print("== grouped CV by semimajor-axis band ==")
    dtv = df.iloc[np.concatenate([tr, va])]
    bands = np.array([1.75, 2.1, 2.5, 2.85, 3.3, 3.71])
    grouped = []
    for k in range(5):
        m_in = (dtv.a_au >= bands[k]) & (dtv.a_au < bands[k + 1])
        m = fit(bp, dtv[~m_in][FEATS], dtv[~m_in].detected)
        p = m.predict_proba(dtv[m_in][FEATS])[:, 1]
        mae = eta_mae(dtv[m_in].diam_km.values, dtv[m_in].detected.values, p)
        grouped.append(dict(band=f"{bands[k]}-{bands[k+1]} au",
                            eta_mae=round(mae, 4), n=int(m_in.sum())))
        print(f"  hold out {grouped[-1]['band']:<14} eta_mae={mae:.4f}")
    random_maes = []
    folds = rng.permutation(len(dtv)) % 5
    for k in range(5):
        m = fit(bp, dtv[folds != k][FEATS], dtv[folds != k].detected)
        p = m.predict_proba(dtv[folds == k][FEATS])[:, 1]
        random_maes.append(eta_mae(dtv[folds == k].diam_km.values,
                                   dtv[folds == k].detected.values, p))
    out["grouped_cv"] = grouped
    out["grouped_cv_mean"] = round(float(np.mean([g["eta_mae"]
                                                  for g in grouped])), 4)
    out["random_cv_mean"] = round(float(np.mean(random_maes)), 4)
    print(f"  grouped mean {out['grouped_cv_mean']} vs "
          f"random mean {out['random_cv_mean']}")

    # ---- 3. the final exam (once) ---------------------------------------
    tuned = fit(bp, pd.concat([dtr, dva])[FEATS],
                pd.concat([dtr, dva]).detected)
    p = tuned.predict_proba(dte[FEATS])[:, 1]
    out["final_test"] = dict(
        logloss=round(float(log_loss(dte.detected, p)), 4),
        eta_mae=round(eta_mae(dte.diam_km.values, dte.detected.values, p), 4))
    joblib.dump(tuned, HERE / "models/gbm_tuned.joblib")
    print(f"== FINAL TEST (untouched 25k): {out['final_test']} ==")

    # ---- 4. real production families ------------------------------------
    print("== real families: emulated vs measured eta(D) ==")
    fam_rows = []
    for zone, fam in FAMILIES:
        cat = os.path.expanduser(
            f"~/synthpop_runs/fitted_eta/{zone}/{fam}/{fam}_synthpop.csv")
        eff = os.path.expanduser(
            f"~/simmer_runs/{zone}/{fam}/{fam}_efficiency_ensemble.csv")
        c = pd.read_csv(cat)
        e = pd.read_csv(eff)
        p = tuned.predict_proba(c[FEATS])[:, 1]
        pred = []
        for _, r in e.iterrows():
            m = (c.diam_km >= r.d_lo) & (c.diam_km < r.d_hi)
            pred.append(p[m].mean() if m.sum() >= 30 else np.nan)
        e["eta_emulated"] = pred
        ok = e.dropna(subset=["eta_emulated"])
        ok = ok[ok.d_mid <= 40]
        mae = float(np.abs(ok.eta_emulated - ok.eta).mean())
        fam_rows.append(dict(zone=zone, family=fam, eta_mae=round(mae, 4),
                             n_bins=len(ok)))
        e.to_csv(HERE / f"data/family_emulated_{fam}.csv", index=False)
        print(f"  {zone} {fam:<12} eta_mae={mae:.4f} over {len(ok)} bins")
    out["families"] = fam_rows
    out["families_mean_eta_mae"] = round(
        float(np.mean([f["eta_mae"] for f in fam_rows])), 4)

    (HERE / "metrics_session3.json").write_text(json.dumps(out, indent=2))
    print("wrote metrics_session3.json")


if __name__ == "__main__":
    main()
