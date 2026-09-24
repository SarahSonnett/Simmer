"""Session 4: interpretability — what did the surrogate actually learn?

Three questions, asked of the TUNED model from session 3:

  1. WHICH features carry the signal? SHAP values (game-theoretic
     per-prediction decompositions, in log-odds units for this model) give
     the global ranking; permutation importance is the cheap model-agnostic
     cross-check. If the two disagree badly, something is wrong.
  2. Does the model REDISCOVER known physics? Dependence plots:
     SHAP(diam) colored by albedo (the D x pV threshold interaction),
     SHAP(a) (the synodic-window plateau geometry), SHAP(pV) and
     SHAP(beaming) (thermal threshold physics). These shapes were never
     told to the model; finding them is the sanity check that it learned
     mechanism-shaped structure, not noise.
  3. The session-2 open question: does the UNCONSTRAINED model contain
     unphysical non-monotonicities in D (all else fixed)? Scan P(detect)
     vs D along many random fixed-feature slices, count and size the
     violations. Verdict decides whether the monotone constraint's accuracy
     cost (0.014 -> 0.020) buys anything real.

Run:  /usr/local/bin/python ml/interpret_model.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
FEATS = ["diam_km", "a_au", "e", "i_deg", "pV", "eta", "b_over_a",
         "pole_beta_deg", "rot_period_h"]
NICE = {"diam_km": "diameter", "a_au": "semimajor axis", "e": "eccentricity",
        "i_deg": "inclination", "pV": "albedo", "eta": "beaming",
        "b_over_a": "shape b/a", "pole_beta_deg": "pole latitude",
        "rot_period_h": "rotation period"}
SEED = 42
BLUE, ORANGE, GREY, RED = "#1d4ed8", "#c2410c", "#6b7280", "#dc2626"


def main():
    df = pd.read_csv(HERE / "data/eta_training.csv")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    dva = df.iloc[order[100_000:125_000]]
    model = joblib.load(HERE / "models/gbm_tuned.joblib")
    out = {}

    # ---- SHAP values on a 10k sample ------------------------------------
    t = time.time()
    samp = dva.sample(10_000, random_state=SEED)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(samp[FEATS])
    if isinstance(sv, list):
        sv = sv[1]
    print(f"SHAP on 10k objects: {time.time()-t:.1f} s")

    mean_abs = np.abs(sv).mean(axis=0)
    rank = sorted(zip(FEATS, mean_abs), key=lambda kv: -kv[1])
    out["shap_mean_abs"] = {k: round(float(v), 4) for k, v in rank}
    print("SHAP ranking:", [f"{k}:{v:.3f}" for k, v in rank])

    # permutation importance cross-check (log-loss based)
    t = time.time()
    pi = permutation_importance(model, samp[FEATS], samp.detected,
                                scoring="neg_log_loss", n_repeats=5,
                                random_state=SEED)
    out["permutation_importance"] = {
        f: round(float(m), 4) for f, m in
        sorted(zip(FEATS, pi.importances_mean), key=lambda kv: -kv[1])}
    print(f"permutation importance: {time.time()-t:.1f} s")

    # ---- summary (beeswarm) plot ----------------------------------------
    plt.figure(figsize=(7.6, 5.2))
    shap.summary_plot(sv, samp[FEATS].rename(columns=NICE), show=False,
                      max_display=9, plot_size=None)
    plt.title("SHAP summary — every dot is one object's per-feature "
              "contribution (log-odds)", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(HERE / "figures/session4_shap_summary.png", dpi=130)
    plt.close("all")

    # ---- dependence panels ----------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    panels = [("diam_km", "pV", axes[0, 0],
               "D x albedo: the threshold interaction"),
              ("a_au", "diam_km", axes[0, 1],
               "semimajor axis: the synodic-window geometry"),
              ("pV", "diam_km", axes[1, 0],
               "albedo: matters only near threshold"),
              ("eta", "diam_km", axes[1, 1],
               "beaming: hotter model = brighter = easier")]
    for feat, color_by, ax, title in panels:
        i, j = FEATS.index(feat), FEATS.index(color_by)
        x, c, y = samp[feat].values, samp[color_by].values, sv[:, i]
        sc = ax.scatter(x, c=np.log10(c) if color_by in ("diam_km", "pV")
                        else c, y=y, s=3, cmap="viridis", alpha=0.5)
        if feat in ("diam_km", "pV"):
            ax.set_xscale("log")
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label(("log10 " if color_by in ("diam_km", "pV") else "")
                     + NICE[color_by], fontsize=8)
        ax.set_xlabel(NICE[feat]); ax.set_ylabel("SHAP (log-odds)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.12)
    fig.suptitle("Session 4 — dependence: the model's learned physics, "
                 "feature by feature", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(HERE / "figures/session4_dependence.png", dpi=130)
    plt.close("all")

    # ---- monotonicity scan: does unconstrained wiggle? -------------------
    gbm = joblib.load(HERE / "models/gbm.joblib")
    dgrid = np.logspace(np.log10(0.5), np.log10(40), 200)
    n_viol, worst = 0, 0.0
    worst_slice = None
    for k in range(400):
        row = dva.sample(1, random_state=k)[FEATS]
        q = pd.concat([row] * 200, ignore_index=True)
        q["diam_km"] = dgrid
        p = gbm.predict_proba(q)[:, 1]
        drops = np.maximum(0, -(np.diff(p)))
        cum = float(drops.sum())
        if (drops > 1e-6).any():
            n_viol += 1
        if cum > worst:
            worst, worst_slice = cum, (row, p.copy())
    out["monotonicity"] = dict(slices=400, slices_with_any_drop=n_viol,
                               worst_cumulative_drop=round(worst, 4))
    print(f"monotonicity: {n_viol}/400 slices have drops; "
          f"worst cumulative drop {worst:.4f} in eta units")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dgrid, worst_slice[1], color=RED, lw=1.7,
            label=f"unconstrained GBM (worst slice of 400;\n"
                  f"cumulative drop {worst:.3f})")
    mono = joblib.load(HERE / "models/gbm_mono.joblib")
    q = pd.concat([worst_slice[0]] * 200, ignore_index=True)
    q["diam_km"] = dgrid
    ax.plot(dgrid, mono.predict_proba(q)[:, 1], color=BLUE, lw=1.7,
            label="monotone-constrained GBM, same slice")
    ax.set_xscale("log"); ax.grid(alpha=0.15)
    ax.set_xlabel("diameter (km), all other features fixed")
    ax.set_ylabel("P(detect)")
    ax.set_title("The monotonicity verdict: worst wiggle found in 400 "
                 "fixed-feature slices", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures/session4_monotonicity.png", dpi=130)

    (HERE / "metrics_session4.json").write_text(json.dumps(out, indent=2))
    print("wrote metrics_session4.json")


if __name__ == "__main__":
    main()
