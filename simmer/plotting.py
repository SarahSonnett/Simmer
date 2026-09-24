"""Matplotlib plots for Simmer products (matplotlib imported lazily)."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def plot_efficiency(combined, path, title: str | None = None):
    """Plot eta(D) with nested statistical and total (stat + systematic) bands.

    ``combined`` is the DataFrame from :meth:`simmer.ensemble.EnsembleResult.combine`
    (needs ``d_mid``, ``eta``, and the ``eta_stat_*`` / ``eta_tot_*`` columns).
    For a single (non-ensemble) run pass an efficiency table and it falls back to
    the Wilson band alone. Writes a PNG; returns its path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = combined["d_mid"].to_numpy()
    eta = combined["eta"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    if "eta_tot_lo" in combined:                       # ensemble: nested bands
        ax.fill_between(d, combined["eta_tot_lo"], combined["eta_tot_hi"],
                        color="#d97706", alpha=0.22, lw=0,
                        label="total (statistical $\\oplus$ systematic)")
        ax.fill_between(d, combined["eta_stat_lo"], combined["eta_stat_hi"],
                        color="#2563eb", alpha=0.30, lw=0, label="statistical (Wilson)")
    elif "eta_lo" in combined:                         # single run: Wilson band only
        ax.fill_between(d, combined["eta_lo"], combined["eta_hi"],
                        color="#2563eb", alpha=0.28, lw=0, label="statistical (Wilson)")

    ax.plot(d, eta, color="#1d4ed8", lw=2.0, zorder=5)
    ax.scatter(d, eta, s=11, color="#1d4ed8", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("diameter (km)")
    ax.set_ylabel("detection efficiency $\\eta$")
    ax.set_ylim(0, 1)
    ax.set_title(title or "detection efficiency $\\eta(D)$")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_sfd(debiased, fit, path, n_observed=None, title=None):
    """Plot the debiased SFD (N_true with total error bars) + the broken-law fit.

    ``debiased`` is the table from :func:`simmer.debias.debias` (needs ``d_mid``,
    ``d_lo``/``d_hi``, ``n_true``, ``n_true_lo``/``n_true_hi``, ``reliable``);
    ``fit`` is a :func:`simmer.sfd_fit.fit_sfd_mc` result. Writes a PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import sfd_fit

    d = debiased["d_mid"].to_numpy()
    d_lo_a = debiased["d_lo"].to_numpy()
    d_hi_a = debiased["d_hi"].to_numpy()
    nt = debiased["n_true"].to_numpy()
    lo, hi = debiased["n_true_lo"].to_numpy(), debiased["n_true_hi"].to_numpy()
    rel = (debiased["reliable"].to_numpy() if "reliable" in debiased
           else np.isfinite(nt))
    mm = rel & np.isfinite(nt) & (nt > 0)
    d_fit = float(fit.get("d_min", 0.0))               # fit floor (separable from completeness)
    d_comp = float(fit.get("completeness", {}).get("D_complete", 0.0))  # H&M reference

    # Cumulative SFD N(>D): sum the reliable per-bin debiased counts from large D
    # downward (suffix sum) -- the published convention (e.g. Vavra & Broz 2026,
    # Bottke et al. 2026) and directly the fitted cumulative slope, N(>D) ~ D**-alpha.
    # Cumulative uncertainty adds the per-bin bands in quadrature (treats bins as
    # independent -- an approximation; the systematic part of the eta band is partly
    # correlated across bins, which this under-states).
    order = np.argsort(d)
    ds = d[order]
    cumulate = lambda v, msk: np.cumsum(
        np.where(msk[order], np.asarray(v, float)[order], 0.0)[::-1])[::-1]
    Ncum = cumulate(nt, mm)
    clo = np.sqrt(cumulate(np.clip(nt - lo, 0, None) ** 2, mm))
    chi = np.sqrt(cumulate(np.clip(hi - nt, 0, None) ** 2, mm))
    mms = mm[order]
    a_s, b_s = mms & (ds >= d_fit), mms & (ds < d_fit)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if n_observed is not None:
        no = np.asarray(n_observed, float)
        okm = np.isfinite(no) & (no > 0)
        nobs_cum = cumulate(no, okm)
        oks = okm[order]
        ax.scatter(ds[oks], nobs_cum[oks], s=24, facecolors="none", edgecolors="#d97706",
                   linewidths=1.4, label=r"observed  $N_{\rm obs}(>D)$")
    yerr = np.vstack([clo, chi])
    ax.errorbar(ds[a_s], Ncum[a_s], yerr=yerr[:, a_s], fmt="o", ms=4.5,
                color="#2563eb", ecolor="#2563eb", elinewidth=1.1, capsize=0,
                label=r"debiased  $N_{\rm true}(>D)$ $\pm$(stat$\oplus$sys$\oplus$Poisson)")
    if b_s.any():
        ax.errorbar(ds[b_s], Ncum[b_s], yerr=yerr[:, b_s], fmt="o", ms=4.5,
                    mfc="none", color="#9ca3af", ecolor="#c7cbd1", elinewidth=1.0,
                    capsize=0, label="below fit floor (not fitted)")
    if d_comp > 0:                                      # H&M completeness reference (value in legend)
        ax.axvline(d_comp, color="#6b7280", ls=":", lw=1.3,
                   label=f"H&M completeness = {d_comp:.1f} km")
    # broken-power-law fit as a cumulative curve: suffix-sum the model's per-bin
    # counts the same way as the data, so both use identical binning.
    best = fit["central"]
    dlog_bins = np.log10(d_hi_a) - np.log10(d_lo_a)
    model_cum = cumulate(sfd_fit.predict_density(best, d) * dlog_bins, mm)
    ax.plot(ds[a_s], model_cum[a_s], color="#0f9d58", lw=2.2, label="broken power-law fit")
    for i, b in enumerate(best["break_diam_km"]):       # define the dashed line in legend
        ax.axvline(b, color="#0f9d58", ls="--", lw=1,
                   label="break diameter" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("diameter  $D$  (km)")
    ax.set_ylabel(r"cumulative number  $N(>D)$")

    # slopes + break locations as an annotation on the plot (upper-right, clear of data)
    a = fit["alpha_median"]
    note = ["cumulative α = " + " → ".join(f"{v:.2f}" for v in a)]
    if best["break_diam_km"]:
        note.append("break = " + ", ".join(f"{v:.1f}" for v in best["break_diam_km"]) + " km")
    ax.text(0.97, 0.96, "\n".join(note), transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=0.85))

    # sample size in the title
    ttl = title or "debiased SFD"
    if n_observed is not None:
        ttl += f"   ($N_{{\\rm obs}}$ = {int(np.nansum(np.asarray(n_observed, float)))})"
    ax.set_title(ttl)

    ax.grid(True, which="both", alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")   # bbox_inches captures the outside legend
    plt.close(fig)
    return Path(path)
