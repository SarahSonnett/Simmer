"""Run a SynthPop *ensemble* through Simmer -> eta(D) with statistical AND
systematic error bands, and the debiased SFD carrying both.

Step 3 (+ 4) of the science chain. SynthPop's ``run_ensemble`` writes M synthetic
realizations of one population (each a bootstrap of that population's measured
properties) plus a ``manifest.json`` listing them. This module runs every member
catalog through the Simmer pipeline, on a **common diameter grid**, and stacks
the per-member eta_m(D):

* **statistical** band -- the Wilson interval *within* a member (counting noise
  at the per-run object count);
* **systematic** band -- the spread of eta_m *across* members (how much the
  result moves as the population's uncertain properties are varied);
* **total** -- the two added in quadrature.

Members are independent full Simmer runs (different orbits + physics), so they run
in parallel -- one process per member, with the survey frames loaded once per
worker. This is the payoff of the per-object deterministic seeding: the result is
identical however the members are distributed across processes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import os
import numpy as np
import pandas as pd

from .config import SimConfig
from . import efficiency, debias as debias_mod, sfd_fit, catalog_io
from .pipeline import simulate, object_summary
from .surveys.base import get_survey


# --- science-floor: model/calibration systematics per member ---------------

@dataclass
class ScienceFloor:
    """Literature-grounded model/calibration systematics, applied per member.

    These sit *on top of* SynthPop's data bootstrap (which the member catalogs
    already carry). Each member gets one coherent draw:

    * ``sensitivity_frac`` -- 1σ multiplicative scaling of the per-band 5σ
      sensitivity limit. Combines the ±10% mag->Jy color-calibration systematic
      for red (thermal) sources (WISE Explanatory Supplement §4.4h; Mainzer et
      al. 2011) with the ~±15% spread in quoted 5σ depths (Wright et al. 2010).
    * ``beaming_sigma`` -- 1σ additive shift of the NEATM beaming parameter η_IR
      (population mean ~1.0). Set to σ/√N ≈ 0.28/√50 ≈ 0.04, the uncertainty on
      the mean from Mainzer et al. 2011's 50 calibrator asteroids.

    Deliberately NOT included, to avoid double-counting: geometric albedo p_V
    (its systematic is the flux calibration propagated -> already in
    ``sensitivity_frac``), emissivity ε (absorbed into η by construction ->
    already in ``beaming_sigma``), and phase slope G (negligible at fixed D).
    """
    sensitivity_frac: float = 0.15
    beaming_sigma: float = 0.04
    seed: int = 0


def _floor_draw(floor: ScienceFloor, member: int):
    """Per-member (sensitivity scale, beaming offset). Member 0 = nominal."""
    if member == 0:
        return 1.0, 0.0
    rng = np.random.default_rng([floor.seed, member])
    sens_scale = float(np.exp(rng.normal(0.0, floor.sensitivity_frac)))  # >0
    beaming_offset = float(rng.normal(0.0, floor.beaming_sigma))
    return sens_scale, beaming_offset


# --- per-member efficiency on a fixed grid ---------------------------------

def _eta_on_grid(diam, detected, bins, z: float = 1.0):
    """Detection efficiency per fixed bin: aligned full-length arrays (NaN where
    a bin is empty), so members stack cleanly."""
    diam = np.asarray(diam, float)
    detected = np.asarray(detected).astype(float)
    idx = np.digitize(diam, bins) - 1
    nbin = len(bins) - 1
    ok = (idx >= 0) & (idx < nbin)
    n_input = np.zeros(nbin)
    n_det = np.zeros(nbin)
    np.add.at(n_input, idx[ok], 1.0)
    np.add.at(n_det, idx[ok], detected[ok])
    p, lo, hi = efficiency.wilson_interval(n_det, n_input, z)   # n=0 -> NaN
    return n_input, n_det, p, lo, hi


# --- worker plumbing (module-level so it pickles under spawn) ---------------

_W: dict = {}


def _init_worker(frames, frames_path, base_cfg, bins, z, floor):
    _W["frames"] = frames if frames is not None else pd.read_csv(frames_path)
    _W["base"] = base_cfg
    _W["bins"] = bins
    _W["z"] = z
    _W["floor"] = floor


def _run_member(task):
    m, catalog_path, name = task
    return _member_eta(m, catalog_path, name, _W["base"], _W["frames"],
                       _W["bins"], _W["z"], _W["floor"])


def _member_eta(m, catalog_path, name, base_cfg, frames, bins, z, floor=None):
    cfg = replace(base_cfg, catalog=Path(catalog_path), name=name)
    catalog = None
    if floor is not None:
        sens_scale, beaming_offset = _floor_draw(floor, m)
        if sens_scale != 1.0:                          # scale the 5σ limit coherently
            base_sens = get_survey(cfg.survey).sensitivity(
                replace(cfg, sensitivity_jy=None))
            cfg = replace(cfg, sensitivity_jy={b: v * sens_scale
                                               for b, v in base_sens.items()})
        if beaming_offset != 0.0:                      # shift the beaming column
            catalog = catalog_io.load_catalog(cfg.catalog)
            catalog = catalog.assign(
                eta=np.clip(catalog["eta"] + beaming_offset, 0.1, None))
    catalog, hits, _ = simulate(cfg, frames, verbose=False, catalog=catalog)
    summ = object_summary(catalog, hits, cfg.min_detections, cfg.detection_band,
                          cfg.tracklet_window_hr, cfg.link_efficiency, cfg.seed,
                          cfg.min_motion_deg_day, cfg.max_motion_deg_day)
    n_input, n_det, p, lo, hi = _eta_on_grid(
        catalog["diam_km"].to_numpy(), summ["detected"].to_numpy(), bins, z)
    return m, n_input, n_det, p, lo, hi


# --- result container ------------------------------------------------------

@dataclass
class EnsembleResult:
    bins: np.ndarray
    eta: np.ndarray          # (M, nbin) per-member efficiency
    eta_lo: np.ndarray       # (M, nbin) per-member Wilson lower
    eta_hi: np.ndarray       # (M, nbin) per-member Wilson upper
    n_input: np.ndarray      # (M, nbin)
    n_detected: np.ndarray   # (M, nbin)

    @property
    def d_mid(self):
        return np.sqrt(self.bins[:-1] * self.bins[1:])

    def combine(self) -> pd.DataFrame:
        """eta(D) with statistical, systematic and total bands.

        Each member's eta_m carries per-member Wilson (statistical) noise *and*
        varies with its bootstrapped fits (systematic). The spread across members
        is therefore the **total**, Var(eta_m) = sys^2 + stat^2; the isolated
        systematic is the excess over the mean per-member statistical variance,
        ``sys = sqrt(max(total^2 - stat^2, 0))``. So the total band is the raw
        cross-member scatter -- it is *not* stat and sys added a second time.
        """
        eta = np.nanmean(self.eta, axis=0)
        # statistical: typical within-member Wilson half-width (~1 sigma)
        stat = np.nanmean((self.eta_hi - self.eta_lo) / 2.0, axis=0)
        # raw cross-member scatter already contains the statistical part, so the
        # isolated systematic is the excess over it; the total floors at stat
        # (a noisy few-member std must never read below the statistical band).
        with np.errstate(invalid="ignore"):
            raw = np.nanstd(self.eta, axis=0, ddof=1)
        sys = np.sqrt(np.clip(raw ** 2 - stat ** 2, 0.0, None))  # isolated systematic
        total = np.sqrt(stat ** 2 + sys ** 2)                    # = max(raw, stat)
        clip = lambda a: np.clip(a, 0.0, 1.0)
        return pd.DataFrame({
            "d_lo": self.bins[:-1], "d_hi": self.bins[1:], "d_mid": self.d_mid,
            "n_members": np.sum(np.isfinite(self.eta), axis=0),
            "n_input_mean": np.nanmean(self.n_input, axis=0),
            "eta": eta,
            "eta_stat_lo": clip(eta - stat), "eta_stat_hi": clip(eta + stat),
            "eta_sys_lo": clip(eta - sys), "eta_sys_hi": clip(eta + sys),
            "eta_tot_lo": clip(eta - total), "eta_tot_hi": clip(eta + total),
        })

    def debias(self, n_observed, eta_min: float = 0.05, z: float = 1.0) -> pd.DataFrame:
        """Debias real observed counts, N_true = N_obs / eta, with stat + total bands.

        ``n_observed`` is the observed count per bin, aligned to ``self.bins``
        (see :func:`simmer.debias.bin_observed`). The inversion is run twice, with
        eta's *statistical* interval and with its *total* (statistical (+)
        systematic) interval; both fold in the Poisson uncertainty of the counts,
        and the total band is >= the statistical band by construction.
        """
        n_obs = np.asarray(n_observed, float)
        comb = self.combine()

        def _debias(lo_col, hi_col):
            eff = pd.DataFrame({
                "d_lo": comb["d_lo"], "d_hi": comb["d_hi"], "d_mid": comb["d_mid"],
                "eta": comb["eta"], "eta_lo": comb[lo_col], "eta_hi": comb[hi_col]})
            return debias_mod.debias(eff, n_obs, eta_min=eta_min, z=z)

        stat = _debias("eta_stat_lo", "eta_stat_hi")   # Poisson (+) Wilson-stat
        total = _debias("eta_tot_lo", "eta_tot_hi")    # Poisson (+) (stat (+) sys)
        out = stat.rename(columns={"n_true_lo": "n_true_stat_lo",
                                   "n_true_hi": "n_true_stat_hi"})
        out["n_true_tot_lo"] = total["n_true_lo"]
        out["n_true_tot_hi"] = total["n_true_hi"]
        return out

    def fit_sfd(self, n_observed, max_breaks: int = 2, eta_min: float = 0.05,
                z: float = 1.0) -> dict:
        """Fit a (broken) power law to the debiased SFD, with ensemble CIs.

        The central fit (BIC-selected number of breaks) is fit to the combined
        debiased SFD with its total band. Every member is then refit *with the
        same number of breaks*, and the 16/84th percentiles of the per-member
        cumulative slopes and break diameters give confidence intervals that
        carry the statistical + systematic uncertainty. Returns the central fit
        plus ``alpha`` and ``break_diam`` medians and [lo, hi] bands.
        """
        n_obs = np.asarray(n_observed, float)
        db = self.debias(n_obs, eta_min=eta_min, z=z)
        central = sfd_fit.fit_sfd(
            db["d_lo"].to_numpy(), db["d_hi"].to_numpy(), db["n_true"].to_numpy(),
            db["n_true_tot_lo"].to_numpy(), db["n_true_tot_hi"].to_numpy(),
            reliable=db["reliable"].to_numpy(), max_breaks=max_breaks)
        k = central["best"]["n_breaks"]

        d_lo, d_hi = self.bins[:-1], self.bins[1:]
        alphas, breaks = [], []
        for mi in range(self.eta.shape[0]):
            eta_m, elo, ehi = self.eta[mi], self.eta_lo[mi], self.eta_hi[mi]
            with np.errstate(divide="ignore", invalid="ignore"):
                n_m = n_obs / eta_m
                n_lo, n_hi = n_obs / ehi, n_obs / elo
            rel = np.isfinite(eta_m) & (eta_m >= eta_min) & (elo > 0)
            try:
                f = sfd_fit.fit_sfd(d_lo, d_hi, n_m, n_lo, n_hi, reliable=rel,
                                    force_breaks=k)["best"]
            except ValueError:
                continue
            alphas.append(f["alpha"])
            breaks.append(f["break_diam_km"])

        alphas = np.array(alphas)
        pct = lambda a, p: np.nanpercentile(a, p, axis=0).tolist()
        res = {"central": central["best"], "candidates": central["candidates"],
               "n_breaks": k, "n_member_fits": len(alphas),
               "alpha_median": pct(alphas, 50),
               "alpha_lo": pct(alphas, 16), "alpha_hi": pct(alphas, 84)}
        if k > 0 and breaks:
            b = np.array(breaks)
            res.update(break_diam_median=pct(b, 50),
                       break_diam_lo=pct(b, 16), break_diam_hi=pct(b, 84))
        return res

    def save(self, out_dir, name: str, n_observed=None, max_breaks: int = 2,
             eta_min: float = 0.05, z: float = 1.0):
        """Write the ensemble products to disk. Returns the paths written.

        Always writes the combined efficiency table. If ``n_observed`` (real
        detected counts per bin) is given, also writes the debiased SFD and the
        (broken) power-law fit -- slopes, break diameters and their CIs -- as
        JSON.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [out_dir / f"{name}_efficiency_ensemble.csv"]
        self.combine().to_csv(written[0], index=False)
        if n_observed is not None:
            db_path = out_dir / f"{name}_debiased_sfd.csv"
            self.debias(n_observed, eta_min=eta_min, z=z).to_csv(db_path, index=False)
            fit_path = out_dir / f"{name}_sfd_fit.json"
            fit = self.fit_sfd(n_observed, max_breaks=max_breaks,
                               eta_min=eta_min, z=z)
            fit_path.write_text(json.dumps(fit, indent=2, default=float))
            written += [db_path, fit_path]
        return written


# --- driver ----------------------------------------------------------------

def _common_bins(catalog_paths, n_bins):
    lo, hi = np.inf, -np.inf
    for p in catalog_paths:
        d = pd.read_csv(p, usecols=["diam_km"])["diam_km"].to_numpy()
        d = d[d > 0]
        lo, hi = min(lo, d.min()), max(hi, d.max())
    return np.geomspace(lo, hi, n_bins + 1)


def run_ensemble(manifest: Path, base_cfg: SimConfig,
                 frames: pd.DataFrame | None = None,
                 frames_path: Path | None = None,
                 bins=None, n_bins: int = 20, n_workers: int | None = None,
                 z: float = 1.0, floor: "ScienceFloor | None" = None,
                 verbose: bool = True) -> EnsembleResult:
    """Run every member of a SynthPop ensemble through Simmer and stack eta(D).

    ``manifest`` is a SynthPop ``*_ensemble_manifest.json``. Supply the survey
    frames once (``frames`` in memory, or ``frames_path`` to load per worker).
    Set ``n_workers=1`` to run serially. Pass a :class:`ScienceFloor` to add the
    model/calibration systematics (sensitivity + beaming) on top of the catalogs'
    data bootstrap -- widening the systematic band, leaving the central eta and
    the statistical band unchanged.

    With ``n_workers > 1`` this spawns processes, so the calling script **must**
    guard its entry point with ``if __name__ == "__main__":`` (a macOS/Windows
    ``spawn`` requirement). ``frames_path`` is preferred over ``frames`` for the
    parallel path -- each worker loads the frames once rather than paying to
    pickle a large DataFrame.

    MEMORY: every worker runs an independent ``run_simmer`` with its own copy of
    the frames (no shared memory under ``spawn``), so budget ~1.3 GB per worker
    against the near-ecliptic cadence -- roughly flat in object count, since the
    field-of-view match refines one time-bin at a time. The default ``n_workers``
    is half the cores; raise it if you have the RAM to spare, and lower it when
    other heavy jobs are running so the machine does not run out of memory.
    """
    manifest = Path(manifest)
    man = json.loads(manifest.read_text())
    mdir = manifest.parent
    tasks = [(e["member"], mdir / e["catalog"], f"{man['name']}_m{e['member']:03d}")
             for e in man["members"]]

    def log(msg):
        if verbose:
            print(f"[sim-ensemble:{man['name']}] {msg}")

    if bins is None:
        bins = _common_bins([t[1] for t in tasks], n_bins)
    bins = np.asarray(bins, float)
    nbin = len(bins) - 1
    M = len(tasks)
    log(f"{M} members, {nbin} diameter bins "
        f"[{bins[0]:.2f}, {bins[-1]:.2f}] km")

    if n_workers is None:
        # Half the cores by default: each worker keeps its own full copy of the
        # frames, so an aggressive count can exhaust RAM (especially alongside
        # other heavy jobs). Raise it explicitly when you have the memory.
        n_workers = max(1, min(M, (os.cpu_count() or 2) // 2))

    eta = np.full((M, nbin), np.nan)
    eta_lo = np.full((M, nbin), np.nan)
    eta_hi = np.full((M, nbin), np.nan)
    n_input = np.zeros((M, nbin))
    n_detected = np.zeros((M, nbin))

    def _store(res):
        m, ni, nd, p, lo, hi = res
        eta[m], eta_lo[m], eta_hi[m] = p, lo, hi
        n_input[m], n_detected[m] = ni, nd

    if n_workers == 1:
        fr = frames if frames is not None else (
            pd.read_csv(frames_path) if frames_path else None)
        for m, cat, name in tasks:
            _store(_member_eta(m, cat, name, base_cfg, fr, bins, z, floor))
            log(f"member {m:3d}/{M} done")
    else:
        log(f"running on {n_workers} workers")
        with ProcessPoolExecutor(
                max_workers=n_workers, initializer=_init_worker,
                initargs=(frames, frames_path, base_cfg, bins, z, floor)) as ex:
            for res in ex.map(_run_member, tasks):
                _store(res)
                log(f"member {res[0]:3d}/{M} done")

    return EnsembleResult(bins=bins, eta=eta, eta_lo=eta_lo, eta_hi=eta_hi,
                          n_input=n_input, n_detected=n_detected)
