"""Census-measured P_known(D) for the debiased-NEOWISE leg (Phase-3 rerun).

See docs/phase3_rerun_design.md: this corrects ONLY the debiased leg
(N_obs/eta), which was built against the 2010-era catalog; the optical
extension leg keeps its own (2024-epoch) completeness model.

P_known(D) for a population = its beta-residence-weighted average of the
census's class-resolved 2010-epoch discovery-completeness map
(docs/pdisc_albedo.csv): sin(beta) = sin(i) x sin(u), u uniform. Below the
census's smallest D bin the map is held flat (flagged caveat; the extension's
own reliability gates govern there anyway).
"""
import os, glob, numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TABLE = os.path.join(_ROOT, "docs", "pdisc_albedo.csv")
DARK_COMPLEXES = {"C", "B", "P", "D", "X", "T"}     # albedo-group mapping
_cache = {}


def _grid(cls):
    if cls in _cache:
        return _cache[cls]
    t = pd.read_csv(_TABLE)
    t = t[t.cls == cls]
    bb = np.sort(t.b0.unique())
    db = np.sort(t.d0.unique())
    bs = np.array([0.5 * (b + t[t.b0 == b].b1.iloc[0]) for b in bb])
    ds = np.array([np.sqrt(d * t[t.d0 == d].d1.iloc[0]) for d in db])
    G = np.full((len(bs), len(ds)), np.nan)
    for _, r in t.iterrows():
        G[np.searchsorted(bb, r.b0), np.searchsorted(db, r.d0)] = r.p2010
    for i in range(G.shape[0]):                      # fill sparse large-D cells
        row = G[i]
        if np.isnan(row).all():
            row[:] = 0.98
        else:
            last = row[~np.isnan(row)][-1]
            row[np.isnan(row)] = max(last, 0.98)
    _cache[cls] = (bs, ds, G)
    return _cache[cls]


def _interp(cls, D, beta):
    bs, ds, G = _grid(cls)
    i = np.clip(np.interp(np.abs(beta), bs, np.arange(len(bs))), 0, len(bs) - 1)
    j = np.clip(np.interp(np.log(np.maximum(D, ds[0])), np.log(ds),
                          np.arange(len(ds))), 0, len(ds) - 1)
    i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
    i1, j1 = np.minimum(i0 + 1, len(bs) - 1), np.minimum(j0 + 1, len(ds) - 1)
    wi, wj = i - i0, j - j0
    return (G[i0, j0] * (1 - wi) * (1 - wj) + G[i1, j0] * wi * (1 - wj)
            + G[i0, j1] * (1 - wi) * wj + G[i1, j1] * wi * wj)


def pknown_curve(d_grid, sini_values, complex_letter, n_res=8, seed=17):
    """P_known,2010(D) on d_grid for a population.

    sini_values : member proper sin(i) array (a scalar works for a narrow
    family); complex_letter : broad complex ('C','S','X',...) mapped to the
    dark/bright census class.
    """
    cls = "dark" if str(complex_letter)[:1].upper() in DARK_COMPLEXES else "bright"
    s = np.atleast_1d(np.asarray(sini_values, float))
    s = s[np.isfinite(s) & (s > 0) & (s < 0.95)]
    if s.size == 0:
        s = np.array([0.1])
    rng = np.random.default_rng(seed)
    u = rng.uniform(-np.pi / 2, np.pi / 2, (s.size, n_res))
    beta = np.degrees(np.arcsin(np.abs(s[:, None] * np.sin(u)))).ravel()
    return np.array([_interp(cls, np.full_like(beta, d), beta).mean()
                     for d in np.atleast_1d(d_grid)])


def family_sini(famid):
    """Member sin(i) from the AFP membership files (None if absent)."""
    cand = glob.glob(os.path.expanduser(
        f"~/Desktop/work/MBA_SFDs/AFP_families/a{int(famid)}v*.txt"))
    if not cand:
        return None
    el = pd.read_csv(cand[0], sep=r"\s+", header=None)
    return pd.to_numeric(el[4], errors="coerce").dropna().to_numpy()
