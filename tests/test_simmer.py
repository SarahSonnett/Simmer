"""Unit tests for Simmer. Runnable with pytest or directly:

    /usr/local/bin/python tests/test_simmer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simmer import ephemeris, thermal, lightcurve, detect, geometry, seeding, bandpass  # noqa: E402
from simmer import run_simmer, efficiency, debias                         # noqa: E402
from simmer import optical, allsurvey                                     # noqa: E402
from simmer.config import SimConfig                                       # noqa: E402
from simmer.catalog_io import load_catalog, REQUIRED_COLUMNS              # noqa: E402
from simmer.surveys import neowise                                        # noqa: E402
from simmer.surveys.base import get_survey                                # noqa: E402


# --- ephemeris -------------------------------------------------------------

def test_kepler_solves_identity():
    # M = E - e sinE; recompute M from the solved E.
    rng = np.random.default_rng(0)
    M = rng.uniform(-np.pi, np.pi, 1000)
    e = rng.uniform(0, 0.6, 1000)
    E = ephemeris.solve_kepler(M, e)
    M_back = E - e * np.sin(E)
    wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
    assert np.max(np.abs(wrap(M_back - M))) < 1e-8


def test_circular_orbit_radius_constant():
    # A circular orbit (e=0) stays at r = a as M advances.
    elements = {"a_au": np.array([2.5]), "e": np.array([0.0]),
                "i_deg": np.array([5.0]), "node_deg": np.array([10.0]),
                "argperi_deg": np.array([0.0]), "M_deg": np.array([0.0])}
    for dt in (0.0, 100.0, 365.0):
        xyz = ephemeris.elements_to_helio_xyz(elements, 2455197.5, 2455197.5 + dt)
        assert abs(np.linalg.norm(xyz) - 2.5) < 1e-9


# --- thermal (NEATM) -------------------------------------------------------

def test_neatm_scalings():
    base = dict(diam_km=50.0, pV=0.1, eta=1.0, r_au=2.7, delta_au=1.8,
                alpha_deg=15.0, band="W3")
    f0 = thermal.flux_band(**base)[0]
    # flux ~ D^2
    f_big = thermal.flux_band(**{**base, "diam_km": 100.0})[0]
    assert abs(f_big / f0 - 4.0) < 0.05
    # flux ~ 1/delta^2
    f_far = thermal.flux_band(**{**base, "delta_au": 3.6})[0]
    assert abs(f_far / f0 - 0.25) < 0.05
    # closer to the Sun is hotter -> brighter
    f_close = thermal.flux_band(**{**base, "r_au": 2.0})[0]
    assert f_close > f0
    assert f0 > 0


def test_subsolar_temperature_sane():
    T = thermal.subsolar_temperature(0.1, 2.7, 1.0)
    assert 150 < T < 250          # MBA dayside temperatures, K


# --- bandpass / RSR color correction --------------------------------------

def test_reference_spectrum_color_correction_is_unity():
    # For the WISE reference spectrum F_nu ~ nu^-2, the reported flux must equal
    # F_nu(lam_iso) and f_c must be 1 (both by construction).
    for band in ("W3", "W4"):
        bp = bandpass.get_bandpass(band)
        lam, w = bp.lam_m, bp.R / bp.lam_m
        # reported flux of a source with F_nu = (lam/lam_iso)^2 (so F(iso)=1)
        num = np.trapezoid((lam / bp.lam_iso_m) ** 2 * w, lam)
        f_rep = num / bp.ref_norm
        assert abs(f_rep - 1.0) < 1e-9


def test_wise_flux_scalings_and_band_ratio():
    base = dict(diam_km=50.0, pV=0.1, eta=1.0, r_au=2.7, delta_au=1.8,
                alpha_deg=15.0)
    f3 = thermal.flux_band_wise(band="W3", **base)[0]
    f4 = thermal.flux_band_wise(band="W4", **base)[0]
    assert f3 > 0 and f4 > 0
    # D^2 and 1/Delta^2 scalings survive the RSR convolution
    f3_big = thermal.flux_band_wise(band="W3", **{**base, "diam_km": 100.0})[0]
    assert abs(f3_big / f3 - 4.0) < 0.05
    # a ~200 K MBA is brighter at 22 um (W4) than 12 um (W3)
    assert f4 > f3


def test_flux_lut_matches_direct_quadrature():
    # the (T_ss, alpha) lookup must reproduce the direct surface quadrature to
    # interpolation accuracy across a realistic MBA range of geometry.
    rng = np.random.default_rng(0)
    n = 2000
    kw = dict(diam_km=rng.uniform(2, 100, n), pV=rng.uniform(0.03, 0.4, n),
              eta=rng.uniform(0.8, 1.3, n), r_au=rng.uniform(1.9, 3.6, n),
              delta_au=rng.uniform(1.0, 4.0, n), alpha_deg=rng.uniform(0, 30, n))
    for band in ("W3", "W4"):
        direct = thermal.flux_band_wise(band=band, **kw)
        lut = thermal.flux_band_wise_lut(band=band, **kw)
        rel = np.abs(lut - direct) / direct
        assert rel.max() < 3e-3            # < 0.3% worst case
        assert np.median(rel) < 5e-4


def test_blackbody_color_correction_reasonable():
    # f_c for asteroid-temperature blackbodies should be O(1) and, for W3's wide
    # band, differ from unity by more than a few percent for a cool source.
    fc_w3 = bandpass.color_correction_blackbody(200.0, "W3")[0]
    fc_w4 = bandpass.color_correction_blackbody(200.0, "W4")[0]
    assert 0.7 < fc_w3 < 1.3 and 0.7 < fc_w4 < 1.3
    # hot (Rayleigh-Jeans, F_nu ~ nu^2) vs cool gives different corrections
    fc_hot = bandpass.color_correction_blackbody(5000.0, "W3")[0]
    assert abs(fc_hot - fc_w3) > 0.02


# --- lightcurve ------------------------------------------------------------

def test_amplitude_zero_for_sphere():
    # b/a = 1 (sphere) -> no variation at any aspect.
    amp = lightcurve.amplitude_mag(np.array([1.0, 1.0]), np.array([90.0, 30.0]))
    assert np.allclose(amp, 0.0, atol=1e-12)


def test_amplitude_max_equator_on():
    # Most elongated signal is seen equator-on (theta=90), least pole-on (0).
    amp_equator = lightcurve.amplitude_mag(np.array([0.5]), np.array([90.0]))[0]
    amp_pole = lightcurve.amplitude_mag(np.array([0.5]), np.array([0.0]))[0]
    assert amp_pole == 0.0
    assert amp_equator > 0.5      # a:b=2 equator-on -> 2.5*log10(2) ~ 0.75 mag


def test_aspect_angle_folds_to_90():
    los = np.array([[0.0, 0.0, 1.0]])
    # pole pointing straight away vs straight toward give the same aspect (0).
    t_up = lightcurve.aspect_angle_deg(np.array([90.0]), np.array([0.0]), los)[0]
    t_dn = lightcurve.aspect_angle_deg(np.array([-90.0]), np.array([0.0]), los)[0]
    assert abs(t_up) < 1e-6 and abs(t_dn) < 1e-6


def test_ellipsoid_lightcurve_double_peaked_and_broad_max():
    # The rotating-ellipsoid curve is double-peaked (2 maxima / rotation) with
    # broad maxima and narrow minima -- not a symmetric sinusoid.
    b, asp = 0.4, 90.0
    phase = np.linspace(0, 1, 1000, endpoint=False)
    flux = lightcurve.modulate_flux(np.ones(1000), b, asp, phase)
    prev, nxt = np.roll(flux, 1), np.roll(flux, -1)
    n_max = int(np.sum((flux > prev) & (flux > nxt)))
    n_min = int(np.sum((flux < prev) & (flux < nxt)))
    assert n_max == 2 and n_min == 2
    # broad max / narrow min: midway (phase 0.375) between a max (0.25) and a
    # min (0.5) the flux stays ABOVE the average of the two -- it lingers high.
    fmax = lightcurve.modulate_flux(1.0, b, asp, np.array([0.25]))[0]
    fmin = lightcurve.modulate_flux(1.0, b, asp, np.array([0.5]))[0]
    fmid = lightcurve.modulate_flux(1.0, b, asp, np.array([0.375]))[0]
    assert fmid > 0.5 * (fmax + fmin)


def test_ellipsoid_amplitude_matches_sheppard_jewitt():
    # peak-to-peak of the physical curve must equal the Sheppard & Jewitt dm.
    rng = np.random.default_rng(0)
    b = rng.uniform(0.35, 0.95, 50)
    asp = rng.uniform(5, 90, 50)
    ph = np.linspace(0, 1, 400, endpoint=False)
    for bi, ai in zip(b, asp):
        f = lightcurve.modulate_flux(np.ones_like(ph), bi, ai, ph)
        dm_curve = 2.5 * np.log10(f.max() / f.min())
        dm_sj = lightcurve.amplitude_mag(np.array([bi]), np.array([ai]))[0]
        assert abs(dm_curve - dm_sj) < 3e-3
        assert abs(f.mean() - 1.0) < 5e-3          # mean flux preserved


def test_rotation_phase_coherent_in_time():
    # Same object at two close times -> phase advances by dt/period; a full
    # period later the modulation repeats.
    cfg = SimConfig(name="t", rot_period_min_hr=6.0, rot_period_max_hr=6.0)
    rot = lightcurve.assign_rotation(1, cfg)
    assert abs(rot.period_hr[0] - 6.0) < 1e-9          # degenerate range -> exact
    p0 = lightcurve.rotational_phase(0.0, 0.0, rot.period_hr, rot.phase0)[0]
    p_full = lightcurve.rotational_phase(0.25, 0.0, rot.period_hr, rot.phase0)[0]  # +6h
    assert abs(((p_full - p0) % 1.0)) < 1e-9           # one full rotation later


# --- detection -------------------------------------------------------------

def test_detection_threshold_and_decimation():
    hits = pd.DataFrame({
        "band": ["W3"] * 1000,
        "flux_jy": np.concatenate([np.full(500, 0.01), np.full(500, 1e-6)]),
        "obj_idx": np.arange(1000),
        "frame_idx": np.zeros(1000, dtype=int),
    })
    cfg = SimConfig(name="t", bad_pixel_fraction=0.0, photometric_noise=False)
    out = detect.apply_detection(hits, {"W3": 0.001}, cfg)
    assert out["above_threshold"].sum() == 500     # only the bright half
    assert out["detected"].sum() == 500            # no decimation

    cfg2 = SimConfig(name="t", bad_pixel_fraction=0.01, photometric_noise=False)
    out2 = detect.apply_detection(hits, {"W3": 0.001}, cfg2)
    # ~1% of the 500 above-threshold dropped, deterministically
    assert 480 <= out2["detected"].sum() <= 500
    assert out2["detected"].sum() < out2["above_threshold"].sum()
    # keyed by (obj_idx, frame_idx): identical result on a reordered frame
    shuf = hits.sample(frac=1.0, random_state=7).reset_index(drop=True)
    out3 = detect.apply_detection(shuf, {"W3": 0.001}, cfg2)
    assert out3["detected"].sum() == out2["detected"].sum()


def test_photometric_noise_soft_threshold():
    lim, n = 0.002, 6000
    at_limit = pd.DataFrame({"band": ["W3"] * n, "flux_jy": np.full(n, lim),
                             "obj_idx": np.arange(n), "frame_idx": np.arange(n)})
    cfg = SimConfig(name="t", bad_pixel_fraction=0.0, photometric_noise=True,
                    photometric_frac_err=0.0)
    out = detect.apply_detection(at_limit, {"W3": lim}, cfg)
    # a source exactly at the 5-sigma limit is detected ~50% of the time
    assert 0.46 < out["above_threshold"].mean() < 0.54
    # bright sources always pass, deeply-faint never
    hi_lo = pd.DataFrame({"band": ["W3"] * n,
                          "flux_jy": np.where(np.arange(n) % 2 == 0, 12 * lim, 0.01 * lim),
                          "obj_idx": np.arange(n), "frame_idx": np.arange(n)})
    o2 = detect.apply_detection(hi_lo, {"W3": lim}, cfg)
    a = o2["above_threshold"].to_numpy()
    assert a[0::2].all() and not a[1::2].any()
    # deterministic / batch-invariant
    o3 = detect.apply_detection(at_limit.sample(frac=1.0, random_state=3).reset_index(drop=True),
                                {"W3": lim}, cfg)
    assert o3["above_threshold"].sum() == out["above_threshold"].sum()


# --- ingestion contract ----------------------------------------------------

def test_catalog_contract(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    df = pd.DataFrame({c: [1.0] for c in REQUIRED_COLUMNS})
    df["id"] = ["x"]
    df["b_over_a"] = [0.7]
    df["diam_km"] = [10.0]
    p = d / "x_synthpop.csv"
    df.to_csv(p, index=False)
    loaded = load_catalog(p)
    assert list(loaded.columns) == list(df.columns) or set(REQUIRED_COLUMNS) <= set(loaded.columns)

    # bad b_over_a rejected
    df.loc[0, "b_over_a"] = 1.5
    df.to_csv(p, index=False)
    try:
        load_catalog(p)
        assert False, "expected ValueError for b_over_a > 1"
    except ValueError:
        pass


# --- survey registry / prefilter ------------------------------------------

def test_survey_registry_and_prefilter():
    survey = get_survey("neowise_cryo")
    assert survey.default_bands == ("W3", "W4")
    frames = neowise.synthetic_frames(n_scans=100, seed=2)
    cfg = SimConfig(name="t", max_ecl_lat_deg=30.0)
    kept = neowise._prefilter(frames, cfg)
    assert len(kept) > 0
    assert set(kept["band"]) <= {"W3", "W4"}


# --- seeding ---------------------------------------------------------------

def test_hash01_uniform_and_batch_invariant():
    idx = np.arange(200_000)
    u = seeding.hash01(0, seeding.STREAM_ROT_PHASE, idx)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01                 # ~uniform
    # keyed by index -> a shuffled query returns the same per-key values
    perm = np.random.default_rng(0).permutation(idx)
    u_perm = seeding.hash01(0, seeding.STREAM_ROT_PHASE, perm)
    assert np.allclose(u_perm, u[perm])
    # different streams are decorrelated
    v = seeding.hash01(0, seeding.STREAM_ROT_PERIOD, idx)
    assert abs(np.corrcoef(u, v)[0, 1]) < 0.01


# --- FOV footprint ---------------------------------------------------------

def test_point_in_quad_and_edge_inset():
    # unit square corners in perimeter order
    qx = np.array([[-1.0, 1.0, 1.0, -1.0]])
    qy = np.array([[-1.0, -1.0, 1.0, 1.0]])
    assert geometry._point_in_quad(np.array([0.0]), np.array([0.0]), qx, qy)[0]
    assert not geometry._point_in_quad(np.array([2.0]), np.array([0.0]), qx, qy)[0]
    # per-band inset shrink factor: W3 (9 px / 1016) vs W4 (5 px / 508)
    cfg = SimConfig(name="t")
    s = geometry._inset_scale(np.array(["W3", "W4"]), cfg)
    assert abs(s[0] - (1 - 9 / 508)) < 1e-12
    assert abs(s[1] - (1 - 5 / 254)) < 1e-12
    # a point just inside the W4 edge is rejected once the 5-px border is applied
    near_edge = 1.0 - 0.5 * (5 / 254)                 # within the inset margin
    inside_raw = geometry._point_in_quad(np.array([near_edge]), np.array([0.0]), qx, qy)[0]
    cx = qx.mean(1, keepdims=True); cy = qy.mean(1, keepdims=True)
    qxi = cx + s[1] * (qx - cx); qyi = cy + s[1] * (qy - cy)
    inside_inset = geometry._point_in_quad(np.array([near_edge]), np.array([0.0]), qxi, qyi)[0]
    assert inside_raw and not inside_inset


def test_match_frames_quad_vs_circular_consistent():
    # One object, one frame whose footprint straddles the object; check the quad
    # path and the corner-less fallback both associate it, with real geometry.
    cat = pd.DataFrame({
        "id": ["o0"], "diam_km": [30.0], "b_over_a": [0.8],
        "pole_beta_deg": [10.0], "pole_lambda_deg": [40.0],
        "a_au": [2.6], "e": [0.1], "i_deg": [5.0], "node_deg": [50.0],
        "argperi_deg": [20.0], "M_deg": [123.0], "pV": [0.1], "eta": [1.0], "H": [12.0],
    })
    cfg = SimConfig(name="t")
    # place a frame exactly at the object's computed sky position
    el = {k: cat[k].to_numpy() for k in
          ("a_au", "e", "i_deg", "node_deg", "argperi_deg", "M_deg")}
    t_jd = cfg.epoch_jd + 30.0
    geo = ephemeris.observe(el, cfg.epoch_jd, t_jd)
    ra0, dec0 = float(geo.ra_deg[0]), float(geo.dec_deg[0])
    mjd = t_jd - 2400000.5
    half = 0.35  # deg, a box comfortably containing the centre
    base = dict(mjd=[mjd], ra=[ra0], dec=[dec0], band=["W3"])
    corners = dict(
        ra1=[ra0 - half], dec1=[dec0 - half], ra2=[ra0 + half], dec2=[dec0 - half],
        ra3=[ra0 + half], dec3=[dec0 + half], ra4=[ra0 - half], dec4=[dec0 + half])
    quad = geometry.match_frames(cat, pd.DataFrame({**base, **corners}), cfg)
    circ = geometry.match_frames(cat, pd.DataFrame(base), cfg)
    assert len(quad) == 1 and len(circ) == 1
    assert quad["r_au"][0] > 1.0 and quad["delta_au"][0] > 0.0
    assert set(["frame_idx", "alpha_deg", "los_x"]).issubset(quad.columns)


def test_match_frames_high_inclination_coverage_warning():
    import warnings
    frames = neowise.synthetic_frames(n_scans=50, seed=3)
    base = dict(id=["o0"], diam_km=[30.0], b_over_a=[0.8], pole_beta_deg=[10.0],
                pole_lambda_deg=[40.0], a_au=[2.6], e=[0.1], node_deg=[50.0],
                argperi_deg=[20.0], M_deg=[123.0], pV=[0.1], eta=[1.0], H=[12.0])
    cfg = SimConfig(name="t", max_ecl_lat_deg=30.0)
    # i > cut -> warns about possible under-coverage
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        geometry.match_frames(pd.DataFrame({**base, "i_deg": [35.0]}), frames, cfg)
        assert any("inclination" in str(x.message) for x in w)
    # i < cut -> no coverage warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        geometry.match_frames(pd.DataFrame({**base, "i_deg": [5.0]}), frames, cfg)
        assert not any("inclination" in str(x.message) for x in w)


# --- all-surveys small-size extension --------------------------------------

def test_jedicke_bright_fraction_monotone():
    # no bias when albedo ratio is 1; over-representation of the bright complex when >1
    assert abs(allsurvey.jedicke_bright_fraction(0.3, 1.0, 2.0) - 0.3) < 1e-12
    assert allsurvey.jedicke_bright_fraction(0.3, 4.0, 2.0) > 0.3
    # inverse limits: f=0 -> 0, f=1 -> 1 regardless of R
    assert allsurvey.jedicke_bright_fraction(0.0, 5.0, 2.0) == 0.0
    assert abs(allsurvey.jedicke_bright_fraction(1.0, 5.0, 2.0) - 1.0) < 1e-12


def test_hm_rollover_completeness_dark_rolls_off_later():
    d = np.geomspace(1.0, 20.0, 40)
    bright = optical.AlbedoModel(np.full(200, 0.25))
    dark = optical.AlbedoModel(np.full(200, 0.05))
    Cb = allsurvey.hm_rollover_completeness(d, 16.0, bright)
    Cd = allsurvey.hm_rollover_completeness(d, 16.0, dark)
    # monotone non-decreasing, bounded (0,1]
    assert np.all(np.diff(Cb) >= -1e-9) and Cb.max() <= 1.0 and Cb.min() > 0
    # at fixed D below the limit, a dark family is LESS complete than a bright one
    mid = (d > 2) & (d < 6)
    assert np.mean(Cd[mid]) < np.mean(Cb[mid])


def test_fit_logistic_completeness_recovers_shape():
    d = np.geomspace(1.0, 30.0, 30)
    true = 1.0 / (1.0 + np.exp(-6.0 * (np.log10(d) - np.log10(3.0))))
    fn = allsurvey.fit_logistic_completeness(d, true)
    got = fn(d)
    assert np.max(np.abs(got - true)) < 0.05
    assert np.all(np.diff(got) >= -1e-6)              # monotone increasing


def test_hm_h_limit_finds_peak():
    rng = np.random.default_rng(0)
    # H distribution rising then rolling over near 17.0
    H = np.concatenate([rng.uniform(10, 17, 4000), rng.normal(17.0, 0.3, 2000)])
    hlim = allsurvey.hm_h_limit(H)
    assert 16.5 <= hlim <= 17.5


def test_extend_sfd_reliability_floor():
    d = np.geomspace(0.5, 20.0, 60)
    d_complete = 4.0
    n_known = np.maximum(200.0 * (d.min() / d) ** 1.5, 1.0)   # rises toward small D
    n_true = np.where(d >= d_complete, n_known * 1.1, np.nan)
    dark = optical.AlbedoModel(np.full(200, 0.05))           # dark -> incomplete early
    out = allsurvey.extend_sfd(d, n_known, n_true, d_complete, 17.0, dark, c_floor=0.05)
    # reliability floor is >= the smallest grid D and unreliable bins are flagged below it
    assert out["d_reliable_min"] >= d.min()
    assert np.all(out["reliable"][d >= d_complete])          # overlap always reliable
    # no reported extension divides by less than the floor -> N_true stays finite/bounded
    assert np.all(np.isfinite(out["n_true"]))
    assert out["n_true"].max() < 1e7                         # no divide-by-~0 blow-up


def test_extend_sfd_lower_bound_and_overlap():
    d = np.geomspace(1.0, 20.0, 50)
    d_complete = 4.0
    # synthetic: known counts rise toward small D; "true" known only above completeness
    n_known = 50.0 * (d[::-1].cumsum()[::-1] / d.sum())   # any decreasing-in-D cumulative
    n_known = np.maximum(n_known, 1.0)
    n_true = np.where(d >= d_complete, n_known * 1.15, np.nan)  # 87% complete in overlap
    alb = optical.AlbedoModel(np.concatenate([np.full(150, 0.22), np.full(50, 0.05)]))
    out = allsurvey.extend_sfd(d, n_known, n_true, d_complete, 16.5, alb)
    # hard lower bound everywhere
    assert np.all(out["n_true"] >= out["lower_bound"] - 1e-6)
    # above completeness reproduces the NEOWISE debiased input
    above = d >= d_complete
    assert np.allclose(out["n_true_hm"][above], n_true[above], equal_nan=False)
    # below completeness the correction lifts N_true above the raw known count
    below = d < d_complete
    assert np.all(out["n_true"][below] >= n_known[below] - 1e-6)
    assert np.mean(out["n_true"][below]) > np.mean(n_known[below])
    # both completeness forms are valid probabilities
    for key in ("C_hm", "C_log"):
        assert np.all((out[key] > 0) & (out[key] <= 1.0 + 1e-9))


def test_apply_lightcurve_flag(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    m = 60
    cat = pd.DataFrame({
        "id": [f"x_{k}" for k in range(m)], "diam_km": rng.uniform(20, 80, m),
        "b_over_a": rng.uniform(0.4, 0.6, m),        # elongated -> visible lightcurve
        "pole_beta_deg": rng.uniform(-30, 30, m), "pole_lambda_deg": rng.uniform(0, 360, m),
        "a_au": rng.uniform(2.2, 3.0, m), "e": rng.uniform(0, 0.1, m),
        "i_deg": rng.uniform(0, 15, m), "node_deg": rng.uniform(0, 360, m),
        "argperi_deg": rng.uniform(0, 360, m), "M_deg": rng.uniform(0, 360, m),
        "pV": rng.uniform(0.05, 0.2, m), "eta": np.full(m, 1.0), "H": rng.uniform(9, 13, m)})
    cp = d / "x_synthpop.csv"; cat.to_csv(cp, index=False)
    frames = neowise.synthetic_frames(n_scans=6000, seed=2, days=40)
    for tag, lc in (("e", True), ("s", False)):
        run_simmer(SimConfig(name=tag, catalog=cp, out_dir=d, seed=0, apply_lightcurve=lc),
                   frames=frames, verbose=False)
    e = pd.read_csv(d / "e_detections.csv")
    sp = pd.read_csv(d / "s_detections.csv")
    assert len(e) > 0 and len(sp) > 0
    # spheres: flux is the unmodulated NEATM mean; ellipsoid: modulated
    assert np.allclose(sp["flux_jy"], sp["flux_thermal_jy"])
    assert not np.allclose(e["flux_jy"], e["flux_thermal_jy"])


def test_wilson_interval_edges():
    p, lo, hi = efficiency.wilson_interval(np.array([0, 50, 100]),
                                           np.array([100, 100, 100]), z=1.0)
    assert p[0] == 0 and lo[0] <= 1e-12 and hi[0] > 0     # k=0 -> eta 0, upper > 0
    assert p[2] == 1 and hi[2] >= 1 - 1e-12 and lo[2] < 1  # k=n -> eta 1, lower < 1
    assert abs(p[1] - 0.5) < 1e-12 and lo[1] < 0.5 < hi[1]
    # a tighter (larger-N) sample gives a narrower interval
    _, lo2, hi2 = efficiency.wilson_interval(5000, 10000, z=1.0)
    _, lo3, hi3 = efficiency.wilson_interval(50, 100, z=1.0)
    assert (hi2 - lo2) < (hi3 - lo3)


def test_efficiency_function_recovers_truth():
    rng = np.random.default_rng(0)
    n = 40000
    D = np.exp(rng.uniform(np.log(0.5), np.log(20), n))      # uniform in log D
    truth = 1.0 / (1.0 + np.exp(-(np.log(D) - np.log(3.0)) * 3))  # logistic, D50=3 km
    det = rng.random(n) < truth
    eff = efficiency.efficiency_function(D, det, n_bins=16)
    # monotonic rise and recovery of the underlying curve at bin centres
    assert (eff["eta"].diff().dropna() >= -0.03).all()
    exp = 1.0 / (1.0 + np.exp(-(np.log(eff["d_mid"]) - np.log(3.0)) * 3))
    assert np.median(np.abs(eff["eta"] - exp)) < 0.03
    # Wilson interval brackets eta and mostly brackets the truth
    assert (eff["eta_lo"] <= eff["eta"]).all() and (eff["eta"] <= eff["eta_hi"]).all()
    covered = ((eff["eta_lo"] <= exp) & (exp <= eff["eta_hi"])).mean()
    assert covered > 0.6


def test_efficiency_function_all_equal_diameter():
    # a run where every object is the same size must not crash the auto-binning
    D = np.full(500, 20.0)
    det = np.arange(500) % 2 == 0
    eff = efficiency.efficiency_function(D, det)
    assert len(eff) >= 1 and abs(eff["eta"].iloc[0] - 0.5) < 0.05


def test_poisson_interval_shape_and_edges():
    k, lo, hi = debias.poisson_interval(np.array([0.0, 1.0, 100.0]), z=1.0)
    assert lo[0] == 0.0 and 1.5 < hi[0] < 2.2          # k=0 -> [0, ~1.84]
    assert (lo <= k).all() and (hi >= k).all()          # interval brackets k
    # large k -> both sides approach the sqrt(k) Gaussian limit (upper stays a
    # touch wider than lower, since Poisson is right-skewed)
    assert 9.0 < (hi[2] - 100) < 12.0 and 8.0 < (100 - lo[2]) < 11.0


def test_debias_recovers_true_counts():
    # A steep power-law "truth", detected through a logistic efficiency; debias
    # should invert back to the true counts within the reported band.
    rng = np.random.default_rng(3)
    edges = np.geomspace(0.5, 20.0, 15)
    d_mid = np.sqrt(edges[:-1] * edges[1:])
    n_true = (2000 * d_mid ** -2.5).round()             # true counts per bin
    eta_true = 1.0 / (1.0 + np.exp(-(np.log(d_mid) - np.log(3.0)) * 3))
    n_obs = rng.binomial(n_true.astype(int), eta_true)  # what the survey sees
    # efficiency measured from an independent uniform-in-log-D simulation
    m = 40000
    D = np.exp(rng.uniform(np.log(0.5), np.log(20), m))
    et = 1.0 / (1.0 + np.exp(-(np.log(D) - np.log(3.0)) * 3))
    eff = efficiency.efficiency_function(D, rng.random(m) < et, bins=edges)

    out = debias.debias(eff, n_obs, eta_min=0.05)
    ok = out["reliable"].to_numpy()
    # recovered counts bracket the truth in the reliable (well-sampled) bins
    within = ((out["n_true_lo"] <= n_true) & (n_true <= out["n_true_hi"])).to_numpy()
    assert within[ok].mean() > 0.7
    # point estimate is unbiased in the log: median ratio near 1
    ratio = (out["n_true"].to_numpy()[ok] / n_true[ok])
    assert abs(np.median(ratio) - 1.0) < 0.15


def test_debias_flags_and_bounds_small_efficiency():
    eff = pd.DataFrame({
        "d_lo": [0.5, 5.0], "d_hi": [5.0, 20.0], "d_mid": [1.5, 10.0],
        "eta": [0.001, 0.8], "eta_lo": [0.0, 0.78], "eta_hi": [0.01, 0.82],
    })
    out = debias.debias(eff, [3, 400], eta_min=0.05)
    assert not out["reliable"].iloc[0] and out["reliable"].iloc[1]
    assert np.isinf(out["n_true_hi"].iloc[0])           # eta_lo=0 -> unbounded
    assert np.isfinite(out["n_true_hi"].iloc[1])
    assert (out["n_true_lo"] >= 0).all()


def test_bin_observed_matches_edges():
    eff = pd.DataFrame({"d_lo": [1.0, 2.0, 4.0], "d_hi": [2.0, 4.0, 8.0],
                        "d_mid": [1.4, 2.8, 5.6]})
    counts = debias.bin_observed([1.5, 1.9, 3.0, 5.0, 6.0, 7.0], eff)
    assert list(counts) == [2, 1, 3]


def _build_sim_ensemble(d, n_members=3):
    """Write n_members member catalogs + a manifest, mimicking SynthPop output."""
    import json
    rng = np.random.default_rng(7)
    m = 150
    base = dict(
        id=[f"o_{k}" for k in range(m)], diam_km=np.exp(rng.uniform(np.log(2), np.log(40), m)),
        b_over_a=rng.uniform(0.4, 1.0, m),
        pole_beta_deg=rng.uniform(-40, 40, m), pole_lambda_deg=rng.uniform(0, 360, m),
        a_au=rng.uniform(2.2, 3.2, m), e=rng.uniform(0, 0.15, m),
        i_deg=rng.uniform(0, 15, m), node_deg=rng.uniform(0, 360, m),
        argperi_deg=rng.uniform(0, 360, m), M_deg=rng.uniform(0, 360, m),
        H=rng.uniform(10, 16, m))
    members = []
    for k in range(n_members):
        cat = pd.DataFrame(base)
        # per-member albedo/beaming perturbation (stands in for a bootstrap draw)
        cat["pV"] = np.clip(rng.uniform(0.05, 0.2, m) * (1 + 0.15 * rng.standard_normal()), 0.02, None)
        cat["eta"] = np.full(m, 1.0 + 0.1 * rng.standard_normal())
        name = f"T_m{k:03d}_synthpop.csv"
        cat.to_csv(d / name, index=False)
        members.append({"member": k, "nominal": k == 0, "catalog": name})
    (d / "T_ensemble_manifest.json").write_text(
        json.dumps({"name": "T", "n_members": n_members, "members": members}))
    return d / "T_ensemble_manifest.json"


def test_sim_ensemble_combines_stat_and_sys():
    import tempfile
    from simmer import ensemble
    d = Path(tempfile.mkdtemp())
    manifest = _build_sim_ensemble(d, n_members=4)
    frames = neowise.synthetic_frames(n_scans=6000, seed=2, days=40)
    bins = np.geomspace(2.0, 40.0, 8)
    res = ensemble.run_ensemble(manifest, SimConfig(name="T", out_dir=d, seed=0, min_detections=1, link_efficiency=1.0, min_motion_deg_day=None, max_motion_deg_day=None),
                                frames=frames, bins=bins, n_workers=1, verbose=False)
    assert res.eta.shape == (4, len(bins) - 1)
    comb = res.combine()
    fin = np.isfinite(comb["eta"]).to_numpy()
    assert fin.any()
    # every band brackets eta and is clipped to [0, 1]
    for lo, hi in (("eta_stat_lo", "eta_stat_hi"), ("eta_sys_lo", "eta_sys_hi"),
                   ("eta_tot_lo", "eta_tot_hi")):
        assert (comb[lo] <= comb["eta"] + 1e-9).all()
        assert (comb["eta"] <= comb[hi] + 1e-9).all()
        assert (comb[lo] >= -1e-12).all() and (comb[hi] <= 1 + 1e-12).all()
    # the isolated systematic is contained within the total scatter
    sys_w = (comb["eta_sys_hi"] - comb["eta_sys_lo"]).to_numpy()[fin]
    tot_w = (comb["eta_tot_hi"] - comb["eta_tot_lo"]).to_numpy()[fin]
    assert (sys_w <= tot_w + 1e-9).all()


def test_sim_ensemble_debias_has_stat_and_sys_bands():
    import tempfile
    from simmer import ensemble
    d = Path(tempfile.mkdtemp())
    manifest = _build_sim_ensemble(d, n_members=4)
    frames = neowise.synthetic_frames(n_scans=6000, seed=2, days=40)
    bins = np.geomspace(2.0, 40.0, 8)
    res = ensemble.run_ensemble(manifest, SimConfig(name="T", out_dir=d, seed=0, min_detections=1, link_efficiency=1.0, min_motion_deg_day=None, max_motion_deg_day=None),
                                frames=frames, bins=bins, n_workers=1, verbose=False)
    n_obs = np.array([50, 40, 30, 20, 10, 5, 2], float)   # observed counts per bin
    out = res.debias(n_obs)
    for c in ("n_true", "n_true_stat_lo", "n_true_stat_hi",
              "n_true_tot_lo", "n_true_tot_hi"):
        assert c in out.columns
    m = out["reliable"].to_numpy() & np.isfinite(out["n_true"]).to_numpy()
    tot = (out["n_true_tot_hi"] - out["n_true_tot_lo"]).to_numpy()[m]
    stat = (out["n_true_stat_hi"] - out["n_true_stat_lo"]).to_numpy()[m]
    assert (tot >= stat - 1e-6).all()
    lo = out["n_true_tot_lo"].to_numpy()
    assert (lo[np.isfinite(lo)] >= 0).all()


def test_science_floor_widens_systematic_band():
    import tempfile
    from simmer import ensemble
    from simmer.ensemble import ScienceFloor
    d = Path(tempfile.mkdtemp())
    manifest = _build_sim_ensemble(d, n_members=6)
    frames = neowise.synthetic_frames(n_scans=6000, seed=2, days=40)
    bins = np.geomspace(2.0, 40.0, 8)
    base = SimConfig(name="T", out_dir=d, seed=0, min_detections=1, link_efficiency=1.0, min_motion_deg_day=None, max_motion_deg_day=None)
    res0 = ensemble.run_ensemble(manifest, base, frames=frames, bins=bins,
                                 n_workers=1, verbose=False)
    resF = ensemble.run_ensemble(manifest, base, frames=frames, bins=bins,
                                 n_workers=1, verbose=False,
                                 floor=ScienceFloor(sensitivity_frac=0.4, beaming_sigma=0.15))
    # the floor injects extra per-member perturbations -> more cross-member
    # scatter in eta_m (measured directly; robust to the excess-variance clip)
    std0 = np.nanmean(np.nanstd(res0.eta, axis=0, ddof=1))
    stdF = np.nanmean(np.nanstd(resF.eta, axis=0, ddof=1))
    assert stdF > std0
    for c in (res0.combine(), resF.combine()):      # bands stay valid
        assert (c["eta_tot_lo"] >= -1e-9).all() and (c["eta_tot_hi"] <= 1 + 1e-9).all()
    # member 0 stays nominal (unperturbed) under the floor
    assert _floor_nominal_unchanged()


def _floor_nominal_unchanged():
    from simmer.ensemble import _floor_draw, ScienceFloor
    f = ScienceFloor(sensitivity_frac=0.4, beaming_sigma=0.15)
    return _floor_draw(f, 0) == (1.0, 0.0) and _floor_draw(f, 1) != (1.0, 0.0)


def test_plot_efficiency_writes_png():
    import tempfile
    from simmer import plotting
    d = np.geomspace(1, 40, 20)
    eta = 0.8 / (1 + np.exp(-(np.log10(d) - np.log10(2)) * 4))
    comb = pd.DataFrame({"d_mid": d, "eta": eta,
                         "eta_stat_lo": eta - 0.01, "eta_stat_hi": eta + 0.01,
                         "eta_sys_lo": eta - 0.006, "eta_sys_hi": eta + 0.006,
                         "eta_tot_lo": eta - 0.012, "eta_tot_hi": eta + 0.012})
    out = Path(tempfile.mkdtemp()) / "eff.png"
    plotting.plot_efficiency(comb, out, title="T")
    assert out.exists() and out.stat().st_size > 5000
    # single-run fallback (Wilson band only) also works
    single = pd.DataFrame({"d_mid": d, "eta": eta,
                           "eta_lo": eta - 0.02, "eta_hi": eta + 0.02})
    out2 = Path(tempfile.mkdtemp()) / "eff_single.png"
    plotting.plot_efficiency(single, out2)
    assert out2.exists() and out2.stat().st_size > 5000


def _sfd_edges(n_bins=20, lo=1.0, hi=50.0):
    edges = np.geomspace(lo, hi, n_bins + 1)
    d_lo, d_hi = edges[:-1], edges[1:]
    d_mid = np.sqrt(d_lo * d_hi)
    dlog = np.log10(d_hi) - np.log10(d_lo)
    return d_lo, d_hi, d_mid, dlog


def test_sfd_fit_single_power_law_recovers_slope():
    from simmer import sfd_fit
    d_lo, d_hi, d_mid, dlog = _sfd_edges()
    alpha = 2.5                                    # cumulative slope
    n = 1e5 * d_mid ** (-alpha) * dlog
    out = sfd_fit.fit_sfd(d_lo, d_hi, n, max_breaks=2)
    assert out["best"]["n_breaks"] == 0           # no spurious break on a pure power law
    assert abs(out["best"]["alpha"][0] - alpha) < 0.02
    # predict_density reproduces the input per-dex density
    model = sfd_fit.predict_density(out["best"], d_mid) * dlog
    assert np.allclose(model, n, rtol=0.02)


def test_sfd_fit_detects_break_and_recovers_slopes():
    from simmer import sfd_fit
    d_lo, d_hi, d_mid, dlog = _sfd_edges()
    u = np.log10(d_mid)
    ub = np.log10(8.0)
    a1, a2 = 2.0, 4.0                              # shallow below the break, steep above
    y = 5.0 - a1 * u - (a2 - a1) * np.maximum(0.0, u - ub)
    n = 10 ** y * dlog
    best = sfd_fit.fit_sfd(d_lo, d_hi, n, max_breaks=2)["best"]
    assert best["n_breaks"] == 1
    assert abs(best["alpha"][0] - a1) < 0.25 and abs(best["alpha"][1] - a2) < 0.25
    assert 6.0 < best["break_diam_km"][0] < 11.0


def test_ensemble_fit_sfd_recovers_broken_law():
    from simmer.ensemble import EnsembleResult
    from simmer import efficiency
    rng = np.random.default_rng(0)
    bins = np.geomspace(1.0, 50.0, 19)
    d_lo, d_hi = bins[:-1], bins[1:]
    d_mid = np.sqrt(d_lo * d_hi)
    dlog = np.log10(d_hi) - np.log10(d_lo)
    nbin = len(d_mid)
    u = np.log10(d_mid)
    ub, a1, a2 = np.log10(8.0), 2.2, 4.2
    N_true = 10 ** (6.0 - a1 * u - (a2 - a1) * np.maximum(0.0, u - ub)) * dlog
    eta_true = 0.85 / (1 + np.exp(-(u - np.log10(2.0)) * 4))         # plateau ~0.85
    n_obs = np.round(N_true * eta_true)

    M, n_input = 30, 4000
    eta = np.zeros((M, nbin)); elo = np.zeros((M, nbin)); ehi = np.zeros((M, nbin))
    ni = np.full((M, nbin), float(n_input)); nd = np.zeros((M, nbin))
    for m in range(M):
        p = np.clip(eta_true * (1 + 0.02 * rng.standard_normal()), 0, 1)  # small systematic
        k = rng.binomial(n_input, p)
        _, lo, hi = efficiency.wilson_interval(k, n_input)
        eta[m], elo[m], ehi[m], nd[m] = k / n_input, lo, hi, k
    res = EnsembleResult(bins=bins, eta=eta, eta_lo=elo, eta_hi=ehi,
                         n_input=ni, n_detected=nd)

    out = res.fit_sfd(n_obs, max_breaks=2)
    assert out["n_breaks"] == 1
    assert abs(out["alpha_median"][0] - a1) < 0.4
    assert abs(out["alpha_median"][1] - a2) < 0.4
    assert 6.0 < out["break_diam_median"][0] < 11.0
    assert out["alpha_lo"][0] <= out["alpha_median"][0] <= out["alpha_hi"][0]

    # save() writes the efficiency table, debiased SFD and the fit JSON
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    paths = res.save(d, "T", n_observed=n_obs)
    assert len(paths) == 3 and all(p.exists() for p in paths)
    saved = json.loads((d / "T_sfd_fit.json").read_text())
    assert saved["n_breaks"] == 1 and "alpha_median" in saved


def test_sfd_fit_force_breaks():
    from simmer import sfd_fit
    d_lo, d_hi, d_mid, dlog = _sfd_edges()
    n = 1e5 * d_mid ** (-3.0) * dlog              # a pure power law
    forced = sfd_fit.fit_sfd(d_lo, d_hi, n, force_breaks=1)["best"]
    assert forced["n_breaks"] == 1                # honours the forced structure
    assert len(forced["alpha"]) == 2


def test_object_summary_min_detections_and_band():
    from simmer.pipeline import object_summary
    catalog = pd.DataFrame({"id": ["a", "b", "c"]})
    hits = pd.DataFrame({                              # a: 6 W3; b: 3 W3 + 3 W4; c: 2 W3
        "id": ["a"] * 6 + ["b"] * 6 + ["c"] * 2,
        "band": ["W3"] * 6 + ["W3"] * 3 + ["W4"] * 3 + ["W3"] * 2,
        "detected": [True] * 14})
    det = lambda md, b: set(object_summary(catalog, hits, md, b,
                            tracklet_window_hr=None, link_efficiency=1.0)
                            .query("detected").id)
    assert det(1, None) == {"a", "b", "c"}            # >=1 any band: all three
    assert det(5, "W3") == {"a"}                       # >=5 W3: only a (6 W3)
    assert det(5, None) == {"a", "b"}                  # >=5 any: a(6), b(6 total)
    assert det(3, "W3") == {"a", "b"}                  # >=3 W3: a(6), b(3)


def test_object_summary_tracklet_window():
    from simmer.pipeline import object_summary
    catalog = pd.DataFrame({"id": ["clustered", "spread"]})
    # Both objects are detected 5 times in W3. "clustered" gets all 5 within a
    # single ~day (one apparition -> links); "spread" gets 1 detection every 5
    # days (never 5 within a 36 h window -> no tracklet), even though its
    # survey-wide count is also 5.
    t_cluster = [0.0, 0.2, 0.5, 0.9, 1.2]                       # days, all < 36 h apart
    t_spread = [0.0, 5.0, 10.0, 15.0, 20.0]                     # days, one per 5 d
    hits = pd.DataFrame({
        "id": ["clustered"] * 5 + ["spread"] * 5,
        "band": ["W3"] * 10,
        "mjd": t_cluster + t_spread,
        "detected": [True] * 10})
    linked = lambda win: set(object_summary(
        catalog, hits, 5, "W3", tracklet_window_hr=win, link_efficiency=1.0)
        .query("detected").id)
    assert linked(36.0) == {"clustered"}              # window: only the apparition links
    assert linked(None) == {"clustered", "spread"}    # survey-wide count: both pass


def test_object_summary_motion_gate():
    from simmer.pipeline import object_summary
    catalog = pd.DataFrame({"id": ["mba", "fast", "slow"]})
    # 5 detections each within ~half a day; positions advance at a set rate so the
    # apparent motion is ~0.25 (mba, in band), ~10 (fast, trailed), ~0.01 (slow,
    # near-stationary) deg/day. Gate keeps only the main-belt-rate object.
    t = [0.0, 0.1, 0.2, 0.3, 0.4]                       # days
    rows = []
    for oid, rate in (("mba", 0.25), ("fast", 10.0), ("slow", 0.01)):
        for k, tk in enumerate(t):
            rows.append({"id": oid, "band": "W3", "mjd": tk, "detected": True,
                         "ra_deg": 180.0 + rate * tk, "dec_deg": 0.0})
    hits = pd.DataFrame(rows)
    keep = set(object_summary(catalog, hits, 5, "W3", tracklet_window_hr=36.0,
               link_efficiency=1.0, min_motion_deg_day=0.06, max_motion_deg_day=3.2)
               .query("detected").id)
    assert keep == {"mba"}                              # fast trailed + slow stationary rejected
    nogate = set(object_summary(catalog, hits, 5, "W3", tracklet_window_hr=36.0,
                 link_efficiency=1.0).query("detected").id)
    assert nogate == {"mba", "fast", "slow"}            # no gate: all three link


def test_observed_sfd_dedups_by_object():
    import tempfile, os
    from simmer import observed
    with tempfile.TemporaryDirectory() as d:
        fam = os.path.join(d, "fam.txt"); open(fam, "w").write("100\n200\n")
        # NEOWISE rows: number at col 2, diameter at col 11. Object 100 appears in
        # THREE rows (~5 km), object 200 once (~10 km), 300 is not a family member.
        rows = []
        for num, D in [(100, 5.0), (100, 5.1), (100, 4.9), (200, 10.0), (300, 7.0)]:
            r = ["0"] * 12; r[2] = str(num); r[11] = str(D); rows.append(",".join(r))
        neo = os.path.join(d, "neo.csv"); open(neo, "w").write("\n".join(rows) + "\n")
        n_obs, diam, n_matched = observed.observed_sfd(fam, neo, bins=[3, 7, 15], cryo_jd=None)
        assert n_matched == 2                          # unique objects, not 4 rows
        assert list(n_obs) == [1, 1]                   # obj 100 counted ONCE, not 3x


def test_optical_diameter_from_H():
    from simmer import optical
    assert optical.diameter_km(10, 0.1) > optical.diameter_km(15, 0.1)     # brighter -> bigger
    assert optical.diameter_km(15, 0.25) < optical.diameter_km(15, 0.05)   # higher albedo -> smaller
    assert abs(optical.diameter_km(15.0, 0.25) - 1329/np.sqrt(0.25)*1e-3) < 1e-9


def test_optical_albedo_model_and_mc():
    from simmer import optical
    rng = np.random.default_rng(0)
    pv = np.concatenate([rng.normal(0.06, 0.01, 300), rng.normal(0.25, 0.03, 200)])
    am = optical.AlbedoModel(pv)
    assert 0.30 < am.frac_bright(0.10) < 0.50            # ~40% bright complex
    H = np.array([15.0, 16.0, 17.0]); meas = np.array([0.20, np.nan, np.nan])
    D = optical.sample_diameters(H, meas, am, rng, n_mc=50, h_sigma=0.3)
    assert D.shape == (3, 50) and np.all(D > 0)
    med, lo, hi = optical.cumulative_known(D, [1.0, 2.0, 5.0])
    assert med[0] >= med[1] >= med[2] and np.all(lo <= hi)  # cumulative decreases with D


def test_optical_conditional_albedo():
    from simmer import optical
    rng = np.random.default_rng(0)
    pv_pool = np.array([0.05, 0.30] * 100)                # bimodal: dark + bright
    eta = optical.eta_from_efficiency([1.0, 3.0, 6.0], [0.1, 0.25, 1.0])
    # H=15: bright pV=0.30 -> D~2.4 km (eta~0.2, NEOWISE-missable); dark 0.05 -> D~5.9 km (eta~1)
    s = optical.sample_conditional_albedo(np.array([15.0]), pv_pool, eta, rng, n_mc=500)
    assert s.shape == (1, 500)
    assert (s > 0.2).mean() > 0.85      # optical-only members assigned mostly BRIGHT (small D)


def test_optical_family_albedo_model():
    from simmer import optical
    ids = [1, 2, 3, 4, 5]
    pv = {1: 0.30, 2: 0.28, 3: 0.32, 4: 0.06, 5: 0.07}    # large bright, small dark (skewed)
    D = {1: 10.0, 2: 9.0, 3: 11.0, 4: 2.0, 5: 2.2}
    am = optical.family_albedo_model(ids, pv, D, d_complete=5.0, min_n=3)
    assert am.n == 3 and am.median() > 0.25              # only the 3 above-Dc members
    am2 = optical.family_albedo_model(ids, pv, D, d_complete=5.0, min_n=10)
    assert am2.n == 5                                     # too few above -> fall back to all


def test_optical_known_diameters_thermal_vs_optical():
    from simmer import optical
    rng = np.random.default_rng(0)
    cat = pd.DataFrame({"H": [14.0, 14.0]}, index=[1, 2])   # both H=14
    neo_D = {1: 5.0}                                         # member 1 thermal; 2 optical-only
    am = optical.AlbedoModel(np.full(50, 0.15))
    D_on = optical.known_diameters([1, 2], cat, {}, neo_D, am, rng, n_mc=100,
                                   apply_h_correction=True)
    D_off = optical.known_diameters([1, 2], cat, {}, neo_D, am, rng, n_mc=100,
                                    apply_h_correction=False)
    assert D_on.shape == (2, 100)
    assert abs(np.median(D_on[0]) - 5.0) < 0.5              # member 1 uses thermal D ~5 km
    assert np.median(D_on[1]) < np.median(D_off[1])        # H-correction shrinks optical D


def test_optical_load_and_member_H():
    import tempfile, os
    from simmer import optical
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prop.dat")
        open(f, "w").write(
            "2.20 0.1 0.10 0.1 0.05 0.1 10 -20 15.30 100 00042 42\n"
            "3.10 0.1 0.20 0.1 0.15 0.1 10 -20 12.10 100 A0001 100001\n")
        cat = optical.load_proper_catalog(f)
        assert cat.loc[42, "H"] == 15.30 and cat.loc[100001, "H"] == 12.10
        mh = optical.member_H([42, 100001, "2013PL32"], cat, {42: 0.2})
        assert mh.loc[42, "H"] == 15.30 and mh.loc[42, "pV"] == 0.2
        assert not mh.loc["2013PL32", "numbered"] and np.isnan(mh.loc["2013PL32", "H"])


def test_designations_canonical():
    from simmer import designations as dz
    c = dz.canonical
    assert c("00005") == 5 and c("5") == 5             # plain numbered
    assert c("A0001") == 100001                         # MPC-packed numbered >=100000
    assert c("z9999") == 619999
    assert c("100001") == 100001
    assert c("K13P32L") == "2013PL32"                   # MPC-packed provisional
    assert c("2013PL32") == "2013PL32"                  # human provisional
    assert c("2013 PL32") == "2013PL32"
    assert c("J94V03X") == "1994VX3"
    assert c("K13P32L") == c("2013 PL32")               # packed == human -> same key
    assert c("garbage!!") is None


def test_observed_sfd_matches_packed_and_provisional():
    import tempfile, os
    from simmer import observed
    with tempfile.TemporaryDirectory() as d:
        fam = os.path.join(d, "fam.txt")
        open(fam, "w").write("100\n100001\n2013PL32\n")   # plain, >=100000, provisional
        rows = []                                          # NEOWISE rows in MPC-packed form
        for desig, D in [("00100", 5.0), ("A0001", 4.0), ("K13P32L", 3.0), ("99999", 9.0)]:
            r = ["0"] * 12; r[2] = desig; r[11] = str(D); rows.append(",".join(r))
        neo = os.path.join(d, "neo.csv"); open(neo, "w").write("\n".join(rows) + "\n")
        n_obs, diam, n_matched = observed.observed_sfd(fam, neo, bins=[2.5, 3.5, 4.5, 6], cryo_jd=None)
        assert n_matched == 3               # 00100, A0001(=100001), K13P32L(=2013PL32); 99999 not a member
        assert list(n_obs) == [1, 1, 1]     # one each at 3, 4, 5 km


def test_observed_sfd_cryo_filter():
    import tempfile, os
    from simmer import observed
    with tempfile.TemporaryDirectory() as d:
        fam = os.path.join(d, "fam.txt"); open(fam, "w").write("100\n200\n")
        rows = []                                          # col5 = fit-epoch JD
        for num, D, ep in [(100, 5.0, 2455300.0), (200, 5.0, 2456800.0)]:  # cryo, reactivation
            r = ["0"] * 12; r[2] = str(num); r[5] = str(ep); r[11] = str(D)
            rows.append(",".join(r))
        neo = os.path.join(d, "neo.csv"); open(neo, "w").write("\n".join(rows) + "\n")
        n_obs, diam, n_matched = observed.observed_sfd(fam, neo, bins=[3, 7])   # cryo default
        assert n_matched == 1 and list(n_obs) == [1]       # only the cryo-epoch object
        _, _, n_all = observed.observed_sfd(fam, neo, bins=[3, 7], cryo_jd=None)
        assert n_all == 2                                  # both without the filter


def test_designation_canonicalization_unifies_all_forms():
    """Every identifier form of the same object must map to one canonical id
    (internal-space and packed forms are the historic traps)."""
    from simmer import designations as dg
    assert dg.canonical("588") == dg.canonical("00588") == 588
    assert (dg.canonical("2008 RK58") == dg.canonical("2008RK58")
            == dg.canonical("K08R58K") == "2008RK58")
    assert dg.canonical("A0475") == dg.canonical("100475") == 100475


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
