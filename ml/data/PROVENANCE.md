# eta_training.csv provenance

Generated 2026-09-23 19:01 EDT by ml/make_training_set.py
(seed 42) at Simmer commit 20d8076.

- 150,000 synthetic objects, random-sampled over the ranges documented
  in the script docstring; per-object detected flag + n_detections from the
  full production physics chain (SimConfig defaults: neowise_cryo W3/W4,
  snr 5, lightcurve modulation, 1% bad-pixel, >=5-detection linking).
- Frames: cryo_4band_pointings_near_ecliptic.csv (1,241,031 rows).
- Overall detected fraction: 0.4447.
- Wall time: 578 s (single process, frames loaded once) —
  this is the "expensive physics" baseline the surrogate is judged against.
- Regenerate with the command in the script header; byte-reproducible for a
  fixed seed and commit.
- Latitude-configuration note: the sweep samples i up to 35 deg while the
  production chain (emulated here) uses the near-ecliptic frame list with a
  30 deg ecliptic-latitude cut. Rows with i > ~30 deg therefore inherit the
  production configuration's coverage, NOT an all-sky truth — correct for
  emulating Simmer-as-run, but do not reinterpret those rows as absolute
  survey completeness.
