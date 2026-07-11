# PSF wing calibration (satstar flux bias)

Measured 2026-07-10 on Brick F182M (two detectors, ~60 + ~5000 stars, two
independent methods): the real NIRCam PSF carries **10-25% more wing flux
relative to its core than the STPSF model** (the model Airy troughs are too
deep), so masked-core (satstar) fits are systematically too BRIGHT:
bias(r_mask) ~ 1.02/1.11/1.20/1.27 at r_mask = 3/4/8/12 px and still rising;
the unmasked control fit is 1.004. Per-filter (F187N differs) and ~3-10%
frame-to-frame.

Production fix: per-frame self-calibration in
`saturated_star_finding.apply_wing_selfcal` (default ON, env SATSTAR_WINGCAL=0
disables) — corrects satstar CATALOG fluxes; the subtracted model keeps raw
amplitudes (it matches the observed wings).

These scripts are the measurement/validation tooling for the durable
PSF-level path (a `CorrectedGriddedPSFModel` with per-detector/filter C(r)
tables — prototype in demo_2d_fit.py, residual bias -0.03..-0.07 mag):

- `wingbias_experiment.py`  — masked-core fit bias curves on real frames
- `measure_wing_ratio.py`   — stacked-star radial profile vs STPSF
- `measure_fitdomain.py`    — fit-domain (LSQ-relevant) wing ratio
- `build_correction_v2.py`  — blend the two estimators into C(r)
- `demo_2d_fit.py`          — CorrectedGriddedPSFModel prototype + validation

Calibrate C(r>20px) on a SPARSER field than the Brick (crowding-limited
there; current bracket 1.1-3x is the residual uncertainty for the deepest
cores).
