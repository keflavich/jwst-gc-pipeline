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

## M92 deep-mask campaign (2026-07-18)

Sparse-field extension of C(r_mask) to 26 px using M92 (GO-1334,
F090W/F150W, zero background, ~250-350 isolated calibrators per frame,
8 SW detectors x 4 frames per filter):

- `m92_deep_stacks.py`    — stage 1: per-(filter, detector) empirical/model
  2D PSF stacks (151x151, incl. per-frame stacks for error estimates);
  npz outputs in `/orange/adamginsburg/jwst/m92/wingcal_deep_stacks/`
- `m92_deep_calcurve.py`  — stage 2: deep C(r_mask) curves to 26 px in two
  fit-region flavors (fixed 50-px annulus; production `_wing_selfcal`
  window), each with/without the OPD phantom-blob region; outputs in
  `calcurves_deep_m92/`
- `run_m92_deep_campaign.sbatch` — SLURM driver (QOS astronomy-dept-b;
  the whole campaign also runs directly in ~5 min on 8 cores)

Results (`calcurves_deep_m92/summary_deep_m92.json`; `sparse_median` =
median over the 4 sparsest detectors per filter, the headline numbers):

- **F090W** (nrca1/a2/b1/b2): C = 1.25 / 1.28 / 1.26 / 1.25 at
  r_mask = 10/15/20/25 px — a PLATEAU.  Residual of the production
  clamp-at-C(10) extrapolation (`apply_wing_selfcal`, np.interp on clipped
  radii): only −0.01 to −0.03 mag at 15-25 px.
- **F150W** (same detectors): C = 1.02 / 1.11 / 1.10 / 1.12 — mild rise to
  ~15 px then flat; clamp residual −0.08 to −0.11 mag.
- The 4 detectors nearer the cluster core (1.5-2.4k calibrators) instead
  show a steep spurious rise (C(25) = 2.6-4.7, clamp residual up to
  −1.3 mag) with tiny frame-to-frame scatter: crowding contamination of
  the stacked wings, NOT PSF structure.  Deep-C(r) measurements are only
  trustworthy on genuinely sparse fields — the same contamination
  presumably inflates any Brick-side deep extrapolation.
- The phantom OPD blob (model-only, ~3.2% excess flux at r~35 px in all 8
  F090W grids + 4/8 F150W grids; <0.2% in the clean grids) lands INSIDE
  the deep-mask fit annulus and must be excluded from the fit region or it
  dominates the deep-mask bias (with-blob C collapses to ~0.1 at 25 px).

Net: the −0.25 mag deep-mask residual bound from
docs/reports/SATURATED_STAR_PHOTOMETRY_ARTICLE.md §6 item 1 tightens to
~−0.03 mag (F090W-like wings) / ~−0.11 mag (F150W-like) for the clamp
extrapolation itself, per this zero-background measurement.
