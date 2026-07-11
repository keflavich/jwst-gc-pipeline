# Saturated-star flux recovery: the STPSF wing deficit, its proof, and the fix

> **Figures moved:** image files are no longer tracked in this repo; they live in the Overleaf astrometry-paper project (https://www.overleaf.com/project/6a521006b63a11a7e0d80fa0) under `figures/evidence/` (same filenames).

**Date:** 2026-07-10.  **Branch:** `satstar-replace-matching` (PR #61).
**Figures:** `docs/evidence/*_2026-07-10.png` (referenced below).

## 1. The problem

Saturated-star ("satstar") photometry fits the STPSF model to the star's
unmasked wings.  Measured against clean narrow-band references, the resulting
fluxes were wrong by up to ~0.5 mag with magnitude-dependent structure:
unphysical color jumps, a population gap at F187N 14–15, and distinct
saturated "clouds" in the CMDs — in every band-width class (wide, medium,
narrow).

## 2. The claim and its proof: STPSF underpredicts the NIRCam PSF wings

The claim is bold, so it was measured **four independent ways**, in **two
unrelated fields**, on **five filters** and **three detectors**:

1. **Stacked-image comparison (2D)** — `psfwing2d_{f182m,f187n,f212n,f405n}_*.png`:
   ~250 bright unsaturated stars per filter stacked sub-pixel-registered;
   identical stacking applied to the STPSF model evaluated at the same
   detector positions and pixel phases; 2D residual and ratio maps + radial
   profiles.  A model-through-machinery control bounds stacking artifacts;
   a mis-centering control bounds registration bias (<3%).
2. **Fit-domain estimator** — per-annulus LSQ amplitude (what the fit
   actually senses), `f182m_wing_ratio_real_vs_stpsf_2026-07-10.png`.
3. **Masked-core fit experiment** — the production failure mode reproduced
   on unsaturated stars with known fluxes,
   `f182m_wingfit_bias_curves_2026-07-10.png`.
4. **Zero-background cross-validation (M92 globular cluster, GO-1334)** —
   `m92_wing_stacks2d_2026-07-10.png`, `m92_wing_ratio_curves_2026-07-10.png`,
   `m92_wingfit_bias_curves_2026-07-10.png`: 812–1248 stars per config,
   plus an **injected-model control** (recovers 0.95–0.99 everywhere,
   proving the estimators don't invent excesses).

**Result** (real/STPSF wing flux ratio; ±1σ):

| r [px] | Brick F182M | Brick F187N | Brick F212N | Brick F405N | M92 F090W | M92 F150W |
|---|---|---|---|---|---|---|
| 3 | 1.09±0.01 | 1.14±0.02 | 1.14±0.02 | 1.00±0.01 | — | — |
| 5 | 1.24±0.02 | 1.03±0.05 | 1.13±0.03 | 0.95±0.03 | — | — |
| 7–10 | 1.1–1.3 | ~1.0 | ~1.0 | ~1.0 | 1.18–1.30 | 1.31–1.51 |
| 14–20 | (crowding-limited) | — | — | — | 1.25–1.40 | 1.26–1.31 |

The excess is **flux-independent across brightness quartiles** and, on M92's
zero background, **purely multiplicative** (additive term ≈ 0) — i.e. a
property of the real PSF relative to the model, not field systematics.  The
masked-core fit bias it produces matches in sign, magnitude, and radius
dependence in both fields (Brick F182M: +0.02/+0.11/+0.19/+0.26 mag at
r_mask = 3/4/8/12; M92 F090W: +0.03/+0.10/+0.16 at 3/4/6).  Per-filter
structure is real: F187N/F212N excess is confined to small radii, F150W is
*faint*-biased at its Airy-dip radius, F405N is neutral — so **no static
correction table is safe**.

**Discovery (M92):** 12/20 of the M92 STPSF grids contain a phantom
"segment blob" (2.2% of total flux at r≈35 px) inherited from OPD sensing
R2022062004 (a segment-tilt epoch).  It corrupts any fit that reaches r≳25
px with those grids.  Action: rebuild the M92 F090W/F150W grids with the
R2022062102 OPD.  Brick grids are clean (<0.1%).  The per-frame
self-calibration below absorbs even this class of model defect
automatically, because its calibration stars see the same model.

## 3. The fix stack (this branch)

Each layer independently validated on real frames:

1. **Severity gates** (merged, #56): unsaturated stars over-flagged by the
   any-group SATURATED bit no longer enter the satstar channel (45% of the
   SW satstar catalogs were such fakes; consolidated F182M satstars
   2032 → ~780 after the gates).
2. **Mutual-nearest replacement to 0.5″**: satstar fits now reach the
   daophot rows they should replace (positions scatter 0.08–0.15″ vs the
   old 0.05″ radius; 90% of the wide-band "catastrophic clipped" rows have
   their satstar within 0.5″).
3. **Zeroframe-anchored fits**: the DQ-SATURATED rim is recovered from the
   ramp's calibrated first read and the fit uses real profile pixels;
   only the group-0-saturated deep core stays masked.  The recovery scale
   is **brightness-dependent** — cal/group0 drifts 0.229→0.167 from
   g0~2k→20k DN even after the linearity step — so R is calibrated as a
   function of group0 per frame (scalar R mis-scaled rim pixels by up to
   25%).
4. **Per-frame wing self-calibration** (`apply_wing_selfcal`, default ON):
   ~30 bright unsaturated stars per frame fit twice (truth vs core-masked at
   each satstar's own mask radius); each satstar's **catalog** flux is
   divided by the interpolated median ratio (precision ~3–5%).  The
   subtracted **model keeps the raw amplitude** — the fit matches the
   observed wings, so residual images stay clean; only the total-flux
   number carried the model wing deficit.  E2E: ratios measured in
   production (1.027/1.035/1.089 at r=3/4/7 on F182M; 1.076–1.098 on M92
   F090W) match the laboratory curves; 379/379 demo frames corrected.
5. **Flag decoupling**: `is_saturated` now marks saturated-but-unrepaired
   rows (clipped photometry) independently of `replaced_saturated`.

## 4. Certification status — honest accounting

**Metric** (`photometry/saturation_continuity.py`,
`assert_saturation_continuity`): max color jump between saturation classes
across the boundary; goal <0.05 mag, floor <0.10, per band pair — the
multi-band CMD-shape requirement in executable form.

**Catalog-level demo** (384-frame satstar rebuild + re-replacement into the
existing m7 catalogs, `brick_demo_cmd_rebuild_2026-07-10.png`): the fake
cloud is gone and the mid-bright satstars now join the giant branch;
F182M−F187N metric improves 0.74 → ~0.55±0.05.  **The demo cannot reach the
floor**: the *unsaturated-side* photometry it compares against was itself
measured on frames with the OLD (wrong-wing) satstar models subtracted, and
rows overwritten by old fake satstars are unrecoverable without refitting —
run-to-run scatter of the residual metric is ±0.1, dominated by that input
pollution.

**Full certification therefore requires the per-frame re-cataloging**
(purge satstar caches → recat brick F182M/F187N/F212N + o004 wide bands →
run the metric on the fresh catalogs, all three band-width classes).  Every
frame-level component of that chain is validated above.

## 5. Follow-ups

- Recat + metric certification (the step above).
- Rebuild M92 F090W/F150W PSF grids (bad OPD epoch).
- Durable PSF-level correction: `CorrectedGriddedPSFModel` with
  per-(detector, filter) C(r) (prototype residual −0.03..−0.07 mag,
  `scripts/analysis/wing_calibration/`); calibrate C(r>20 px) on a sparse
  field (M92 is ideal — zero background, thousands of calibrators).
- Report the STPSF wing deficit upstream (STScI) with the M92 evidence.
