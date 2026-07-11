# Photometry of Saturated Stars in JWST/NIRCam Crowded Fields: Failure Modes, Wing Calibration, and Recovered Accuracy

**Date:** 2026-07-11.  **Branch:** `satstar-replace-matching` (PR #61).
**Companion technical report:** [SATSTAR_WING_CALIBRATION_REPORT.md](SATSTAR_WING_CALIBRATION_REPORT.md).
**Figures:** `docs/evidence/*_2026-07-{10,11}.png`, linked inline.

## Abstract

Stars brighter than the NIRCam saturation limit dominate the bright end of every
Galactic-center color–magnitude diagram (CMD), and their photometry in this
pipeline was wrong by up to ~50% (0.5 mag), with magnitude-dependent structure
that produced unphysical color jumps, population gaps, and detached "clouds" at
the saturated↔unsaturated boundary. We describe the diagnostic campaign that
localized the errors, the hypotheses tested and rejected along the way, and the
method finally adopted: DQ-severity gating of spurious saturation flags,
peak-based seeding of unflagged charge-migration stars, ramp-zeroframe anchoring
of the saturated rim with a brightness-dependent recovery scale, PSF wing
fitting, and a **per-frame wing self-calibration** that corrects a measured
~10–30% deficit in the STPSF model wings relative to the real PSF. The deficit
is demonstrated four independent ways in two unrelated fields (the Brick,
GC program 2221; and the zero-background globular cluster M92, GO-1334).
After the full stack, saturated-star fluxes agree with narrow-band-implied
truth to ~6%, per-frame calibration precision is 3–5%, and the
F182M−F187N class-continuity metric improves from 0.74 to 0.11 mag
(0.03 mag in the transition bins). Residual systematics — the deep-mask
wing extrapolation (up to −0.25 mag for the most saturated ~1 mag of the
catalog), demo-level certification limits, and unrecoverable clipped rows —
are enumerated with the controls applied to each.

## 1. Introduction

Crowded-field PSF photometry in the Galactic center requires the bright stars
for three reasons: they anchor the flux scale against 2MASS/Spitzer, they
dominate cross-band merging (a wrong bright-star flux corrupts the seed
catalogs of every other band), and their subtracted models set the background
against which faint stars are measured. NIRCam saturates at K~15 in the wide
bands in our readout patterns, so the entire giant branch above the red-clump
region is measured from PSF wings around masked cores ("satstars").

The symptom that started this campaign was categorical: in every CMD pair
(wide−wide, wide−medium, wide−narrow), the saturated population was displaced
from the unsaturated locus by 0.2–0.5 mag in color, with a population gap at
F187N = 14–15 and internally-distinct sub-clouds. Errors of this size are
science-breaking, and the requirement was set accordingly: **no systematic
color shift between saturation classes**, verified per band pair by an
executable metric (`photometry/saturation_continuity.py`).

## 2. Data and diagnostic tooling

- **Fields:** the Brick (JWST 2221, o001/o004; NIRCam F182M/F187N/F212N +
  F405N/F410M/F466N) as the science-representative crowded field; **M92**
  (GO-1334, F090W/F150W) as a zero-background cross-validation field with
  thousands of isolated calibrators and *no* extended emission.
- **Truth references:** narrow-band photometry saturates ~2.5 mag later than
  the wide/medium bands, so F187N/F405N unsaturated measurements provide
  per-star flux truth for stars saturated in F182M/F410M; brightness ordering
  and Spitzer/2MASS provide sanity floors above that.
- **Metric:** `saturation_continuity()` computes the maximum median-color jump
  between saturation classes across the saturation boundary per CMD;
  goal < 0.05 mag, floor < 0.10; wired into the test suite as
  `assert_saturation_continuity`.
- **Class-decomposed CMDs:** every diagnostic CMD splits the population into
  unsaturated / flagged-recovered / zeroframe-anchored / substituted classes
  ([brick_multiband_class_cmds_2026-07-10.png](../evidence/brick_multiband_class_cmds_2026-07-10.png)).

## 3. Hypotheses tested and rejected

The failure was not one bug; it was six stacked mechanisms. The path to that
conclusion ran through explicit hypotheses, most of which died under test.

**H1 — "The fits are bad" (rejected).** Fit quality on the wings is excellent
(qfit ≈ 0.05 on the brightest satstars) and residual images show the model
tracking the observed wings; the visible residual is a saturated-core + spike
remnant, not misconvergence. A separately-reported 0.2″ centroid jitter turned
out to be *stale position bounds on the shared PSF-grid object* between
successive fits — a harness bug, fixed and regression-tested, unrelated to the
flux errors.

**H2 — "Replacement bookkeeping loses the fits" (partially confirmed —
necessary, not sufficient).** Satstar↔catalog matching used a 0.05″ radius,
but satstar positions scatter 0.08–0.15″ relative to daophot centroids; 90% of
catastrophically-clipped wide-band rows had their correct satstar within 0.5″.
Mutual-nearest matching at 0.5″ (with a faint-replacement veto) fixed the
bookkeeping ([brick_cmd_replace_radius_sim_2026-07-10.png](../evidence/brick_cmd_replace_radius_sim_2026-07-10.png)),
and the clouds moved — but did not join the locus. The flux values themselves
were still wrong.

**H3 — "The satstar catalog is contaminated by fakes" (confirmed — necessary,
not sufficient).** The detection path used the any-group SATURATED DQ bit,
which flags pixels that saturate only in late groups and are fully recoverable:
45% of the SW satstar entries were unsaturated stars measured as if their cores
were missing (89% over-flag on MIRI F770W). Component-level severity floors
plus a post-fit implied-peak gate removed them (F182M consolidated satstars
2032 → ~780). A subtlety: narrow-band wing underestimation halved implied
peaks and the gate began rejecting *real* deep satstars (F187N: 157 survivors
vs ~1173 expected); an observed-peak second chance restored them.

**H4 — "Phantoms can be removed by a prominence/core-shape gate" (rejected).**
Deep-pit phantoms on bright-star diffraction spikes are as sharp as real cores;
no prominence threshold separated them without killing real stars. The
severity/implied-peak gating (H3) superseded this approach.

**H5 — "The masked-core bias can be inverted analytically" (rejected).**
Reproducing the production geometry on unsaturated stars of known flux — the
key controlled experiment — showed the daophot 5×5 fit with a 2.5 px masked
core recovers only **18.8%** of the true flux
([f182m_wingfit_bias_curves_2026-07-10.png](../evidence/f182m_wingfit_bias_curves_2026-07-10.png)).
Multiplying by ~5.3 amplifies every noise and normalization error by the same
factor; inversion is unstable. Conclusion: clipped daophot photometry of
saturated stars must be **substituted**, never corrected in place.

**H6 — "The bright 'unsaturated' reference locus is trustworthy" (rejected —
the comparison was inverted).** The wing-calibrated satstars appeared offset
−0.6 mag from the bright gray locus; but the gray locus itself sat ~1.5 mag
*above* the physical saturation limit. Those are **unflagged charge-migration
stars**: peaks above the severity floor with *zero* DQ-SATURATED pixels
(96% of the affected population), whose suppressed cores clip the daophot
photometry by ~0.3 mag. Once identified, the satstar photometry matched the
*faint, genuinely-unsaturated* locus and the F187N-implied truth to 6%.
Fix: peak-based seeding (image peaks above the severity floor, ≥2 px
components, 2 px shoulder dilation) routes these stars into the satstar
channel ([brick_cmd_iter3_peakseed_2026-07-10.png](../evidence/brick_cmd_iter3_peakseed_2026-07-10.png)).

**H7 — "The ZEROFRAME extension recovers the rim" (rejected on a technicality,
then repaired).** The cal-file ZEROFRAME extension is zeroed exactly at the
flagged pixels we need. The ramp file's calibrated first read (SCI[0,0]) is
not; it anchors the DQ-saturated rim as R(g0)×g0. A **scalar** R was then
rejected too: the cal/group0 ratio drifts 0.229 → 0.167 from g0 ~2k → 20k DN
even after linearity correction (charge migration again), so R is calibrated
per frame as a binned function of g0, with a pile-up-plateau ceiling and a
g0 > 0 validity mask (a percentile ceiling failed on dark frames).

**H8 — "The remaining flux error is a field/background systematic"
(rejected).** The decisive test used M92: zero background, no extinction
structure, 812–1248 calibrators per configuration. The wing excess persisted,
was **purely multiplicative** (additive term ≈ 0), and was flux-independent
across brightness quartiles
([m92_wing_stacks2d_2026-07-10.png](../evidence/m92_wing_stacks2d_2026-07-10.png),
[m92_wing_ratio_curves_2026-07-10.png](../evidence/m92_wing_ratio_curves_2026-07-10.png)).
An injected-model control run through the identical machinery recovered
0.95–0.99 everywhere — the estimators do not invent excess. Stacking and
registration artifacts were bounded < 3% by model-through-machinery and
deliberate mis-centering controls.

**The surviving hypothesis:** the **STPSF model underpredicts the real NIRCam
PSF wings** by ~10–30% (filter- and radius-dependent; Airy troughs too deep).
Demonstrated four independent ways — 2D sub-pixel-registered stacks vs the
model through identical machinery
([psfwing2d_summary_2026-07-10.png](../evidence/psfwing2d_summary_2026-07-10.png)),
per-annulus LSQ amplitude (the estimator the fit actually uses), the
masked-core fit experiment (H5), and the M92 cross-validation (H8) — on five
filters and three detectors. The masked-core bias it predicts matches the
production failure in sign, magnitude, and radius dependence in both fields.

**H9 — "A static per-filter correction table fixes it" (rejected).** The
deficit's radial structure differs per filter (F182M rising to +25% at r=5;
F187N/F212N confined to small radii; F405N neutral; M92 F150W *faint*-biased
at its Airy-dip radius), and 12/20 of the M92 STPSF grids carry a phantom OPD
"segment blob" (2.2% of flux at r≈35 px from sensing epoch R2022062004) that
any static table would bake in. A calibration must be measured *per frame,
against the same model grid the fit uses*.

## 4. The adopted method

The production satstar channel now runs, per frame:

1. **Detection with severity gates** — components must contain pixels whose
   implied peak exceeds the per-filter severity floor
   (`SAT_SEVERITY_FLOOR`, single-sourced with the cataloging auto-floor);
   post-fit, `satstar_implied_peak < 0.5×floor` rejects, with an
   observed-peak second chance for narrow bands.
2. **Peak-based seeding** — unflagged charge-migration stars (image peak above
   floor, no DQ-SAT pixels) are seeded into the channel with 2 px shoulder
   dilation of the mask.
3. **Zeroframe anchoring** — the DQ-saturated rim is filled with
   R(g0)×(ramp first read), R calibrated per frame as a binned curve;
   only the genuinely-unmeasured deep core stays masked.
4. **PSF wing fit** — STPSF gridded model, fit to wings + anchored rim.
5. **Per-frame wing self-calibration** (`apply_wing_selfcal`, default ON) —
   ~30 bright *unsaturated* stars in the same frame are fit twice: truth
   (unmasked) and core-masked at each satstar's own mask radius. The median
   truth/masked ratio, interpolated in mask radius, divides each satstar's
   **catalog** flux. The subtracted **model keeps the raw amplitude**: the fit
   matches the observed wings, so residual images stay clean; only the
   total-flux number carried the model deficit. Because the calibrators see
   the same PSF grid, this absorbs model defects (including the M92 OPD blob
   class) automatically. Columns `flux_fit_raw`, `wingcal_ratio`,
   `wingcal_rmask` preserve the audit trail.
6. **Merging: mutual-nearest replacement at 0.5″** with a faint-replacement
   veto (satstar must be ≥0.8× the clipped catalog flux to replace), and
   `is_saturated` decoupled from `replaced_saturated` so clipped-but-unrepaired
   rows remain identifiable.

Masked-core substitution calibration curves per filter (STPSF + stacked-PSF
based, `scripts/analysis/wing_calibration/calcurves/`) quantify the bias as a
function of masked-pixel count for 8 filters
([masked_core_calibration_curves_2026-07-10.png](../evidence/masked_core_calibration_curves_2026-07-10.png));
they motivated and validate the self-calibration but the production correction
is the per-frame measurement, not the static curve (H9).

## 5. Results

**Flux accuracy.** Wing-calibrated satstar fluxes match the F187N-implied
truth to **6%** on the stars where truth exists; production-measured
calibration ratios (1.027/1.035/1.089 at r_mask = 3/4/7 px on F182M;
1.076–1.098 on M92 F090W) reproduce the laboratory curves; 379/379 demo frames
calibrated successfully.

**CMD continuity.** F182M−F187N class-continuity metric: **0.74 → 0.11**
(0.03 in the transition bins) on the catalog-level demo rebuild
([brick_cmd_calibrated_substituted_2026-07-10.png](../evidence/brick_cmd_calibrated_substituted_2026-07-10.png)).
Across the four requested additional pairs
([brick_multipair_cmds_beforeafter_2026-07-11.png](../evidence/brick_multipair_cmds_beforeafter_2026-07-11.png)):

| Pair | Production | Wing-cal + substitution |
|---|---|---|
| F182M−F212N | 1.35 | 0.34 |
| F405N−F410M | undefined (detached clouds) | 0.48 |
| F405N−F466N | undefined (detached clouds) | 0.38 |
| F212N−F405N | undefined (detached clouds) | 0.86* |

In production, three of four pairs had *no measurable continuity* — the
saturated population formed clouds disconnected from the locus. After the fix
the satstars extend the giant branch in all four pairs. (*F212N−F405N spans
the full extinction vector, so its intrinsic color spread inflates the metric;
the demo floor below applies to all four numbers.)

**Demo floor (stated plainly).** These catalog-level numbers are bounded from
below by the demo's construction: the *unsaturated-side* photometry was
measured on frames with the OLD (wrong-wing) satstar models subtracted, and
rows overwritten by pre-gate fake satstars are unrecoverable without
refitting. Run-to-run scatter from that pollution is ±0.1. Reaching the
<0.05 goal requires the full per-frame re-cataloging (queued behind the
production remediation campaign), after which `assert_saturation_continuity`
certifies every band pair in CI.

## 6. Residual systematics and their controls

1. **Deep-mask wing extrapolation (largest known residual).** For the most
   saturated stars (mask radii ≳10 px; roughly the brightest ~1 mag of the
   satstar population, e.g. F410M ≈ 10.75–11.75), the self-calibration
   extrapolates C(r) beyond where crowded-field calibrators constrain it;
   residual ≈ **−0.25 mag** for that stratum. *Control:* stars carry
   `wingcal_rmask`, so the stratum is identifiable; the planned M92
   sparse-field campaign (zero background, thousands of calibrators to r>20 px)
   calibrates C(r>10) directly. Until then these stars are flagged, not
   silently trusted.
2. **Calibration precision, not just accuracy.** Each frame's ratio is a
   median over ~30 stars: 3–5% per frame, averaging down over the 4–8 frames
   contributing to each consolidated satstar (~2–3% effective). *Control:*
   `wingcal_ratio` per row enables outlier-frame rejection at merge time.
3. **Zeroframe recovery scale.** R(g0) is a per-frame empirical curve; its
   scatter propagates into rim pixels (up to 25% per pixel for a *scalar* R —
   the binned curve removes the trend). *Control:* rim pixels are a minority
   of fit pixels and the wing self-cal is applied downstream of the fit;
   zeroframe-anchored rows are class-tagged in every diagnostic CMD.
4. **STPSF grid defects.** The M92 phantom-blob discovery shows model grids
   can carry epoch-specific artifacts at the % level. *Control:* the per-frame
   self-calibration is measured against the same grid, so first-order model
   defects cancel; grid rebuild for the affected M92 epochs is queued.
5. **Gate leakage.** The observed-peak second chance re-admits stars the
   implied-peak gate would reject; a mis-gated fake would carry clipped flux.
   *Control:* faint-replacement veto at merge (a fake satstar fainter than
   0.8× the catalog row cannot replace it) plus the continuity metric as a
   population-level tripwire.
6. **Unrecoverables.** ~12 extreme-clip rows (saturated in group 0 across all
   dithers with no usable zeroframe) cannot be measured; they are flagged
   `is_saturated` without `replaced_saturated` and excluded by
   `confident_star_mask`.

## 7. Expected accuracy and precision vs unsaturated photometry

| Quantity | Unsaturated PSF photometry | Saturated-star channel (this work) |
|---|---|---|
| Flux precision (per star, consolidated) | ~1–2% | ~3–6% (self-cal 3–5%/frame, averaging over frames) |
| Flux accuracy (systematic) | few % (ZP-limited) | ~6% vs narrow-band truth; up to −0.25 mag for the deepest-mask stratum (flagged) |
| Color continuity across the boundary | — (reference) | 0.03–0.11 mag demo; <0.05 goal pending full recat |
| Astrometry | ~2.7 mas (Hosek benchmark) | 3 mas frame-to-frame repeatability; 14 mas median vs clean narrow-band centroids |
| Completeness at the bright end | 0 above saturation | all but ~12 extreme-clip stars per field |

The satstar channel is, by construction, a factor ~3 noisier and a factor ~2
less accurate than unsaturated photometry — but it now sits on the **same flux
scale** (the self-calibration ties every satstar to unsaturated stars in the
same frame), which is what CMD-based science requires.

A note on the astrometry numbers: the 0.08–0.15″ scatter quoted in §3 (H2) is
the satstar-vs-**daophot** offset for the *same* saturated star — and the
daophot centroid of a clipped, charge-migration-skewed core is itself the
corrupted quantity. Measured properly, the wing-constrained satstar positions
perform far better: frame-to-frame repeatability is **3 mas** median (F182M
and F410M, ≥4 dithers per star), and consolidated satstar positions agree with
*clean unsaturated* F187N daophot centroids of the same stars to **14 mas**
median (90th percentile 109 mas, dominated by blends/mismatches at the 0.3″
match radius). So the large replacement radius exists to catch the corrupted
daophot positions being replaced, not because satstar positions are poor.
Satstar astrometry is ~5× worse than the 2.7 mas unsaturated benchmark but
usable — with per-star repeatability as the error estimate — even for
proper-motion work at the long-baseline (multi-year) level.

## 8. Conclusions

1. The 0.5 mag saturated-star errors were six stacked mechanisms; no single
   fix was sufficient, and two of the "reference" populations used to judge
   the errors were themselves corrupted (H3, H6).
2. The load-bearing physical finding is the STPSF wing deficit (~10–30%,
   filter-dependent), established with controls in a zero-background field;
   it will be reported upstream to STScI with the M92 evidence.
3. The adopted correction is *self*-calibration — per frame, against the same
   model grid, at the same mask radius — rather than any static table; it is
   robust to model-grid defects by construction.
4. Certification is executable (`assert_saturation_continuity`) and blocked
   only on the full re-cataloging run; every frame-level component has been
   independently validated.

## Postscript (2026-07-11): H10 — sub-floor suppression strips

Referee review of this article predicted, and targeted forensics on two
anomalous CMD boxes confirmed, a tenth mechanism: **core suppression is
continuous in well fill, and the pipeline switched estimators discontinuously
at the severity floor.** Every band has an unflagged strip
[flagging floor, floor + ~1.1 mag] where stellar peaks reach 0.4–1.6× the
severity floor with zero DQ-SATURATED pixels; daophot photometry there is
clipped by up to ~0.4 mag. Evidence
([brick_degenerate_pair_trends_2026-07-11.png](../evidence/brick_degenerate_pair_trends_2026-07-11.png)):
**near-degenerate color pairs** (F405N−F410M at 4.05/4.10 μm, F182M−F187N at
1.82/1.87 μm — intrinsic+reddened color nearly constant, so any
magnitude-dependent drift is instrumental) break exactly at the floors:
−0.10 → −0.49 below F410M 13.3; +0.18 → +0.72 below F187N 14.7. Cal-frame
peak measurements put the strip stars at 0.4–0.8× (F410M) and 0.67–1.64×
(F182M) of the floor; wing-annulus photometry (2.5–7 px, fully linear)
shows the F410M strip catalog fluxes ≥0.17 mag too faint while F405N at the
same magnitudes is clean. SW strips are migration-dominated (a 9×9 cal
aperture recovers +0.15 mag over the PSF fit); LW strip flux is mostly not
locally conserved (+0.06 mag).

Fixes (this branch): (i) **sub-floor seeding** — DQ-clean components peaking
in [0.35×floor, floor) enter the satstar channel with an amplitude-derived
mask (bright pixels + 2 px shoulder; 2–50 px size window), so the flux comes
from linear-regime wings; the post-fit implied-peak gate arbitrates
(suppressed stars' wings imply peaks above threshold and are kept;
unsuppressed sub-floor stars are rejected and keep their daophot values);
(ii) a latent short-circuit fixed — frames with zero DQ-SATURATED components
previously skipped seeding entirely; (iii) **degenerate-pair flatness**
(`degenerate_pair_flatness` / `assert_degenerate_pair_flatness`) added as a
certification gate: the released catalog's near-degenerate colors must be
magnitude-flat to <0.05, which detects any residual strip, taper error, or
flux-scale drift independent of locus models and saturation flags.

The affected-window map for other pairs (strips at F212N 15.6–16.7,
F182M 14.0–15.0, all three LW bands 12.0–13.3, and the o004 wide bands at
their own floors) explains the residual wiggles in the wide-pair CMDs and is
covered by the same gate.
