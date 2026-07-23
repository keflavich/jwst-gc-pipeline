# GDC-vs-CRDS astrometry experiment — results

Date: 2026-07-23.  Branch `feature/jayander-gdc` (PR #154).  Pure
post-processing (no re-reduction, no SLURM): each per-exposure m1 catalog gets
a second, STDGDC-corrected sky position per star, and both variants are pushed
through the same m2-level measurement machinery side by side.

**One-line verdict: the STDGDC (Jay Anderson JWST1PASS) distortion swap
produces no measurable improvement on any metric — relative (module-seam,
consensus scatter) or absolute (VIRAC2/Gaia tie) — at the ~1–1.5 mas
per-exposure noise floor of these fields, and slightly degrades the Hosek/L2
agreement on arches NRCA4.  The real residual astrometric structure (the
2.7–5 mas brick A/B module-seam terms) is inter-detector affine placement,
which a distortion-field swap cannot touch (and, by the frame-ownership
design of this integration, must not).**

## 1. Setup

**Variants per per-exposure (m1) catalog:**

- **CRDS** — the existing `skycoord_centroid` (CRDS `assign_wcs` gwcs + the
  VIRAC2-tied offsets machinery; the production solution).
- **GDC** — STDGDC forward maps, affine-anchored per frame to the frame's own
  crf WCS (`gdc_wcs.GDCSkySolution`), applied as a per-star *delta* on
  `skycoord_centroid`: bulk pointing/scale/rotation are IDENTICAL to CRDS by
  construction; only the higher-order (non-affine) distortion field differs.
  Library version `auto` (v2 preferred).
- **GDC v1** — additionally, on a sensitivity subset (arches NRCA4, brick
  nrcb1), the v1 library solutions.

**Data** (per-exposure per-detector m1 `daophot_basic` catalogs + matching
`*_destreak_o001_crf.fits` frames):

| set | exp x det | catalogs | notes |
|---|---|---|---|
| arches F212N (jw02045 o001 vgroup02101) | 12 x 8 | 96 | RAOFFSET=0 (no offsets table applied) |
| brick F212N (jw02221 o001 vgroup05101) | 24 x 8 | 192 | m2cycle2 offsets baked |
| brick F182M (jw02221 o001 vgroup07101) | 24 x 8 | 192 | catalog SNAPSHOT 2026-07-22 (originals regenerating) |

**Quality cut** mirrors m2 (`visit_consensus.select_reliable_stars`):
`qfit <= 0.1`, S/N >= 10, finite x/y.

**Per-frame verification.**  Every catalog's `skycoord_centroid` was checked
same-star against its crf frame's on-disk WCS at (`x_fit`,`y_fit`) — an
identity check, not a cross-catalog match; > 5 mas would mean a
regenerated-crf/catalog mismatch and drop the frame.  **All 480 frames
passed** (median 0.6–0.7 mas, max 0.78 mas), so no F182M snapshot exposure
was dropped for crf vintage.

| field/filter | frames | kept/total stars | cat-vs-crf check (med/max mas) | affine rms v2 (med/max mas) | GDC delta field p95/max (mas) | per-star delta med (mas) |
|---|---|---|---|---|---|---|
| arches F212N | 96 | 566,860/2,581,476 | 0.60/0.73 | 0.77/0.95 | 1.4/4.2 | 0.59 |
| brick F212N | 192 | 233,077/4,364,493 | 0.61/0.76 | 0.77/0.95 | 1.4/4.2 | 0.59 |
| brick F182M | 192 | 1,125,472/6,962,256 | 0.68/0.78 | 0.79/0.96 | 1.4/5.7 | 0.57 |

The CRDS-vs-GDC distortion delta itself is small: per-frame affine-anchor rms
~0.8 mas, per-star median ~0.6 mas, p95 ~1.4 mas, max ~4–6 mas at detector
corners.  So the *ceiling* on any possible GDC improvement was a few mas at
frame edges — exactly where the module-overlap test looks.

**Astrometry-rules compliance.**  All offsets measured with `measure_offset`
(offset-histogram, sweep ON) for detection and `local_residual_map`
(matched-pair, only after a verified small tie) for precise same-star values;
dense-reference histogram peaks are reported as detection only (see
`histogram-vs-samestar-offset-bias`).  No NN-median against a dense catalog
anywhere.

### Library defect found (fixed in this PR)

`STDGDC_NRCB4_F212N[_2].fits` (v1 AND v2) store **unmeasured border pixels as
0** — the y=2047 row (x<=1535) and the x=2047 column (y>=1535), 2047 px.  Read
raw, that is a ~2048-px "correction"; one poisoned anchor-grid point corrupted
the whole-frame affine fit (affine rms **11.9 arcsec** instead of 0.8 mas on
every arches+brick nrcb4/F212N frame).  `GDCSkySolution` now masks invalid
map samples (|correction| > 50 px) out of the anchor fit, NaNs them in the
delta map, warns (`n_anchor_invalid`), and falls back to the frame's original
WCS position for stars on invalid regions (`n_star_fallback`).  Regression
tests added (synthetic holed map).  No other F182M/F212N library file has
holes.

## 2. Measurement A — relative, frame-overlap (module seam)

Reference-free pair measurement of every geometrically overlapping frame pair
(`interframe_overlap` footprint gate -> swept `measure_offset` -> verified
small -> `local_residual_map`: one giant cell = the pair's same-star offset;
2"-cell maps where the density allows).  Pair kinds: **AB** = cross-module
(the headline; brick only — arches is a single pointing with sub-arcsec
dithers and has NO cross-module overlaps), **xdet** = same module different
detector (sampled, cap 120), **same** = same detector different exposure
(control — the GDC field cancels between same-detector frames at small
dither; cap 120).

| field/filter | pair kind | n meas | same-star med off (mas) | p90 | rms dRA/dDec | 2" cell med / p90 (mas) | contrast med |
|---|---|---|---|---|---|---|---|
| arches F212N | xdet crds | 120 | 13.40 | 17.39 | 11.19/8.04 | 14.41 / 15.92 | 112 |
| arches F212N | xdet gdc | 120 | 13.03 | 17.70 | 11.23/7.84 | 13.05 / 16.05 | 111 |
| arches F212N | same crds | 120 | 4.47 | 17.92 | 4.34/7.77 | 4.05 / 7.76 | 1184 |
| arches F212N | same gdc | 120 | 4.48 | 17.86 | 4.33/7.75 | 4.10 / 7.77 | 1183 |
| brick F212N | AB crds | 119 | 4.99 | 5.59 | 1.74/4.29 | n/a (too sparse) | 52 |
| brick F212N | AB gdc | 119 | 4.89 | 5.59 | 1.69/4.25 | n/a | 53 |
| brick F212N | xdet crds | 120 | 2.76 | 5.19 | 2.48/2.34 | n/a | 120 |
| brick F212N | xdet gdc | 120 | 2.82 | 6.18 | 2.84/2.46 | n/a | 121 |
| brick F212N | same crds | 120 | 1.10 | 2.31 | 1.13/1.01 | n/a | 270 |
| brick F212N | same gdc | 120 | 1.07 | 2.10 | 1.12/1.04 | n/a | 261 |
| brick F182M | AB crds | 156 | 2.74 | 3.15 | 1.43/2.18 | 1.91 / 3.44 | 252 |
| brick F182M | AB gdc | 156 | 2.85 | 4.07 | 1.35/2.80 | 2.31 / 3.64 | 238 |
| brick F182M | xdet crds | 120 | 1.49 | 3.37 | 1.30/1.52 | 1.77 / 3.16 | 343 |
| brick F182M | xdet gdc | 120 | 1.67 | 3.59 | 1.50/1.60 | 1.83 / 3.38 | 358 |
| brick F182M | same crds | 120 | 1.07 | 2.45 | 1.32/0.82 | 1.14 / 2.59 | 679 |
| brick F182M | same gdc | 120 | 1.07 | 2.39 | 1.31/0.82 | 1.05 / 2.44 | 684 |

Per-pair same-star sem is ~0.25 mas (med 76 matched stars/pair, F212N), so
these medians are precise.  **The brick A/B seam offsets are COHERENT and
detector-pair dependent** (CRDS, b−a convention, med over pairs):

| field | detector pair | n pairs | dRA (mas) | dDec (mas) | GDC change |
|---|---|---|---|---|---|
| brick F212N | nrca3–nrcb4 | 64 | −0.2 | **−5.4** | ~none (−5.4 → −5.3) |
| brick F212N | nrca4–nrcb3 | 55 | +2.5 | −2.5 | ~none (−2.5 → −2.4) |
| brick F182M | nrca3–nrcb4 | 64 | +1.5 | −1.3 | ~none |
| brick F182M | nrca4–nrcb3 | 64 | +1.1 | +2.7 | slightly worse (+2.9) |
| brick F182M | nrca3–nrcb3 | 12 | +2.6 | +1.6 | worse (+3.7) |
| brick F182M | nrca4–nrcb4 | 16 | +0.6 | +2.9 | worse (+5.0) |

**Verdict A: no improvement.**  AB medians move 4.99 → 4.89 mas (F212N,
−2%) and 2.74 → 2.85 mas (F182M, +4%, p90 3.15 → 4.07 — mildly worse);
same-detector controls are unchanged (as designed).  The seam terms are
static inter-detector *affine placement* differences (filter/epoch dependent
— they differ between F212N and F182M), which the affine-anchored GDC
preserves by construction.  This is the SIAF-placement/self-cal class of
error (`siaf-accuracy-network-selfcal`), not a distortion-map deficiency.

Arches note: with no offsets table applied (RAOFFSET=0), per-exposure
pointing errors (~4–18 mas) dominate every arches pair measurement
identically in both variants — arches stage A is insensitive to distortion.

## 3. Measurement B — per-visit consensus scatter (the m2 metric)

The real m2 machinery (`visit_consensus.build_visit_consensus`) per module
and variant: per-star cross-exposure position scatter + every exposure's
swept-histogram tie vs the consensus.

| field/filter | module | variant | consensus stars | median scatter (mas) | p90 | exp-vs-consensus med/max (mas) | unverified/misaligned |
|---|---|---|---|---|---|---|---|
| arches F212N | nrca | crds | 38,357 | 1.84 | 2.76 | 3.46/17.22 | 0/44 |
| arches F212N | nrca | gdc | 38,359 | 1.85 | 2.76 | 3.45/17.21 | 0/44 |
| arches F212N | nrcb | crds | 33,924 | 1.57 | 2.62 | 3.56/19.48 | 0/42 |
| arches F212N | nrcb | gdc | 33,922 | 1.57 | 2.64 | 3.54/19.41 | 0/42 |
| brick F212N | nrca | crds | 9,855 | 1.42 | 2.41 | 0.77/1.53 | 0/0 |
| brick F212N | nrca | gdc | 9,855 | 1.42 | 2.43 | 0.80/1.58 | 0/0 |
| brick F212N | nrcb | crds | 11,067 | 1.43 | 2.57 | 0.88/1.72 | 0/0 |
| brick F212N | nrcb | gdc | 11,067 | 1.43 | 2.64 | 0.85/1.65 | 0/0 |
| brick F182M | nrca | crds | 44,448 | 1.20 | 2.17 | 0.58/1.56 | 0/0 |
| brick F182M | nrca | gdc | 44,452 | 1.16 | 2.26 | 0.60/1.49 | 0/0 |
| brick F182M | nrcb | crds | 54,509 | 1.14 | 2.11 | 0.83/1.75 | 0/0 |
| brick F182M | nrcb | gdc | 54,507 | 1.15 | 2.25 | 0.76/1.86 | 0/0 |

**Verdict B: unchanged** (|change| <= 0.04 mas, both signs).  Expected: the
consensus is dominated by same-detector cross-exposure comparisons at
sub-arcsec dither, where any static distortion field cancels.  (The arches
"44/48 misaligned" rows are the uncorrected per-exposure pointing — 3.5 mas
median, up to 19 mas vs the 2 mas m2 tolerance — identical in both variants;
see anomaly 2.)

## 4. Measurement C — absolute vs VIRAC2 / Gaia-sparse

Input = the de-duplicated m2 consensus catalogs (module A+B union) per
variant.  The dense-VIRAC histogram peak is DETECTION only; the precise bulk
is the same-star matched-pair value (giant cell), plus a 20"-cell residual
map (2" cells lack matched-pair density: only ~6k JWST-x-VIRAC pairs per
field).  Gaia-sparse is the non-blocking diagnostic cross-check.

| field/filter | variant | ref | hist peak dRA/dDec (mas, detection) | contrast | same-star dRA/dDec (mas) | sem | n pairs | 20" cell robust scatter dRA/dDec (mas) | n cells |
|---|---|---|---|---|---|---|---|---|---|
| arches F212N | crds | VIRAC2 | −8.9/−29.8 | 45 | −8.5/−25.8 | 0.83 | 7,496 | 9.4/14.6 | 101 |
| arches F212N | gdc | VIRAC2 | −8.7/−30.0 | 45 | −8.4/−25.9 | 0.83 | 7,494 | 9.4/14.0 | 101 |
| arches F212N | crds | Gaia-sparse | −10.0/−26.1 | 82 | −8.9/−24.9 | 1.16 | 166 | n/a | 0 |
| arches F212N | gdc | Gaia-sparse | −10.0/−26.3 | 84 | −9.2/−24.6 | 1.13 | 167 | n/a | 0 |
| brick F212N | crds | VIRAC2 | −6.3/+6.3 | 112 | +0.1/−0.6 | 1.07 | 6,234 | 9.9/7.5 | 140 |
| brick F212N | gdc | VIRAC2 | −6.3/+6.1 | 116 | +0.0/−0.5 | 1.07 | 6,230 | 9.6/7.5 | 140 |
| brick F212N | crds | Gaia-sparse | −2.4/+2.1 | 42 | −1.5/+2.3 | 1.31 | 130 | n/a | 0 |
| brick F212N | gdc | Gaia-sparse | −2.3/+1.8 | 42 | −1.6/+2.0 | 1.31 | 130 | n/a | 0 |
| brick F182M | crds | VIRAC2 | +6.0/+6.1 | 33 | −0.4/−0.0 | 0.99 | 5,866 | 8.8/8.4 | 134 |
| brick F182M | gdc | VIRAC2 | +6.2/+6.0 | 33 | −0.5/−0.1 | 0.99 | 5,866 | 8.8/8.3 | 134 |
| brick F182M | crds | Gaia-sparse | −2.2/+1.2 | 30 | −2.0/+1.1 | 1.41 | 118 | n/a | 0 |
| brick F182M | gdc | Gaia-sparse | −2.4/+1.0 | 32 | −1.8/+0.9 | 1.41 | 118 | n/a | 0 |

**Verdict C: identical** — same-star bulk differences <= 0.15 mas; 20"-cell
robust scatter differences <= 0.6 mas (that scatter itself is the known
~5–10 mas VIRAC2 local wander + per-star VIRAC noise, not JWST distortion).
Bonus confirmations: (i) the brick dense-VIRAC histogram peak reads ~6–9 mas
where the same-star value is ~0 — the documented dense-reference histogram
bias, again with filter-dependent sign; (ii) brick's m2cycle2 frames are tied
to VIRAC2 at <= 1 mas in both filters.

## 5. Measurement D — Hosek/peppar L2 (arches NRCA4)

Our NRCA4-only m2-like consensus (12 exposures; 10,834 stars, median
consensus scatter 0.85/0.88 mas crds/gdc) vs Hosek's L2
`combo_starlist_F212N_NRCA4` (74,896 sources, t=2023.635) via the benchmark
scorer (path-fixed copy of `compare_hosek_L2.py`; 1:1 mutual NN < 80 mas,
median-subtracted residuals — the metric of the 2026-06-30 report whose
baseline was 2.7 mas median sep on the m7 catalog).  Epoch: the L2 averages
the SAME exposures we use (single epoch) — no PM term, matching the prior
comparison.  Hosek's pipeline applies these same STDGDC maps (peppar), so a
dominant CRDS-distortion error should make the GDC variant agree better.

| metric | CRDS | GDC |
|---|---|---|
| 1:1 mutual matches (<80 mas) | 10,406 | 10,407 |
| median separation | **1.9 mas** | **2.2 mas** |
| robust scatter dRA/dDec | **1.3 / 0.9 mas** | 1.6 / 1.2 mas |
| raw scatter dRA/dDec (incl. saturated) | 4.6 / 4.8 mas | 4.7 / 4.8 mas |
| 6-par-affine-removed robust dRA/dDec | 1.34 / 0.89 mas | 1.46 / 1.23 mas |
| instr-mag offset (ours−Hosek) | −3.931 | −3.931 |
| bright (m<−6) mag scatter | 0.238 (N=1419) | 0.238 |

**Verdict D: GDC slightly DEGRADES the Hosek agreement** (median sep 1.9 →
2.2 mas; robust scatter up ~0.2–0.3 mas in both axes, and still up after
removing a free 6-parameter affine).  Reading: at the ~1 mas level, our CRDS
per-exposure solution already matches Hosek's effective L2 frame; Hosek's
exposure-to-frame polynomial transforms absorb net distortion, so
re-applying raw STDGDC on our side only injects the CRDS-vs-STDGDC delta
field (~0.6 mas med, up to ~4 mas at edges) as extra disagreement.  (The
1.9 mas here vs 2.7 mas in the prior report reflects the cleaner
per-exposure consensus input vs the m7 merged catalog, not a pipeline
change.)  Photometry columns reproduce the known −3.9 mag normalization and
0.24 mag bright-star scatter — untouched by this experiment.

## 6. Measurement E — v1 vs v2 library sensitivity

Per-star difference between v1- and v2-anchored solutions (same frames):

| subset | frames | stars | median (mas) | p95 | max |
|---|---|---|---|---|---|
| arches F212N nrca4 | 12 | 83,501 | 0.092 | 0.201 | 0.435 |
| brick F212N nrcb1 | 24 | 34,053 | 0.124 | 0.281 | 0.473 |
| brick F182M nrcb1 | 24 | 167,494 | 0.102 | 0.219 | 0.442 |

The two library versions are interchangeable at the ~0.1 mas level here;
`auto` (v2-preferred) is fine.

## 7. Overall verdict

Per metric, GDC vs CRDS:

| metric | improvement? | size | significant? |
|---|---|---|---|
| A: brick A/B seam (same-star pair offsets) | no | −0.1 mas (F212N) / +0.1–0.9 mas worse (F182M p90) | no (pair sem 0.25 mas; coherent seam terms unchanged) |
| A: cross-detector / same-detector controls | no | <= 0.2 mas either sign | no |
| B: m2 consensus scatter | no | <= 0.04 mas | no |
| C: absolute VIRAC2/Gaia tie (bulk + 20" cells) | no | <= 0.15 mas bulk, <= 0.6 mas cell scatter | no |
| D: Hosek L2 agreement | **worse** | +0.3 mas median sep, +0.2–0.3 mas scatter | yes (10k matches; consistent across raw/robust/affine-removed) |
| E: v1 vs v2 | n/a | 0.1 mas | — |

The CRDS distortion solution for these SW filters is already good to the
~1 mas level at which these fields can measure; the STDGDC field differs
from it by only ~0.6 mas (median star) / ~4 mas (corners), and swapping it
in buys nothing.  The real, coherent residuals this experiment DID measure —
the 2.7–5 mas, filter-dependent, detector-pair-dependent A/B seam terms —
are inter-detector affine placements (SIAF class), invisible to any
distortion-map swap under the frame-ownership anchoring this integration
(correctly) uses.  Fixing the seam needs per-detector affine/offset ties
(the `siaf-accuracy-network-selfcal` direction), not STDGDC.

**Recommendation: do not adopt the GDC starlist correction for production;
keep the package as an opt-in diagnostic.**

## 8. Anomalies found along the way

1. **STDGDC library defect** — NRCB4/F212N v1+v2 border holes (stored as 0);
   now masked in `GDCSkySolution` (this PR).  Survey of all F182M/F212N
   library files found no other holes.
2. **Arches absolute frame is off by ~27 mas** — the merged arches F212N
   consensus sits at (−8.5, −25.8) mas vs VIRAC2 AND (−8.9, −24.9) mas vs
   Gaia-sparse (same-star, sem ~1 mas; the two refs agree, so it is real and
   JWST-side).  Consistent with RAOFFSET=0: no offsets table has ever been
   applied to these arches products.  Additionally 44/48 (A) and 42/48 (B)
   exposures are >2 mas (median 3.5, max 19 mas) off their visit consensus —
   uncorrected per-exposure pointing.  Arches needs an offsets table +
   realignment before release-grade use.
3. **Brick A/B seam terms** (measurement A): coherent, precise (sem ~0.25
   mas), detector-pair- and filter-dependent: F212N nrca3–nrcb4 = −5.4 mas
   Dec, nrca4–nrcb3 = (+2.5, −2.5); F182M 1–3 mas with different signs.
   Right at the m7 cross-band gate scale (<5 mas) — worth a per-detector
   affine tie at m2 eventually.
4. **Dense-histogram bias reconfirmed** (measurement C): brick histogram
   peaks read 6–9 mas vs same-star ~0, sign differs by filter — exactly the
   `histogram-vs-samestar-offset-bias` artifact class.

## 9. Caveats

- **F182M snapshot vintage**: m1 catalogs are a 2026-07-22 snapshot taken
  while the originals regenerate; every snapshot catalog was verified
  same-star against the current on-disk crf WCS (max 0.78 mas), so the
  catalog/frame pairing is consistent.
- **Arches is single-band, single-pointing**: no cross-module overlaps; its
  relative metrics are dominated by uncorrected per-exposure pointing, so the
  AB seam test rests on brick alone.
- **LW uncovered**: the STDGDC library has no LW solutions except F277W —
  this experiment says nothing about F323N–F480M distortion.
- **Affine anchoring is also a blind spot by design**: any CRDS error in
  per-detector scale/rotation/placement is absorbed into the anchor and NOT
  tested — only the non-affine distortion difference is.  (That is the
  correct design for this pipeline: the frame is owned by the offsets
  machinery.)
- **quintuplet F212N replication**: not run (time); the cached machinery
  makes it a one-command follow-up.
- 2"-cell maps vs VIRAC are underpopulated (~6k matched pairs/field); the
  cell scatter is reported at 20".

## 10. Reproduction

Scripts + cached corrected catalogs (per-frame CRDS+GDC positions, all 480
frames) live in the session scratchpad
(`.../scratchpad/gdc_experiment/{scripts,corrected,results}`):
`stage1_correct.py` (corrections + verification), `stage_a_overlap.py`,
`stage_b_consensus.py`, `stage_c_absolute.py`, `stage_d_hosek.py`
(+ path-fixed `compare_hosek_L2_fixed.py` copy), `stage_e_v1v2.py`; all
outputs are JSON under `results/`.  Reruns skip cached frames.

## 11. Delta-field figures (2026-07-23)

Per-detector quiver maps of the CRDS->GDC distortion delta
(`GDCSkySolution.delta_map()`, 24x24 grid, one representative crf frame per
detector: brick F212N exposure 00001 x 8 SW detectors + arches F212N NRCA4;
NRCB4's unmeasured STDGDC border cells marked), plus a per-detector
median/max |delta| summary.  Generated by
`jwst_gc_pipeline/astrometry_gdc/make_vector_figures.py`; the repo bans
tracked PNGs (PR #73), so the figures are hosted on a gist and embedded in
the PR #154 discussion:

- vector fields: <https://gist.githubusercontent.com/keflavich/888700b865b93770b99fe2712db75682/raw/gdc_delta_vecfield_f212n.png>
- summary: <https://gist.githubusercontent.com/keflavich/888700b865b93770b99fe2712db75682/raw/gdc_delta_summary_f212n.png>

| panel | median \|delta\| (mas) | p95 | max | invalid cells | affine rms (mas) |
|---|---|---|---|---|---|
| brick F212N nrca1 | 0.79 | 1.74 | 3.65 | 0 | 0.98 |
| brick F212N nrca2 | 0.62 | 1.35 | 3.35 | 0 | 0.78 |
| brick F212N nrca3 | 0.64 | 1.48 | 4.13 | 0 | 0.86 |
| brick F212N nrca4 | 0.63 | 1.32 | 3.52 | 0 | 0.80 |
| brick F212N nrcb1 | 0.54 | 1.54 | 3.76 | 0 | 0.79 |
| brick F212N nrcb2 | 0.58 | 1.24 | 2.03 | 0 | 0.72 |
| brick F212N nrcb3 | 0.57 | 1.21 | 3.76 | 0 | 0.76 |
| brick F212N nrcb4 | 0.71 | 1.44 | 3.79 | 23 | 0.87 |
| arches F212N nrca4 | 0.63 | 1.32 | 3.52 | 0 | 0.80 |

The delta field is a static (detector, filter) property: arches NRCA4 and
brick NRCA4 give the same map to the quoted precision (the affine anchor
removes pointing, and |delta| is rotation-invariant), confirming the
correction is frame-independent.  Structure is coherent (e.g. the nrca1
central swirl, edge/corner ramps on every detector), sub-mas over most of
each detector, and 2-4 mas only at edges/corners.

## 12. MGC pixel-area magnitude correction (2026-07-23)

**Question: does applying the STDGDC MGC (pixel-area magnitude correction,
`STDGDC.pixel_area_mag`, added to the instrumental mag in the peppar
convention) double-count something our PSF photometry already handles?  And
how big is it?**  Analysis script:
`jwst_gc_pipeline/astrometry_gdc/mgc_pixel_area_analysis.py`.

### (0) Library finding: v2 files have NO MGC

Every **v2** STDGDC file (all 16 F212N+F182M SW files) stores MGC
identically **zero** -- only the **v1** files carry the pixel-area map.  The
peppar-style `version='auto'` (v2-preferred) load therefore silently applies
no MGC.  The v1 NRCB4/F212N MGC also carries border-hole garbage (values to
-3.2 mag), masked below (same class as the section-1 XGC/YGC holes).

### (a) MGC magnitude scale (v1; valid cells; mmag)

Not <<mmag: p5-p95 spans +-9 to +-20 mmag, extrema to 29 mmag, pk-pk 27-52
mmag per detector.  Achromatic (F212N == F182M to ~0.1 mmag -- it is
geometry, not throughput):

| filter | detector | median | p5 | p95 | max abs | pk-pk | invalid px |
|---|---|---|---|---|---|---|---|
| F212N | NRCA1 | -0.5 | -18.3 | +14.4 | 22.1 | 38.9 | 0 |
| F212N | NRCA2 | -0.5 | -12.1 | +8.8 | 16.2 | 26.8 | 0 |
| F212N | NRCA3 | -0.6 | -19.9 | +16.7 | 29.2 | 51.3 | 0 |
| F212N | NRCA4 | -0.7 | -14.7 | +11.8 | 22.2 | 38.2 | 0 |
| F212N | NRCB1 | -0.5 | -12.1 | +8.9 | 15.9 | 26.7 | 0 |
| F212N | NRCB2 | -0.6 | -18.2 | +14.4 | 21.5 | 38.4 | 0 |
| F212N | NRCB3 | -0.7 | -14.6 | +11.5 | 22.4 | 38.4 | 0 |
| F212N | NRCB4 | -0.8 | -19.9 | +16.7 | 28.4 | 51.5 | 22427 |
| F182M | NRCA1 | -0.6 | -18.3 | +14.3 | 22.1 | 38.9 | 0 |
| F182M | NRCA2 | -0.5 | -12.2 | +8.7 | 16.3 | 26.9 | 0 |
| F182M | NRCA3 | -0.6 | -19.9 | +16.6 | 29.2 | 51.3 | 0 |
| F182M | NRCA4 | -0.8 | -14.7 | +11.8 | 22.2 | 38.2 | 0 |
| F182M | NRCB1 | -0.6 | -12.2 | +8.8 | 16.0 | 26.7 | 0 |
| F182M | NRCB2 | -0.7 | -18.3 | +14.4 | 21.6 | 38.5 | 0 |
| F182M | NRCB3 | -0.8 | -14.7 | +11.4 | 22.4 | 38.3 | 0 |
| F182M | NRCB4 | -0.7 | -19.9 | +16.7 | 28.5 | 51.4 | 0 |

### (b) Our pipeline does NOT apply a per-pixel area term

Evidence chain (all in this branch's tree):

1. The m1 fit runs directly on the crf SCI array in **MJy/sr** (surface
   brightness): `crowdsource_catalogs_long.py:1964-1976` (`load_data`
   returns `im1['SCI'].data` untouched) -> `cataloging.py:1440` ->
   `cataloging.py:225` (`result = phot(data, ...)`).  No multiplication by
   the crf `AREA` extension anywhere on that path; the only `area`
   references in the photometry package are the cutout machinery *copying*
   the extension through (`crowdsource_catalogs_long.py:439,495,507`).
2. The PSF model absorbs nothing spatial: every ePSF in the gridded-PSF
   files the fit uses is normalized to **unit sum** (measured: 6 brick
   F182M/F212N oversample1 grids, 377-533 ePSFs each, sum spread 0.000
   mmag).  So `flux_fit` is the star's locally-summed surface brightness.
3. Flux calibration multiplies by ONE constant pixel area per frame:
   `merge_catalogs.py:1663-1700` (daophot path) and `1476-1483`
   (crowdsource) and `2403` (satstar) use `wcs.proj_plane_pixel_area()` --
   the CD-matrix pixel area, spatially constant (SIP does not enter it).

Because the flat field divides out the per-pixel solid-angle response, a
point source on a larger-than-nominal pixel is *under*-measured by
`2.5*log10(area_rel)` when calibrated with a constant pixel area -- exactly
the error MGC corrects (peppar adds MGC, and MGC is negative where pixels
are large; signs consistent).

### (c) MGC is the same information as the CRDS AREA reference file

Per-detector correlation of the v1 MGC map against the crf `AREA` extension
(per-pixel area relative to nominal, brick F212N frames), sampled every 8 px:

| detector | pearson r | slope MGC/(2.5 log10 area_rel) | MGC rms | area-term rms | residual rms | area pk-pk (mmag) |
|---|---|---|---|---|---|---|
| NRCA1 | -1.000 | -0.996 | 10.4 | 10.5 | 0.2 | 39.4 |
| NRCA2 | -1.000 | -1.006 | 6.8 | 6.7 | 0.2 | 25.2 |
| NRCA3 | -1.000 | -0.998 | 11.3 | 11.3 | 0.2 | 52.4 |
| NRCA4 | -1.000 | -0.999 | 8.0 | 8.0 | 0.2 | 38.2 |
| NRCB1 | -0.999 | -0.992 | 6.7 | 6.8 | 0.2 | 25.9 |
| NRCB2 | -1.000 | -1.003 | 10.5 | 10.5 | 0.2 | 38.5 |
| NRCB3 | -1.000 | -0.995 | 7.9 | 8.0 | 0.2 | 38.8 |
| NRCB4 | -1.000 | -0.996 | 11.4 | 11.4 | 0.2 | 52.4 |

`MGC = -2.5*log10(area_rel)` to 0.2 mmag rms with slope -1.00 on every
detector: MGC carries **no information beyond the CRDS AREA reference
file** (to <1 mmag).

### (d) Verdict

**Applying MGC (or equivalently the AREA map) is (ii) COMPLEMENTARY, not
double-counting**: our unit-normalized-ePSF fit on MJy/sr frames calibrated
with a constant per-frame pixel area currently *inherits* the full local
pixel-area error.  The uncorrected spatial photometric systematic is
**~7-11 mmag rms per detector, extrema ~30 mmag, pk-pk 27-52 mmag** --
above the mmag level, though well below the current 0.24 mag bright-star
scatter floor (Hosek benchmark).  If a correction is ever adopted, prefer
the pipeline-native crf `AREA` extension (identical information, no v1/v2
library ambiguity, defined for every filter/detector including LW); note
the v2-STDGDC MGC is empty, so any peppar-style `auto` application applies
nothing.
