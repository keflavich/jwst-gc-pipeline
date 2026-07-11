# The inter-detector differential velocity aberration residual in JWST/NIRCam WCS: measurement and magnitude

**Authors:** A. Ginsburg et al. (JWST GC pipeline team), with agent-assisted analysis
**Date:** 2026-07-11
**Contact:** adamginsburg@ufl.edu
**Software version audited:** `jwst` 1.21.0.dev314 (commit `61bd2fe47`), CRDS context `jwst_1253.pmap`
**Related upstream issue:** [spacetelescope/jwst#9400](https://github.com/spacetelescope/jwst/issues/9400) (JP-3987)

## Summary

The `assign_wcs` differential velocity aberration (DVA) correction is applied
as a scale about **each detector's own aperture reference point**. This
corrects the aberration *within* each detector but leaves the aberration of
the detector *separations* in the delivered WCS. The residual is a
per-detector rigid shift `−(1 − VA_SCALE) × (ref_d − C)` (C = any
exposure-common point), i.e. an apparent plate-scale error of
`1 − VA_SCALE ≈ 1e-4` across the focal-plane aperture layout: **±9–13 mas at
NIRCam module lever arms**. We measure this signature directly, at high
significance, in two independent JWST programs: the fitted inter-detector
scale equals each program's own `1 − VA_SCALE` value. Because `VA_SCALE`
swings by ~±1e-4 over the year, the apparent module separation of NIRCam
changes by up to ~25 mas between epochs — a direct systematic for
multi-epoch/proper-motion science. The term survives into standard Level-3
products: `tweakreg` moves all detectors of an exposure as a rigid group and
the operational NIRCam `pars-tweakregstep` files fit `shift`-only.

## 1. The issue in the code

All references to `jwst` commit `61bd2fe47` (1.21.0.dev314); the relevant
code is unchanged in contemporary releases.

1. `jwst/lib/set_telescope_pointing.py` L1402–1461 (`calc_gs2gsapp`,
   implementing Eq. 40 of JWST-STScI-003222): the attitude-level VA
   correction is a **rigid rotation** — "The velocity aberration correction
   is applied in the direction of the guide star." A rotation moves all
   apertures together and cannot rescale aperture separations. It enters the
   attitude at ~L2552 (`m_eci2gs = M_ics2idl @ m_gs2gsapp @ m_eci2gsics`).
2. The per-aperture `RA_REF`/`DEC_REF` keywords therefore encode the SIAF
   (physical, unaberrated) aperture separations attached to a
   guide-star-de-aberrated attitude — while the photons arrive along
   aberrated directions.
3. `jwst/assign_wcs/nircam.py` L100–108: the imaging WCS pipeline inserts
   `va_corr = pointing.dva_corr_model(va_scale, v2_ref=wcsinfo.v2_ref,
   v3_ref=wcsinfo.v3_ref)` — each detector's **own** reference point.
4. `jwst/assign_wcs/pointing.py` L305–353 (`dva_corr_model`): L334 builds
   `Scale(va_scale) & Scale(va_scale)`; L348–351 compose it with
   `Shift((1−va_scale)·v2_ref) & Shift((1−va_scale)·v3_ref)`. Net transform:
   `v′ = va·(v − ref_d) + ref_d` — a scale about the detector's own
   reference. (L342–343: with `v2_ref = v3_ref = 0` the same function
   returns a scale about the V1 origin — the globally consistent variant is
   the degenerate case of the existing code.)
5. `jwst/assign_wcs/pointing.py` L35–62 (`v23tosky`): a pure rotation.

No stage in the chain applies the DVA scale to the *separations between*
aperture reference points. The residual internal inconsistency of the
delivered WCS, for a detector with reference `ref_d` and any exposure-common
point `C`, is a rigid per-detector shift (exactly constant across the
detector, since the intra-detector part is corrected):

    Δ_d = (1 − VA_SCALE) × (ref_d − C)

For NIRCam, detector references span ~±125″ (SW) / ~±90″ (LW) about the
instrument center, so `|Δ| ≈ 9–13 mas` per detector at `1−VA_SCALE ≈ 1e-4`,
anti-symmetric between modules A and B — i.e. it presents as a
**module A/B offset** of ~16–25 mas that is not a SIAF error.

## 2. Measurement method: network self-calibration

We measured the effect on two JWST programs covering the same field (the
"Brick", G0.253+0.016): **PID 2221** obs 001 (2022-08-28; F182M, F187N,
F212N, F405N, F410M; `VA_SCALE = 0.99990819`) and **PID 1182** obs 004
(2022-09-14/19, two visits; F115W, F200W; `VA_SCALE ≈ 0.99990`).

Per-exposure, per-detector PSF-photometry catalogs (bright stars, fit
quality `qfit < 0.15`, S/N > 20, 50–5000 stars per catalog) were compared
pairwise: for every pair of overlapping catalogs, stars were matched (2-D
offset-histogram peak for the rigid estimate, then tight mutual matching)
and the median matched-star offset recorded. Each pair gives one observation

    Δ_ij = (att_j − att_i) + (S_dj − S_di)

where `att_i` is the attitude (pointing) error of exposure *i* — common to
all ten detectors of the exposure, because NIRCam exposes them
simultaneously — and `S_d` is the per-detector placement error. A weighted
least-squares solve over all pairs separates the two (gauges: mean attitude
= 0; mean detector shift = 0 per band). The solve is **invariant to any
rigid per-exposure WCS manipulation**, so alignment history cannot
contaminate it, and it uses no external reference catalog.

Solves performed: 2221 F212N / F187N / F182M separately (192 catalogs,
~3600–4200 usable pairs each), 2221 F405N+F410M jointly (96 catalogs), 1182
F115W+F200W jointly (384 catalogs, both visits), and a direct cross-proposal
2221×1182 joint solve (13,151 pairs). Post-fit pair residuals: 0.6–0.75 mas
rms (2221 solves), MAD 1.4–1.7 mas (solves involving the 1182 wide bands).

The fitted per-detector shifts were then decomposed against the detector
center positions into a global rotation + a global scale + per-detector
residuals.

## 3. Results

**(a) The fitted inter-detector scale equals `1 − VA_SCALE`, per program:**

| Dataset | fitted scale | predicted `1 − VA_SCALE` |
|---|---|---|
| 2221 F212N | +9.9e-5 | 9.18e-5 |
| 2221 F187N | +9.8e-5 | 9.18e-5 |
| 2221 F182M | +9.7e-5 | 9.18e-5 |
| 1182 F115W | +1.06e-4 | ~1.00e-4 |
| 1182 F200W | +1.05e-4 | ~1.00e-4 |

Fit uncertainty ≈ 1.5e-5. Two independent programs, five filters: the scale
tracks each program's own `VA_SCALE`. Raw per-detector shifts reach ±13–15
mas (Dec, anti-symmetric between modules A and B) — the full solved tables
are in the repository (`analysis/solution_*.npz` in the companion astrometry
paper; per-band tables reproduced in the pipeline PR).

**(b) The LW module split shows the same signature:** F405N/F410M
(NRCALONG/NRCBLONG, ~±90″ levers): A−B ≈ 18.5 mas in Dec (±9.2 per module),
with F405N and F410M internally consistent to 0.5 mas.

**(c) After removing scale + per-visit roll (0.1–5″), the residual
per-detector placements are 1–2.5 mas and static:** they repeat across all
five datasets (different filters, epochs, visits, guide stars, proposals)
with an rms-about-the-mean of 0.2–0.9 mas per detector. In other words, the
underlying SIAF/CRDS distortion solution is excellent — sub-mas
intra-detector residuals, 1–2.5 mas static detector placements — and the
dominant inter-detector inconsistency in delivered products is the DVA
term.

**(d) Filter offsets are applied correctly:** per-detector shift tables
agree across filters to ≤2 mas, including F200W (whose filteroffset is
~170 mas — applied correctly to ~1% of itself).

## 4. Why the term survives to Level-3 products

- `tweakreg` aligns all detectors of an exposure as one **rigid group**:
  `group_id` (`jwst/datamodels/utils/group_id.py` L20–41) is built from
  program/observation/visit/visitgroup/sequence/activity/**exposure** with no
  detector component, and the documentation states the underlying
  assumption explicitly: "it is assumed that the relative positions of
  (e.g., NIRCam) detectors do not change."
- The operational CRDS `pars-tweakregstep` files for NIRCam set
  `fitgeometry='shift'`, so even the group-level fit has no scale degree of
  freedom (and `abs_refcat` is unset by default).

Hence standard Level-3 mosaics inherit the ±9–13 mas per-detector term.
High-precision astrometry pipelines that fit ≥6-parameter linear transforms
per exposure (e.g. the hst1pass/jwst1pass lineage and the
LMC/globular-cluster calibration works) absorb the term silently in their
per-exposure scale parameters, which is why it has not been widely reported
despite being present in all NIRCam imaging.

**Epoch dependence (the science hazard):** `VA_SCALE − 1` varies with the
angle between the pointing and the barycentric velocity apex, swinging by
~±1e-4 over the year. Uncorrected, the apparent NIRCam module separation
changes by up to ~25 mas between epochs. Any multi-epoch measurement that
trusts the delivered WCS detector geometry (proper motions, parallax,
astrometric binaries) inherits an epoch-dependent, field-position-dependent
systematic of this size.

## 5. Proposed correction

The missing term is deterministic and available from the headers of each
file. Two equivalent remedies:

1. **Upstream (preferred):** apply the DVA scale about a single
   exposure-common point rather than each aperture's own reference — e.g.
   pass the guider or V1 reference into `dva_corr_model` (the function
   already supports it: the `v2_ref=v3_ref=0` branch is a scale about the
   V1 origin), or equivalently de-aberrate the per-aperture pointing
   keywords in `set_telescope_pointing`. This is the fixed-point question
   already raised in
   [spacetelescope/jwst#9400](https://github.com/spacetelescope/jwst/issues/9400);
   the measurements above supply its empirical magnitude.
2. **Downstream (implemented in this repository, opt-in):** a per-detector
   rigid shift `−(1 − VA_SCALE) × (RA/DEC_REF − RA/DEC_V1)` applied to both
   the GWCS and the FITS SIP header
   (`jwst_gc_pipeline/reduction/dva_correction.py`), idempotent, applied
   before any reference-catalog tie. The common point (V1) differs from the
   guider only by a common rigid shift, which any downstream tie absorbs.
   Validation: re-running the network self-calibration on corrected frames
   must fit an inter-detector scale of ~0.

If the upstream fix lands, the downstream correction must be disabled
(marker keyword `DVACORR` + the network-self-cal scale test guard against
double-correction).

## 6. Reproducibility

- Network self-calibration and decomposition scripts:
  `scripts/analysis/siaf_selfcal/{network_selfcal.py, decompose_selfcal.py}`
  (this repository); solved per-detector tables (`analysis/solution_*.npz`)
  in the Brick astrometry-paper repository.
- Inputs: per-exposure, per-detector DAOPHOT-style PSF catalogs from the
  JWST GC pipeline; any equivalently structured per-detector catalogs
  reproduce the measurement.
- Source-trace line numbers: `jwst` commit `61bd2fe47`.
