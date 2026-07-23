# astrometry_gdc — Jay Anderson STDGDC distortion corrections

Starlist-level integration of Jay Anderson's JWST1PASS "STDGDC" NIRCam
geometric-distortion-correction maps
(<https://www.stsci.edu/~jayander/JWST1PASS/LIB/GDCs/STDGDCs/NIRCam/>),
mirrored at `/orange/adamginsburg/jwst/distortion/jayander_stdgdc/NIRCam`
(override with `$JWST_GDC_ROOT`; a root with or without the trailing `NIRCam`
level is accepted).

Nothing in the main pipeline imports this package — it is **opt-in**
(`python -m jwst_gc_pipeline.astrometry_gdc.correct_catalog ...`).

## File format (verified on NRCB1/F212N and NRCBL/F277W)

| HDU | name | shape | BSCALE | content |
|-----|------|-------|--------|---------|
| 0 | — | — | — | metadata: `XGC_0`/`YGC_0` (corrected-frame coordinate of the first reverse-map pixel; `LTV = 1 − *GC_0`), `NDIM_*` |
| 1 | XGC | 2048×2048 | 1e−5 | forward X map: `XGC[j, i]` = corrected **1-based** X of raw 1-based pixel (X=i+1, Y=j+1) |
| 2 | YGC | 2048×2048 | 1e−5 | forward Y map, same convention |
| 3 | MGC | 2048×2048 | 1e−4 | pixel-area correction in **magnitudes**, added to the instrumental mag |
| 4 | XCG | e.g. 2056×2057 | 1e−5 | reverse X map on the corrected-frame grid, `LTV1`/`LTV2` offsets |
| 5 | YCG | 〃 | 1e−5 | reverse Y map |

HDU1/2/3 COMMENT cards embed 5×5 spot-value tables ("X-FORWARD DISTORTION
MAPPING" etc. at X,Y ∈ {1, 512, 1024, 1536, 2048}); these are hard-coded as
regression fixtures in `tests/test_stdgdc.py` and pin the `[y, x]`
orientation, the 1-based value convention, and the index origin.
Reverse-map sampling: corrected 1-based (xc, yc) → 0-based array index
`[yc + LTV2 − 1, xc + LTV1 − 1]`; measured forward→reverse round-trip
~5e−5 pix (BSCALE-quantisation limited).

## Application convention (from peppar)

Extracted from Matt Hosek's peppar
(`/blue/adamginsburg/adamginsburg/repos/peppar`,
`peppar/combo_starlists.py::apply_distortion_to_starlists`, lines 876–1043);
full citations in the `stdgdc.py` module docstring. Summary:

- maps sampled `[y, x]` with `RegularGridInterpolator` on 0-based
  `np.arange` grids, evaluated at the **raw photutils (0-based)** positions
  directly (lines 926–935, 964–966) — which is the correct index because map
  index i ↔ raw 1-based pixel i+1;
- peppar interpolates cubic (line 877); we default to bilinear (difference
  measured < 1e−3 pix on these smooth maps; both exact at grid nodes);
- corrected values replace `x`/`y` (originals kept as `x_raw`/`y_raw`;
  lines 970–974). peppar keeps the returned 1-based-convention values as-is;
  **this package subtracts 1** so input and output share the 0-based
  convention;
- the MGC magnitude correction **is** applied by peppar
  (`starlist['m'] += mag_offset`, lines 966, 975); available here as
  `STDGDC.pixel_area_mag`;
- the reverse maps are unused (commented out) in peppar with a "don't
  actually reverse" note (lines 937–947) — with the LTV offsets applied they
  do reverse correctly (~5e−5 pix), so `STDGDC.reverse` is provided;
- file resolution (lines 896–919): `SWC/<FILT>/` per SW filter with a
  preferred `v2/` subdir when present; LW has **only F277W**
  (`NRCA5/NRCALONG → NRCAL`, `NRCB5/NRCBLONG → NRCBL`). The library mixes
  `STDGDC_<DET>_<FILT>` and `STDGDC_<FILT>_<DET>` naming, so resolution
  globs on the detector token.

## Coverage

SW solutions exist for: F070W, F090W, F115W, F140M, F150W, F182M, F200W,
F210M, F212N (v2 variants for F182M/F200W/F212N). LW: **F277W only**
(NRCAL/NRCBL).

GC-program bands:

| band | STDGDC? |
|------|---------|
| F182M | yes (v1+v2) |
| F212N | yes (v1+v2) |
| F187N | **no** — optional loud-warning fallback to F212N (`fallback_filter=`, default OFF) |
| F323N, F360M, F405N, F410M, F444W, F466N, F470N, F480M | **no** — only the F277W solution exists for LW; treat any LW use as a proxy experiment |

## Integration design

This is an **affine-anchored starlist correction, NOT a CRDS reference-file
swap**: frames' on-disk WCS and drizzling are untouched.

`gdc_wcs.GDCSkySolution` samples the frame's existing WCS (gwcs when
loadable, else the SCI FITS-SIP approximation) on a sparse grid (default
32×32), pushes the same grid through the GDC forward maps, and fits a
6-parameter affine mapping GDC-corrected pixels → tangent-plane offsets of
the original sky positions. The least-squares affine (with intercept)
preserves the frame's mean position/scale/rotation by construction — the
frame's pointing stays owned by the VIRAC2-tied offsets machinery — and only
the higher-order distortion field changes. The fit residual field is
returned as the CRDS-vs-GDC distortion delta map (`delta_map()`,
diagnostics).

Measured on a real brick frame (`jw02221001001_05101_00001_nrcb1_cal.fits`,
F212N, v1): CRDS-vs-GDC delta field mean 0.60 mas, max 3.8 mas, affine rms
0.75 mas; v1-vs-v2 solutions differ by 0.2–0.4 mas.

CLI:

```
python -m jwst_gc_pipeline.astrometry_gdc.correct_catalog \
    --catalog <m1 per-exposure fits> --cal <cal/crf frame> [--inplace-cols] \
    [--gdc-version auto|v1|v2] [--fallback-filter F212N]
```

adds `skycoord_gdc_ra`/`skycoord_gdc_dec` (deg) + provenance meta
(`GDCFILE`, `GDCVERS`, `GDCAFFX/Y`, `GDCRMS`, ...); it never overwrites
existing skycoord columns.

## Experiment results (2026-07-23)

**Run — see `GDC_EXPERIMENT_REPORT.md` in this directory.**  Outcome: no
measurable improvement on any relative or absolute metric (arches+brick,
F212N+F182M, 480 frames); the Hosek/L2 agreement is slightly degraded; the
brick A/B seam terms are inter-detector affine placement, untouchable by a
distortion swap.  Recommendation: keep this package as an opt-in diagnostic,
do not adopt for production.

## Intended experiment

1. **Brick module-overlap**: build GDC-corrected sky positions for the m1
   per-exposure catalogs on both modules and test whether the A↔B
   module-overlap residuals (and the epoch-dependent inter-detector term
   noted in the SIAF self-cal memory) shrink relative to the CRDS solution.
2. **Arches Hosek comparison**: re-run the m2-stage comparison of our F212N
   per-exposure astrometry against Hosek's L2 catalog (benchmark-team
   workspace) with `skycoord_gdc_*` in place of the CRDS positions; Hosek's
   own reduction applies these same STDGDC maps via peppar, so agreement
   should isolate distortion-model differences from PSF-fit differences.

Both experiments compare catalogs star-by-star through existing matched-pair
machinery — no NN-median-vs-dense-reference measurement anywhere (see
ASTROMETRY RULE #1 in the repo CLAUDE.md).
