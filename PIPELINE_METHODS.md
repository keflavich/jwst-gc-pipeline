<!--
  DOC SYNC — last reviewed 2026-06-10, repo commit 17d0861 (+ uncommitted fixes).
  This is the SCIENCE-NARRATIVE doc: it describes the photometry method the way a
  journal "Observations & Photometry" section would, in terms of steps and their
  intent, with no code, file names, or flag names. Its implementation companion is
  PHOTOMETRY_PIPELINE.md (flags, tokens, outputs, function names). KEEP THE TWO IN
  SYNC: when the algorithm changes, update BOTH and bump the stamp in BOTH. This
  file = "what & why"; PHOTOMETRY_PIPELINE.md = "how & where".
-->

# PSF photometry method

This document describes, at the level of a publication methods section, how the
pipeline measures point-source photometry in crowded JWST/NIRCam fields toward
the Galactic Center. It is deliberately code-free; for the concrete commands,
output products, and tunable parameters see `PHOTOMETRY_PIPELINE.md`.

## Motivation and overall strategy

The Galactic Center fields are extremely crowded and sit on bright, structured
diffuse emission, with a wide dynamic range that includes saturated stars. In
this regime an automatic iterative PSF fitter that simultaneously re-detects
sources and refines their positions is numerically fragile: for isolated bright
stars the free-position fit can let a centroid wander off the true peak and
settle at a spuriously inflated flux, producing a model whose central surface
brightness exceeds that of the data — impossible for a single positive PSF, and
something the usual goodness-of-fit statistic passes.

We therefore replace the black-box iterative fitter with an **explicit sequence
of single-pass fits**. Each pass performs one well-defined detect → fit →
subtract → re-detect cycle, and every intermediate product (catalog, residual
image, background estimate) is written out and inspectable. This makes the
photometry reproducible and auditable, and lets us insert physically motivated
quality controls between passes.

A guiding principle throughout: **the fit is always performed on the individual
calibrated exposures (frames), never on the combined mosaic.** The mosaic's role is to *detect* sources and to
build cleaner detection images between passes; the flux of every source is
measured on the native detector frames, where the PSF is well defined. Those
frames are the **distorted** ones — they carry a 3rd-order SIP approximation plus
the authoritative GWCS, while the mosaic is the rectified product; the point is
that fitting happens before resampling, rather than that the frames are
undistorted. One qualification: from m5 onward the
mosaic-derived smoothed background is reprojected onto each frame and subtracted
*before* fitting (see §Background), so the mosaic does enter the reported fluxes
by that route.

## Point-spread function

We use **theoretical, well-constrained PSF models**, spatially varying across the
detector. The crowding and variable background make empirical PSF construction
unreliable here, whereas the JWST optical model is stable and accurately known. The PSF is strictly positive by
construction, a fact the pipeline enforces on every flux it reports.

## Detection: a progressively cleaned co-add

Source detection uses a standard star-finding algorithm (a DAOFIND-style
matched-filter peak finder) that, by construction, only responds to **positive**
signal. The key idea is that the image we run the finder on gets *progressively
cleaner* from one pass to the next:

1. First the raw mosaic, to find the obvious sources.
2. Then the mosaic with the current source models subtracted (the residual), so
   that fainter stars hidden in the wings of brighter ones become local maxima,
   and so become detectable.
3. Then the residual with the diffuse background also removed, exposing sources
   that were buried in structured emission.

Detections are filtered on a local signal-to-noise estimate and on peak shape
(roundness/sharpness) to suppress diffuse-emission bumps and cosmic-ray-like
artifacts. At each pass the new detections are **unioned with the previous pass's
vetted catalog**. The seed list usually grows, and it can also shrink: the seed is
the *vetted* subset, so a source the vetting stage rejects (or a non-positive fit)
drops out of the next pass's seed. Its model is also left out of the vetted
residual mosaic, so it shows up there as an unsubtracted source.

## Per-pass fitting procedure

For each pass and each frame the following steps are applied:

1. **Saturated stars** are fit and subtracted first with a dedicated
   saturated-star treatment, so neighboring faint sources are fit on clean sky
   rather than on those broad wings. (They are reinserted with their saturated-star flux
   for the final catalog.)

2. **Joint fitting of blends — available, OFF by default.** Sources closer
   together than a few times the PSF FWHM *can* be fit simultaneously as a group
   rather than independently (`--group`, with `--manual-group-min-sep-fwhm`,
   default 2.0). Fitting close pairs independently causes each to over-subtract in
   the valley between them, driving the fainter source artificially negative;
   joint fitting removes that bias at a cost that rises steeply with group size.
   **`--group` defaults to off and only one submitter passes it**
   (`submit_cataloging_m8_partial.sbatch`), so the shipped catalogs are fit
   source-by-source. The other submitters pass `--max-group-size` alone; it takes
   effect only alongside `--group`.

3. **Single-pass PSF fit.** Each (group of) source(s) is fit with the
   spatially-varying PSF. The flux is constrained to be non-negative.

4. **Overshoot check and forced refit.** Even with a single pass, a
   free-position fit can occasionally walk a centroid off its star and inflate
   the flux. We detect this physically: we compare each fitted source's rendered
   model **peak** to the local **data** peak, and flag any source whose model
   peak exceeds the data peak by more than a set margin (impossible for a true
   single positive PSF). Each flagged source is then **re-measured as forced
   photometry**: its position is pinned at the trusted detection (seed) position,
   because the drifted fit position is the failure being corrected, and only its
   flux is solved. With the position fixed this flux has an exact, single-step
   closed-form solution, so the centroid stays where it was pinned. This forced
   flux is likewise constrained to be non-negative.

5. **Positivity enforcement.** Because the PSF is strictly positive, a
   non-positive fitted flux is unphysical (a "negative-peak" model that *adds*
   light to the residual instead of removing a star). Such fits are discarded
   outright, and are also excluded from the seed list passed to the next pass, so
   a transient over-subtraction stops at the pass that produced it.

6. **Deduplication and artifact rejection.** Near-duplicate detections are
   merged (keeping the better-fit instance), and sources lying on saturated-star
   diffraction features are rejected.

## Background estimation

After a pass, the per-frame source models are subtracted and the frames are
recombined into a residual mosaic. From this residual we build a smoothed
**diffuse-background** map, having first masked source positions so that bright
stars stay out of the "background". The mask is the **union of the vetted
catalog and the independent i2d detection seed**. Masking only the fitted list is
self-reinforcing: a star dropped by vetting is left unmasked, is absorbed into the
background, is then subtracted away, and is lost for good (ngc6334 F405N: the
reconstructed background reached 164 at a 298-peak star). This background is
subtracted from
the frames before the next fitting pass, so that faint-source fluxes are measured
against the true local sky rather than against structured nebular emission. The
background is recomputed from the cleanest available residual at each stage. The
guard against a runaway (the background absorbing real extended emission) is the
union mask above. The per-stage `[bg]` line reports the output name, the number
of masked sources and the mask radius — those three items only, so a
background-to-background delta is neither computed nor logged.

## Cross-frame merging and vetting

Within a pass, the per-frame catalogs are matched and merged into a single
multi-frame catalog, recording for each source its detection provenance
(specifically, the earliest pass in which it was found). The merged catalog is
then vetted to separate genuine point sources from bumps in extended emission,
using a combination of fit quality, peak surface brightness relative to the local
background, and local signal-to-noise. The vetted catalog defines the source
models used to build that pass's residual and background, and seeds the next
pass.

## Multi-filter cross-band stage

When more than one filter is processed together, a final detect/fit pass seeds
the fit from the per-filter source lists, deduplicated so a star seen in several
bands is seeded once, so that a source detected confidently in one band is measured
in the others even where it is individually marginal. **The seed requires detection
in at least two bands** within a tight positional tolerance
(`--manual-crossband-seed-min-filters`, default **2**, at
`--manual-crossband-seed-max-sep-mas` 30); the permissive single-band union is the
legacy path, reachable with `=1` and not recommended.

The per-band catalogs are then cross-matched into a single multi-filter table.
A closing **forced cross-band fill** step revisits that table: wherever a source
is a non-saturated non-detection in a given band, its flux in that band is
re-measured by forced photometry at the matched position (the position is fixed;
only the flux is solved), so a cross-band non-detection carries a measured value
or a genuine per-source noise limit. This fill is recorded in a separate catalog,
leaving the independently detected measurements as they are.

## Outputs and quality controls

The pipeline emits, for each pass: the merged source catalog (carrying sky
coordinates, fluxes, fit-quality metrics, and the detection-provenance label),
the residual mosaic (point sources removed), a model mosaic (for display, with
the saturated stars added back so it overlays against the data), and the diffuse-
background map. Because every pass is explicit and its products are saved, the
photometry can be inspected stage by stage.

The principal quality safeguards are: rejection of non-positive fitted flux in
the detect/fit passes (the PSF is strictly positive, so `flux_fit ≤ 0` is a
negative-peak model); the model-vs-data peak overshoot test that catches
centroid-walk inflation a goodness-of-fit statistic passes, together with a
final overshoot **drop** for fits whose model peak still exceeds 5× the local data
peak after the refit; and source-masked background estimation to avoid the
background-eats-source feedback loop. Joint fitting of close blends is available
but off by default (§Per-pass fitting).

**One deliberate exception to positivity:** the m8 forced cross-band fill fits with
the *signed* estimator (`nonnegative=False`) and inverse-variance-averages the
result, so a non-detection carries a two-sided noise estimate rather than a hard
zero. Fluxes reported by m8 for non-detections may therefore be negative.

## Known limitations

- The passes are inherently ordered (each detects on the previous pass's
  recombined image — the raw co-add at m3, the residual thereafter; m12 is
  per-frame only), so throughput is ultimately limited by the serial
  recombination (resampling) between passes. The per-exposure fits *within* a
  pass are embarrassingly parallel and can be distributed across many machines;
  only the between-pass recombination is a barrier.
- The cross-band seed requires a ≥2-band coincidence by default, so a source
  genuinely detectable in only one band is seeded there only when
  `--manual-crossband-seed-min-filters=1` is passed.
- Faint sources blended into the wings of much brighter or saturated stars are
  detected but may still be lost at the fitting/deduplication stage. The
  forced cross-band fill recovers such sources where they were detected in at
  least one other band. A fit-side deblending stage exists for saturated cores
  (`--deblend-satstars`, `reduction/satstar_deblend.py`, on for gc2211) but is off
  by default; sources lost in every band around *unsaturated* bright stars stay
  lost.
