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
not flagged by the usual goodness-of-fit statistic.

We therefore replace the black-box iterative fitter with an **explicit sequence
of single-pass fits**. Each pass performs one well-defined detect → fit →
subtract → re-detect cycle, and every intermediate product (catalog, residual
image, background estimate) is written out and inspectable. This makes the
photometry reproducible and auditable, and lets us insert physically motivated
quality controls between passes.

A guiding principle throughout: **the fit is always performed on the individual
calibrated exposures (frames), never on the combined mosaic.** The mosaic is
used only to *detect* sources and to build cleaner detection images between
passes; the flux of every source is always measured on the native, undistorted
frames where the PSF is well defined.

## Point-spread function

We use **theoretical, well-constrained PSF models** (spatially varying across
the detector) rather than empirically derived ones. The crowding and variable
background make empirical PSF construction unreliable here, whereas the JWST
optical model is stable and accurately known. The PSF is strictly positive by
construction, a fact the pipeline enforces on every flux it reports.

## Detection: a progressively cleaned co-add

Source detection uses a standard star-finding algorithm (a DAOFIND-style
matched-filter peak finder) that, by construction, only responds to **positive**
signal. The key idea is that the image we run the finder on gets *progressively
cleaner* from one pass to the next:

1. First the raw mosaic, to find the obvious sources.
2. Then the mosaic with the current source models subtracted (the residual), so
   that fainter stars hidden in the wings of brighter ones — which are not local
   maxima in the raw image — become detectable.
3. Then the residual with the diffuse background also removed, exposing sources
   that were buried in structured emission.

Detections are filtered on a local signal-to-noise estimate and on peak shape
(roundness/sharpness) to suppress diffuse-emission bumps and cosmic-ray-like
artifacts. At each pass the new detections are **unioned with the previous
pass's vetted catalog**, so the source list grows monotonically and earlier,
trusted sources are never lost.

## Per-pass fitting procedure

For each pass and each frame the following steps are applied:

1. **Saturated stars** are fit and subtracted first with a dedicated
   saturated-star treatment, so their broad wings do not contaminate the fits of
   neighboring faint sources. (They are reinserted with their saturated-star flux
   for the final catalog; they are not re-measured as ordinary point sources.)

2. **Joint fitting of blends.** Sources closer together than a few times the PSF
   FWHM are fit *simultaneously* as a group rather than independently. Fitting
   close pairs independently causes each to over-subtract in the valley between
   them, driving the fainter source artificially negative; joint fitting removes
   this bias.

3. **Single-pass PSF fit.** Each (group of) source(s) is fit with the
   spatially-varying PSF. The flux is constrained to be non-negative.

4. **Overshoot check and forced refit.** Even with a single pass, a
   free-position fit can occasionally walk a centroid off its star and inflate
   the flux. We detect this physically: we compare each fitted source's rendered
   model **peak** to the local **data** peak, and flag any source whose model
   peak exceeds the data peak by more than a set margin (impossible for a true
   single positive PSF). Each flagged source is then **re-measured as forced
   photometry**: its position is pinned at the trusted detection (seed) position
   — *not* the drifted fit position, since the drift is exactly the failure being
   corrected — and only its flux is solved. With the position fixed this flux has
   an exact, single-step closed-form solution, so there is no opportunity for the
   centroid to wander again. This forced flux is likewise constrained to be
   non-negative.

5. **Positivity enforcement.** Because the PSF is strictly positive, a
   non-positive fitted flux is unphysical (a "negative-peak" model that *adds*
   light to the residual instead of removing a star). Such fits are discarded
   outright; crucially they are also excluded from the seed list passed to the
   next pass, so a transient over-subtraction cannot breed a growing population
   of spurious negative sources across iterations.

6. **Deduplication and artifact rejection.** Near-duplicate detections are
   merged (keeping the better-fit instance), and sources lying on saturated-star
   diffraction features are rejected.

## Background estimation

After a pass, the per-frame source models are subtracted and the frames are
recombined into a residual mosaic. From this residual we build a smoothed
**diffuse-background** map, having first masked the locations of fitted sources
so their cores do not bias the background low. This background is subtracted from
the frames before the next fitting pass, so that faint-source fluxes are measured
against the true local sky rather than against structured nebular emission. The
background is recomputed from the cleanest available residual at each stage, and
the change between successive background estimates is logged so that a runaway
(in which the background begins absorbing real extended emission) is visible.

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
the fit with the union of the per-filter source lists — deduplicated so a star
seen in several bands is seeded once — so that a source detected confidently in
one band is measured in the others even where it is individually marginal. (A
stricter cross-band requirement — detection in two or more bands within a tight
positional tolerance — is planned but not yet the default.)

The per-band catalogs are then cross-matched into a single multi-filter table.
A closing **forced cross-band fill** step revisits that table: wherever a source
is a non-saturated non-detection in a given band, its flux in that band is
re-measured by forced photometry at the matched position (the position is fixed;
only the flux is solved), so a cross-band non-detection carries a measured value
or a genuine per-source noise limit rather than an empty cell. This fill is
recorded in a separate catalog and never alters the independently detected
measurements.

## Outputs and quality controls

The pipeline emits, for each pass: the merged source catalog (carrying sky
coordinates, fluxes, fit-quality metrics, and the detection-provenance label),
the residual mosaic (point sources removed), a model mosaic (for display, with
the saturated stars added back so it overlays against the data), and the diffuse-
background map. Because every pass is explicit and its products are saved, the
photometry can be inspected stage by stage.

The principal quality safeguards are: enforcement of non-negative flux
everywhere (the PSF is positive); the model-vs-data peak overshoot test that
catches centroid-walk inflation a goodness-of-fit statistic would miss;
source-masked background estimation to avoid the background-eats-source feedback
loop; and joint fitting of close blends to avoid mutual over-subtraction.

## Known limitations

- The passes are inherently ordered (each detects on the previous pass's
  recombined residual image), so throughput is ultimately limited by the serial
  recombination (resampling) between passes. The per-exposure fits *within* a
  pass are embarrassingly parallel and can be distributed across many machines;
  only the between-pass recombination is a barrier.
- The cross-band seed is a deduplicated union rather than a strict multi-band
  coincidence requirement.
- Faint sources blended into the wings of much brighter or saturated stars are
  detected but may still be lost at the fitting/deduplication stage. The
  forced cross-band fill recovers such sources where they were detected in at
  least one other band; sources lost in every band still require a dedicated
  fit-side deblending stage (forced photometry at detected positions), which is
  not yet applied.
