# CLAUDE.md — jwst-gc-pipeline

Instructions for any agent (or human) working in this repo. **Read this before
touching astrometry, alignment, reference catalogs, or reduction/cataloging.**

When editing jwst-gc-pipeline, use git worktrees.

---

## ⛔ ASTROMETRY RULE #1 — never nearest-neighbour-median against a dense catalog

**Do NOT compute or validate an astrometric offset as the MEDIAN (or mean) of
nearest-neighbour matches (`match_to_catalog_sky` / `search_around_sky` +
`np.median`/`np.mean`) against a DENSE reference (VIRAC2 / VVV / GNS; median NN
spacing ≲ 3").**

Why: when the true shift exceeds the reference's NN spacing (~0.3"), NN pairs the
WRONG star and the median **collapses toward ~0** (or a spurious value). The method
**fabricates false agreement** and has repeatedly fooled *validation* of the GC
fields (a NN-median check "confirms 0.00 fine" on a frame that is really off) — a
recurring failure mode behind the 2221/1182 astrometry errors. (The specific
brick-1182 v001 ~20" error was an offsets-table CURATION collapse — v001 overwritten
with v002's value — not a NN-median measurement; but it is the same class of silent
false-agreement failure, so NN-median against a dense catalog is banned outright.)

This is now **enforced in code** — `measure_offsets.assert_sparse_reference_for_nn_median`
raises `DenseNNMedianAstrometryError` — and by a **grep-guard test**
(`tests/test_no_adhoc_nn_median_astrometry.py`) that fails CI if a NEW file pairs a
NN match with a median/mean. Do not disable either.

### The ONLY sanctioned ways to measure an astrometric offset

1. **Offset-HISTOGRAM stacking** — histogram ALL pairwise offsets within a window,
   take the peak. Density-immune to NN-collapse; correct no matter how large the
   shift. Use `jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset(a, b)`
   (the public, guarded helper) or `scripts/reduction/astrometry_audit.py::xcorr`.
   Use it for DETECTION and for LARGE offsets.
2. **A SPARSE reference** — the Gaia-only subset (`source == b'GaiaDR3'`, medNN
   ~5.7"), never the full dense catalog.

### ⚠ Histogram-stacking is density-immune to NN-collapse but NOT to a dense-reference bias

The offset-histogram peak is unbiased only when the NON-matching pair background is
uniform. Against a **DENSE** reference (full VIRAC2), two catalogs tracing the SAME
clustered stellar field make a correlated, non-uniform wrong-pair background that
**pulls the peak by several mas** — a *different* failure from NN-collapse but the
same result: density fools the estimator. Measured on brick (2026-07-16): histogram
vs dense VIRAC2 read **~9–10 mas with a ±6.5 mas RA term whose SIGN FLIPS between
filters** (an artifact — a real field offset cannot flip sign per filter), while the
**same-star** tie was ~0 in RA / ~−5 mas in Dec. Against SPARSE Gaia the histogram
was clean (matched same-star). The "common-mode ~(±6.5) mas" once blamed on
assign_wcs drift was largely this artifact. (Memory: `histogram-vs-samestar-offset-bias`.)

**So for the PRECISE bulk value against a dense reference:** do NOT use the histogram
peak. Use it only to DETECT the tie is small, then refine **same-star** — a matched-pair
residual via `local_residual_map` (single giant cell), which REFUSES unless a verified
small global tie already exists (nearest pair = the RIGHT star; the sanctioned path,
NOT ad-hoc dense NN-median). `measure_reference_tie` now does this automatically and
reports `bulk_source` (`"same-star"` / `"histogram"`); cycle-N table corrections must
come from the same-star bulk, never the raw histogram against dense VIRAC.

Do **not** hand-roll `match_to_catalog_sky(...).median()` in an ad-hoc script.
Import `measure_offset` (detection/large) or use `local_residual_map` after a verified
tie (precise bulk); never a raw NN-median against the dense catalog.

### ⛔ GC RULE — Gaia is the FRAME, never the reference CATALOG; it must not BLOCK

In the Galactic Center, **Gaia DR3 defines the absolute FRAME (ICRS)** but the Gaia
**catalog is NEVER the reference catalog** — it is far too sparse (Brick footprint:
~1.8k Gaia vs ~113k VIRAC2). **VIRAC2 is the correct GC reference catalog**, measured
with the density-immune offset-histogram tie above. A JWST→Gaia-sparse tie may be
used as a *diagnostic cross-check* but **must never BLOCK** a coherent VIRAC tie, a
correction, or a release gate.

Why: a direct Gaia↔VIRAC2 crossmatch over the whole Brick footprint (same physical
stars, no JWST, epoch 2022.70) shows the two frames **agree to ~2.3 mas** globally,
with only ~5–10 mas spatially-varying local wander and ~40 mas per-star VIRAC
precision. So a JWST→VIRAC vs JWST→Gaia "disagreement" at the ~5–10 mas level is a
**JWST-side** population/crowding effect (few bright Gaia stars → noisy sparse peak),
NOT a catalog conflict. The sparse-Gaia cross-check retains ONLY a GROSS gate
(`REFERENCE_CROSSCHECK_GROSS_MAS`, ~100 mas) to catch a spurious/window-limited
VIRAC peak (the brick-1182 v001 ~700 mas tell) — never a fine ~5–10 mas gate.
(Memory: `gc-gaia-frame-not-catalog`. NOTE this refines item 2 above: Gaia-sparse is
a legitimate *measurement* cross-check, but is NOT a blocker and is NOT the catalog.)

### A bulk offset ≈ 0 does NOT mean "clean"

A field-average / whole-mosaic offset can read ~0 while HALF the mosaic is offset
(brick-1182: visit-001 exposures tile the top half, shifted ~20" from visit-002;
bulk peak washed it out). **Always map the offset PER TILE** (`measure_offset_grid`,
`registration_failsafes.py`) and report per-tile peak-contrast: ≳5 = real tie, ~1 =
broken. A single global number is never sufficient sign-off.

### A LOW contrast can mean "offset ≫ window", NOT "no tie"

A large rigid offset has ZERO true pairs inside a narrow search window, so the peak
is noise (low contrast, and different against a dense vs sparse reference). That is
exactly how brick-1182 v001's ~20" offset first read as ~2"/incoherent. So:
- `measure_offset` now **sweeps the window** (3→10→30→60") by default and takes the
  highest-contrast peak — do not disable `sweep`. A returned `swept=True` /
  `window_arcsec ≫` your expected offset means the frame is grossly shifted.
- On a weak tie, cross-check TWO references (`agree_across_references`, VIRAC vs
  Gaia-only): a real tie agrees; a spurious peak moves.
- **A swept peak near the window EDGE is geometry, not a tie** (issue #158). Two
  adjacent, non-overlapping footprints (two NIRCam modules, two mosaic tiles) have
  a pair-density RIDGE at the lag that slides one onto the other; the search window
  truncates it and the histogram's arg-max lands on the cut. Such a "peak" is sharp,
  clears the contrast floor, and MOVES with the window (W51: off = 54.8/59.8/64.7/
  67.2/75.9/89.0/99.5" at windows 55/60/66/70/80/90/100"). A real tie reads the
  SAME offset at every window that can contain it. Check `window_edge_fraction`
  (off/window; ~1 is the tell) and pass `confirm_windows=True` for any tie you
  expect to be small — it re-measures at an independent window and rejects a peak
  that does not reproduce. One accepted alias CASCADES: the displaced exposure
  enters the consensus union, and every later exposure then ties to it at
  contrast 200, so the guard must reject the FIRST marginal tie.

### Correcting an already-aligned frame after the offsets table changes

`fix_alignment` skips a frame that already has a `RAOFFSET` header (idempotent). If
you CORRECT the offsets table, the stale frame is silently kept (this is how v001
stayed ~20" off). The disagreement guard warns when a frame's baked `RAOFFSET`
differs from the current table value; set `FORCE_REALIGN_ON_DISAGREE=1` to hard-stop.
The fix is to REGENERATE the working copy from `_cal` (destreak overwrite → RAOFFSET
resets → current table applied), never to re-apply on top of the stale shift.

### Stage astrometry checkpoints (in-pipeline failsafe ladder)

Cataloging now re-verifies the astrometry at EVERY merge stage
(`jwst_gc_pipeline/photometry/ASTROMETRY_CHECKPOINTS.md`): at m2 every exposure
is re-measured against its visit consensus (2 mas tol) and the consensus is
tied to VIRAC2/Gaia with multiple independent checks — a real misalignment
CORRECTS the offsets table (with provenance), stale-tags the im0 mosaics
(`*_im0_badastrom.fits`), and STOPS the run for regeneration; at m3–m6 the
solution is FROZEN and any measured shift raises; at the m7 cross-band merge
every filter must agree with the VIRAC2-Ks-nearest anchor to <5 mas with no
significant 2″ cell >15 mas.  Do not disable (`ASTROM_CHECKPOINT=0`) or
override (`ALLOW_LATE_STAGE_ASTROM_SHIFT`, `ALLOW_CROSSFILTER_ASTROM_FAIL`)
without written justification.

### ⛔ ASTROMETRY RULE #2 — read the GWCS; the SIP header is only an approximation

**Do NOT build an `astropy.wcs.WCS` from a detector-frame SCI header for anything
astrometric.** Use:

```python
from jwst_gc_pipeline.frame_wcs import frame_wcs
ww = frame_wcs(filename_or_hdulist)      # GWCS-backed; SIP fallback + warning
```

Every JWST detector-frame product carries **two** WCSes: the **GWCS** in the ASDF
extension (`model.meta.wcs`) — the authoritative full distortion chain — and a
FITS `RA---TAN-SIP` header, which is a *fitted low-order approximation* of it.
SIP cannot represent the JWST distortion; it can only fit it, and both directions
of that fit carry error:

- **the fit residual**: 5–8 mas on every frame written before 2026-07-29,
  *position-dependent and different per detector and per filter*, so no bulk tie
  removes it. Cause: `gwcs.WCS.to_fits()` defaults to `max_pix_error=0.25` **px**
  (STScI uses 0.01), and the reduction re-stamped headers with a bare
  `header.update(ww.to_fits()[0])`. In pixels that is up to ~165 mpix — the same
  error, not a second one: SIP's own forward→inverse round trip closes to 0.000 mpix.
- **off-footprint**: the iterative SIP inverse either **raises `NoConvergence`**
  (the W51 m8 abort) or, with `quiet=True`, returns **finite garbage with no
  warning** — which lands in a catalog instead of stopping the run.

The GWCS has neither problem (inverse exact to <1 mpix; off-footprint → `NaN` on
every call path) and is not slower overall (forward ~1.3× slower, inverse ~1.1× faster).

- Enforced by the grep-guard test `jwst_gc_pipeline/tests/test_no_sip_frame_astrometry.py`.
- Any FITS WCS you *write* goes through `reduction.fits_wcs_sync.sync_header_to_gwcs`,
  which fits at 0.01 px and **verifies** the result against the GWCS.
- Audit on-disk products: `python scripts/release/audit_fits_gwcs_agreement.py --field <f>`.
- `i2d` mosaics are exempt — `resample` writes a rectified plain `RA---TAN` grid
  with no SIP, so `WCS(i2d_header)` is exact.
- `astropy.wcs.WCS(header)` on a frame must always pass `relax=True`: a header whose
  CTYPE lost the `-SIP` suffix still carries `A_*`/`B_*`, and without `relax` the
  distortion is silently dropped (~0.1" at the detector corners).

### Reading list before any astrometry change
- `jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md` — the full flow,
  the two authoring points, no-double-correction rule, epochs, module-lock policy.
- `jwst_gc_pipeline/photometry/ASTROMETRY_CHECKPOINTS.md` — the stage checkpoint
  ladder (visit consensus, frozen stages, cross-filter gate).
- The `brick-1182-*` and `dense-nn-median-guard-enforced` memory notes.

---

## Release gate

Full blocking checklist: **`scripts/release/RELEASE_DEPLOYMENT_CHECKLIST.md`**.

`scripts/release/stage_release.py` runs `registration_failsafes.py --scan` (per-tile,
cross-band + own-catalog) and **REFUSES to stage** any field with a locally
misregistered band. Do not stage around it. The `--allow-registration-fail`
override additionally requires `ALLOW_REGISTRATION_FAIL=1` in the environment — it
exists only for deliberate, justified overrides, not for making a red gate go green.

**⛔ Inter-frame overlap check is BLOCKING and not fully covered by the stock gate.**
The #1 recurring corruption: two overlapping observations/visits/pointings sit
>1 pixel (usually >1″) apart, so the overlap region loses all its stars (or doubles
them) while the bulk offset still reads ~0. You MUST verify that wherever different
frames overlap, their stars match `< 30 mas` — **per pair, PER TILE, reference-free
(JWST-internal, frame-vs-frame)**, with the **swept** estimator.

Two structural blind spots to respect:
- `registration_failsafes.py` matches the mosaic vs its **own merged catalog** —
  both derive from the same `_crf`, so a per-visit residual is self-referential and
  **cancels** (both wrong the same way → agree → PASS). It also searches only ±2.5″
  with no sweep, so it cannot see a >2.5″ overlap offset. A green
  `registration_failsafes` is therefore NOT sufficient.
- A **field-pooled** offset (one number per visit/mosaic) averages a spatially
  varying residual away. The brick-1182 F200W seam (2026-07-12) was a ~90 mas
  visit-001 residual confined to the y=0.5 strip that doubled every star there; the
  whole-field peak read ~50 mas and a 4×4 grid diluted the strip. Map it PER TILE
  (≥12×12) and gate on offset **magnitude**, not just contrast.

The real gate: **`scripts/release/check_interframe_overlap.py --field <f> --scan`**
(reference-free, per-tile, swept; wired BLOCKING into `stage_release.py`). It uses
`jwst_gc_pipeline.photometry.interframe_overlap` (`assert_overlaps_registered`,
`overlap_offset_grid`) and the offset-magnitude gate in
`astrometry_offsets.measure_offset_grid(..., max_off_mas=…)`. See checklist item 0.

---

## Workflow

All pipeline work goes on **worktree branches** (`../jwst-gc-pipeline-<slug>`) pushed
as **pull requests**, one concern per PR. Never accumulate uncommitted changes on the
active `main` working tree (it is the live reduction environment).

## Other conventions
- SLURM: use `--account=astronomy-dept --qos=astronomy-dept-b`. The default
  `adamginsburg` QOS caps cpu=10 and will hang a 16-cpu task.
- New photometry code goes in new modules, not the `crowdsource_catalogs_long.py`
  monolith.
- No bare `try/except`; catch specific exceptions only.

## SLURM job naming (standing rule)

Every submitted job name MUST identify **target + program** (+ **obsid** whenever
the program has multiple observations), plus the stage and filter where
applicable: `<target><program>-o<obsid>-<stage>[-FILTER]`, e.g.
`brick2221-o001-reduce-F182M`, `cloudc2221-o002-cat`, `m4-1979-o002-reduce`.
Pass it at **submit time** (`sbatch --job-name=...`) — the in-script runtime
rename only fires when the job STARTS, and quota-bound jobs sit PENDING for
hours under the generic name, which is exactly when the queue is being watched.
Never leave `reduce`/`catalog` as the visible name. Multiple reduce/catalog
jobs are almost always in flight simultaneously.
