# Refactor Plan — jwst-gc-pipeline

Generated 2026-06-17, executing in worktree `refactor-bloat` (branch `worktree-refactor-bloat`,
based on origin/main @ 318b902).

Codebase ~23k LOC. Four worst offenders: `crowdsource_catalogs_long.py` (6379),
`merge_catalogs.py` (2196), `saturated_star_finding.py` (2035), `cataloging.py` (2010).

Goals:
1. Big intent-comment blocks → regression tests.
2. Repeated patterns → shared functions.
3. Loops that shouldn't be loops → vectorize.
4. Excessive if/elif → dicts / dispatch tables.
5. Untangle MIRI vs NIRCam paths that hide inside shared functions.
6. Replace bespoke primitives with JWST/STScI library equivalents (esp. WCS).

Sequencing: **deletes → shared-naming module → tests (lock behavior) → STScI-redundancy
swaps → MIRI/NIRCam splits → loop/dispatch cleanups → big-function splits.** Tests go in
*before* structural changes so they're verifiable.

---

## Phase 0 — Free wins: delete dead code  [STATUS: DONE]

- [x] `photometry/crowdsource_catalogs_short.py` — `git rm` (long version has nan_to_num guard).
- [x] `reduction/PipelineRerunNIRCAM-SHORT.py` — `git rm`.
- [x] `reduction/crowdsource_fields.py` (0 bytes) — `git rm`.
- [ ] `reduction/PipelineRerunF212N.py` — HELD: orphaned module but possible science
  special-case; awaiting user decision (delete vs move to scripts/).
- [ ] Root scratch / build artifacts → gitignore (TODO).

---

## Phase 1 — Shared naming/token module  [STATUS: partial — function dedup DONE]
Expand `photometry/naming.py`, route through it.
- [x] `MIRI_FILTERS`, `_instrument_from_filter`, `_inst_token`, `_svo_filter_id` moved to
  naming.py; merge_catalogs + crowdsource_long now import them (removed 3 local copies).
- [x] `_bgsub_token`: one flags-core (`_bgsub_token_from_flags`) + options wrapper; merge
  imports the core as `_bgsub_token`. 3 impls → 1. All 53 photometry tests pass.
- [x] cataloging.py inlined uppercase MIRI tuples (2×) → `_L._instrument_from_filter`.
- [ ] TODO: product-stem f-string (15+), `output_tokens` (8×), `detector_tokens_for`.

## Phase 2 — Intent comments → tests
(before structural changes). Top: sat-proximity Star-B filter, position-avg-without-flux_err
(31k-drop bug), residual/model QA policy, MIRI prominence gate, overshoot guard,
no-silent-frame-drops + collision guard, resolve_max_group_size, _max_r_for_source tiers,
ERR/data shape-mismatch trim, _filter_to_wavelength specials.

## Phase 3 — STScI / library redundancy  [STATUS: WCS items CLOSED — not redundant]
**RESOLVED 2026-06-18 against latest origin/main.** The audit's WCS findings were wrong;
they predated `reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md` (added this morning), which
documents the WCS flow as deliberate, correct design:
- `sync_gwcs_to_fits_wcs` is NOT replaceable by `adjust_wcs`: adjust_wcs's own docstring says
  it is "not designed to handle GWCS of resampled images." STScI provides no resampled-i2d WCS
  shifter; the hand-built FITSImagingWCSTransform is the minimal supported path (<0.01 mas,
  idempotent). Doc already says "if a future jwst/gwcs ships one, replace it."
- `fix_alignment` FITS+GWCS double-write and the skipped TweakRegStep are the deliberate
  two-authoring-points / no-double-correction design.
So: NOTHING to change for WCS.

Tier 1 (WCS): CLOSED — deliberate design, see the flow doc.
- `reduction/align_to_catalogs.py:312-465` `realign_to_catalog` + `photometry/measure_offsets.py`
  + `make_reftable.py` — 3 hand-rolled tweakreg/CRVAL-shift aligners → `jwst.tweakreg` /
  `tweakwcs.fit_wcs`. Medium risk (custom cuts + diagnostics to reproduce).
- `PipelineRerunNIRCAM-LONG.py:1072-1095` & `PipelineMIRI.py:516-539` `fix_alignment` — redundant
  FITS-header WCS write after correct `adjust_wcs`+save; `check_wcs` exists only to detect the
  divergence this creates. Drop the FITS splat IF no plain-astropy/CARTA consumer reads it.

Tier 2 (mechanical, low risk):
- `crowdsource_catalogs_long.py:678-690` `_shift_gwcs` → `gwcs.WCS.__getitem__` slicing.
- `reduction/destreak.py:128-172` `add_background_map` → `reproject.reproject_interp`.
- `reduction/make_merged_psf.py:132-190` custom PSF-grid FITS IO → `GriddedPSFModel.read/.write`;
  eliminates `fix_psfs_with_bad_meta`.
- `make_starless_image.py:585-594` regex DS9 parse → `regions.Regions.read(format='ds9')`.
- `PipelineRerunF212N.py:98-128` DQ bit decomp → `stdatamodels.dqflags.dqflags_to_mnemonics`
  (skip if file deleted in Phase 0).

Tier 3 (background/stats, medium, research-tuned — careful, not bit-identical):
- `filtering.py:120-285,488-500` → `photutils.Background2D` + `MADStdBackgroundRMS`.
- `destreak.py:36-95` → `Background2D`.
- `make_starless_image.py` infill/stamp/annulus → astropy.convolution + photutils apertures.

Leave-as-is (confirmed clean): cataloging/merge coord math, psf_fitting bespoke solve,
merge_a_plus_b (reproject, ResampleStep deliberately abandoned).

## Phase 4 — Untangle MIRI vs NIRCam
Per-instrument param struct + MIRI-only hooks + agnostic core. Targets: `get_saturated_stars`
(~250 MIRI lines woven in), `get_psf_model`, `_manual_phot_pass`, `_filter_extended_emission`
(two algos in one fn), `run_manual_pipeline` MIRI tuning, satstar radii. Plus `pipeline_common.py`
for reduction (NIRCAM-LONG vs MIRI 70-80% copy-paste; `check_wcs` byte-identical). BUG: LONG
697-738 dups 662-695 (loads VVV then overwrites back) — test + fix.

## Phase 5 — Loops→vectorize, if/elif→dispatch
KDTree for O(N²) satstar matching, batch PSF grid-search, re-roll 3× unrolled isochrone loop,
dict-backed `_filter_to_wavelength`, `PHASE_SOURCES` dict, `_first_col` resolver, vstack instead
of per-row add_row.

## Phase 6 — Split mega-functions (last, needs Phase-2 tests)
`do_photometry_step` (1755) + `main` + `get_saturated_stars` → modules: cutout.py,
residual_mosaicking.py, seeding.py, postfit_filters.py, io_results.py, cli_config.py, naming.py.

---

## Risk notes
- Big files had active local edits at planning time; origin/main moved ahead (318b902). Re-verify
  WCS-related findings against current code.
- Chunked merge is verified bit-identical to serial — preserve.
- Naming consolidation must reproduce token strings exactly (tests pin some).
- Reduction config dicts are data, not duplication — don't over-merge.
