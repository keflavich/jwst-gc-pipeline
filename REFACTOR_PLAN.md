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

## Phase 4 — Untangle MIRI vs NIRCam  [STATUS: in progress]
Per-instrument param struct + MIRI-only hooks + agnostic core.
- [x] `_filter_extended_emission` (cataloging.py) two-algorithm split → `_emission_keep_miri`
  / `_emission_keep_nircam` named helpers + hoisted the duplicated overshoot-drop.
  Behavior-identical (pinned by test_cataloging_regressions.py, 4 tests, verified before+after).
- [NOTE] `accept_satstar_fit` was already factored out by the user (origin/main 3f5a20d) —
  the satstar accept-gate part of Phase 4 is DONE upstream.
- [ ] `get_saturated_stars` (~1342 lines, MIRI branches at 866-906/1045-1049/1159/1266/1535/
  1615/1931): HOLD — this is the user's hot file (actively edited this week). High conflict
  risk for a big split; coordinate timing or extract only a localized instrument-policy struct.
- [ ] `_manual_phot_pass` MIRI cleanup extraction, `run_manual_pipeline` MIRI tuning overrides.
- [ ] reduction `pipeline_common.py` (NIRCAM-LONG vs MIRI 70-80% copy-paste). NOTE: PipelineRerun
  NIRCAM-LONG was edited on latest origin/main — re-verify the LONG 697-738 double-paste bug
  still exists before fixing.

## Phase 5 — Loops→vectorize, if/elif→dispatch  [STATUS: partial]
- [x] dict-backed `plot_tools._filter_to_wavelength` (was a 5-clause nested ternary copy-pasted
  ~16× in 3 divergent variants) + test (test_plot_tools_regressions.py, 11 pass).
- [x] re-rolled the 3× unrolled isochrone-age loop in `ccds_withiso` (ages 5/7/9 → zip loop,
  matching the form already in `cmds_withiso`).
- [ ] TODO (deferred — hot files / heavier risk): KDTree for O(N²) satstar matching
  (satstar/merge), batch PSF grid-search (satstar), `PHASE_SOURCES` dict (cataloging
  run_manual_pipeline), `_first_col` column resolver (merge_catalogs), vstack instead of
  per-row add_row (merge_catalogs replace_saturated).

## Phase 6 — Split mega-functions  [STATUS: partial — safe extraction done]
- [x] Extracted the pure table column-convention resolvers (`_get_source_xy`,
  `_best_available_xy`, `_has_any_xy_columns`, `_column_to_float_array`,
  `_skycoord_radec_arrays`, `_XY_COLUMN_CANDIDATES`) from crowdsource_catalogs_long.py into
  new `photometry/column_utils.py`; the monolith re-imports them (so `_L.<name>` access from
  cataloging.py is unchanged). Behavior-neutral pure move; covered by existing tests.
- [x] First two extracted modules done: naming.py + column_utils.py.

### Phase 6 carve-up of `do_photometry_step` (1758 lines, 4525-6283) — scope: this fn only
(get_saturated_stars deferred — user's hot file). Contract fully mapped (blocks A-U, 3 method
gates: `not nocrowdsource` / `daophot` / `daophot and not basic_only`; satstar always-on but
no-op without saturated DQ). Milestones, commit each:

- [x] **M0 fixtures** (built into the integration test): synthetic GriddedPSFModel builder,
  synthetic i2d FITS (SCI+ERR+DQ, WCS, headers), minimal-options factory.
- [x] **M1 PSF centralization** (user request): new `psf_paths.py` with
  `resolve_merged_psf_grid_path` (read central→legacy, central naming keyed by
  inst+module+FILTER+oversample+blur). Wired into get_psf_model use_webbpsf=False read +
  webbpsf cache outdir default → `psfs_shared/`. 6 unit tests (test_psf_paths.py) pass; both
  edited files compile. NOTE: grid PRODUCERS (make_merged_psf.py) still write legacy paths —
  follow-up to point them at central so the disk saving actually kicks in for new grids.
- [x] **M2 characterization test** (test_do_photometry_step_integration.py): drives the
  shortest real path (daophot basic-only, unseeded; stubs get_psf_model→synthetic grid,
  load_or_make_satstar_catalog→empty, SvoFps→1-row, save_residual_datamodel→noop,
  savefig→noop) on a synthetic 3-star frame; asserts the basic catalog is written and all 3
  sources recovered <1px. ~4min (webbpsf import). This is the SAFETY NET for M3+.
- [x] **M3 block B → `_output_suffix_tokens`** (namedtuple, unpacked in field order; closes
  Phase-1 "output_tokens ×8"). 4 unit tests + M2 re-run: 5 passed.
- [x] **M4 block J else-branch → `_first_pass_daofinder`** (M2-covered). 2 unit tests + M2: 3 passed.
- [x] **M5a** expanded characterization net: iterative (basic_only=False) + seeded (seed_catalog)
  paths now covered too — 3 tests (basic/iterative/seeded) each recover 3 stars <1px.
- [x] **M5b** extract `_svo_effective_wavelength` + `_make_grouper` (block H). 3 unit + 3
  characterization: 6 passed.
- [x] **M6** extract `_subtract_satstar_model` (block K math; pure, sat-pixel→0 invariant).
  3 unit + 3 characterization: 6 passed.
- [ ] **M7+ remaining big blocks** (need new fixtures FIRST to extract safely):
  - block L seed-assembly (~5145-5470): wide interface + snap/inject branches only fire when an
    iter2 catalog exists on disk. STEP 1: add a seeded-characterization variant that writes
    `{basepath}/catalogs/{filt}_merged_indivexp_merged_iter2_daoiterative_iterative.fits` (skycoord
    col) so the snap branch executes; THEN extract → seeding.py.
  - block Q/T post-fit dedup+filters: stateful (mutates the phot object). Extract a shared
    `_apply_keep_mask_to_phot` + `_postfit_clean` → postfit_filters.py (covered by basic+iterative).
  - block O crowdsource fit: needs a nocrowdsource=False characterization test (WrappedPSFModel
    + fit_im path).

---

## Risk notes
- Big files had active local edits at planning time; origin/main moved ahead (318b902). Re-verify
  WCS-related findings against current code.
- Chunked merge is verified bit-identical to serial — preserve.
- Naming consolidation must reproduce token strings exactly (tests pin some).
- Reduction config dicts are data, not duplication — don't over-merge.
