# Cataloging / merging regression tests

Each test here pins a specific bug or defensive code path documented by a
comment block in the cataloging/merging modules, so parameter tuning and
refactors can't silently re-introduce a fixed failure. Tests build small
synthetic inputs (no real survey data, no network).

Run the whole suite:

```bash
python -m pytest jwst_gc_pipeline/photometry/tests/ -q
```

Note: importing `crowdsource_catalogs_long` / `merge_catalogs` pulls in
photutils/jwst/crowdsource and is slow (tens of seconds, cold). pytest pays
each module import once per session.

**Scope of the map below:** it lists the regression tests that pin a specific
documented comment block. It is **not** an index of the suite — the directory
carries far more tests than appear here (astrometry checkpoints, interframe
overlap, registration gate, obs-token collisions, satstar gates, …). Use
`pytest --collect-only -q` for the full list.

## Coverage map (comment block → test)

| Module | Comment / bug | Test file :: test |
| --- | --- | --- |
| crowdsource_catalogs_long.py::`_resolve_seed_skycoords` | Star B masked `(0,0)` sentinel after vstack | `test_seed_skycoord_resolution.py` (pre-existing) |
| crowdsource_catalogs_long.py::`SeededFinder` | SeededFinder must log silent finite-drops | `test_seed_skycoord_resolution.py` (pre-existing) |
| crowdsource_catalogs_long.py::`_overlap_slices_tuple_shape` | `overlap_slices` ndarray-shape + `e_max==0` ambiguous-truth crash | `test_crowdsource_long_regressions.py::TestOverlapSlicesPatch` |
| crowdsource_catalogs_long.py::`_strip_chunk` | `_chunkXXofYY` suffix stripping (`is_iter3` must still fire) | `…::TestStripChunk` |
| crowdsource_catalogs_long.py::`_resolve_seed_skycoords` (preferred column) | per-filter `skycoord_{filter}` snap must beat plain ra/dec (sickle F480M src 55) | `…::TestResolveSeedPreferredColumn` |
| crowdsource_catalogs_long.py::`_filter_near_saturation` | Star B saturation-proximity radius 5.0→1.0 px | `…::TestFilterNearSaturation` |
| crowdsource_catalogs_long.py::`forced_psf_photometry` | closed-form forced-photometry flux solve | `…::TestForcedPsfPhotometry` |
| merge_catalogs.py::`nanaverage_numpy` / `nanaverage_dask` | `nanaverage` all-zero / all-NaN-weight rows → NaN | `test_merge_catalogs_regressions.py::TestNanaverage` |
| merge_catalogs.py::`combine_singleframe` | ~31k rows lost: NaN flux_err → all-zero pos weights → NaN pos → dropped | `…::test_combine_singleframe_keeps_sources_with_nan_fluxerr` |
| merge_catalogs.py::`merge_catalogs` (`ref_filter`) | missing `ref_filter` must raise (no silent reference switch) | `…::test_merge_catalogs_missing_ref_filter_raises` |
| make_reference_from_pipeline_catalogs.py::`summarize_offsets` | `summarize_offsets` empty-input guard | `test_make_reference_regressions.py::TestSummarizeOffsets` |
| make_reference_from_pipeline_catalogs.py::`initial_spatial_photometric_match` | `initial_spatial_photometric_match` MAD==0 fallback | `…::TestInitialSpatialPhotometricMatchMadFallback` |
| make_reference_from_pipeline_catalogs.py::`_normalize_reference_ks_magnitude` | Ks-column `nanmedian` stacking, drop all-NaN sources | `…::TestNormalizeReferenceKsMagnitude` |
| crowdsource_catalogs_long.py::`resolve_max_group_size` | `--max-group-size` 0 rejected as ambiguous; `unlimited`→None; +int passthrough | `test_crowdsource_long_regressions.py::TestResolveMaxGroupSize` |
| crowdsource_catalogs_long.py::`normalize_vgroup_id` | `normalize_vgroup_id` token/int extraction, idempotent on prefixed | `…::TestNormalizeVgroupId` |
| make_starless_image.py::`_max_r_for_source` | `_max_r_for_source` SNR-tier radius cascade (strict `>`; below SNR_SKIP→0) | `test_make_starless_regressions.py::TestMaxRForSource` |
| make_starless_image.py::`nan_gaussian` | `nan_gaussian` infill across NaN holes; far-from-data weight<0.01 → NaN not 0 | `…::TestNanGaussian` |
| cataloging.py:`_filter_extended_emission` | NIRCam keep = star_like AND SNR cut (`_emission_keep_nircam`) | `test_cataloging_regressions.py::TestFilterExtendedEmissionNircam` |
| cataloging.py:`_filter_extended_emission` | MIRI keep = deep-i2d prominence ALONE (star_like/SNR bypassed); NaN prominence (off-i2d) dropped (`_emission_keep_miri`) | `…::TestFilterExtendedEmissionMiri` |
| plot_tools.py:`_filter_to_wavelength` | filter/color name → effective wavelength: special names (410m405/405m410/182m187/187m182/Hmag/Ksmag) + generic F⟨NNN⟩→NNN/100 um | `test_plot_tools_regressions.py` |
| psf_paths.py | centralized PSF-grid path resolver: central→legacy read order, central key = physics only | `test_psf_paths.py` |
| legacy/crowdsource_step.py:`do_photometry_step` (blocks A-U) | END-TO-END characterization (Phase-6 safety net): basic / iterative / seeded paths each recover 3 injected stars <1px on a synthetic frame (4 external seams stubbed) | `test_do_photometry_step_integration.py` |
| legacy/crowdsource_step.py:`_output_suffix_tokens` | filename-suffix tokens (block B), field-order locked | `test_crowdsource_long_regressions.py::TestOutputSuffixTokens` |
| legacy/crowdsource_step.py:`_first_pass_daofinder` | iter1 DAO threshold = nsigma·min(median err, mad_std data) (block J else) | `…::TestFirstPassDaofinder` |
| cataloging.py:`_filter_extended_emission` | SKY-CLEAN keep tier: on emission-free sky (local 25th-pct annulus floor ≈ dark-sky ref) keep on prominence+S/N, qfit ignored; inert on emission; satstar-proximity guard | `test_sky_clean_keep.py` |
| cataloging.py:`_build_source_masked_bg` | bg must mask the UNION of vetted catalog + i2d detection seed (a vetted-dropped real star must not be absorbed into the bg) | `test_bg_masks_unmodeled_seed_star.py` |
| cataloging.py:`_build_source_masked_bg` | end-to-end: transiently-dropped blended star survives 7 subtract-bg→re-detect iterations with seed masking (ngc6334 F405N loss cycle) | `test_iterative_bg_preserves_blended_star.py` |

## Deferred (need full PSFPhotometry/IterativePSFPhotometry integration fixtures)

These regressions live inside the ~1000-line `do_photometry_step` and other
I/O-heavy paths. They can't be unit-tested without running real PSF
photometry on a synthetic image; documented here so they aren't forgotten.
Good next step: build a tiny synthetic GriddedPSFModel + injected-star image
fixture and drive `do_photometry_step` end-to-end.

- **V12 iter2 source injection, Modes A & B** (the seed-merge + dedup stage in `crowdsource_catalogs_long.py`):
  faint target near a bright neighbour must survive pre-fit dedup as an
  injected iter2 seed (Mode A) or when the union seed is just outside the
  V11 snap radius (Mode B). Needs union + iter2 seed tables driven through
  the seed-merge + dedup stage.
- **V12 satstar snap-to-iter2** (the satstar force-union snap in `crowdsource_catalogs_long.py`):
  force-union satstar entries at SW positions must be snapped to the
  per-filter iter2 position so dedup doesn't keep the unsnapped one.
- **Post-fit dedup thresholds** (the post-fit dedup radius in
  `crowdsource_catalogs_long.py`): 0.5→1.5 (2026-04-24, deep negative residuals) then →1.0 px
  (V13, preserve real adjacent stars 3.32 px apart). Needs two-PSF synthetic
  image + residual-percentile assertion.
- **iter3 `xy_bounds` / finder settings** (the iter3 finder settings in `crowdsource_catalogs_long.py`):
  seed-dominated iter3 must not wander far from seeded positions.
- **`save_photutils_results` filename token** (`crowdsource_catalogs_long.py::save_photutils_results`):
  must emit `{module}` not `{module}{detector}` (no `nrcbnrcb` doubling).
  Buried in a writer that needs a full result table + WCS + im1.
- **`replace_saturated` many-to-one** (`merge_catalogs.py::replace_saturated`): multiple
  catalog rows can share one saturated star; needs an on-disk satstar
  catalog + fwhm table + SvoFps zeropoints.
- **Memory two-phase refactor** (`merge_catalogs.py::combine_singleframe`) and
  **SeededFinder vectorised-skycoord perf**
  (`crowdsource_catalogs_long.py::SeededFinder`): these
  are performance/scaling properties, not correctness assertions.
